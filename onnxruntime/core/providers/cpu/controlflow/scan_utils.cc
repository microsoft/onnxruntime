// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "core/providers/cpu/controlflow/scan_utils.h"

#include "gsl/gsl_algorithm"

#include "core/framework/mldata_type_utils.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace scan {
namespace detail {

void ReadDirections(const OpKernelInfo& info, const std::string& attr_name,
                    std::vector<int64_t>& directions, int64_t num_entries) {
  if (info.GetAttrs<int64_t>(attr_name, directions).IsOK()) {
    ORT_ENFORCE(num_entries < 0 || gsl::narrow_cast<int64_t>(directions.size()) == num_entries,
                "Number of entries in '", attr_name, "' was ", directions.size(),
                " but expected ", num_entries);

    bool valid = std::all_of(directions.cbegin(), directions.cend(),
                             [](int64_t i) { return static_cast<ScanDirection>(i) == ScanDirection::kForward ||
                                                    static_cast<ScanDirection>(i) == ScanDirection::kReverse; });
    ORT_ENFORCE(valid, "Invalid values in '", attr_name, "'. 0 == forward. 1 == reverse.");
  } else {
    // default to forward if we know how many entries there should be
    directions = std::vector<int64_t>(num_entries, static_cast<int64_t>(ScanDirection::kForward));
  }
}

Status AllocateOutput(OpKernelContextInternal& context, const GraphViewer& subgraph,
                      int output_index, bool is_loop_state_var, int64_t batch_size, int64_t sequence_len,
                      std::unique_ptr<OutputIterator>& output_iterator, ScanDirection direction,
                      bool temporary) {
  // use the shape from the subgraph output. we require this to be specified in the model or inferable.
  auto& graph_outputs = subgraph.GetOutputs();
  auto* graph_output = graph_outputs.at(output_index);
  auto* graph_output_shape = graph_output->Shape();

  if (!graph_output_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph must have the shape set for all outputs but ",
                           graph_output->Name(), " did not.");
  }

  TensorShape output_shape{onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape)};
  auto& graph_output_dims{output_shape.GetDims()};

  std::vector<int64_t> scan_output_dims;
  scan_output_dims.reserve(graph_output_dims.size() + 2);

  // v8 has batch size. v9 and later do not.
  bool is_v8 = batch_size > 0;

  if (is_v8) {
    scan_output_dims.push_back(batch_size);
  }

  if (!is_loop_state_var) {
    scan_output_dims.push_back(sequence_len);
  }

  scan_output_dims.insert(scan_output_dims.cend(), graph_output_dims.cbegin(), graph_output_dims.cend());

  if (!temporary) {
    OutputIterator::Create(context, output_index, is_loop_state_var, is_v8, TensorShape(scan_output_dims),
                           output_iterator, direction);
  } else {
    auto mltype = utils::GetMLDataType(*graph_output);

    // the outputs from Scan are constrained to tensors, so we can safely cast to TensorTypeBase
    auto ml_data_type = static_cast<const TensorTypeBase*>(mltype)->GetElementType();

    OutputIterator::Create(context, output_index, is_loop_state_var, is_v8, TensorShape(scan_output_dims),
                           output_iterator, direction, temporary, ml_data_type);
  }

  return Status::OK();
}

Status CreateFeedsFetchesManager(const GraphViewer& subgraph,
                                 int num_variadic_inputs,
                                 std::unordered_map<std::string, const MLValue*>& implicit_inputs,
                                 std::vector<std::string>& subgraph_output_names,
                                 const MLValueNameIdxMap& mlvalue_name_idx_map,
                                 std::unique_ptr<FeedsFetchesManager>& ffm) {
  auto* graph_inputs = &subgraph.GetInputsIncludingInitializers();
  if (static_cast<size_t>(num_variadic_inputs) < graph_inputs->size()) {
    // fallback to just the required inputs.
    graph_inputs = &subgraph.GetInputs();
    ORT_ENFORCE(static_cast<size_t>(num_variadic_inputs) == graph_inputs->size(),
                "Graph::InferAndVerifySubgraphTypes should have already validated that "
                "num_variadic_inputs matched the subgraph inputs or required inputs.");
  }

  auto num_implicit_inputs = implicit_inputs.size();
  auto num_inputs = num_variadic_inputs + num_implicit_inputs;

  std::vector<std::string> feed_names;
  feed_names.reserve(num_inputs);

  // pass explicit graph inputs first. order doesn't actually matter though
  for (int input = 0; input < num_variadic_inputs; ++input) {
    feed_names.push_back((*graph_inputs)[input]->Name());
  }

  for (auto& entry : implicit_inputs) {
    feed_names.push_back(entry.first);
  }

  FeedsFetchesInfo ffi(feed_names, subgraph_output_names);
  auto status = FeedsFetchesManager::Create(feed_names, subgraph_output_names, mlvalue_name_idx_map, ffm);

  return status;
}

Status IterateSequence(OpKernelContextInternal& context,
                       const SessionState& session_state,
                       std::vector<LoopStateVariable>& loop_state_variables,
                       std::vector<MLValueTensorSlicer<const MLValue>::Iterator>& scan_input_stream_iterators,
                       int64_t seq_length,
                       int num_loop_state_variables,
                       int num_variadic_inputs,
                       int num_variadic_outputs,
                       std::unordered_map<std::string, const MLValue*>& implicit_inputs,
                       std::vector<std::unique_ptr<OutputIterator>>& output_iterators,
                       FeedsFetchesManager* ffm,
                       const FeedsFetchesManager* cached_ffm) {
  Status status = Status::OK();

  auto num_implicit_inputs = implicit_inputs.size();
  auto num_inputs = num_variadic_inputs + num_implicit_inputs;

  std::vector<MLValue> feeds;
  std::vector<MLValue> fetches;
  std::unordered_map<size_t, IExecutor::CustomAllocator> fetch_allocators;

  feeds.resize(num_inputs);
  fetches.resize(num_variadic_outputs);

  // add implicit inputs and pass in implicit inputs as feeds. we're going to pass in the explicit inputs
  // first in each iteration though so offset by num_variadic_inputs
  int i = 0;
  for (auto& entry : implicit_inputs) {
    ORT_ENFORCE(entry.second, "All implicit inputs should have MLValue instances by now. ", entry.first, " did not.");
    feeds[num_variadic_inputs + i] = *entry.second;
    ++i;
  }

  int64_t seq_no = 0;
  for (; seq_no < seq_length; ++seq_no) {
    for (int input = 0; input < num_variadic_inputs; ++input) {
      if (input < num_loop_state_variables) {
        // add loop state variable input
        feeds[input] = loop_state_variables[input].Input();
      } else {
        // add sliced input
        auto& iterator = scan_input_stream_iterators[input - num_loop_state_variables];
        feeds[input] = *iterator;

        ++iterator;
      }
    }

    fetches.clear();

    for (int output = 0, end = num_variadic_outputs; output < end; ++output) {
      if (output < num_loop_state_variables) {
        // add loop state variable output
        fetches.push_back(loop_state_variables[output].Output());
      } else {
        auto& iterator = *output_iterators[output];

        if (iterator.FinalOutputAllocated()) {
          // add MLValue from sliced output
          auto& mlvalue = *iterator;
          fetches.push_back(mlvalue);
        } else {
          // use a custom allocator that will forward the allocation request to the Scan context
          // and add the sequence length dimension. this avoids using a temporary value for the first output
          fetch_allocators[output] =
              [&iterator](const TensorShape& shape, MLValue& mlvalue) {
                return iterator.AllocateSubgraphOutput(shape, mlvalue);
              };

          // also need a dummy empty entry in fetches so the order matches the output names
          fetches.push_back({});
        }
      }
    }

    // Create Executor and run graph.
    if (cached_ffm) {
      status = utils::ExecuteGraphWithCachedInfo(session_state, *cached_ffm, feeds, fetches, fetch_allocators,
                                                 /*sequential_execution*/ true, context.GetTerminateFlag(),
                                                 context.Logger());
    } else {
      status = utils::ExecuteGraph(session_state, *ffm, feeds, fetches, fetch_allocators,
                                   /*sequential_execution*/ true, context.GetTerminateFlag(), context.Logger(),
                                   /*cache_copy_info*/ true);
      // we can now use the cached info
      cached_ffm = ffm;
    }

    ORT_RETURN_IF_ERROR(status);

    // cycle the LoopStateVariable input/output in preparation for the next iteration
    std::for_each(loop_state_variables.begin(), loop_state_variables.end(), [](LoopStateVariable& v) { v.Next(); });

    // and move the output iterators.
    for (int output = num_loop_state_variables; output < num_variadic_outputs; ++output) {
      ++(*output_iterators[output]);
    }

    if (seq_no == 0) {
      // we only ever use custom allocators on the first iteration as the final output is always allocated during that
      fetch_allocators.clear();
    }
  }

  return status;
}

MLValue AllocateTensorInMLValue(const MLDataType data_type, const TensorShape& shape, AllocatorPtr& allocator) {
  auto new_tensor = std::make_unique<Tensor>(data_type,
                                             shape,
                                             allocator);

  return MLValue{new_tensor.release(),
                 DataTypeImpl::GetType<Tensor>(),
                 DataTypeImpl::GetType<Tensor>()->GetDeleteFunc()};
};

void CalculateTransposedShapeForInput(const TensorShape& original_shape, int64_t axis,
                                      std::vector<int64_t>& permutations, std::vector<int64_t>& transposed_shape) {
  int64_t rank = original_shape.NumDimensions();
  const auto& dims = original_shape.GetDims();

  permutations.reserve(rank);
  permutations.push_back(axis);

  transposed_shape.reserve(rank);
  transposed_shape.push_back(dims[axis]);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != axis) {
      permutations.push_back(i);
      transposed_shape.push_back(dims[i]);
    }
  }
}

void CalculateTransposedShapeForOutput(const TensorShape& original_shape, int64_t axis,
                                       std::vector<int64_t>& permutations, std::vector<int64_t>& transposed_shape) {
  int64_t rank = original_shape.NumDimensions();
  const auto& dims = original_shape.GetDims();

  permutations.reserve(rank);
  transposed_shape.reserve(rank);

  for (int64_t i = 1; i <= axis; ++i) {
    permutations.push_back(i);
    transposed_shape.push_back(dims[i]);
  }

  permutations.push_back(0);
  transposed_shape.push_back(dims[0]);

  for (int64_t i = axis + 1; i < rank; ++i) {
    permutations.push_back(i);
    transposed_shape.push_back(dims[i]);
  }
}

LoopStateVariable::LoopStateVariable(const MLValue& original_value,
                                     MLValue& final_value,
                                     const int64_t sequence_len,
                                     AllocatorPtr& allocator)
    : sequence_len_{sequence_len},
      original_value_{original_value},
      final_value_{final_value} {
  auto& tensor = original_value.Get<Tensor>();
  auto& shape = tensor.Shape();

  // allocate a new Tensor in an MLValue with the same shape and type as the tensor in original_value.
  // the Tensor will own the buffer, and the MLValue will own the Tensor.
  // the MLValue returned by Input()/Output() gets copied into the execution frame feeds/fetches
  // with the Tensor being used via a shared_ptr (so remains valid during execution and is cleaned up
  // automatically at the end).
  // TODO: Could allocate one large chunk for all the loop state variable buffers in ScanImpl, although that
  // may make it harder to parallelize processing of the batch in the future.

  // if length is > 1, we need a_ for the first output location. otherwise we use final_value for the output.
  if (sequence_len_ > 1) {
    a_ = AllocateTensorInMLValue(tensor.DataType(), shape, allocator);
  }

  // if length is > 2, we need b_ for the second output location
  if (sequence_len_ > 2) {
    b_ = AllocateTensorInMLValue(tensor.DataType(), shape, allocator);
  }
}

const MLValue& LoopStateVariable::Input() const {
  if (iteration_num_ == 0)
    return original_value_;

  return iteration_num_ % 2 == 1 ? a_ : b_;
}

MLValue& LoopStateVariable::Output() {
  if (iteration_num_ + 1 == sequence_len_) {
    return final_value_;
  }

  return iteration_num_ % 2 == 1 ? b_ : a_;
}

void LoopStateVariable::Next() {
  ORT_ENFORCE(iteration_num_ < sequence_len_, "Misuse of LoopStateVariable. Attempt to move beyond end of sequence");
  ++iteration_num_;
}

// fill in a symbolic dimension in the overall output using the output shape from an iteration of the subgraph
static Status MakeShapeConcrete(const TensorShape& per_iteration_shape, TensorShape& final_shape) {
  auto num_dims_per_iteration = per_iteration_shape.NumDimensions();
  auto final_shape_offset = final_shape.NumDimensions() - num_dims_per_iteration;
  for (size_t i = 0; i < num_dims_per_iteration; ++i) {
    auto existing_value = final_shape[i + final_shape_offset];
    if (existing_value == -1) {
      final_shape[i + final_shape_offset] = per_iteration_shape[i];
    } else {
      if (existing_value != per_iteration_shape[i]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Mismatch between expected shape and shape from first output",
                               final_shape, " is not compatible with ", per_iteration_shape);
      }
    }
  }

  return Status::OK();
}

OutputIterator::OutputIterator(OpKernelContextInternal& context,
                               int output_index,
                               bool is_loop_state_var,
                               bool is_v8,
                               TensorShape final_shape,
                               ScanDirection direction,
                               bool temporary,
                               MLDataType data_type)
    : context_{context},
      is_v8_{is_v8},
      output_index_{output_index},
      final_shape_{final_shape},
      is_loop_state_var_{is_loop_state_var},
      direction_{direction},
      cur_iteration_{0},
      temporary_{temporary},
      data_type_{data_type} {
  is_concrete_shape_ = final_shape_.Size() >= 0;

  if (is_v8) {
    // there are one or two dimensions being iterated depending on whether it's a loop state variable or scan input.
    auto num_iteration_dims = is_loop_state_var_ ? 1 : 2;
    num_iterations_ = final_shape_.Slice(0, num_iteration_dims).Size();
  } else {
    // batch dimension is not handled in v9 and later so for a loop state var there are no iterations, and for
    // the scan outputs we use dimension 0 which is the sequence length.
    if (is_loop_state_var)
      num_iterations_ = 1;
    else
      num_iterations_ = final_shape_[0];
  }
}

Status OutputIterator::Initialize() {
  Status status = Status::OK();

  if (is_loop_state_var_ && !is_concrete_shape_) {
    // copy the shape from the input initial value which will have a concrete shape.
    // +1 to skip the sequence_len input if v8
    auto* input = context_.Input<Tensor>(is_v8_ ? output_index_ + 1 : output_index_);
    status = MakeShapeConcrete(input->Shape(), final_shape_);
    ORT_RETURN_IF_ERROR(status);

    is_concrete_shape_ = true;
  }

  if (is_concrete_shape_) {
    status = AllocateFinalBuffer();
    ORT_RETURN_IF_ERROR(status);
  } else {
    // delay until the first subgraph execution calls AllocateSubgraphOutput.
  }

  return Status::OK();
}

Status OutputIterator::AllocateFinalBuffer() {
  // make sure a single buffer for the full output is created upfront.
  // we slice this into per-iteration pieces using MLValueTensorSlicer.
  if (!temporary_) {
    // we can write directly to the Scan output
    auto* tensor = context_.Output(output_index_, final_shape_);

    if (!tensor) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for output #", output_index_);
    }

    // get the output tensor we just created as an MLValue
    final_output_mlvalue_ = context_.GetOutputMLValue(output_index_);
  } else {
    // we need to do a transpose at the end so need to write to a temporary buffer when executing the subgraph.
    AllocatorPtr alloc;
    auto status = context_.GetTempSpaceAllocator(&alloc);
    ORT_RETURN_IF_ERROR(status);

    temporary_final_output_mlvalue_ = AllocateTensorInMLValue(data_type_, final_shape_, alloc);
    final_output_mlvalue_ = &temporary_final_output_mlvalue_;
  }

  // if it's v8 there's always a batch size dimension so we need a slicer to hide that from each iteration
  if (is_v8_) {
    if (is_loop_state_var_) {
      // only one entry is required as we slice on a single dimension
      slicer_iterators_.push_back((direction_ == ScanDirection::kForward)
                                      ? MLValueTensorSlicer<MLValue>::Create(*final_output_mlvalue_).begin()
                                      : MLValueTensorSlicer<MLValue>::Create(*final_output_mlvalue_).rbegin());
    } else {
      auto batch_size = final_shape_[0];
      for (int i = 0; i < batch_size; ++i) {
        // the slicer handles the sequence dimension (dim 1) so create an entry for each batch
        slicer_iterators_.push_back((direction_ == ScanDirection::kForward)
                                        ? MLValueTensorSlicer<MLValue>::Create(*final_output_mlvalue_, 1, i).begin()
                                        : MLValueTensorSlicer<MLValue>::Create(*final_output_mlvalue_, 1, i).rbegin());
      }
    }

    cur_slicer_iterator_ = slicer_iterators_.begin();
  } else {
    // nothing to slice for a loop state var. slice on dimension 0 (sequence) for the scan outputs.
    if (!is_loop_state_var_) {
      slicer_iterators_.push_back((direction_ == ScanDirection::kForward)
                                      ? MLValueTensorSlicer<MLValue>::Create(*final_output_mlvalue_).begin()
                                      : MLValueTensorSlicer<MLValue>::Create(*final_output_mlvalue_).rbegin());
      cur_slicer_iterator_ = slicer_iterators_.begin();
    }
  }

  return Status::OK();
}

Status OutputIterator::AllocateSubgraphOutput(const TensorShape& shape, MLValue& mlvalue) {
  ORT_ENFORCE(!is_concrete_shape_, "If shape was concrete we shouldn't be using a custom allocator");

  // update the final shape now that we can fill in the symbolic dimension with an actual value
  auto status = MakeShapeConcrete(shape, final_shape_);
  ORT_RETURN_IF_ERROR(status);

  is_concrete_shape_ = true;
  status = AllocateFinalBuffer();
  ORT_RETURN_IF_ERROR(status);

  // get MLValue from operator*()
  mlvalue = **this;

  return Status::OK();
}

MLValue& OutputIterator::operator*() {
  ORT_ENFORCE(cur_iteration_ < num_iterations_);
  ORT_ENFORCE(is_concrete_shape_,
              "Expected AllocateSubgraphOutput to have been called to before we read the MLValue from the iterator.");

  // for v8 both outputs and loop state vars use slicers. for v9 only outputs do
  if (is_v8_ || !is_loop_state_var_)
    return **cur_slicer_iterator_;
  else
    return *final_output_mlvalue_;
}

OutputIterator& OutputIterator::operator++() {
  if (cur_iteration_ < num_iterations_) {
    ORT_ENFORCE(is_concrete_shape_,
                "Expected AllocateSubgraphOutput to have been called to before we increment the iterator");

    ++cur_iteration_;

    if (is_v8_) {
      // if not a loop state var, see if we just finished the current sequence (dim 1) and need to move to the
      // next iterator. otherwise increment the current one
      if (!is_loop_state_var_ && cur_iteration_ % final_shape_[1] == 0) {
        ++cur_slicer_iterator_;
      } else {
        ++(*cur_slicer_iterator_);
      }
    } else if (!is_loop_state_var_) {
      // v9 output uses iterator (v9 loop state vars do not)
      ++(*cur_slicer_iterator_);
    }
  }

  return *this;
}

}  // namespace detail
}  // namespace scan
}  // namespace onnxruntime

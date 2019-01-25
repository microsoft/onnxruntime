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

Status IterateSequence(OpKernelContextInternal& context,
                       const SessionState& session_state,
                       const GraphViewer& subgraph,
                       std::vector<LoopStateVariable>& loop_state_variables,
                       std::vector<MLValueTensorSlicer<const MLValue>::Iterator>& scan_input_stream_iterators,
                       int64_t seq_length,
                       int num_loop_state_variables,
                       int num_variadic_inputs,
                       int num_variadic_outputs,
                       std::unordered_map<std::string, const MLValue*>& implicit_inputs,
                       std::vector<std::string>& subgraph_output_names,
                       std::vector<std::unique_ptr<OutputIterator>>& output_iterators) {
  Status status = Status::OK();

  // prefer matching all inputs to the subgraph as per the Scan spec,
  auto* graph_inputs = &subgraph.GetInputsIncludingInitializers();
  if (static_cast<size_t>(num_variadic_inputs) < graph_inputs->size()) {
    // fallback to just the required inputs.
    graph_inputs = &subgraph.GetInputs();
    ORT_ENFORCE(static_cast<size_t>(num_variadic_inputs) == graph_inputs->size(),
                "Graph::InferAndVerifySubgraphTypes should have already validated that "
                "num_variadic_inputs matched the subgraph inputs or required inputs.");
  }

  NameMLValMap feeds;
  std::vector<MLValue> fetches;
  feeds.reserve(num_variadic_inputs + implicit_inputs.size());
  fetches.resize(num_variadic_outputs);

  // pass in implicit inputs as feeds.
  for (auto& entry : implicit_inputs) {
    ORT_ENFORCE(entry.second, "All implicit inputs should have MLValue instances by now. ", entry.first, " did not.");
    feeds[entry.first] = *entry.second;
  }

  int64_t seq_no = 0;
  for (; seq_no < seq_length; ++seq_no) {
    for (int input = 0; input < num_variadic_inputs; ++input) {
      // the ordering of the Scan inputs should match the ordering of the subgraph inputs
      auto name = (*graph_inputs)[input]->Name();

      if (input < num_loop_state_variables) {
        // add loop state variable input
        feeds[name] = loop_state_variables[input].Input();
      } else {
        // add sliced input
        auto& iterator = scan_input_stream_iterators[input - num_loop_state_variables];
        feeds[name] = *iterator;

        ++iterator;
      }
    }

    fetches.clear();

    // one or more outputs have symbolic dimensions and need the first fetch to be copied to the OutputIterator
    bool have_symbolic_dim_in_output = false;

    for (int output = 0, end = num_variadic_outputs; output < end; ++output) {
      if (output < num_loop_state_variables) {
        // add loop state variable output
        fetches.push_back(loop_state_variables[output].Output());
      } else {
        // add MLValue from sliced output
        auto& iterator = *output_iterators[output];
        auto& mlvalue = *iterator;
        fetches.push_back(mlvalue);

        // mlvalue.IsAllocated will be false when the OutputIterator is using a temporary MLValue
        // and not the overall output buffer.
        have_symbolic_dim_in_output = seq_no == 0 &&
                                      (mlvalue.IsAllocated() == false ||
                                       have_symbolic_dim_in_output);  // don't unset
      }
    }

    // Create Executor and run graph.
    // TODO: Consider pulling ExecutionFrame up from within SequentialExecutor
    // and separating it out a bit so we can maybe just update the feeds/fetches in the frame on each iteration.
    // Many of the other pieces are constant across usages.
    // Not sure how best to handle the memory pattern side of things though.
    // For now just making it work. Optimization and refinement will follow.
    //SequentialExecutor executor{context.GetTerminateFlag()};
    //status = executor.Execute(session_state, feeds, subgraph_output_names, fetches, context.Logger());
    //ORT_RETURN_IF_ERROR(status);

    status = utils::ExecuteGraph(session_state, feeds, subgraph_output_names, fetches, /*sequential_execution*/ true,
                                 context.GetTerminateFlag(), context.Logger());
    ORT_RETURN_IF_ERROR(status);

    // cycle the LoopStateVariable input/output in preparation for the next iteration
    std::for_each(loop_state_variables.begin(), loop_state_variables.end(), [](LoopStateVariable& v) { v.Next(); });

    // and move the output iterators.
    for (int output = num_loop_state_variables; output < num_variadic_outputs; ++output) {
      auto& iterator = *output_iterators[output];

      // copy data from the fetch to the iterator so it can setup the overall output when the iterator is incremented.
      // if the iterator is already using the overall output buffer IsAllocated() will be true and no copy is required.
      if (have_symbolic_dim_in_output && (*iterator).IsAllocated() == false) {
        *iterator = fetches[output];
      }

      ++iterator;
    }
  }

  return status;
}

MLValue AllocateTensorInMLValue(const MLDataType data_type, const TensorShape& shape, AllocatorPtr& allocator) {
  auto new_tensor = std::make_unique<Tensor>(data_type,
                                             shape,
                                             allocator->Alloc(shape.Size() * data_type->Size()),
                                             allocator->Info(),
                                             allocator);

  return MLValue{new_tensor.release(),
                 DataTypeImpl::GetType<Tensor>(),
                 DataTypeImpl::GetType<Tensor>()->GetDeleteFunc()};
};

void CalculateTransposedShape(const TensorShape& input_shape, int64_t axis,
                              std::vector<int64_t>& permutations, std::vector<int64_t>& output_shape) {
  int64_t rank = input_shape.NumDimensions();
  const auto& dims = input_shape.GetDims();

  permutations.reserve(rank);
  permutations.push_back(axis);

  output_shape.reserve(rank);
  output_shape.push_back(dims[axis]);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != axis) {
      permutations.push_back(i);
      output_shape.push_back(dims[i]);
    }
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
    // use first_output_
  }

  return Status::OK();
}

Status OutputIterator::AllocateFinalBuffer() {
  // make sure a single buffer for the full output is created upfront.
  // we slice this into per-iteration pieces in Execute using MLValueTensorSlicer.
  // get the output tensor we just created as an MLValue
  if (!temporary_) {
    // we can write directly to the Scan output
    auto* tensor = context_.Output(output_index_, final_shape_);

    if (!tensor) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for output #", output_index_);
    }

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

Status OutputIterator::MakeConcrete() {
  ORT_ENFORCE(first_output_.IsAllocated(), "First usage of OutputIterator did not result in any output.");
  Status status = Status::OK();

  auto& tensor = first_output_.Get<Tensor>();
  auto& tensor_shape = tensor.Shape();

  // update the final shape
  status = MakeShapeConcrete(tensor_shape, final_shape_);
  ORT_RETURN_IF_ERROR(status);

  is_concrete_shape_ = true;
  status = AllocateFinalBuffer();
  ORT_RETURN_IF_ERROR(status);

  // copy first output to final buffer
  auto input_span = gsl::make_span<const gsl::byte>(static_cast<const gsl::byte*>(tensor.DataRaw()), tensor.Size());

  auto output = (**this).GetMutable<Tensor>();
  auto output_span = gsl::make_span<gsl::byte>(static_cast<gsl::byte*>(output->MutableDataRaw()), output->Size());

  gsl::copy(input_span, output_span);

  // release the MLValue we used for the first output
  first_output_ = {};

  return status;
}

MLValue& OutputIterator::operator*() {
  ORT_ENFORCE(cur_iteration_ < num_iterations_);

  if (is_concrete_shape_)
    // for v8 both outputs and loop state vars use slicers. for v9 only outputs do
    if (is_v8_ || !is_loop_state_var_)
      return **cur_slicer_iterator_;
    else
      return *final_output_mlvalue_;
  else
    return first_output_;
}

OutputIterator& OutputIterator::operator++() {
  if (cur_iteration_ < num_iterations_) {
    if (!is_concrete_shape_) {
      // we should have an output now, so convert to using the overall output buffer and slicers
      auto status = MakeConcrete();
      ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    }

    ++cur_iteration_;

    if (is_v8_) {
      // if not a loop state var, see if we just finished the current sequence (dim 1)
      if (!is_loop_state_var_ && cur_iteration_ % final_shape_[1] == 0) {
        ++cur_slicer_iterator_;
      } else {
        ++(*cur_slicer_iterator_);
      }
    } else if (!is_loop_state_var_) {
      // v9 output uses iterator
      ++(*cur_slicer_iterator_);
    }
  }

  return *this;
}

}  // namespace detail
}  // namespace scan
}  // namespace onnxruntime

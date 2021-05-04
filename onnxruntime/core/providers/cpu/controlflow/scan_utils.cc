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

#include "gsl/gsl"

#include "core/framework/mldata_type_utils.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/framework/session_options.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace scan {
namespace detail {

Info::Info(const Node& node, const GraphViewer& subgraph_in, int num_scan_inputs_in, bool is_v8)
    : subgraph(subgraph_in), num_scan_inputs(num_scan_inputs_in) {
  num_inputs = static_cast<int>(node.InputDefs().size());
  num_variadic_inputs = is_v8 ? num_inputs - 1 : num_inputs;  // allow for sequence_lens input in v8
  num_loop_state_variables = num_variadic_inputs - num_scan_inputs;

  num_outputs = static_cast<int>(node.OutputDefs().size());
  num_scan_outputs = num_outputs - num_loop_state_variables;

  num_implicit_inputs = static_cast<int>(node.ImplicitInputDefs().size());

  auto& graph_inputs = subgraph.GetInputs();
  auto num_subgraph_inputs = static_cast<int>(graph_inputs.size());
  ORT_ENFORCE(num_variadic_inputs == num_subgraph_inputs,
              "The subgraph in 'body' requires ", num_subgraph_inputs,
              " inputs but Scan was only given ", num_variadic_inputs);

  subgraph_input_names.reserve(num_inputs);
  subgraph_output_names.reserve(num_outputs);
  for (const auto& input : graph_inputs) {
    subgraph_input_names.push_back(input->Name());
  }

  for (const auto& output : subgraph.GetOutputs()) {
    subgraph_output_names.push_back(output->Name());
  }
}

void ReadDirections(const OpKernelInfo& info, const std::string& attr_name,
                    std::vector<int64_t>& directions, size_t num_entries) {
  if (info.GetAttrs<int64_t>(attr_name, directions).IsOK()) {
    ORT_ENFORCE(directions.size() == num_entries,
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
                      std::unique_ptr<OutputIterator>& output_iterator,
                      const scan::detail::DeviceHelpers::CreateMutableSlicer& create_slicer_func,
                      const scan::detail::DeviceHelpers::ZeroData& zero_data_func,
                      ScanDirection direction,
                      bool temporary) {
  // use the shape from the subgraph output. we require this to be specified in the model or inferable.
  auto& graph_outputs = subgraph.GetOutputs();
  auto* graph_output = graph_outputs.at(output_index);
  auto* graph_output_shape = graph_output->Shape();

  if (!graph_output_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph must have the shape set for all outputs but ",
                           graph_output->Name(), " did not.");
  }

  TensorShape output_shape = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape);
  auto& graph_output_dims(output_shape.GetDims());

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

  std::copy(graph_output_dims.cbegin(), graph_output_dims.cend(), std::back_inserter(scan_output_dims));

  if (!temporary) {
    OutputIterator::Create(context, output_index, is_loop_state_var, is_v8, TensorShape(scan_output_dims),
                           create_slicer_func, zero_data_func,
                           output_iterator, direction);
  } else {
    auto mltype = utils::GetMLDataType(*graph_output);

    // the outputs from Scan are constrained to tensors, so we can safely cast to TensorTypeBase
    auto ml_data_type = static_cast<const TensorTypeBase*>(mltype)->GetElementType();

    OutputIterator::Create(context, output_index, is_loop_state_var, is_v8, TensorShape(scan_output_dims),
                           create_slicer_func, zero_data_func,
                           output_iterator, direction, temporary, ml_data_type);
  }

  return Status::OK();
}

Status CreateFeedsFetchesManager(const Node& node,
                                 const Info& info,
                                 const SessionState& session_state,
                                 const SessionState& subgraph_session_state,
                                 bool is_v8,
                                 std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager) {
  // we need the names of the Scan inputs to determine what device they are available on,
  // so first create a list using those value
  std::vector<std::string> feed_names;
  feed_names.reserve(info.num_variadic_inputs + info.num_implicit_inputs);

  const auto& scan_inputs = node.InputDefs();
  int start = is_v8 ? 1 : 0;  // skip sequence_lens for v8
  for (int i = start; i < info.num_inputs; ++i) {
    feed_names.push_back(scan_inputs[i]->Name());
  }

  for (auto& entry : node.ImplicitInputDefs()) {
    feed_names.push_back(entry->Name());
  }

  // find locations. use session_state as they're coming from Scan inputs
  std::vector<OrtDevice> feed_locations;
  ORT_RETURN_IF_ERROR(controlflow::detail::FindDevicesForValues(session_state, feed_names, feed_locations));

  // now update the feed names to use the subgraph input names so we know what devices they're needed on
  for (int i = 0; i < info.num_variadic_inputs; ++i) {
    feed_names[i] = info.subgraph_input_names[i];
  }

  std::unique_ptr<FeedsFetchesManager> ffm;
  ORT_RETURN_IF_ERROR(FeedsFetchesManager::Create(feed_names, info.subgraph_output_names,
                                                  subgraph_session_state.GetOrtValueNameIdxMap(), ffm));
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *ffm));

  // we provide fetches using memory allocated by Scan, so provide locations based on the Scan output locations
  std::vector<const OrtMemoryInfo*> fetch_locations;
  fetch_locations.reserve(info.num_outputs);

  for (const auto& output : node.OutputDefs()) {
    const auto& alloc_info = utils::FindMemoryInfoForValue(session_state, output->Name());
    fetch_locations.push_back(&alloc_info);
  }

  utils::FinalizeFeedFetchCopyInfo(*ffm, feed_locations, fetch_locations);

  feeds_fetches_manager = std::move(ffm);

  return Status::OK();
}

Status IterateSequence(OpKernelContextInternal& context, const SessionState& session_state,
                       std::vector<LoopStateVariable>& loop_state_variables,
                       std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator>& scan_input_stream_iterators,
                       int64_t seq_length, int num_loop_state_variables, int num_variadic_inputs,
                       int num_variadic_outputs, const std::vector<const OrtValue*>& implicit_inputs,
                       std::vector<std::unique_ptr<OutputIterator>>& output_iterators,
                       const FeedsFetchesManager& ffm) {
  Status status = Status::OK();

  auto num_implicit_inputs = implicit_inputs.size();
  auto num_inputs = num_variadic_inputs + num_implicit_inputs;

  std::vector<OrtValue> feeds;
  std::vector<OrtValue> fetches;
  std::unordered_map<size_t, IExecutor::CustomAllocator> fetch_allocators;

  feeds.resize(num_inputs);
  fetches.resize(num_variadic_outputs);

  // add implicit inputs and pass in implicit inputs as feeds. we're going to pass in the explicit inputs
  // first in each iteration though so offset by num_variadic_inputs
  for (size_t i = 0; i < num_implicit_inputs; ++i) {
    feeds[num_variadic_inputs + i] = *implicit_inputs[i];
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
          // add OrtValue from sliced output
          auto& ort_value = *iterator;
          fetches.push_back(ort_value);
        } else {
          // need a dummy empty entry in fetches so the order matches the output names
          size_t i = fetches.size();
          fetches.emplace_back();

          // use a custom allocator that will forward the allocation request to the Scan context
          // and add the sequence length dimension. this avoids using a temporary value for the first output
          fetch_allocators[output] = [i, &iterator, &fetches](const TensorShape& shape, const OrtMemoryInfo& location,
                                                              OrtValue& ort_value, bool& allocated) {
            auto status = iterator.AllocateFinalOutput(shape);
            ORT_RETURN_IF_ERROR(status);

            const OrtValue& value = *iterator;

            // for now we only allocate on CPU as currently all 'Scan' outputs are on CPU.
            // if that does not match the required device we don't update the provided OrtValue and return false for
            // 'allocated'. the execution frame will allocate a buffer on the required device, and the fetches copy
            // logic in utils::ExecuteSubgraph will handle moving it to CPU (and into the tensor we allocated here)
            if (value.Get<Tensor>().Location().device == location.device) {
              // update OrtValue with a current slice from the iterator.
              ort_value = value;
              allocated = true;
            } else {
              // put the allocated value into fetches so the copy logic in utils::ExecuteGraphImpl can use it
              fetches[i] = value;
            }

            return Status::OK();
          };
        }
      }
    }

    // Create Executor and run graph.
    status = utils::ExecuteSubgraph(session_state, ffm, feeds, fetches, fetch_allocators,
                                    ExecutionMode::ORT_SEQUENTIAL, context.GetTerminateFlag(), context.Logger());

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

OrtValue AllocateTensorInMLValue(const MLDataType data_type, const TensorShape& shape, AllocatorPtr& allocator) {
  auto new_tensor = std::make_unique<Tensor>(data_type,
                                                     shape,
                                                     allocator);

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  return OrtValue{new_tensor.release(), ml_tensor,
                  ml_tensor->GetDeleteFunc()};
};

void CalculateTransposedShapeForInput(const TensorShape& original_shape, int64_t axis,
                                      std::vector<size_t>& permutations, std::vector<int64_t>& transposed_shape) {
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
                                       std::vector<size_t>& permutations, std::vector<int64_t>& transposed_shape) {
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

LoopStateVariable::LoopStateVariable(const OrtValue& original_value, OrtValue& final_value, const int64_t sequence_len,
                                     AllocatorPtr& allocator)
    : sequence_len_{sequence_len}, original_value_{original_value}, final_value_{final_value} {
  auto& tensor = original_value.Get<Tensor>();
  auto& shape = tensor.Shape();

  // Allocate a new Tensor in an OrtValue with the same shape and type as the tensor in original_value.
  // the Tensor will own the buffer, and the OrtValue will own the Tensor.
  // the OrtValue returned by Input()/Output() gets copied into the execution frame feeds/fetches
  // with the Tensor being used via a shared_ptr (so remains valid during execution and is cleaned up
  // automatically at the end).
  //
  // Note: The allocator comes from the EP for the Scan node, so will allocate on the default device for that EP.

  // if length is > 1, we need a_ for the first output location. otherwise we use final_value for the output.
  if (sequence_len_ > 1) {
    a_ = AllocateTensorInMLValue(tensor.DataType(), shape, allocator);
  }

  // if length is > 2, we need b_ for the second output location
  if (sequence_len_ > 2) {
    b_ = AllocateTensorInMLValue(tensor.DataType(), shape, allocator);
  }
}

const OrtValue& LoopStateVariable::Input() const {
  if (iteration_num_ == 0)
    return original_value_;

  return iteration_num_ % 2 == 1 ? a_ : b_;
}

OrtValue& LoopStateVariable::Output() {
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
                               const scan::detail::DeviceHelpers::CreateMutableSlicer& create_slicer_func,
                               const scan::detail::DeviceHelpers::ZeroData& zero_data_func,
                               ScanDirection direction,
                               bool temporary,
                               MLDataType data_type)
    : context_(context),
      is_v8_(is_v8),
      output_index_(output_index),
      final_shape_(final_shape),
      is_loop_state_var_(is_loop_state_var),
      direction_(direction),
      cur_iteration_(0),
      temporary_(temporary),
      data_type_{data_type},
      create_slicer_func_(create_slicer_func),
      zero_data_func_(zero_data_func) {
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
    // delay until the first subgraph execution calls AllocateFinalOutput.
  }

  return Status::OK();
}

Status OutputIterator::AllocateFinalBuffer() {
  // make sure a single buffer for the full output is created upfront.
  // we slice this into per-iteration pieces using OrtValueTensorSlicer.
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
    auto status = context_.GetTempSpaceAllocator(&alloc);  // get allocator for the EP running this (CPU or CUDA)
    ORT_RETURN_IF_ERROR(status);

    temporary_final_output_mlvalue_ = AllocateTensorInMLValue(data_type_, final_shape_, alloc);
    final_output_mlvalue_ = &temporary_final_output_mlvalue_;
  }

  // if it's v8 there's always a batch size dimension so we need a slicer to hide that from each iteration
  if (is_v8_) {
    if (is_loop_state_var_) {
      // only one entry is required as we slice on a single dimension
      slicer_iterators_.push_back((direction_ == ScanDirection::kForward)
                                      ? create_slicer_func_(*final_output_mlvalue_, 0, 0).begin()
                                      : create_slicer_func_(*final_output_mlvalue_, 0, 0).rbegin());
    } else {
      auto batch_size = final_shape_[0];
      for (int i = 0; i < batch_size; ++i) {
        // the slicer handles the sequence dimension (dim 1) so create an entry for each batch
        slicer_iterators_.push_back(
            (direction_ == ScanDirection::kForward)
                ? create_slicer_func_(*final_output_mlvalue_, 1, i).begin()
                : create_slicer_func_(*final_output_mlvalue_, 1, i).rbegin());
      }
    }

    cur_slicer_iterator_ = slicer_iterators_.begin();
  } else {
    // nothing to slice for a loop state var. slice on dimension 0 (sequence) for the scan outputs.
    if (!is_loop_state_var_) {
      slicer_iterators_.push_back((direction_ == ScanDirection::kForward)
                                      ? create_slicer_func_(*final_output_mlvalue_, 0, 0).begin()
                                      : create_slicer_func_(*final_output_mlvalue_, 0, 0).rbegin());
      cur_slicer_iterator_ = slicer_iterators_.begin();
    }
  }

  return Status::OK();
}

Status OutputIterator::AllocateFinalOutput(const TensorShape& shape) {
  ORT_ENFORCE(!is_concrete_shape_, "If shape was concrete we shouldn't be using a custom allocator");

  // update the final shape now that we can fill in the symbolic dimension with an actual value
  auto status = MakeShapeConcrete(shape, final_shape_);
  ORT_RETURN_IF_ERROR(status);

  is_concrete_shape_ = true;
  status = AllocateFinalBuffer();
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

OrtValue& OutputIterator::operator*() {
  ORT_ENFORCE(cur_iteration_ < num_iterations_);
  ORT_ENFORCE(is_concrete_shape_,
              "Expected AllocateFinalOutput to have been called to before we read the OrtValue from the iterator.");

  // for v8 both outputs and loop state vars use slicers. for v9 only outputs do
  if (is_v8_ || !is_loop_state_var_)
    return **cur_slicer_iterator_;

  return *final_output_mlvalue_;
}

OutputIterator& OutputIterator::operator++() {
  if (cur_iteration_ < num_iterations_) {
    ORT_ENFORCE(is_concrete_shape_,
                "Expected AllocateFinalOutput to have been called to before we increment the iterator");

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

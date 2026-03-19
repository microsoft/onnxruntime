// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "core/providers/cpu/controlflow/scan.h"
#include "core/providers/cpu/controlflow/scan_utils.h"
#include "core/providers/cpu/controlflow/utils.h"

#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"

#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/tensor/transpose.h"

#include <gsl/gsl>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
using namespace onnxruntime::scan::detail;

namespace onnxruntime {
/*
ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    9,
    OpSchema()
        .SetDoc(scan_9_doc)
        .Input(
            0,
            "initial_state_and_scan_inputs",
            "Initial values of the loop's N state variables followed by M scan_inputs",
            "V",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "final_state_and_scan_outputs",
            "Final values of the loop's N state variables followed by K scan_outputs",
            "V",
            OpSchema::Variadic,
            false)
        .Attr(
            "body",
            "The graph run each iteration. It has N+M inputs: "
            "(loop state variables..., scan_input_elts...). It has N+K outputs: "
            "(loop state variables..., scan_output_elts...). Each "
            "scan_output is created by concatenating the value of the specified "
            "scan_output_elt value at the end of each iteration of the loop. It is an error"
            " if the dimensions of these values change across loop iterations.",
            AttributeProto::GRAPH,
            true)
        .Attr(
            "num_scan_inputs",
            "An attribute specifying the number of scan_inputs M. ",
            AttributeProto::INT,
            true)
        .Attr(
            "scan_input_directions",
            "An optional list of M flags. The i-th element of the list specifies the direction "
            "to be scanned for the i-th scan_input tensor: 0 indicates forward direction and 1 "
            "indicates reverse direction. "
            "If omitted, all scan_input tensors will be scanned in the forward direction.",
            AttributeProto::INTS,
            false)
        .Attr(
            "scan_output_directions",
            "An optional list of K flags, one for each scan_output. The i-th element of the list "
            "specifies whether the i-th scan_output should be constructed by appending or "
            "prepending a new value in each iteration: 0 indicates appending and 1 "
            "indicates prepending. "
            "If omitted, all scan_output tensors will be produced by appending a value "
            "in each iteration.",
            AttributeProto::INTS,
            false)
        .Attr(
            "axes",
            "An optional list of M flags. The i-th element of the list specifies the axis "
            "to be scanned (the sequence axis) for the i-th scan_input. If omitted, 0 will "
            "be used as the scan axis for every scan_input.",
            AttributeProto::INTS,
            false)
        .TypeConstraint("I", {"tensor(int64)"}, "Int64 tensor")
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
}
*/

class ScanImpl {
 public:
  ScanImpl(OpKernelContextInternal& context,
           const SessionState& session_state,
           const Scan<9>::Info& info,
           const gsl::span<const int64_t>& input_directions,
           const gsl::span<const int64_t>& output_directions,
           const gsl::span<const int64_t>& input_axes,
           const gsl::span<const int64_t>& output_axes,
           const scan::detail::DeviceHelpers& device_helpers,
           bool use_var_len_output = false,
           gsl::span<const int64_t> output_lengths = {});

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute(const FeedsFetchesManager& ffm);

 private:
  // validate inputs and setup batch size and max sequence length.
  Status ValidateInput();

  Status ValidateSubgraphInput(int start_input, int end_input,
                               const std::vector<const NodeArg*>& graph_inputs);

  // setup inputs to subgraph, transposing if necessary
  Status SetupInputs();

  Status AllocateOutputTensors();
  Status CreateLoopStateVariables(std::vector<LoopStateVariable>& loop_state_variables);
  Status TransposeOutput();

  // Variable-length scan output methods: allows per-iteration scan outputs to differ
  // in the concatenation-axis dimension.
  Status AllocateLoopStateOutputs();
  Status ExecuteVarLen(const FeedsFetchesManager& ffm);
  Status ConcatenateScanOutputs(std::vector<std::vector<OrtValue>>& scan_output_per_iteration);

  // Pre-allocated variable-length path: when output_lengths are provided, pre-allocates
  // the final output before the scan loop and writes scan outputs in-place.
  Status AllocatePreAllocOutputs();
  Status ExecutePreAlloc(const FeedsFetchesManager& ffm);

  using ConstTensorSlicerIterators = std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator>;
  using MutableTensorSlicerIterators = std::vector<OrtValueTensorSlicer<OrtValue>::Iterator>;

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const Scan<9>::Info& info_;

  int64_t sequence_len_ = -1;

  gsl::span<const int64_t> input_directions_;
  gsl::span<const int64_t> output_directions_;
  gsl::span<const int64_t> input_axes_from_attribute_;
  gsl::span<const int64_t> output_axes_from_attribute_;
  TensorShapeVector input_axes_;

  // inputs for graph. either original input value or transposed input if an axis other than 0 was specified
  std::vector<OrtValue> inputs_;
  std::vector<std::unique_ptr<OutputIterator>> output_iterators_;
  const std::vector<const OrtValue*>& implicit_inputs_;

  const scan::detail::DeviceHelpers& device_helpers_;

  // Offset to add when accessing variadic inputs from the OpKernelContext.
  // 0 for standard Scan; equals num_non_variadic_inputs for the contrib op.
  int input_offset_;

  // When true, uses the variable-length output implementation that collects per-iteration
  // scan outputs and concatenates them at the end, allowing the concatenation-axis dimension
  // to vary across iterations.
  bool use_var_len_output_;

  // When non-empty, specifies the expected total size of each scan output along the
  // concatenation axis, enabling pre-allocation of the final output tensor.
  gsl::span<const int64_t> output_lengths_;
};

template <>
void Scan<9>::Init(const OpKernelInfo& info) {
  // make sure the attribute was present even though we don't need it here.
  // The GraphProto is loaded as a Graph instance by main Graph::Resolve,
  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("body", &proto).IsOK());
  (void)proto;

  ORT_ENFORCE(info.GetAttr<int64_t>("num_scan_inputs", &num_scan_inputs_).IsOK());

  auto num_loop_state_vars = info.GetInputCount() - num_non_variadic_inputs_ - num_scan_inputs_;
  auto num_scan_outputs = info.GetOutputCount() - num_loop_state_vars;

  ReadDirections(info, "scan_input_directions", input_directions_, onnxruntime::narrow<size_t>(num_scan_inputs_));
  if (!use_var_len_output_) {
    ReadDirections(info, "scan_output_directions", output_directions_, onnxruntime::narrow<size_t>(num_scan_outputs));
  }

  if (info.GetAttrs("scan_input_axes", input_axes_).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(input_axes_.size()) == num_scan_inputs_,
                "Number of entries in 'scan_input_axes' was ", input_axes_.size(), " but expected ", num_scan_inputs_);
  } else {
    input_axes_.resize(onnxruntime::narrow<size_t>(num_scan_inputs_), 0);
  }

  if (info.GetAttrs("scan_output_axes", output_axes_).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(output_axes_.size()) == num_scan_outputs,
                "Number of entries in 'scan_output_axes' was ", output_axes_.size(), " but expected ",
                num_scan_outputs);
  } else {
    output_axes_.resize(onnxruntime::narrow<size_t>(num_scan_outputs), 0);
  }

  device_helpers_.transpose_func = [](const gsl::span<const size_t>& permutations, const Tensor& input,
                                      Tensor& output, Stream* /*no stream needed for cpu*/) -> Status {
    return TransposeBase::DoTranspose(permutations, input, output);
  };

  device_helpers_.set_data_to_zero_func = [](void* data, size_t size_in_bytes) -> Status {
    memset(data, 0, size_in_bytes);
    return Status::OK();
  };
}

template <>
Status Scan<9>::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                           const std::string& attribute_name,
                                           const SessionState& subgraph_session_state) {
  ORT_ENFORCE(info_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
  ORT_UNUSED_PARAMETER(attribute_name);

  const auto& node = Node();
  info_ = std::make_unique<Scan<9>::Info>(node, subgraph_session_state.GetGraphViewer(),
                                          static_cast<int>(num_scan_inputs_),
                                          num_non_variadic_inputs_);

  auto status = scan::detail::CreateFeedsFetchesManager(node, *info_, session_state, subgraph_session_state,
                                                        /* is_v8 */ false, feeds_fetches_manager_);

  return status;
}

template <>
Status Scan<9>::Compute(OpKernelContext* ctx) const {
  ORT_ENFORCE(feeds_fetches_manager_ && info_,
              "CreateFeedsFetchesManager must be called prior to execution of graph.");

  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("body");
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");

  // Read optional output_lengths input (only for contrib ScanVarLen with num_non_variadic_inputs_ > 0)
  gsl::span<const int64_t> output_lengths;
  if (num_non_variadic_inputs_ > 0 && use_var_len_output_) {
    auto* output_lengths_tensor = ctx->Input<Tensor>(0);
    if (output_lengths_tensor != nullptr) {
      output_lengths = output_lengths_tensor->DataAsSpan<int64_t>();
    }
  }

  ScanImpl scan_impl{*ctx_internal, *session_state, *info_, input_directions_, output_directions_,
                     input_axes_, output_axes_, device_helpers_, use_var_len_output_, output_lengths};

  auto status = scan_impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = scan_impl.Execute(*feeds_fetches_manager_);

  return status;
}

ScanImpl::ScanImpl(OpKernelContextInternal& context,
                   const SessionState& session_state,
                   const Scan<9>::Info& info,
                   const gsl::span<const int64_t>& input_directions,
                   const gsl::span<const int64_t>& output_directions,
                   const gsl::span<const int64_t>& input_axes,
                   const gsl::span<const int64_t>& output_axes,
                   const scan::detail::DeviceHelpers& device_helpers,
                   bool use_var_len_output,
                   gsl::span<const int64_t> output_lengths)
    : context_(context),
      session_state_(session_state),
      info_(info),
      input_directions_(input_directions),
      output_directions_(output_directions),
      input_axes_from_attribute_(input_axes),
      output_axes_from_attribute_(output_axes),
      implicit_inputs_(context_.GetImplicitInputs()),
      device_helpers_(device_helpers),
      input_offset_(info_.num_non_variadic_inputs),
      use_var_len_output_(use_var_len_output),
      output_lengths_(output_lengths) {
  inputs_.reserve(info_.num_scan_inputs);
  input_axes_.reserve(info_.num_scan_inputs);
}

Status ScanImpl::Initialize() {
  auto status = ValidateInput();
  ORT_RETURN_IF_ERROR(status);

  status = SetupInputs();
  ORT_RETURN_IF_ERROR(status);

  if (use_var_len_output_) {
    if (!output_lengths_.empty()) {
      status = AllocatePreAllocOutputs();
    } else {
      status = AllocateLoopStateOutputs();
    }
  } else {
    status = AllocateOutputTensors();
  }
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

Status ScanImpl::ValidateSubgraphInput(int start_input, int end_input,
                                       const std::vector<const NodeArg*>& graph_inputs) {
  // sequence dim is all that's required as a scalar input will only have that
  auto min_dims_required = 1;

  for (int i = start_input; i < end_input; ++i) {
    auto& input_tensor = *context_.Input<Tensor>(i + input_offset_);
    const auto& input_shape = input_tensor.Shape();

    if (input_shape.NumDimensions() < static_cast<size_t>(min_dims_required))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid scan input:", graph_inputs[i]->Name(),
                             " Expected ", min_dims_required,
                             " dimensions or more but input had shape of ", input_shape);

    auto seq_len_dim = input_axes_[static_cast<ptrdiff_t>(i) - info_.num_loop_state_variables];
    auto this_seq_len = input_shape[onnxruntime::narrow<size_t>(seq_len_dim)];

    if (sequence_len_ < 0) {
      sequence_len_ = this_seq_len;
    } else {
      if (sequence_len_ != this_seq_len) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Scan inputs have inconsistent sequence lengths. Previous value was ",
                               sequence_len_, " but input '", graph_inputs[i]->Name(),
                               "' dimension ", seq_len_dim, " has length of ", this_seq_len);
      }
    }
  }

  return Status::OK();
}

Status ScanImpl::ValidateInput() {
  // validate/calculate the input axes values and populate input_axes_.
  // we already checked that input_axes_from_attribute_.size() == info_.num_scan_inputs
  for (int i = 0; i < info_.num_scan_inputs; ++i) {
    auto axis = input_axes_from_attribute_[i];

    // zero is always valid, so only do extra checks for non-zero values
    if (axis != 0) {
      int64_t input_rank = context_.Input<Tensor>(i + info_.num_loop_state_variables + input_offset_)->Shape().NumDimensions();
      // check axis is valid for input_rank and also handle any negative axis value
      if (axis >= -input_rank && axis < input_rank)
        axis = HandleNegativeAxis(axis, input_rank);
      else
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid value in scan_input_axes for input ", i,
                               " of ", axis, ". Input tensor rank was ", input_rank);
    }

    input_axes_.push_back(axis);
  }

  // we're not guaranteed to have complete output shapes, so delay checking output_axes_from_attribute_
  // values until after execution.

  // no validation for loop state variables.

  // validate the scan inputs
  auto status = ValidateSubgraphInput(info_.num_loop_state_variables, info_.num_variadic_inputs,
                                      info_.subgraph.GetInputs());
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

Status ScanImpl::SetupInputs() {
  auto status = Status::OK();
  AllocatorPtr alloc;

  for (int i = 0; i < info_.num_scan_inputs; ++i) {
    auto sequence_dim = input_axes_[i];

    if (sequence_dim == 0) {
      // no transpose required
      inputs_.push_back(*context_.GetInputMLValue(i + info_.num_loop_state_variables + input_offset_));
    } else {
      auto& input_tensor = *context_.Input<Tensor>(i + info_.num_loop_state_variables + input_offset_);
      const auto& input_shape = input_tensor.Shape();

      InlinedVector<size_t> permutations;
      TensorShapeVector new_shape;
      CalculateTransposedShapeForInput(input_shape, sequence_dim, permutations, new_shape);

      if (!alloc) {
        status = context_.GetTempSpaceAllocator(&alloc);
        ORT_RETURN_IF_ERROR(status);
      }

      OrtValue transpose_output = scan::detail::AllocateTensorInMLValue(input_tensor.DataType(), new_shape, alloc);

      status = device_helpers_.transpose_func(permutations, input_tensor, *transpose_output.GetMutable<Tensor>(),
                                              context_.GetComputeStream());
      ORT_RETURN_IF_ERROR(status);

      inputs_.push_back(transpose_output);
    }
  }

  return status;
}

Status ScanImpl::AllocateOutputTensors() {
  Status status = Status::OK();
  auto& graph_outputs = info_.subgraph.GetOutputs();

  if (graph_outputs.size() != static_cast<size_t>(info_.num_outputs)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph in 'body' produces ", graph_outputs.size(),
                           " outputs but Scan expects ", info_.num_outputs);
  }

  std::unique_ptr<OutputIterator> output_iter;

  for (int i = 0; i < info_.num_loop_state_variables; ++i) {
    status = AllocateOutput(context_, info_.subgraph, i, true, -1, sequence_len_, output_iter,
                            device_helpers_.create_mutable_slicer_func, device_helpers_.set_data_to_zero_func,
                            ScanDirection::kForward, false, input_offset_);
    ORT_RETURN_IF_ERROR(status);
    output_iterators_.push_back(std::move(output_iter));
  }

  for (int i = info_.num_loop_state_variables, end = info_.num_outputs; i < end; ++i) {
    ScanDirection direction = ScanDirection::kForward;
    const int scan_output_index = i - info_.num_loop_state_variables;
    if (static_cast<size_t>(scan_output_index) < output_directions_.size()) {
      direction = static_cast<ScanDirection>(output_directions_[scan_output_index]);
    }

    // if we need to transpose later, we need to use a temporary output buffer when executing the subgraph
    bool temporary = output_axes_from_attribute_[scan_output_index] != 0;

    status = AllocateOutput(context_, info_.subgraph, i, false, -1, sequence_len_, output_iter,
                            device_helpers_.create_mutable_slicer_func, device_helpers_.set_data_to_zero_func,
                            direction, temporary);
    ORT_RETURN_IF_ERROR(status);

    output_iterators_.push_back(std::move(output_iter));
  }

  return Status::OK();
}

Status ScanImpl::CreateLoopStateVariables(std::vector<LoopStateVariable>& loop_state_variables) {
  AllocatorPtr alloc;
  auto status = context_.GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  loop_state_variables.reserve(info_.num_loop_state_variables);

  for (int i = 0; i < info_.num_loop_state_variables; ++i) {
    const OrtValue& input_mlvalue = *context_.GetInputMLValue(i + input_offset_);
    OrtValue* output_mlvalue = context_.GetOutputMLValue(i);
    ORT_ENFORCE(output_mlvalue, "Output OrtValue has not been created for loop state variable output ", i);

    loop_state_variables.push_back(LoopStateVariable(input_mlvalue, *output_mlvalue, sequence_len_, alloc));
  }

  return status;
}

Status ScanImpl::Execute(const FeedsFetchesManager& ffm) {
  if (use_var_len_output_) {
    if (!output_lengths_.empty()) {
      return ExecutePreAlloc(ffm);
    }
    return ExecuteVarLen(ffm);
  }

  Status status = Status::OK();

  std::vector<LoopStateVariable> loop_state_variables;
  status = CreateLoopStateVariables(loop_state_variables);
  ORT_RETURN_IF_ERROR(status);

  // Setup input OrtValue streams
  std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator> scan_input_stream_iterators;
  scan_input_stream_iterators.reserve(static_cast<size_t>(info_.num_variadic_inputs) - info_.num_loop_state_variables);

  for (int i = 0, end = info_.num_scan_inputs; i < end; ++i) {
    const auto& ort_value = inputs_[i];

    // forward
    if (input_directions_[i] == static_cast<int64_t>(ScanDirection::kForward)) {
      // the iterator is self contained, so we don't need to keep the OrtValueTensorSlicer instance around
      scan_input_stream_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 0, 0).begin());
    } else {  // reverse
      scan_input_stream_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 0, 0).rbegin());
    }
  }

  // Call the subgraph for each item in the sequence
  status = IterateSequence(context_, session_state_, loop_state_variables, scan_input_stream_iterators,
                           sequence_len_, info_.num_loop_state_variables, info_.num_variadic_inputs, info_.num_outputs,
                           implicit_inputs_, output_iterators_, ffm);

  ORT_RETURN_IF_ERROR(status);

  status = TransposeOutput();

  return status;
}

Status ScanImpl::TransposeOutput() {
  auto status = Status::OK();

  for (int i = 0; i < info_.num_scan_outputs; ++i) {
    auto axis = output_axes_from_attribute_[i];

    if (axis != 0) {
      auto output_index = i + info_.num_loop_state_variables;
      const OrtValue& temporary_output_mlvalue = output_iterators_[output_index]->GetOutput();
      const auto& temporary_output_tensor = temporary_output_mlvalue.Get<Tensor>();

      int64_t output_rank = temporary_output_tensor.Shape().NumDimensions();

      // check axis is valid for input_rank and also handle any negative axis value
      if (axis >= -output_rank && axis < output_rank)
        axis = HandleNegativeAxis(axis, output_rank);
      else
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid value in scan_output_axes for output ", i,
                               " of ", axis, ". Output tensor rank was ", output_rank);

      InlinedVector<size_t> permutations;
      TensorShapeVector new_shape;
      CalculateTransposedShapeForOutput(temporary_output_tensor.Shape(), axis, permutations, new_shape);

      Tensor* output = context_.Output(output_index, new_shape);
      ORT_ENFORCE(output, "Outputs from Scan are not optional and should never be null.");

      status = device_helpers_.transpose_func(permutations, temporary_output_tensor, *output,
                                              context_.GetComputeStream());
      ORT_RETURN_IF_ERROR(status);
    }
  }

  return status;
}

Status ScanImpl::AllocateLoopStateOutputs() {
  Status status = Status::OK();
  auto& graph_outputs = info_.subgraph.GetOutputs();

  if (graph_outputs.size() != static_cast<size_t>(info_.num_outputs)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph in 'body' produces ", graph_outputs.size(),
                           " outputs but Scan expects ", info_.num_outputs);
  }

  std::unique_ptr<OutputIterator> output_iter;

  for (int i = 0; i < info_.num_loop_state_variables; ++i) {
    status = AllocateOutput(context_, info_.subgraph, i, true, -1, sequence_len_, output_iter,
                            device_helpers_.create_mutable_slicer_func, device_helpers_.set_data_to_zero_func,
                            ScanDirection::kForward, false, input_offset_);
    ORT_RETURN_IF_ERROR(status);
    output_iterators_.push_back(std::move(output_iter));
  }

  return Status::OK();
}

Status ScanImpl::ExecuteVarLen(const FeedsFetchesManager& ffm) {
  Status status = Status::OK();

  std::vector<LoopStateVariable> loop_state_variables;
  status = CreateLoopStateVariables(loop_state_variables);
  ORT_RETURN_IF_ERROR(status);

  // Setup input OrtValue streams
  std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator> scan_input_stream_iterators;
  scan_input_stream_iterators.reserve(static_cast<size_t>(info_.num_variadic_inputs) - info_.num_loop_state_variables);

  for (int i = 0, end = info_.num_scan_inputs; i < end; ++i) {
    const auto& ort_value = inputs_[i];

    if (input_directions_[i] == static_cast<int64_t>(ScanDirection::kForward)) {
      scan_input_stream_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 0, 0).begin());
    } else {
      scan_input_stream_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 0, 0).rbegin());
    }
  }

  // Iterate sequence, collecting per-iteration scan outputs
  std::vector<std::vector<OrtValue>> scan_output_per_iteration;
  status = IterateSequenceVarLen(context_, session_state_, loop_state_variables, scan_input_stream_iterators,
                                 sequence_len_, info_.num_loop_state_variables, info_.num_variadic_inputs, info_.num_outputs,
                                 implicit_inputs_, scan_output_per_iteration, ffm);
  ORT_RETURN_IF_ERROR(status);

  // Concatenate per-iteration outputs into final scan outputs (handles transpose and direction)
  status = ConcatenateScanOutputs(scan_output_per_iteration);
  ORT_RETURN_IF_ERROR(status);

  return status;
}

Status ScanImpl::ConcatenateScanOutputs(std::vector<std::vector<OrtValue>>& scan_output_per_iteration) {
  auto& graph_outputs = info_.subgraph.GetOutputs();

  for (int i = 0; i < info_.num_scan_outputs; ++i) {
    auto& per_iteration_outputs = scan_output_per_iteration[i];
    int output_index = i + info_.num_loop_state_variables;

    // Resolve the concat axis for this scan output
    int64_t concat_axis = output_axes_from_attribute_[i];

    if (per_iteration_outputs.empty()) {
      // Zero iterations: create empty output with concat-axis dimension = 0
      auto* graph_output = graph_outputs.at(output_index);
      auto* graph_output_shape = graph_output->Shape();

      TensorShapeVector output_dims;
      if (graph_output_shape) {
        auto tensor_shape = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape);
        auto dims = tensor_shape.GetDims();
        output_dims.assign(dims.begin(), dims.end());

        auto output_rank = static_cast<int64_t>(output_dims.size());
        if (concat_axis < 0) concat_axis += output_rank;
        if (concat_axis >= 0 && concat_axis < output_rank) {
          output_dims[concat_axis] = 0;
        }
        // Replace symbolic dims with 0
        for (auto& dim : output_dims) {
          if (dim < 0) dim = 0;
        }
      }

      if (output_dims.empty()) {
        output_dims.push_back(0);
      }

      ORT_IGNORE_RETURN_VALUE(context_.Output(output_index, TensorShape(output_dims)));
      continue;
    }

    // Get shape info from first per-iteration output
    const auto& first_tensor = per_iteration_outputs.front().Get<Tensor>();
    const auto& first_shape = first_tensor.Shape();
    auto rank = static_cast<int64_t>(first_shape.NumDimensions());

    ORT_RETURN_IF(rank == 0,
                  "Scan output ", i, " has rank 0. Variable-length concatenation requires rank >= 1.");

    // Resolve negative axis
    if (concat_axis < 0) concat_axis += rank;
    ORT_RETURN_IF(concat_axis < 0 || concat_axis >= rank,
                  "Invalid value in scan_output_axes for output ", i,
                  " of ", output_axes_from_attribute_[i], ". Output tensor rank was ", rank);

    // Compute total size along concat axis and validate other dimensions match
    int64_t total_concat_dim = 0;
    for (size_t iter = 0; iter < per_iteration_outputs.size(); ++iter) {
      const auto& iter_shape = per_iteration_outputs[iter].Get<Tensor>().Shape();

      if (static_cast<int64_t>(iter_shape.NumDimensions()) != rank) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Scan output ", i, " has inconsistent rank across iterations. ",
                               "Expected ", rank, " but iteration ", iter, " produced rank ",
                               iter_shape.NumDimensions());
      }

      for (int64_t d = 0; d < rank; ++d) {
        if (d == concat_axis) {
          total_concat_dim += iter_shape[d];
        } else if (iter_shape[d] != first_shape[d]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Scan output ", i, " has inconsistent shape in dimension ", d,
                                 " across iterations. Expected ", first_shape[d],
                                 " but iteration ", iter, " produced ", iter_shape[d]);
        }
      }
    }

    // Build concatenated output shape
    TensorShapeVector concat_dims(first_shape.GetDims().begin(), first_shape.GetDims().end());
    concat_dims[concat_axis] = total_concat_dim;
    TensorShape concat_shape(concat_dims);

    Tensor* output = context_.Output(output_index, concat_shape);
    ORT_ENFORCE(output, "Failed to create output tensor for scan output ", i);

    auto num_iters = static_cast<int64_t>(per_iteration_outputs.size());

    // Compute strides for axis-aware concatenation.
    // For axis 0 this reduces to sequential memcpy of whole tensors.
    int64_t outer_size = 1;
    for (int64_t d = 0; d < concat_axis; ++d) outer_size *= first_shape[d];

    size_t elem_size = first_tensor.DataType()->Size();
    int64_t inner_count = 1;
    for (int64_t d = concat_axis + 1; d < rank; ++d) inner_count *= first_shape[d];
    size_t inner_bytes = static_cast<size_t>(inner_count) * elem_size;
    size_t dest_stride = static_cast<size_t>(total_concat_dim) * inner_bytes;

    auto* dest = static_cast<uint8_t*>(output->MutableDataRaw());

    for (int64_t outer = 0; outer < outer_size; ++outer) {
      size_t dest_offset = outer * dest_stride;
      for (int64_t iter = 0; iter < num_iters; ++iter) {
        const auto& iter_tensor = per_iteration_outputs[iter].Get<Tensor>();
        int64_t iter_concat_dim = iter_tensor.Shape()[concat_axis];
        size_t chunk_bytes = static_cast<size_t>(iter_concat_dim) * inner_bytes;
        size_t src_offset = outer * chunk_bytes;
        memcpy(dest + dest_offset,
               static_cast<const uint8_t*>(iter_tensor.DataRaw()) + src_offset,
               chunk_bytes);
        dest_offset += chunk_bytes;
      }
    }
  }

  return Status::OK();
}

Status ScanImpl::AllocatePreAllocOutputs() {
  // Same as AllocateLoopStateOutputs: only allocate loop state variable outputs.
  // Scan outputs are allocated in ExecutePreAlloc after the first iteration reveals
  // the per-iteration output shapes (for non-concat dimensions).
  Status status = Status::OK();
  auto& graph_outputs = info_.subgraph.GetOutputs();

  if (graph_outputs.size() != static_cast<size_t>(info_.num_outputs)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph in 'body' produces ", graph_outputs.size(),
                           " outputs but Scan expects ", info_.num_outputs);
  }

  ORT_RETURN_IF(static_cast<int64_t>(output_lengths_.size()) != info_.num_scan_outputs,
                "output_lengths has ", output_lengths_.size(), " entries but expected ", info_.num_scan_outputs);

  std::unique_ptr<OutputIterator> output_iter;

  for (int i = 0; i < info_.num_loop_state_variables; ++i) {
    status = AllocateOutput(context_, info_.subgraph, i, true, -1, sequence_len_, output_iter,
                            device_helpers_.create_mutable_slicer_func, device_helpers_.set_data_to_zero_func,
                            ScanDirection::kForward, false, input_offset_);
    ORT_RETURN_IF_ERROR(status);
    output_iterators_.push_back(std::move(output_iter));
  }

  return Status::OK();
}

Status ScanImpl::ExecutePreAlloc(const FeedsFetchesManager& ffm) {
  Status status = Status::OK();

  std::vector<LoopStateVariable> loop_state_variables;
  status = CreateLoopStateVariables(loop_state_variables);
  ORT_RETURN_IF_ERROR(status);

  // Setup input OrtValue streams
  std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator> scan_input_stream_iterators;
  scan_input_stream_iterators.reserve(static_cast<size_t>(info_.num_variadic_inputs) - info_.num_loop_state_variables);

  for (int i = 0, end = info_.num_scan_inputs; i < end; ++i) {
    const auto& ort_value = inputs_[i];

    if (input_directions_[i] == static_cast<int64_t>(ScanDirection::kForward)) {
      scan_input_stream_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 0, 0).begin());
    } else {
      scan_input_stream_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 0, 0).rbegin());
    }
  }

  // Pre-allocated output tensors for each scan output.
  // Allocated on the first iteration after we know the per-iteration shape.
  struct PreAllocScanOutput {
    Tensor* output_tensor = nullptr;
    int64_t rank = 0;
    int64_t concat_axis = 0;
    int64_t outer_size = 1;       // product of dims before concat_axis
    size_t inner_bytes = 0;       // bytes per inner unit (product of dims after concat_axis * elem_size)
    size_t dest_stride = 0;       // bytes per outer block in the output
    int64_t axis_offset = 0;      // sum of concat dims from iterations processed so far
  };
  std::vector<PreAllocScanOutput> prealloc_outputs(info_.num_scan_outputs);

  auto num_implicit_inputs = implicit_inputs_.size();
  auto num_inputs = info_.num_variadic_inputs + static_cast<int>(num_implicit_inputs);

  std::vector<OrtValue> feeds;
  std::vector<OrtValue> fetches;
  std::unordered_map<size_t, IExecutor::CustomAllocator> fetch_allocators;

  feeds.resize(num_inputs);

  // add implicit inputs (constant across iterations)
  for (size_t i = 0; i < num_implicit_inputs; ++i) {
    feeds[info_.num_variadic_inputs + i] = *implicit_inputs_[i];
  }

  for (int64_t seq_no = 0; seq_no < sequence_len_; ++seq_no) {
    for (int input = 0; input < info_.num_variadic_inputs; ++input) {
      if (input < info_.num_loop_state_variables) {
        feeds[input] = loop_state_variables[input].Input();
      } else {
        auto& iterator = scan_input_stream_iterators[static_cast<ptrdiff_t>(input) - info_.num_loop_state_variables];
        feeds[input] = *iterator;
        ++iterator;
      }
    }

    fetches.clear();

    for (int output = 0; output < info_.num_outputs; ++output) {
      if (output < info_.num_loop_state_variables) {
        fetches.push_back(loop_state_variables[output].Output());
      } else {
        // Leave empty: subgraph will allocate fresh memory
        fetches.emplace_back();
      }
    }

    status = utils::ExecuteSubgraph(session_state_, ffm, feeds, fetches, fetch_allocators,
                                    ExecutionMode::ORT_SEQUENTIAL, context_.GetTerminateFlag(), context_.Logger(),
                                    context_.GetComputeStream());
    ORT_RETURN_IF_ERROR(status);

    // Cycle loop state variables
    std::for_each(loop_state_variables.begin(), loop_state_variables.end(), [](LoopStateVariable& v) { v.Next(); });

    // Process scan outputs
    for (int i = 0; i < info_.num_scan_outputs; ++i) {
      int fetch_index = i + info_.num_loop_state_variables;
      int output_index = i + info_.num_loop_state_variables;
      auto& pa = prealloc_outputs[i];
      const auto& iter_tensor = fetches[fetch_index].Get<Tensor>();
      const auto& iter_shape = iter_tensor.Shape();

      if (seq_no == 0) {
        // First iteration: determine shape and allocate
        pa.rank = static_cast<int64_t>(iter_shape.NumDimensions());
        ORT_RETURN_IF(pa.rank == 0,
                      "Scan output ", i, " has rank 0. Variable-length concatenation requires rank >= 1.");

        pa.concat_axis = output_axes_from_attribute_[i];
        if (pa.concat_axis < 0) pa.concat_axis += pa.rank;
        ORT_RETURN_IF(pa.concat_axis < 0 || pa.concat_axis >= pa.rank,
                      "Invalid value in scan_output_axes for output ", i,
                      " of ", output_axes_from_attribute_[i], ". Output tensor rank was ", pa.rank);

        // Build the pre-allocated output shape: replace concat dim with output_lengths_[i]
        TensorShapeVector concat_dims(iter_shape.GetDims().begin(), iter_shape.GetDims().end());
        concat_dims[pa.concat_axis] = output_lengths_[i];
        TensorShape concat_shape(concat_dims);

        pa.output_tensor = context_.Output(output_index, concat_shape);
        ORT_ENFORCE(pa.output_tensor, "Failed to create output tensor for scan output ", i);

        // Compute strides for axis-aware writes
        pa.outer_size = 1;
        for (int64_t d = 0; d < pa.concat_axis; ++d) pa.outer_size *= iter_shape[d];

        size_t elem_size = iter_tensor.DataType()->Size();
        int64_t inner_count = 1;
        for (int64_t d = pa.concat_axis + 1; d < pa.rank; ++d) inner_count *= iter_shape[d];
        pa.inner_bytes = static_cast<size_t>(inner_count) * elem_size;
        pa.dest_stride = static_cast<size_t>(output_lengths_[i]) * pa.inner_bytes;

        pa.axis_offset = 0;
      } else {
        // Validate non-concat dims match first iteration
        ORT_RETURN_IF(static_cast<int64_t>(iter_shape.NumDimensions()) != pa.rank,
                      "Scan output ", i, " has inconsistent rank across iterations. ",
                      "Expected ", pa.rank, " but iteration ", seq_no, " produced rank ",
                      iter_shape.NumDimensions());

        auto output_dims = pa.output_tensor->Shape().GetDims();
        for (int64_t d = 0; d < pa.rank; ++d) {
          if (d != pa.concat_axis && iter_shape[d] != output_dims[d]) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "Scan output ", i, " has inconsistent shape in dimension ", d,
                                   " across iterations. Expected ", output_dims[d],
                                   " but iteration ", seq_no, " produced ", iter_shape[d]);
          }
        }
      }

      // Copy this iteration's data into the pre-allocated buffer at the right position
      int64_t iter_concat_dim = iter_shape[pa.concat_axis];
      size_t chunk_bytes = static_cast<size_t>(iter_concat_dim) * pa.inner_bytes;
      auto* dest = static_cast<uint8_t*>(pa.output_tensor->MutableDataRaw());
      auto* src = static_cast<const uint8_t*>(iter_tensor.DataRaw());

      for (int64_t outer = 0; outer < pa.outer_size; ++outer) {
        size_t dest_offset = outer * pa.dest_stride + static_cast<size_t>(pa.axis_offset) * pa.inner_bytes;
        size_t src_offset = outer * chunk_bytes;
        ORT_RETURN_IF(dest_offset + chunk_bytes > pa.output_tensor->SizeInBytes(),
                      "Scan output ", i, " exceeded pre-allocated size.");
        memcpy(dest + dest_offset, src + src_offset, chunk_bytes);
      }
      pa.axis_offset += iter_concat_dim;
    }
  }

  // Verify scan outputs: axis_offset should match output_lengths
  for (int i = 0; i < info_.num_scan_outputs; ++i) {
    auto& pa = prealloc_outputs[i];
    ORT_RETURN_IF(pa.output_tensor && pa.axis_offset != output_lengths_[i],
                  "Scan output ", i, " size mismatch: total concat dim was ", pa.axis_offset,
                  " but output_lengths specified ", output_lengths_[i], ".");
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   9,
                                   10,
                                   KernelDefBuilder()
                                       // 'I' is in the ONNX spec but is not actually used for any inputs or outputs
                                       //.TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                   Scan<9>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   11,
                                   15,
                                   KernelDefBuilder()
                                       // 'I' is in the ONNX spec but is not actually used for any inputs or outputs
                                       //.TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                   Scan<9>);

// Opset 16 starts to support BFloat16 type for the type constraint "V"
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   16, 18,
                                   KernelDefBuilder()
                                       // 'I' is in the ONNX spec but is not actually used for any inputs or outputs
                                       //.TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                   Scan<9>);

// Opset 19 starts to support float 8 types for the type constraint "V"
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   19, 20,
                                   KernelDefBuilder()
                                       // 'I' is in the ONNX spec but is not actually used for any inputs or outputs
                                       // .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypesIRv9()),
                                   Scan<9>);

// Opset 21 starts to support 4-bit int types for the type constraint "V"
// TODO(adrianlizarraga): Implement int4 and uint4 support.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   21,
                                   22,
                                   KernelDefBuilder()
                                       // 'I' is in the ONNX spec but is not actually used for any inputs or outputs
                                       // .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypesIRv9()),
                                   Scan<9>);

// Opset 23 added support for float4e2m1.
// TODO: Add support for float4e2m1.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   23,
                                   23,
                                   KernelDefBuilder()
                                       // 'I' is in the ONNX spec but is not actually used for any inputs or outputs
                                       // .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypesIRv9()),
                                   Scan<9>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   24,
                                   24,
                                   KernelDefBuilder()
                                       // 'I' is in the ONNX spec but is not actually used for any inputs or outputs
                                       // .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypesIRv9()),
                                   Scan<9>);

// Opset 25
ONNX_CPU_OPERATOR_KERNEL(Scan,
                         25,
                         KernelDefBuilder()
                             // 'I' is in the ONNX spec but is not actually used for any inputs or outputs
                             // .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorTypesIRv9()),
                         Scan<9>);
}  // namespace onnxruntime

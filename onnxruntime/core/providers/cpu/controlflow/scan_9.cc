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

#include "gsl/gsl"

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

template <>
struct Scan<9>::Info : public scan::detail::Info {
  Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in, int num_scan_inputs_in)
      : scan::detail::Info(node, subgraph_in, num_scan_inputs_in, /* is_v8 */ false) {}
};

class ScanImpl {
 public:
  ScanImpl(OpKernelContextInternal& context,
           const SessionState& session_state,
           const Scan<9>::Info& info,
           const std::vector<int64_t>& input_directions,
           const std::vector<int64_t>& output_directions,
           const std::vector<int64_t>& input_axes,
           const std::vector<int64_t>& output_axes,
           const scan::detail::DeviceHelpers& device_helpers);

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

  using ConstTensorSlicerIterators = std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator>;
  using MutableTensorSlicerIterators = std::vector<OrtValueTensorSlicer<OrtValue>::Iterator>;

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const Scan<9>::Info& info_;

  int64_t sequence_len_ = -1;

  const std::vector<int64_t>& input_directions_;
  const std::vector<int64_t>& output_directions_;
  const std::vector<int64_t>& input_axes_from_attribute_;
  const std::vector<int64_t>& output_axes_from_attribute_;
  std::vector<int64_t> input_axes_;

  // inputs for graph. either original input value or transposed input if an axis other than 0 was specified
  std::vector<OrtValue> inputs_;
  std::vector<std::unique_ptr<OutputIterator>> output_iterators_;
  const std::vector<const OrtValue*>& implicit_inputs_;

  const scan::detail::DeviceHelpers& device_helpers_;
};

template <>
Scan<9>::Scan(const OpKernelInfo& info) : IControlFlowKernel(info) {
  // make sure the attribute was present even though we don't need it here.
  // The GraphProto is loaded as a Graph instance by main Graph::Resolve,
  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("body", &proto).IsOK());
  (void)proto;

  ORT_ENFORCE(info.GetAttr<int64_t>("num_scan_inputs", &num_scan_inputs_).IsOK());

  auto num_loop_state_vars = info.GetInputCount() - num_scan_inputs_;
  auto num_scan_outputs = info.GetOutputCount() - num_loop_state_vars;

  ReadDirections(info, "scan_input_directions", input_directions_, num_scan_inputs_);
  ReadDirections(info, "scan_output_directions", output_directions_, num_scan_outputs);

  if (info.GetAttrs<int64_t>("scan_input_axes", input_axes_).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(input_axes_.size()) == num_scan_inputs_,
                "Number of entries in 'scan_input_axes' was ", input_axes_.size(), " but expected ", num_scan_inputs_);
  } else {
    input_axes_ = std::vector<int64_t>(num_scan_inputs_, 0);
  }

  if (info.GetAttrs<int64_t>("scan_output_axes", output_axes_).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(output_axes_.size()) == num_scan_outputs,
                "Number of entries in 'scan_output_axes' was ", output_axes_.size(), " but expected ",
                num_scan_outputs);
  } else {
    output_axes_ = std::vector<int64_t>(num_scan_outputs, 0);
  }

  device_helpers_.transpose_func = [](const std::vector<size_t>& permutations, const Tensor& input,
                                      Tensor& output) -> Status {
    return TransposeBase::DoTranspose(permutations, input, output);
  };

  device_helpers_.set_data_to_zero_func = [](void* data, size_t size_in_bytes) -> Status {
    memset(data, 0, size_in_bytes);
    return Status::OK();
  };
}

// we need this to be in the .cc so 'unique_ptr<Info> info_' can be handled
template <>
Scan<9>::~Scan() = default;

template <>
Status Scan<9>::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                           const std::string& attribute_name,
                                           const SessionState& subgraph_session_state) {
  ORT_ENFORCE(info_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
  ORT_UNUSED_PARAMETER(attribute_name);

  const auto& node = Node();
  info_ = std::make_unique<Scan<9>::Info>(node, subgraph_session_state.GetGraphViewer(),
                                                  static_cast<int>(num_scan_inputs_));

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

  ScanImpl scan_impl{*ctx_internal, *session_state, *info_, input_directions_, output_directions_,
                     input_axes_, output_axes_, device_helpers_};

  auto status = scan_impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = scan_impl.Execute(*feeds_fetches_manager_);

  return status;
}

ScanImpl::ScanImpl(OpKernelContextInternal& context,
                   const SessionState& session_state,
                   const Scan<9>::Info& info,
                   const std::vector<int64_t>& input_directions,
                   const std::vector<int64_t>& output_directions,
                   const std::vector<int64_t>& input_axes,
                   const std::vector<int64_t>& output_axes,
                   const scan::detail::DeviceHelpers& device_helpers)
    : context_(context),
      session_state_(session_state),
      info_(info),
      input_directions_(input_directions),
      output_directions_(output_directions),
      input_axes_from_attribute_(input_axes),
      output_axes_from_attribute_(output_axes),
      implicit_inputs_(context_.GetImplicitInputs()),
      device_helpers_(device_helpers) {
  inputs_.reserve(info_.num_scan_inputs);
  input_axes_.reserve(info_.num_scan_inputs);
}

Status ScanImpl::Initialize() {
  auto status = ValidateInput();
  ORT_RETURN_IF_ERROR(status);

  status = SetupInputs();
  ORT_RETURN_IF_ERROR(status);

  status = AllocateOutputTensors();
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

Status ScanImpl::ValidateSubgraphInput(int start_input, int end_input,
                                       const std::vector<const NodeArg*>& graph_inputs) {
  // sequence dim is all that's required as a scalar input will only have that
  auto min_dims_required = 1;

  for (int i = start_input; i < end_input; ++i) {
    auto& input_tensor = *context_.Input<Tensor>(i);
    const auto& input_shape = input_tensor.Shape();

    if (input_shape.NumDimensions() < static_cast<size_t>(min_dims_required))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid scan input:", graph_inputs[i]->Name(),
                             " Expected ", min_dims_required,
                             " dimensions or more but input had shape of ", input_shape);

    auto seq_len_dim = input_axes_[i - info_.num_loop_state_variables];
    auto this_seq_len = input_shape[seq_len_dim];

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
      int64_t input_rank = context_.Input<Tensor>(i + info_.num_loop_state_variables)->Shape().NumDimensions();
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
  auto status = ValidateSubgraphInput(info_.num_loop_state_variables, info_.num_inputs, info_.subgraph.GetInputs());
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
      inputs_.push_back(*context_.GetInputMLValue(i + info_.num_loop_state_variables));
    } else {
      auto& input_tensor = *context_.Input<Tensor>(i + info_.num_loop_state_variables);
      const auto& input_shape = input_tensor.Shape();

      std::vector<size_t> permutations;
      std::vector<int64_t> new_shape;
      CalculateTransposedShapeForInput(input_shape, sequence_dim, permutations, new_shape);

      if (!alloc) {
        status = context_.GetTempSpaceAllocator(&alloc);
        ORT_RETURN_IF_ERROR(status);
      }

      OrtValue transpose_output = scan::detail::AllocateTensorInMLValue(input_tensor.DataType(), new_shape, alloc);

      status = device_helpers_.transpose_func(permutations, input_tensor, *transpose_output.GetMutable<Tensor>());
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
                            device_helpers_.create_mutable_slicer_func, device_helpers_.set_data_to_zero_func);
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
    const OrtValue& input_mlvalue = *context_.GetInputMLValue(i);
    OrtValue* output_mlvalue = context_.GetOutputMLValue(i);
    ORT_ENFORCE(output_mlvalue, "Output OrtValue has not been created for loop state variable output ", i);

    loop_state_variables.push_back(LoopStateVariable(input_mlvalue, *output_mlvalue, sequence_len_, alloc));
  }

  return status;
}

Status ScanImpl::Execute(const FeedsFetchesManager& ffm) {
  Status status = Status::OK();

  std::vector<LoopStateVariable> loop_state_variables;
  status = CreateLoopStateVariables(loop_state_variables);
  ORT_RETURN_IF_ERROR(status);

  // Setup input OrtValue streams
  std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator> scan_input_stream_iterators;
  scan_input_stream_iterators.reserve(info_.num_inputs - info_.num_loop_state_variables);

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
                           sequence_len_, info_.num_loop_state_variables, info_.num_inputs, info_.num_outputs,
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

      std::vector<size_t> permutations;
      std::vector<int64_t> new_shape;
      CalculateTransposedShapeForOutput(temporary_output_tensor.Shape(), axis, permutations, new_shape);

      Tensor* output = context_.Output(output_index, new_shape);
      ORT_ENFORCE(output, "Outputs from Scan are not optional and should never be null.");

      status = device_helpers_.transpose_func(permutations, temporary_output_tensor, *output);
      ORT_RETURN_IF_ERROR(status);
    }
  }

  return status;
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   9,
                                   10,
                                   KernelDefBuilder()
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                   Scan<9>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_KERNEL(Scan,
                         11,
                         KernelDefBuilder()
                             .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                         Scan<9>);
}  // namespace onnxruntime

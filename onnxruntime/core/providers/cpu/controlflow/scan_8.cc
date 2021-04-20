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

#include "core/providers/cpu/tensor/utils.h"

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
    8,
    OpSchema()
    .SetDoc(scan_ver1_doc)
    .Input(
        0,
        "sequence_lens",
        "Optional tensor specifying lengths of the sequences in a batch. "
        "If this input is not specified, all sequences are assumed to be of "
        "the maximum sequence length (the dimension of the sequence axis of "
        "the scan_input tensors).",
        "I",
        OpSchema::Optional)
    .Input(
        1,
        "initial_state_and_scan_inputs",
        "Initial values of the loop's N state variables followed by M scan_inputs",
        "V",
        OpSchema::Variadic)
    .Output(
        0,
        "final_state_and_scan_outputs",
        "Final values of the loop's N state variables followed by K scan_outputs",
        "V",
        OpSchema::Variadic)
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
        "directions",
        "An optional list of M flags. The i-th element of the list specifies the direction "
        "to be scanned for the i-th scan_input tensor: 0 indicates forward direction and 1 "
        "indicates reverse direction. "
        "If omitted, all scan_input tensors will be scanned in the forward direction.",
        AttributeProto::INTS,
        false)
    .TypeConstraint("I", { "tensor(int64)" }, "Int64 tensor")
    .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types"));
*/

class Scan8Impl {
 public:
  Scan8Impl(OpKernelContextInternal& context,
            const SessionState& session_state,
            const Scan<8>::Info& info,
            const std::vector<int64_t>& directions,
            const scan::detail::DeviceHelpers& device_helpers);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute(const FeedsFetchesManager& cached_ffm);

 private:
  // validate inputs and setup batch size and max sequence length.
  Status ValidateInput();
  Status ValidateSubgraphInput(int start_input, int end_input, bool is_loop_state_var,
                               const std::vector<const NodeArg*>& graph_inputs);

  Status AllocateOutputTensors();
  Status CreateLoopStateVariables(std::vector<std::vector<LoopStateVariable>>& loop_state_variables);

  using ConstTensorSlicerIterators = std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator>;
  using MutableTensorSlicerIterators = std::vector<OrtValueTensorSlicer<OrtValue>::Iterator>;

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const Scan<8>::Info& info_;

  int64_t batch_size_ = -1;
  int64_t max_sequence_len_ = -1;

  const std::vector<int64_t>& directions_;
  const Tensor* sequence_lens_tensor_;
  std::vector<int64_t> sequence_lens_;

  std::vector<std::unique_ptr<OutputIterator>> output_iterators_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  const scan::detail::DeviceHelpers& device_helpers_;
};

template <>
void Scan<8>::Init(const OpKernelInfo& info) {
  // make sure the attribute was present even though we don't need it here.
  // The GraphProto is loaded as a Graph instance by main Graph::Resolve,
  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("body", &proto).IsOK());
  (void)proto;

  ORT_ENFORCE(info.GetAttr<int64_t>("num_scan_inputs", &num_scan_inputs_).IsOK());

  ReadDirections(info, "directions", input_directions_, num_scan_inputs_);

  device_helpers_.transpose_func = [](const std::vector<size_t>&, const Tensor&, Tensor&) -> Status {
    ORT_NOT_IMPLEMENTED("Scan<8> spec does not support transpose of output. This should never be called.");
  };

  device_helpers_.set_data_to_zero_func = [](void* data, size_t size_in_bytes) -> Status {
    memset(data, 0, size_in_bytes);
    return Status::OK();
  };
}

template <>
Status Scan<8>::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                           const std::string& attribute_name,
                                           const SessionState& subgraph_session_state) {
  ORT_ENFORCE(info_ == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");
  ORT_UNUSED_PARAMETER(attribute_name);

  const auto& node = Node();
  info_ = onnxruntime::make_unique<Scan<8>::Info>(node, subgraph_session_state.GetGraphViewer(),
                                                  static_cast<int>(num_scan_inputs_));

  auto status = scan::detail::CreateFeedsFetchesManager(node, *info_, session_state, subgraph_session_state,
                                                        /* is_v8 */ true, feeds_fetches_manager_);

  return status;
}

template <>
Status Scan<8>::Compute(OpKernelContext* ctx) const {
  ORT_ENFORCE(feeds_fetches_manager_ && info_,
              "CreateFeedsFetchesManager must be called prior to execution of graph.");

  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("body");
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");

  Scan8Impl scan_impl{*ctx_internal, *session_state, *info_, input_directions_, device_helpers_};

  auto status = scan_impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = scan_impl.Execute(*feeds_fetches_manager_);

  return status;
}

Scan8Impl::Scan8Impl(OpKernelContextInternal& context,
                     const SessionState& session_state,
                     const Scan<8>::Info& info,
                     const std::vector<int64_t>& directions,
                     const scan::detail::DeviceHelpers& device_helpers)
    : context_(context),
      session_state_(session_state),
      info_(info),
      directions_(directions),
      implicit_inputs_(context_.GetImplicitInputs()),
      device_helpers_(device_helpers) {
  // optional first input so may be nullptr
  sequence_lens_tensor_ = context.Input<Tensor>(0);
}

Status Scan8Impl::Initialize() {
  auto status = ValidateInput();
  ORT_RETURN_IF_ERROR(status);

  status = AllocateOutputTensors();
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

// get the Scan input that is used in a call to the subgraph as a Tensor,
// skipping over the optional arg to the Scan operator
static const Tensor& GetSubgraphInputTensor(const OpKernelContext& context, int index) {
  // skip the optional sequence_lens input
  return *context.Input<Tensor>(index + 1);
}

// get the Scan input that is used in a call to the subgraph as an OrtValue,
// skipping over the optional arg to the Scan operator
static const OrtValue& GetSubgraphInputMLValue(const OpKernelContextInternal& context, int index) {
  // skip the optional sequence_lens input
  return *context.GetInputMLValue(index + 1);
}

// Validate that the subgraph input has valid shapes
Status Scan8Impl::ValidateSubgraphInput(int start_input, int end_input, bool is_loop_state_var,
                                        const std::vector<const NodeArg*>& graph_inputs) {
  // first dim is batch size. optional sequence dim. dim/s for the data.
  // if there is no dim for the data treat it as a scalar.
  bool has_seq_len_dim = !is_loop_state_var;
  auto min_dims_required = has_seq_len_dim ? 2 : 1;

  for (int i = start_input; i < end_input; ++i) {
    auto& input_tensor = GetSubgraphInputTensor(context_, i);
    const auto& input_shape = input_tensor.Shape();

    if (input_shape.NumDimensions() < static_cast<size_t>(min_dims_required))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid scan input:", graph_inputs[i]->Name(),
                             " Expected ", min_dims_required,
                             " dimensions or more but input had shape of ", input_shape);

    auto this_batch_size = input_shape[0];

    if (batch_size_ < 0)
      batch_size_ = this_batch_size;
    else {
      if (batch_size_ != this_batch_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Scan inputs have inconsistent batch size. Previous value was ",
                               batch_size_, " but ", graph_inputs[i]->Name(), " has batch size of ",
                               this_batch_size);
      }
    }

    if (has_seq_len_dim) {
      auto this_seq_len = input_shape[1];

      if (max_sequence_len_ < 0) {
        max_sequence_len_ = this_seq_len;
      } else {
        if (max_sequence_len_ != this_seq_len) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Scan inputs have inconsistent sequence lengths. Previous value was ",
                                 max_sequence_len_, " but ", graph_inputs[i]->Name(),
                                 " has length of ", this_seq_len);
        }
      }
    }
  }

  return Status::OK();
}

Status Scan8Impl::ValidateInput() {
  auto& graph_inputs = info_.subgraph.GetInputs();

  // process any loop state variables, which will set the batch size
  auto status = ValidateSubgraphInput(0, info_.num_loop_state_variables, true, graph_inputs);
  ORT_RETURN_IF_ERROR(status);

  // process the scan inputs. sets/validates batch size and sequence length
  status = ValidateSubgraphInput(info_.num_loop_state_variables, info_.num_variadic_inputs, false, graph_inputs);
  ORT_RETURN_IF_ERROR(status);

  if (sequence_lens_tensor_ != nullptr) {
    auto num_entries = sequence_lens_tensor_->Shape().Size();
    if (num_entries != batch_size_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_lens length of ", num_entries,
                             " did not match batch size of ", batch_size_);
    }

    auto d = sequence_lens_tensor_->DataAsSpan<int64_t>();
    sequence_lens_.assign(d.cbegin(), d.cend());

    if (!std::all_of(sequence_lens_.cbegin(), sequence_lens_.cend(),
                     [this](int64_t value) { return value > 0 && value <= max_sequence_len_; })) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Invalid entries in sequence_lens. Max sequence length was ", max_sequence_len_);
    }

  } else {
    sequence_lens_ = std::vector<int64_t>(batch_size_, max_sequence_len_);
  }

  return Status::OK();
}

Status Scan8Impl::AllocateOutputTensors() {
  Status status = Status::OK();
  auto& graph_outputs = info_.subgraph.GetOutputs();

  if (graph_outputs.size() != static_cast<size_t>(info_.num_outputs)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph in 'body' produces ", graph_outputs.size(),
                           " outputs but Scan expects ", info_.num_outputs);
  }

  std::unique_ptr<OutputIterator> output_iter;

  for (int i = 0; i < info_.num_loop_state_variables; ++i) {
    status = AllocateOutput(context_, info_.subgraph, i, true, batch_size_, max_sequence_len_, output_iter,
                            device_helpers_.create_mutable_slicer_func, device_helpers_.set_data_to_zero_func);
    ORT_RETURN_IF_ERROR(status);
    output_iterators_.push_back(std::move(output_iter));
  }

  for (int i = info_.num_loop_state_variables, end = info_.num_outputs; i < end; ++i) {
    status = AllocateOutput(context_, info_.subgraph, i, false, batch_size_, max_sequence_len_, output_iter,
                            device_helpers_.create_mutable_slicer_func, device_helpers_.set_data_to_zero_func);
    ORT_RETURN_IF_ERROR(status);
    output_iterators_.push_back(std::move(output_iter));
  }

  return Status::OK();
}

// setup the loop state variables for each batch item
Status Scan8Impl::CreateLoopStateVariables(std::vector<std::vector<LoopStateVariable>>& batch_loop_state_variables) {
  // Setup loop state variables
  // 1. Slice the input/output loop state variable tensors provided to Scan into the per-batch-item chunks
  //    (slice on the first dimension which is the batch size).
  // 2. For each batch item, create the LoopStateVariable instances that can be used to pass state between
  //    each iteration of the subgraph. This minimizes copying of data during each iteration.

  std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator> loop_state_input_iterators;
  loop_state_input_iterators.reserve(info_.num_loop_state_variables);

  // create the input and output slice iterator for each loop state variable.
  for (int i = 0; i < info_.num_loop_state_variables; ++i) {
    const OrtValue& ort_value = GetSubgraphInputMLValue(context_, i);
    OrtValue* p_mlvalue = context_.GetOutputMLValue(i);

    ORT_ENFORCE(p_mlvalue, "Output OrtValue has not been created for loop state variable output ", i);

    loop_state_input_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 0, 0).begin());
  }

  batch_loop_state_variables.clear();
  batch_loop_state_variables.resize(batch_size_);

  AllocatorPtr alloc;
  auto status = context_.GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  // setup the loop state variables for each batch row
  for (int64_t b = 0; b < batch_size_; ++b) {
    std::vector<LoopStateVariable>& variables = batch_loop_state_variables[b];
    variables.reserve(info_.num_loop_state_variables);

    for (int i = 0; i < info_.num_loop_state_variables; ++i) {
      auto& input_iter = loop_state_input_iterators[i];
      auto& output_iter = *output_iterators_[i];

      variables.emplace_back(*input_iter, *output_iter, sequence_lens_[b], alloc);

      ++input_iter;
      ++output_iter;
    }
  }

  return status;
}

Status Scan8Impl::Execute(const FeedsFetchesManager& ffm) {
  Status status = Status::OK();

  // for each batch item, std::vector of LoopStateVariables
  std::vector<std::vector<LoopStateVariable>> batch_loop_state_variables;
  status = CreateLoopStateVariables(batch_loop_state_variables);
  ORT_RETURN_IF_ERROR(status);

  for (int64_t b = 0; b < batch_size_; ++b) {
    auto sequence_len = sequence_lens_[b];

    // Setup input OrtValue streams
    std::vector<OrtValueTensorSlicer<const OrtValue>::Iterator> scan_input_stream_iterators;
    scan_input_stream_iterators.reserve(info_.num_variadic_inputs - info_.num_loop_state_variables);

    for (int i = info_.num_loop_state_variables, end = info_.num_variadic_inputs; i < end; ++i) {
      const auto& ort_value = GetSubgraphInputMLValue(context_, i);

      // forward
      if (directions_[i - info_.num_loop_state_variables] == static_cast<int64_t>(ScanDirection::kForward)) {
        // the iterator is self contained, so we don't need to keep the OrtValueTensorSlicer instance around
        scan_input_stream_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 1, b).begin());
      } else {  // reverse
        scan_input_stream_iterators.push_back(device_helpers_.create_const_slicer_func(ort_value, 1, b).rbegin());
        // need to skip past the empty entries at the end of the input if sequence length is short
        auto offset = max_sequence_len_ - sequence_len;
        if (offset > 0) {
          // reverse iterator so += moves backwards through the input
          scan_input_stream_iterators.back() += offset;
        }
      }
    }

    // Call the subgraph for each item in the sequence
    status = IterateSequence(context_, session_state_, batch_loop_state_variables[b], scan_input_stream_iterators,
                             sequence_len, info_.num_loop_state_variables, info_.num_variadic_inputs, info_.num_outputs,
                             implicit_inputs_, output_iterators_, ffm);

    // zero out any remaining values in the sequence
    for (int64_t i = sequence_len; i < max_sequence_len_; ++i) {
      for (int output = info_.num_loop_state_variables; output < info_.num_outputs; ++output) {
        auto& iterator = *output_iterators_[output];
        iterator.ZeroOutCurrent();
        ++iterator;
      }
    }

    ORT_RETURN_IF_ERROR(status);
  }

  return status;
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Scan,
                                   8, 8,
                                   KernelDefBuilder()
                                       .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                   Scan<8>);

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/controlflow/scan.h"

#include "core/framework/framework_common.h"
#include "core/framework/mlvalue_tensor_slicer.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"

#include "core/providers/cpu/tensor/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

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

ONNX_CPU_OPERATOR_KERNEL(Scan,
                         8,
                         KernelDefBuilder()
                             .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                         Scan);

/**
Class to provide input/output MLValue instances for a loop state variable.
The MLValue flips between two internal temporary buffers to minimize copies.
*/
class LoopStateVariable {
 public:
  LoopStateVariable(const MLValue& original_value, MLValue& final_value, const int64_t sequence_len,
                    AllocatorPtr& allocator);

  // get current Input MLValue
  const MLValue& Input() const;

  // get current Output MLValue
  MLValue& Output();

  // move to next usage of the loop state variable. call after each iteration of the subgraph.
  void Next();

 private:
  int64_t iteration_num_{0};
  const int64_t sequence_len_;

  // copy original and final value from temporary MLValue provided by iterator
  const MLValue original_value_;
  MLValue final_value_;

  /* we use original_value and final_value once, 
     and alternate between a_ and b_ as input/output for each iteration to avoid copies

    Iteration   Input             Output
    0           original_value    a_
    1           a_                b_
    2           b_                a_
    ...
    seq len - 1 <previous output> final_value
    */
  MLValue a_;
  MLValue b_;
};

class ScanImpl {
 public:
  ScanImpl(OpKernelContextInternal& context,
           const SessionState& session_state,
           int64_t num_scan_inputs,
           const std::vector<int64_t>& directions);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute();

 private:
  // validate inputs and setup batch size and max sequence length.
  Status ValidateInput();
  Status ValidateSubgraphInput(int start_input, int end_input, bool has_seq_len_dim,
                               const std::vector<const NodeArg*>& graph_inputs);

  Status AllocateOutput(int index, bool has_sequence_len);
  Status AllocateOutputTensors();
  Status CreateLoopStateVariables(std::vector<std::vector<LoopStateVariable>>& loop_state_variables);

  using ConstTensorSlicerIterators = std::vector<MLValueTensorSlicer<const MLValue>::Iterator>;
  using MutableTensorSlicerIterators = std::vector<MLValueTensorSlicer<MLValue>::Iterator>;

  Status IterateSequence(std::vector<LoopStateVariable>& loop_state_variables,
                         ConstTensorSlicerIterators& scan_input_stream_iterators,
                         MutableTensorSlicerIterators& scan_output_stream_iterators,
                         int64_t seq_length);

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const GraphViewer& graph_viewer_;

  int num_loop_state_variables_;
  int num_variadic_inputs_;
  int num_variadic_outputs_;

  int64_t batch_size_ = -1;
  int64_t max_sequence_len_ = -1;

  const std::vector<int64_t>& directions_;
  const Tensor* sequence_lens_tensor_;
  std::vector<int64_t> sequence_lens_;

  std::vector<std::string> output_names_;

  std::unordered_map<std::string, const MLValue*> implicit_inputs_;
};

Status Scan::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("body");
  ONNXRUNTIME_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");

  // TODO:
  //       Consider how usage of ExecutionFrame and SequentialExecutor can be optimized
  //         - initial implementation is focused on making it work, rather than optimizing.

  ScanImpl scan_impl{*ctx_internal, *session_state, num_scan_inputs_, directions_};

  auto status = scan_impl.Initialize();
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  status = scan_impl.Execute();

  return status;
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
  auto allocate_tensor_in_mlvalue = [&]() {
    auto new_tensor = std::make_unique<Tensor>(tensor.DataType(),
                                               shape,
                                               allocator->Alloc(shape.Size() * tensor.DataType()->Size()),
                                               allocator->Info(),
                                               allocator);

    return MLValue{new_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc()};
  };

  // if length is > 1, we need a_ for the first output location. otherwise we use final_value for the output.
  if (sequence_len_ > 1) {
    a_ = allocate_tensor_in_mlvalue();
  }

  // if length is > 2, we need b_ for the second output location
  if (sequence_len_ > 2) {
    b_ = allocate_tensor_in_mlvalue();
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
  ONNXRUNTIME_ENFORCE(iteration_num_ < sequence_len_, "Misuse of LoopStateVariable. Attempt to move beyond end of sequence");
  ++iteration_num_;
}

ScanImpl::ScanImpl(OpKernelContextInternal& context,
                   const SessionState& session_state,
                   int64_t num_scan_inputs,
                   const std::vector<int64_t>& directions)
    : context_{context},
      session_state_{session_state},
      graph_viewer_{*session_state.GetGraphViewer()},
      directions_{directions},
      implicit_inputs_{context_.GetImplicitInputs()} {
  //optional first input so may be nullptr
  sequence_lens_tensor_ = context.Input<Tensor>(0);

  num_variadic_inputs_ = context_.NumVariadicInputs(1);
  num_variadic_outputs_ = context_.OutputCount();

  num_loop_state_variables_ = num_variadic_inputs_ - gsl::narrow_cast<int>(num_scan_inputs);
}

Status ScanImpl::Initialize() {
  auto status = ValidateInput();
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  auto& graph_outputs = graph_viewer_.GetOutputs();
  output_names_.reserve(graph_outputs.size());

  // save list of subgraph output names in their provided order to use when fetching the results
  // from each subgraph execution. the Scan outputs will match this order.
  for (auto& output : graph_outputs) {
    output_names_.push_back(output->Name());
  }

  status = AllocateOutputTensors();
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  return Status::OK();
}

// get the Scan input that is used in a call to the subgraph as a Tensor,
// skipping over the optional arg to the Scan operator
static const Tensor& GetSubgraphInputTensor(const OpKernelContext& context, int index) {
  // skip the optional sequence_lens input
  return *context.Input<Tensor>(index + 1);
}

// get the Scan input that is used in a call to the subgraph as an MLValue,
// skipping over the optional arg to the Scan operator
static const MLValue& GetSubgraphInputMLValue(const OpKernelContextInternal& context, int index) {
  // skip the optional sequence_lens input
  return *context.GetInputMLValue(index + 1);
}

// Validate that the subgraph input has valid shapes
Status ScanImpl::ValidateSubgraphInput(int start_input, int end_input, bool has_seq_len_dim,
                                       const std::vector<const NodeArg*>& graph_inputs) {
  // first dim is batch size. optional sequence dim. dim/s for the data
  auto min_dims_required = has_seq_len_dim ? 3 : 2;

  for (int i = start_input; i < end_input; ++i) {
    auto& input_tensor = GetSubgraphInputTensor(context_, i);
    const auto& input_shape = input_tensor.Shape();

    if (input_shape.NumDimensions() < min_dims_required)
      return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid scan input:", graph_inputs[i]->Name(),
                                     " Expected ", min_dims_required,
                                     " dimensions or more but input had shape of ", input_shape);

    auto this_batch_size = input_shape[0];

    if (batch_size_ < 0)
      batch_size_ = this_batch_size;
    else {
      if (batch_size_ != this_batch_size) {
        return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Scan inputs have inconsistent batch size. Previous value was ",
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
          return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Scan inputs have inconsistent sequence lengths. Previous value was ",
                                         max_sequence_len_, " but ", graph_inputs[i]->Name(),
                                         " has length of ", this_seq_len);
        }
      }
    }
  }

  return Status::OK();
}

Status ScanImpl::ValidateInput() {
  auto& graph_inputs = graph_viewer_.GetInputs();
  auto num_graph_inputs = graph_inputs.size();

  if (num_graph_inputs != num_variadic_inputs_) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "The subgraph in 'body' expects ", num_graph_inputs,
                                   " inputs but Scan was only given ", num_variadic_inputs_);
  }

  // process any loop state variables, which will set the batch size
  auto status = ValidateSubgraphInput(0, num_loop_state_variables_, false, graph_inputs);
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  // process the scan inputs. sets/validates batch size and sequence length
  status = ValidateSubgraphInput(num_loop_state_variables_, num_variadic_inputs_, true, graph_inputs);
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  if (sequence_lens_tensor_ != nullptr) {
    auto num_entries = sequence_lens_tensor_->Shape().Size();

    if (num_entries != batch_size_) {
      return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_lens length of ", num_entries,
                                     " did not match batch size of ", batch_size_);
    }

    auto d = sequence_lens_tensor_->DataAsSpan<int64_t>();
    sequence_lens_.assign(d.cbegin(), d.cend());

    if (std::all_of(sequence_lens_.cbegin(), sequence_lens_.cend(),
                    [this](int64_t value) { return value > 0 && value <= max_sequence_len_; }) == false) {
      return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                     "Invalid entries in sequence_lens. Max sequence length was ", max_sequence_len_);
    }

  } else {
    sequence_lens_ = std::vector<int64_t>(batch_size_, max_sequence_len_);
  }

  return Status::OK();
}

Status ScanImpl::AllocateOutput(int index, bool has_sequence_len_dimension) {
  // use the shape from the subgraph output. we require this to be specified in the model or inferable.
  auto& graph_outputs = graph_viewer_.GetOutputs();
  auto* graph_output = graph_outputs.at(index);
  auto* graph_output_shape = graph_output->Shape();
  if (!graph_output_shape) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph must have the shape set for all outputs but ",
                                   graph_output->Name(), " did not.");
  }

  TensorShape output_shape{onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape)};
  auto& graph_output_dims{output_shape.GetDims()};

  std::vector<int64_t> scan_output_dims;
  scan_output_dims.reserve(graph_output_dims.size() + 2);

  scan_output_dims.push_back(batch_size_);

  if (has_sequence_len_dimension) {
    scan_output_dims.push_back(max_sequence_len_);
  }

  scan_output_dims.insert(scan_output_dims.cend(), graph_output_dims.cbegin(), graph_output_dims.cend());

  // make sure a single buffer for the full output is created upfront.
  // we slice this into per-iteration pieces in Execute using MLValueTensorSlicer.
  auto* tensor = context_.Output(index, TensorShape(scan_output_dims));

  if (!tensor)
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for ", graph_output->Name());

  // zero out the output so that any short sequences have deterministic values in unused slots.
  // strictly speaking this isn't required, and alternatively we could fill with zeros when we
  // encounter a short sequence and are creating output, but one memset is easy, involves
  // less code complexity, and should be relatively cheap.
  memset(tensor->MutableDataRaw(), 0, tensor->Size());

  return Status::OK();
}

Status ScanImpl::AllocateOutputTensors() {
  Status status = Status::OK();
  auto& graph_outputs = graph_viewer_.GetOutputs();

  if (graph_outputs.size() != num_variadic_outputs_) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph in 'body' produces ", graph_outputs.size(),
                                   " outputs but Scan expects ", num_variadic_outputs_);
  }

  // TODO: Need to handle shape/type inference for subgraphs.
  // For now copy shape from subgraph output and expand based on batch size and sequence length

  for (int i = 0; i < num_loop_state_variables_; ++i) {
    const bool has_sequence_len_dimension = false;  // loop state variables don't have a sequence_len dimension;
    status = AllocateOutput(i, has_sequence_len_dimension);
    ONNXRUNTIME_RETURN_IF_ERROR(status);
  }

  for (int i = num_loop_state_variables_, end = num_variadic_outputs_; i < end; ++i) {
    status = AllocateOutput(i, true);
    ONNXRUNTIME_RETURN_IF_ERROR(status);
  }

  return Status::OK();
}

// setup the loop state variables for each batch item
Status ScanImpl::CreateLoopStateVariables(std::vector<std::vector<LoopStateVariable>>& batch_loop_state_variables) {
  // Setup loop state variables
  // 1. Slice the input/output loop state variable tensors provided to Scan into the per-batch-item chunks
  //    (slice on the first dimension which is the batch size).
  // 2. For each batch item, create the LoopStateVariable instances that can be used to pass state between
  //    each iteration of the subgraph. This minimizes copying of data during each iteration.

  std::vector<MLValueTensorSlicer<const MLValue>::Iterator> loop_state_input_iterators;
  std::vector<MLValueTensorSlicer<MLValue>::Iterator> loop_state_output_iterators;
  loop_state_input_iterators.reserve(num_loop_state_variables_);
  loop_state_output_iterators.reserve(num_loop_state_variables_);

  // create the input and output slice iterator for each loop state variable.
  for (int i = 0; i < num_loop_state_variables_; ++i) {
    const MLValue& mlvalue = GetSubgraphInputMLValue(context_, i);
    MLValue* p_mlvalue = context_.GetOutputMLValue(i);

    ONNXRUNTIME_ENFORCE(p_mlvalue, "Output MLValue has not been created for loop state variable output ", i);

    loop_state_input_iterators.push_back(MLValueTensorSlicer<const MLValue>::Create(mlvalue).begin());
    loop_state_output_iterators.push_back(MLValueTensorSlicer<MLValue>::Create(*p_mlvalue).begin());
  }

  batch_loop_state_variables.clear();
  batch_loop_state_variables.resize(batch_size_);

  AllocatorPtr alloc;
  auto status = context_.GetTempSpaceAllocator(&alloc);
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  // setup the loop state variables for each batch row
  for (int64_t b = 0; b < batch_size_; ++b) {
    std::vector<LoopStateVariable>& variables = batch_loop_state_variables[b];
    variables.reserve(num_loop_state_variables_);

    for (int i = 0; i < num_loop_state_variables_; ++i) {
      auto& input_iter = loop_state_input_iterators[i];
      auto& output_iter = loop_state_output_iterators[i];

      variables.push_back(LoopStateVariable(*input_iter, *output_iter, sequence_lens_[b], alloc));

      ++input_iter;
      ++output_iter;
    }
  }

  return status;
}

Status ScanImpl::Execute() {
  Status status = Status::OK();

  // for each batch item, std::vector of LoopStateVariables
  std::vector<std::vector<LoopStateVariable>> batch_loop_state_variables;
  status = CreateLoopStateVariables(batch_loop_state_variables);
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  for (int64_t b = 0; b < batch_size_; ++b) {
    // Setup input MLValue streams
    std::vector<MLValueTensorSlicer<const MLValue>::Iterator> scan_input_stream_iterators;
    scan_input_stream_iterators.reserve(num_variadic_inputs_ - num_loop_state_variables_);

    for (int i = num_loop_state_variables_, end = num_variadic_inputs_; i < end; ++i) {
      const auto& mlvalue = GetSubgraphInputMLValue(context_, i);

      // forward
      if (directions_[i - num_loop_state_variables_] == static_cast<int64_t>(Scan::Direction::kForward)) {
        // the iterator is self contained, so we don't need to keep the MLValueTensorSlicer instance around
        scan_input_stream_iterators.push_back(MLValueTensorSlicer<const MLValue>::Create(mlvalue, 1, b).begin());
      } else {  // reverse
        scan_input_stream_iterators.push_back(MLValueTensorSlicer<const MLValue>::Create(mlvalue, 1, b).rbegin());
      }
    }

    // Setup output MLValue streams
    std::vector<MLValueTensorSlicer<MLValue>::Iterator> scan_output_stream_iterators;
    scan_output_stream_iterators.reserve(num_variadic_outputs_);

    for (int i = num_loop_state_variables_, end = num_variadic_outputs_; i < end; ++i) {
      MLValue* p_mlvalue = context_.GetOutputMLValue(i);
      ONNXRUNTIME_ENFORCE(p_mlvalue, "Output MLValue has not been created for output ", i);

      scan_output_stream_iterators.push_back(MLValueTensorSlicer<MLValue>::Create(*p_mlvalue, 1, b).begin());
    }

    // Call the subgraph for each item in the sequence
    status = IterateSequence(batch_loop_state_variables[b],
                             scan_input_stream_iterators,
                             scan_output_stream_iterators,
                             sequence_lens_[b]);

    ONNXRUNTIME_RETURN_IF_ERROR(status);
  }

  return status;
}

Status ScanImpl::IterateSequence(std::vector<LoopStateVariable>& loop_state_variables,
                                 ConstTensorSlicerIterators& scan_input_stream_iterators,
                                 MutableTensorSlicerIterators& scan_output_stream_iterators,
                                 int64_t seq_length) {
  Status status = Status::OK();
  auto& graph_inputs = graph_viewer_.GetInputs();
  NameMLValMap feeds;
  std::vector<MLValue> fetches;

  feeds.reserve(num_variadic_inputs_ + implicit_inputs_.size());
  fetches.resize(num_variadic_outputs_);

  // pass in implicit inputs as feeds.
  for (auto& entry : implicit_inputs_) {
    ONNXRUNTIME_ENFORCE(entry.second, "All implicit inputs should have MLValue instances by now. ",
                        entry.first, " did not.");
    feeds[entry.first] = *entry.second;
  }

  // as we fill all the outputs with 0 initially, just iterate seq_length not max_seq_length_
  // as we don't need to pad the output for a short sequence here.
  for (int64_t seq_no = 0; seq_no < seq_length; ++seq_no) {
    for (int input = 0; input < num_variadic_inputs_; ++input) {
      // the ordering of the Scan inputs should match the ordering of the subgraph inputs
      auto name = graph_inputs[input]->Name();

      if (input < num_loop_state_variables_) {
        // add loop state variable input
        feeds[name] = loop_state_variables[input].Input();
      } else {
        // add sliced input
        auto& iterator = scan_input_stream_iterators[input - num_loop_state_variables_];
        feeds[name] = *iterator;

        ++iterator;
      }
    }

    fetches.clear();

    for (int output = 0, end = num_variadic_outputs_; output < end; ++output) {
      if (output < num_loop_state_variables_) {
        // add loop state variable output
        fetches.push_back(loop_state_variables[output].Output());
      } else {
        // add sliced output
        auto& iterator = scan_output_stream_iterators[output - num_loop_state_variables_];
        fetches.push_back(*iterator);
        ++iterator;
      }
    }

    // Create Executor and run graph.
    // TODO: Consider pulling ExecutionFrame up from within SequentialExecutor
    // and separating it out a bit so we can maybe just update the feeds/fetches in the frame on each iteration.
    // Many of the other pieces are constant across usages.
    // Not sure how best to handle the memory pattern side of things though.
    // For now just making it work. Optimization and refinement will follow.
    SequentialExecutor executor;
    status = executor.Execute(session_state_, feeds, output_names_, fetches, context_.Logger());
    ONNXRUNTIME_RETURN_IF_ERROR(status);

    // cycle the LoopStateVariable input/output in preparation for the next iteration
    std::for_each(loop_state_variables.begin(), loop_state_variables.end(), [](LoopStateVariable& v) { v.Next(); });
  }

  return status;
}

}  // namespace onnxruntime

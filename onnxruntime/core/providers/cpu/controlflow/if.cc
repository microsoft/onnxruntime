// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/controlflow/if.h"

#include "core/framework/execution_frame.h"
#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"

#include "core/framework/tensorprotoutils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

/*
ONNX_OPERATOR_SET_SCHEMA(
    If,
    1,
    OpSchema()
        .SetDoc("If conditional")
        .Input(0, "cond", "Condition for the if", "B")
        .Output(
            0,
            "outputs",
            "Values that are live-out to the enclosing scope. The return values in "
            "the `then_branch` and `else_branch` must be of the same shape and same "
            "data type.",
            "V",
            OpSchema::Variadic)
        .Attr(
            "then_branch",
            "Graph to run if condition is true. Has N outputs: values you wish to "
            "be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the else_branch.",
            AttributeProto::GRAPH)
        .Attr(
            "else_branch",
            "Graph to run if condition is false. Has N outputs: values you wish to"
            " be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the then_branch.",
            AttributeProto::GRAPH)
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("B", {"tensor(bool)"}, "Only bool"));
*/

ONNX_CPU_OPERATOR_KERNEL(If,
                         1,
                         KernelDefBuilder()
                             .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                         If);

class IfImpl {
 public:
  IfImpl(OpKernelContextInternal& context,
         const SessionState& session_state);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute();

 private:
  Status AllocateOutputTensors();

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const GraphViewer& subgraph_;

  int num_outputs_;
  std::vector<std::string> subgraph_output_names_;
  std::unordered_map<std::string, const MLValue*> implicit_inputs_;

  enum class AllocationType {
    Delayed,  // allocation of If output will be done by subgraph execution
    IfOutput
  };

  // track where the fetches provided to subgraph execution were allocated.
  std::vector<std::pair<AllocationType, MLValue>> outputs_;
};

Status If::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);

  auto condition = *ctx->Input<Tensor>(0)->Data<bool>();

  auto attribute = condition ? "then_branch" : "else_branch";
  auto* session_state = ctx_internal->SubgraphSessionState(attribute);
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for '", attribute, "' attribute.");

  IfImpl impl{*ctx_internal, *session_state};

  auto status = impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = impl.Execute();

  return status;
}

IfImpl::IfImpl(OpKernelContextInternal& context,
               const SessionState& session_state)
    : context_{context},
      session_state_{session_state},
      subgraph_{*session_state.GetGraphViewer()},
      implicit_inputs_{context_.GetImplicitInputs()} {
  num_outputs_ = context_.OutputCount();
}

Status IfImpl::Initialize() {
  auto& graph_outputs = subgraph_.GetOutputs();
  auto num_subgraph_outputs = graph_outputs.size();

  if (num_subgraph_outputs != num_outputs_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'If' node has ", num_outputs_,
                           " outputs which doesn't match the subgraph's ", num_subgraph_outputs, " outputs.");
  }

  subgraph_output_names_.reserve(num_subgraph_outputs);

  // save list of subgraph output names in their provided order to use when fetching the results
  // from each subgraph execution. the If outputs will match this order.
  for (auto& output : graph_outputs) {
    subgraph_output_names_.push_back(output->Name());
  }

  auto status = AllocateOutputTensors();
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

Status IfImpl::AllocateOutputTensors() {
  Status status = Status::OK();
  int index = 0;

  for (auto& graph_output : subgraph_.GetOutputs()) {
    auto* graph_output_shape = graph_output->Shape();
    if (!graph_output_shape) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph must have the shape set for all outputs but ",
                             graph_output->Name(), " did not.");
    }

    TensorShape output_shape{onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape)};

    // if size < 0 we have a symbolic dimension and need to use a temporary MLValue in the subgraph execution
    if (output_shape.Size() < 0) {
      // we still need a value to put in the feeds we give to the execution frame, so just use an empty MLValue
      outputs_.push_back({AllocationType::Delayed, {}});
    } else {
      auto* tensor = context_.Output(index, output_shape);

      if (!tensor)
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for ", graph_output->Name());

      outputs_.push_back({AllocationType::IfOutput, *context_.GetOutputMLValue(index)});
    }

    ++index;
  }

  return Status::OK();
}

Status IfImpl::Execute() {
  Status status = Status::OK();

  NameMLValMap feeds;

  feeds.reserve(implicit_inputs_.size());
  auto& mlvalue_name_idx_map = session_state_.GetMLValueNameIdxMap();

  // pass in implicit inputs as feeds.
  for (auto& entry : implicit_inputs_) {
    ORT_ENFORCE(entry.second, "All implicit inputs should have MLValue instances by now. ",
                entry.first, " did not.");

    // prune to values that are in this subgraph as the implicit inputs cover both 'then' and 'else' subgraphs.
    // alternatively we could track implicit inputs on a per-attribute basis in the node, but that
    // would make that tracking a bit more complicated.
    int idx;
    if (mlvalue_name_idx_map.GetIdx(entry.first, idx).IsOK()) {
      feeds[entry.first] = *entry.second;
    }
  }
  std::vector<MLValue> fetches;
  std::unordered_map<size_t, IExecutor::CustomAllocator> fetch_allocators;
  fetches.reserve(num_outputs_);

  for (int i = 0; i < num_outputs_; ++i) {
    fetches.push_back(outputs_[i].second);

    if (outputs_[i].first == AllocationType::Delayed) {
      // functor to forward the allocation request from the subgraph to the If node's context so that the
      // allocation plan for the If node's output is used.
      fetch_allocators[i] =
          [this, i](const TensorShape& shape, MLValue& mlvalue) {
            // allocate
            auto* tensor = context_.Output(i, shape);

            if (!tensor)
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for If output ", i);

            // return MLValue for allocated tensor
            mlvalue = *context_.GetOutputMLValue(i);
            return Status::OK();
          };
    }
  }

  status = utils::ExecuteGraph(session_state_, feeds, subgraph_output_names_, fetches, fetch_allocators,
                               /*sequential_execution*/ true, context_.GetTerminateFlag(), context_.Logger());
  ORT_RETURN_IF_ERROR(status);

  return status;
}

}  // namespace onnxruntime

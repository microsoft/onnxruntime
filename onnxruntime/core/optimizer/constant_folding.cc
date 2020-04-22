// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/constant_folding.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

Status ConstantFolding::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    InitializedTensorSet constant_inputs;

    // we currently constant fold using the CPU EP only.
    // if the node is assigned to a different EP we can run it if it's an ONNX op as we have CPU based implementations
    // for all ONNX ops. if it's from a different domain we can't.
    // NOTE: This is in addition to the IsSupportedProvider check below which will optionally do further filtering
    // on the EPs we constant fold for.
    auto ep_type = node->GetExecutionProviderType();
    bool cpu_ep = ep_type == kCpuExecutionProvider;
    if (!cpu_ep && node->Domain() != kOnnxDomain) {
      continue;
    }

    // Check if constant folding can be applied on this node.
    if (!graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        excluded_op_types_.find(node->OpType()) != excluded_op_types_.end() ||
        // constant folding does not support executing a node that includes subgraphs (control flow operators,
        // such as If/Loop/Scan, fall into this category). individual nodes in the subgraph will be processed
        // by the Recurse call above
        node->ContainsSubgraph() || !graph_utils::AllNodeInputsAreConstant(graph, *node, constant_inputs)) {
      continue;
    }

    // override the EP while setting up OptimizerExecutionFrame::Info so that it will use the CPU kernel for Compute.
    if (!cpu_ep) {
      node->SetExecutionProviderType(kCpuExecutionProvider);
    }

    // Create execution frame for executing constant nodes.
    OptimizerExecutionFrame::Info info({node}, constant_inputs);

    // undo the EP change in case something fails prior to node removal
    if (!cpu_ep) {
      node->SetExecutionProviderType(ep_type);
    }

    std::vector<int> fetch_mlvalue_idxs;
    for (const auto* node_out : node->OutputDefs()) {
      fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
    }

    OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

    auto* kernel = info.GetKernel(node->Index());
    if (kernel == nullptr)
      continue;
    OpKernelContext op_kernel_context(&frame, kernel, nullptr, logger);

    ORT_RETURN_IF_ERROR(kernel->Compute(&op_kernel_context));

    std::vector<OrtValue> fetches;
    ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));

    // Go over all output node args and substitute them with the newly computed tensors, which will be
    // added to the graph as initializers.
    ORT_ENFORCE(fetches.size() == node->OutputDefs().size());
    bool unsupported_output_type = false;
    for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
      OrtValue& ort_value = fetches[fetch_idx];

      if (!ort_value.IsTensor()) {
        LOGS(logger, WARNING) << "Unsupported output type of " << ort_value.Type()
                              << ". Can't constant fold " << node->OpType() << " node '" << node->Name() << "'";
        unsupported_output_type = true;
        break;
      }

      // Build the TensorProto that corresponds to the computed OrtValue and add it as initializer to the graph.
      const auto* constant_arg_out = node->OutputDefs()[fetch_idx];
      ORT_ENFORCE(ort_value.IsTensor());
      const Tensor& out_tensor = ort_value.Get<Tensor>();
      ONNX_NAMESPACE::TensorProto out_tensorproto = utils::TensorToTensorProto(out_tensor, constant_arg_out->Name());

      graph.AddInitializedTensor(out_tensorproto);
    }

    if (unsupported_output_type)
      continue;

    // Remove the output edges of the constant node and then remove the node itself.
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());

    // The output nodes already have the right input arg, since we used the same name in the initializer.
    // We could remove unused graph initializers here, but Graph::Resolve() will take care of it.

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/execution_provider.h"

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/murmurhash3.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

std::vector<std::unique_ptr<ComputeCapability>>
IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                  const IKernelLookup& kernel_lookup,
                                  const GraphOptimizerRegistry&,
                                  IResourceAccountant*) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (const auto& node : graph.Nodes()) {
    if (const KernelCreateInfo* kernel_create_info = kernel_lookup.LookUpKernel(node);
        kernel_create_info != nullptr) {
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    } else if (node.GetExecutionProviderType().empty()) {
      // No kernel was found in this EP's registry for a still-unassigned node. Logging the op type,
      // domain and resolved opset makes it obvious why a node is not claimed by the EP (and therefore
      // falls back to another provider such as the CPU EP). This is the root-cause signal for
      // unexpected CPU fallbacks, e.g. an op registered only for a different opset range or
      // domain/layout. Skipping nodes already assigned to another EP keeps the output readable.
      LOGS_DEFAULT(VERBOSE) << "EP [" << Type() << "] has no kernel for node '" << node.Name()
                            << "': " << node.OpType() << "(" << node.SinceVersion() << ") domain=["
                            << node.Domain() << "] -> not claimed by this EP.";
    }
  }

  return result;
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& /*fused_nodes_and_graphs*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                        "IExecutionProvider::Compile with FusedNodeAndGraph is not implemented by " + type_);
}

#endif
}  // namespace onnxruntime

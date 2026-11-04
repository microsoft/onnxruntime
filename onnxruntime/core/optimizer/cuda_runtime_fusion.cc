// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/cuda_runtime_fusion_internal.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include "core/graph/graph_viewer.h"

namespace onnxruntime::cuda_runtime_fusion {
namespace detail {

Status ProbeTarget(Graph& graph, const rewrite::Replacement& recipe,
                   const rewrite::Replacement::TensorBindings& bindings,
                   const NodeAttributes& attributes, size_t output_count,
                   const IExecutionProvider::IKernelLookup& lookup, bool& available) {
  rewrite::Replacement::Instantiation instantiated;
  ORT_RETURN_IF_ERROR(recipe.Instantiate(graph, bindings, attributes, instantiated));
  ORT_RETURN_IF_NOT(instantiated.nodes.size() == 1 && instantiated.outputs.size() == output_count,
                    "CUDA runtime fusion target recipe must emit one node with the expected outputs.");
  InlinedVector<const NodeArg*> outputs(instantiated.outputs.begin(), instantiated.outputs.end());
  graph.SetOutputs(outputs);
  ORT_RETURN_IF_ERROR(graph.Resolve());
  available = lookup.LookUpKernel(*graph.GetNode(instantiated.nodes.front())) != nullptr;
  return Status::OK();
}

std::unique_ptr<ComputeCapability> MakeCapability(gsl::span<const NodeIndex> source_ids,
                                                  OptimizationFunction optimization_func) {
  auto subgraph = std::make_unique<IndexedSubGraph>();
  subgraph->nodes.assign(source_ids.begin(), source_ids.end());
  auto parent = std::make_unique<ComputeCapability>(std::move(subgraph));
  auto optimization = std::make_unique<ComputeCapability>(std::make_unique<IndexedSubGraph>());
  optimization->sub_graph->nodes.assign(source_ids.begin(), source_ids.end());
  optimization->optimization_func = std::move(optimization_func);
  parent->nodes_to_optimize.push_back(std::move(optimization));
  return parent;
}

}  // namespace detail

Status Select(const GraphViewer& viewer, const IExecutionProvider::IKernelLookup& kernel_lookup,
              const GraphOptimizerRegistry& registry, const std::string& rule_id,
              std::vector<std::unique_ptr<ComputeCapability>>& capabilities) {
  ORT_RETURN_IF_NOT(capabilities.empty(), "CUDA runtime fusion requires initially empty output.");
  ORT_RETURN_IF_NOT(rule_id == kMobiusBf16Rule || rule_id == kUnitSwishRule,
                    "Unknown CUDA runtime fusion rule: ", rule_id);
  const auto* config = rule_id == kUnitSwishRule ? kUnitSwishConfig : kCausalConvConfig;
  const auto setting = registry.GetSessionOptions().config_options.GetConfigOrDefault(config, "0");
  ORT_RETURN_IF_NOT(setting == "0" || setting == "1", config, " must be exactly 0 or 1.");
  if (!registry.SupportsCapabilityOptimizations()) return Status::OK();
  if (registry.GetSessionOptions().graph_optimization_level < TransformerLevel::Level2) return Status::OK();
  if (setting == "0" || viewer.GetFilterInfo() || viewer.IsSubgraph()) return Status::OK();
  if (rule_id == kUnitSwishRule) {
    return detail::SwishFusion::Discover(viewer, kernel_lookup, *registry.GetLogger(), capabilities);
  }
  return detail::CausalConvFusion::Discover(viewer, kernel_lookup, *registry.GetLogger(), capabilities);
}

}  // namespace onnxruntime::cuda_runtime_fusion

#endif

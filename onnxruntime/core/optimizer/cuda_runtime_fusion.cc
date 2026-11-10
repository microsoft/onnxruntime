// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/cuda_runtime_fusion_internal.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>

#include "contrib_ops/cuda/fusion/cuda_source_replacements.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime::contrib::cuda::fusion {

gsl::span<const SourceReplacement* const> CudaSourceReplacements() {
  static const InlinedVector<const SourceReplacement*> sources = [] {
    InlinedVector<const SourceReplacement*> collected;
    for (const auto& source : ScaledSiLUSources()) collected.push_back(&source);
    for (const auto& source : CausalConvWithStateSources()) collected.push_back(&source);
    return collected;
  }();
  return sources;
}

Status ValidateCudaSourceReplacements(gsl::span<const SourceReplacement* const> sources,
                                      gsl::span<const OpKey> schemas,
                                      gsl::span<const SourcePolicyRecord> policies,
                                      gsl::span<const OpKey> legacy) {
  InlinedVector<ExecutableSource> metadata;
  for (const auto* source : sources) {
    ORT_RETURN_IF_NOT(source && source->discover && !source->config.empty() &&
                          !source->rule_id.empty() && source->rule_id == source->body.rule.name,
                      "Malformed CUDA source replacement declaration.");
    ORT_RETURN_IF_NOT(source->body.rule.replacement.Parsed(),
                      "CUDA v0 source replacement requires a parsed replacement DAG; callback-only bodies "
                      "cannot be declared Available.");
    ORT_RETURN_IF_NOT(SupportsCudaReplacement(*source->body.rule.replacement.Parsed(),
                                              source->body.rule.pattern.Outputs().size()),
                      "CUDA v0 source replacement requires one target with identity output mappings.");
    metadata.push_back(source->Metadata());
  }
  return ValidateSourceDeclarations(schemas, policies, metadata, legacy);
}

bool SupportsCudaReplacement(const rewrite::Replacement& recipe, size_t output_count) {
  if (!recipe.Plan() || recipe.Plan()->size() != 1 || recipe.Outputs().size() != output_count) return false;
  if (recipe.Plan()->front().output_count != output_count) return false;
  for (size_t i = 0; i < output_count; ++i) {
    const auto& output = recipe.Outputs()[i];
    if (output.node != 0 || output.slot != i) return false;
  }
  return true;
}
}  // namespace onnxruntime::contrib::cuda::fusion

namespace onnxruntime::cuda_runtime_fusion {
namespace detail {

Status ProbeTarget(Graph& graph, const rewrite::Replacement& recipe,
                   const rewrite::Replacement::TensorBindings& bindings,
                   const NodeAttributes& attributes, size_t output_count,
                   const IExecutionProvider::IKernelLookup& lookup, bool& available) {
  available = false;
  if (!contrib::cuda::fusion::SupportsCudaReplacement(recipe, output_count)) return Status::OK();
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
  const auto sources = contrib::cuda::fusion::CudaSourceReplacements();
  const auto found = std::find_if(sources.begin(), sources.end(),
                                  [&](const auto* source) { return source->rule_id == rule_id; });
  ORT_RETURN_IF_NOT(found != sources.end(), "Unknown CUDA runtime fusion rule: ", rule_id);
  const auto& source = **found;
  const auto& config = source.config;
  const auto setting = registry.GetSessionOptions().config_options.GetConfigOrDefault(config, "0");
  ORT_RETURN_IF_NOT(setting == "0" || setting == "1", config, " must be exactly 0 or 1.");
  if (!registry.SupportsCapabilityOptimizations()) return Status::OK();
  if (registry.GetSessionOptions().graph_optimization_level < TransformerLevel::Level2) return Status::OK();
  if (setting == "0" || viewer.GetFilterInfo() || viewer.IsSubgraph()) return Status::OK();
  return source.discover(source, viewer, kernel_lookup, *registry.GetLogger(), capabilities);
}

}  // namespace onnxruntime::cuda_runtime_fusion

#endif

#include "core/optimizer/function_extractor.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/optimizer/function_extractor_matcher.h"
#include "core/optimizer/function_extractor_pattern.h"

namespace onnxruntime {
namespace {

using function_extractor_internal::CompiledFunctionPattern;
using function_extractor_internal::ReplacementPlan;

common::Status ApplyReplacementPlan(
    Graph& graph,
    const ONNX_NAMESPACE::FunctionProto& function_proto,
    const ReplacementPlan& plan,
    bool& call_added) {
  call_added = false;
  InlinedHashSet<NodeIndex> removable(plan.removable_node_indices.begin(),
                                      plan.removable_node_indices.end());

  for (auto node_index = plan.removable_node_indices.rbegin();
       node_index != plan.removable_node_indices.rend(); ++node_index) {
    Node* node = graph.GetNode(*node_index);
    ORT_RETURN_IF(node == nullptr, "FunctionExtractor replacement plan became stale during application.");
    graph_utils::RemoveNodeOutputEdges(graph, *node);
  }
  for (auto node_index = plan.removable_node_indices.rbegin();
       node_index != plan.removable_node_indices.rend(); ++node_index) {
    ORT_RETURN_IF_NOT(graph.RemoveNode(*node_index),
                      "FunctionExtractor failed to remove a planned node.");
  }

  Node& call = graph.AddNode(plan.generated_call_name,
                             function_proto.name(),
                             "Function call created by FunctionExtractor",
                             plan.call_inputs,
                             plan.call_outputs,
                             nullptr,
                             function_proto.domain());
  call_added = true;
  call.SetOverload(function_proto.overload());
  call.SetLayeringAnnotation(plan.layering_annotation);

  for (size_t input_index = 0; input_index < plan.call_inputs.size(); ++input_index) {
    const auto* input = plan.call_inputs[input_index];
    const auto producer_edge = std::find_if(
        plan.explicit_input_edges.begin(), plan.explicit_input_edges.end(),
        [&](const graph_utils::GraphEdge& edge) { return edge.arg_name == input->Name(); });
    if (producer_edge != plan.explicit_input_edges.end() &&
        removable.find(producer_edge->src_node) == removable.end()) {
      graph.AddEdge(producer_edge->src_node, call.Index(), producer_edge->src_arg_index,
                    static_cast<int>(input_index));
    }
  }

  for (const auto& edge : plan.explicit_output_edges) {
    const auto output = std::find_if(
        plan.call_outputs.begin(), plan.call_outputs.end(),
        [&](const NodeArg* value) { return value->Name() == edge.arg_name; });
    ORT_RETURN_IF(output == plan.call_outputs.end(),
                  "FunctionExtractor plan contains an edge from a private output.");
    graph.AddEdge(call.Index(), edge.dst_node,
                  static_cast<int>(std::distance(plan.call_outputs.begin(), output)),
                  edge.dst_arg_index);
  }
  return common::Status::OK();
}

}  // namespace

namespace function_extractor_internal {

FunctionExtractionResult ExtractGraph(
    Graph& graph,
    const NormalizedFunctionPattern& normalized_pattern,
    const FunctionExtractorOptions& options,
    const ExtractionControls& controls) {
  FunctionExtractionResult result;
  if (!normalized_pattern.construction_status.IsOK()) {
    result.status = normalized_pattern.construction_status;
    return result;
  }
  if (graph.GraphResolveNeeded()) {
    result.status = ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "FunctionExtractor requires a resolved graph; GraphResolveNeeded() is true.");
    return result;
  }
  for (const auto& node : graph.Nodes()) {
    if (node.Op() == nullptr) {
      result.status = ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "FunctionExtractor requires every target node to have a resolved schema.");
      return result;
    }
  }

  result.status = ValidateRegisteredFunction(normalized_pattern, graph);
  if (!result.status.IsOK()) return result;

  CompiledFunctionPattern compiled_pattern;
  result.status = CompileFunctionPattern(normalized_pattern, graph, compiled_pattern);
  if (!result.status.IsOK()) return result;

  const size_t pass_cap =
      controls.maximum_passes.value_or(static_cast<size_t>(graph.NumberOfNodes()));
  std::unordered_set<std::string> literal_initializers_to_preserve;
  for (size_t pass = 0; pass <= pass_cap; ++pass) {
    std::vector<ReplacementPlan> selected_plans;
    {
      TargetGraphSnapshot snapshot;
      result.status = BuildTargetGraphSnapshot(graph, compiled_pattern, options, snapshot);
      if (!result.status.IsOK()) return result;

      std::vector<ReplacementPlan> discovered_plans;
      result.status = DiscoverReplacementPlans(
          compiled_pattern, snapshot, options, discovered_plans);
      if (!result.status.IsOK()) return result;

      std::vector<size_t> selected_indices;
      result.status = SelectNonConflictingPlans(discovered_plans, selected_indices);
      if (!result.status.IsOK()) return result;
      if (selected_indices.empty()) {
        result.status = common::Status::OK();
        return result;
      }
      selected_plans.reserve(selected_indices.size());
      for (const auto index : selected_indices) {
        selected_plans.push_back(std::move(discovered_plans[index]));
      }
      for (auto& plan : selected_plans) {
        plan.generated_call_name = graph.GenerateNodeName(normalized_pattern.function_proto.name());
      }
      result.status = PrevalidatePlans(graph, compiled_pattern, selected_plans);
      if (!result.status.IsOK()) return result;
    }

    if (pass >= pass_cap) {
      result.status = ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "FunctionExtractor reached its defensive pass cap despite strict node-count decrease.");
      return result;
    }

    std::sort(selected_plans.begin(), selected_plans.end(),
              [](const ReplacementPlan& lhs, const ReplacementPlan& rhs) {
                return lhs.primary_root_topological_position > rhs.primary_root_topological_position;
              });
    for (const auto& plan : selected_plans) {
      for (const auto& witness : plan.literal_witnesses) {
        if (witness.is_initializer) {
          literal_initializers_to_preserve.insert(witness.target_value->Name());
        }
      }
      bool call_added = false;
      result.status =
          ApplyReplacementPlan(graph, normalized_pattern.function_proto, plan, call_added);
      if (call_added) ++result.replacements_applied;
      if (!result.status.IsOK()) return result;
    }

    graph.SetGraphResolveNeeded().SetGraphProtoSyncNeeded();
    Graph::ResolveOptions resolve_options;
    resolve_options.initializer_names_to_preserve = &literal_initializers_to_preserve;
    result.status = controls.resolve_graph != nullptr
                        ? controls.resolve_graph(graph, resolve_options)
                        : graph.Resolve(resolve_options);
    if (!result.status.IsOK()) return result;

    result.status = ValidateRegisteredFunction(normalized_pattern, graph);
    if (!result.status.IsOK()) return result;
    result.status = CompileFunctionPattern(normalized_pattern, graph, compiled_pattern);
    if (!result.status.IsOK()) return result;
  }

  result.status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "FunctionExtractor pass invariant failure.");
  return result;
}

}  // namespace function_extractor_internal

struct FunctionExtractor::Impl {
  Impl(const ONNX_NAMESPACE::FunctionProto& function_proto, FunctionExtractorOptions extractor_options)
      : owned_function_proto(function_proto),
        options(std::move(extractor_options)),
        normalized_pattern(
            function_extractor_internal::NormalizeFunctionPattern(owned_function_proto, options)),
        construction_status(normalized_pattern.construction_status) {}

  ONNX_NAMESPACE::FunctionProto owned_function_proto;
  FunctionExtractorOptions options;
  function_extractor_internal::NormalizedFunctionPattern normalized_pattern;
  common::Status construction_status;
};

FunctionExtractor::FunctionExtractor(
    const ONNX_NAMESPACE::FunctionProto& function_proto,
    FunctionExtractorOptions options)
    : impl_(std::make_unique<Impl>(function_proto, std::move(options))) {}

FunctionExtractor::~FunctionExtractor() = default;

FunctionExtractionResult FunctionExtractor::Extract(Model& model) {
  return Extract(model.MainGraph());
}

FunctionExtractionResult FunctionExtractor::Extract(Graph& graph) {
  return function_extractor_internal::ExtractGraph(
      graph, impl_->normalized_pattern, impl_->options);
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

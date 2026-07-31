#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/function_extractor_pattern.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace function_extractor_internal {

struct ProducerSlot {
  NodeIndex node_index{};
  size_t output_index{};
};

struct ConsumerSlot {
  NodeIndex node_index{};
  size_t input_index{};
};

struct TargetGraphSnapshot {
  const Graph* graph{};
  std::unique_ptr<GraphViewer> graph_viewer;
  InlinedVector<NodeIndex> topological_node_indices;
  InlinedHashMap<NodeIndex, size_t> topological_positions;
  InlinedHashMap<const NodeArg*, ProducerSlot> producers;
  InlinedHashMap<const NodeArg*, InlinedVector<ConsumerSlot>> explicit_consumers;
  InlinedHashMap<const NodeArg*, InlinedVector<NodeIndex>> implicit_consumers;
  InlinedHashSet<const NodeArg*> graph_outputs;
  InlinedHashMap<std::string, const ONNX_NAMESPACE::TensorProto*> constant_initializers;
};

enum class ValueVisitState : uint8_t {
  Unseen,
  Scheduled,
  Processed,
};

struct LiteralWitness {
  PatternValueId pattern_value_id{kMissingPatternValue};
  const NodeArg* target_value{};
  NodeIndex constant_node_index{};
  bool is_initializer{};
  std::string fingerprint;
};

struct MatchState {
  InlinedVector<NodeIndex> pattern_node_to_target;
  InlinedHashMap<NodeIndex, PatternNodeId> target_node_to_pattern;
  InlinedVector<const NodeArg*> pattern_value_to_target;
  InlinedVector<ValueVisitState> value_visit_states;
  InlinedVector<const NodeArg*> formal_input_bindings;
  InlinedVector<LiteralWitness> literal_witnesses;
  size_t processed_binding_count{};
};

struct ReplacementPlan {
  size_t primary_root_topological_position{};
  InlinedVector<NodeIndex> removable_node_indices;
  InlinedVector<NodeArg*> call_inputs;
  InlinedVector<NodeArg*> call_outputs;
  InlinedVector<LiteralWitness> literal_witnesses;
  std::vector<graph_utils::GraphEdge> matched_input_edges;
  std::vector<graph_utils::GraphEdge> explicit_input_edges;
  std::vector<graph_utils::GraphEdge> explicit_output_edges;
  InlinedHashMap<const NodeArg*, InlinedVector<NodeIndex>> implicit_consumers;
  std::string layering_annotation;
  std::string generated_call_name;
};

struct MatcherDiagnostics {
  size_t output_root_tuples_considered{};
  size_t worklist_bindings_scheduled{};
  size_t worklist_bindings_processed{};
  size_t structurally_matched_candidates{};
  size_t accepted_candidates{};
};

common::Status BuildTargetGraphSnapshot(
    const Graph& graph,
    const CompiledFunctionPattern& compiled_pattern,
    const FunctionExtractorOptions& options,
    TargetGraphSnapshot& snapshot);

common::Status DiscoverReplacementPlans(
    const CompiledFunctionPattern& compiled_pattern,
    const TargetGraphSnapshot& snapshot,
    const FunctionExtractorOptions& options,
    std::vector<ReplacementPlan>& plans,
    MatcherDiagnostics* diagnostics = nullptr);

common::Status SelectNonConflictingPlans(
    gsl::span<const ReplacementPlan> plans,
    std::vector<size_t>& selected_plan_indices);

common::Status PrevalidatePlans(
    const Graph& graph,
    const CompiledFunctionPattern& compiled_pattern,
    gsl::span<const ReplacementPlan> plans);

}  // namespace function_extractor_internal
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

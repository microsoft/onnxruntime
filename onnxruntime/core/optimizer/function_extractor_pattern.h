#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/graph/graph.h"
#include "core/optimizer/function_extractor.h"

namespace onnxruntime {
namespace function_extractor_internal {

using PatternNodeId = size_t;
using PatternValueId = size_t;
constexpr PatternValueId kMissingPatternValue = std::numeric_limits<PatternValueId>::max();
constexpr PatternNodeId kNoPatternNode = std::numeric_limits<PatternNodeId>::max();

struct PatternValueConsumer {
  PatternNodeId node_id{kNoPatternNode};
  size_t input_index{};
};

struct LiteralDescriptor {
  ONNX_NAMESPACE::TensorProto tensor;
  std::string fingerprint;
  size_t byte_count{};
};

// Flags are independent: a value may be both produced and formal output, while
// formal inputs and normalized literals are terminal leaves with no producer.
struct PatternValue {
  std::string name;
  ONNX_NAMESPACE::TypeProto type;
  PatternNodeId producer_node_id{kNoPatternNode};
  size_t producer_output_index{};
  InlinedVector<PatternValueConsumer> consumers;
  bool has_type{};
  bool is_formal_input{};
  bool is_formal_output{};
  bool is_literal{};
  LiteralDescriptor literal;
};

struct PatternNode {
  size_t source_node_proto_index{};
  InlinedVector<PatternValueId> input_value_ids;
  InlinedVector<PatternValueId> output_value_ids;
};

// Owns the context-free validated pattern. Values use stable numeric IDs, and
// every operation is connected, acyclic, and backward-reachable from an output.
struct NormalizedFunctionPattern {
  ONNX_NAMESPACE::FunctionProto function_proto;
  InlinedVector<PatternValueId> formal_input_value_ids;
  InlinedVector<PatternValueId> formal_output_value_ids;
  InlinedVector<PatternValue, 1> values;
  InlinedVector<PatternNode> nodes;
  InlinedVector<PatternNodeId> reverse_topological_node_ids;
  common::Status construction_status{common::Status::OK()};
};

// Per-Graph semantic resolution of a pattern node. Schema pointers are borrowed
// from the Graph/model registry and remain valid only while that context lives.
struct ResolvedPatternNode {
  PatternNodeId pattern_node_id{kNoPatternNode};
  std::string canonical_domain;
  std::string op_type;
  std::string overload;
  int since_version{-1};
  const ONNX_NAMESPACE::OpSchema* schema{};
  NodeAttributes effective_attributes;
  size_t input_arity{};
  size_t output_arity{};
  std::string function_fingerprint;
  bool transitively_pure{};
};

struct FormalOutputProducerGroup {
  PatternNodeId producer_node_id{kNoPatternNode};
  InlinedVector<size_t> formal_output_indices;
  InlinedVector<size_t> producer_output_indices;
};

struct CompiledFunctionPattern {
  // Non-owning. The referenced normalized pattern must outlive this object.
  const NormalizedFunctionPattern* normalized_pattern{};
  InlinedVector<ResolvedPatternNode> resolved_nodes;
  InlinedVector<FormalOutputProducerGroup> formal_output_producer_groups;
};

NormalizedFunctionPattern NormalizeFunctionPattern(
    const ONNX_NAMESPACE::FunctionProto& function_proto,
    const FunctionExtractorOptions& options);

common::Status CompileFunctionPattern(
    const NormalizedFunctionPattern& normalized_pattern,
    const Graph& graph,
    CompiledFunctionPattern& compiled_pattern);

common::Status ValidateRegisteredFunction(
    const NormalizedFunctionPattern& normalized_pattern,
    const Graph& graph);

bool IsV1PureOperator(const ResolvedPatternNode& node);

bool AreAttributesSemanticallyEqual(const NodeAttributes& lhs, const NodeAttributes& rhs);

common::Status NormalizeConstantAttributes(
    const NodeAttributes& attributes,
    ONNX_NAMESPACE::TensorProto& tensor);

common::Status CompareTensorLiterals(
    const ONNX_NAMESPACE::TensorProto& lhs,
    const ONNX_NAMESPACE::TensorProto& rhs,
    size_t max_literal_bytes,
    bool& equal,
    const std::filesystem::path* rhs_model_path = nullptr);

std::string CanonicalFunctionFingerprint(const ONNX_NAMESPACE::FunctionProto& function_proto);

}  // namespace function_extractor_internal
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

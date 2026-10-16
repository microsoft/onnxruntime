// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {
namespace rewrite {

namespace detail {
class PatternParser;
}

struct Value {
  std::string name;
  int handle = -1;
  Value() = default;
  Value(int id) : handle(id) {}
  Value(std::string label, int id) : name(std::move(label)), handle(id) {}
  // Handles index Match::values; names are diagnostic and do not affect identity.
  operator int() const { return handle; }
};

struct AttributeMatcher {
  std::string name;
  std::function<bool(const ONNX_NAMESPACE::AttributeProto&)> predicate;
  std::string capture;
  bool Matches(const ONNX_NAMESPACE::AttributeProto& value) const { return predicate(value); }
  static AttributeMatcher Equal(std::string name, int64_t value);
  static AttributeMatcher Equal(std::string name, int value) { return Equal(std::move(name), int64_t{value}); }
  static AttributeMatcher Equal(std::string name, float value);
  static AttributeMatcher Equal(std::string name, double value) {
    return Equal(std::move(name), static_cast<float>(value));
  }
  static AttributeMatcher Equal(std::string name, std::string value);
};

struct Operator {
  std::string name;
  std::string domain;
};

using Literal = std::variant<float, int64_t, InlinedVector<float>, InlinedVector<int64_t>>;

using Dimension = std::variant<std::monostate, int64_t, std::string>;

struct TypeConstraint {
  std::string value_name;
  int32_t element_type = 0;  // Zero means unconstrained.
  std::string element_variable;
  InlinedVector<Dimension> dimensions;
  bool any_rank = false;
  bool prefix = false;
  int32_t attribute_type = 0;  // Scalar/list attribute interpretation, when applicable.
  bool tensor_type = true;
  bool absent = false;
};

struct OpsetRequirement {
  std::string domain;
  int min_version;
  int max_version;  // Inclusive model/function import range, not schema since-versions.
};

// A straight-line definition describes a DAG, not necessarily a chain.
// Reusing a Value requires exactly the same NodeArg, including for input variables.
class Pattern {
 public:
  explicit Pattern(std::string_view text);
  Value NamedValue(std::string_view name) const;
  struct NodePattern {
    Operator op;
    InlinedVector<Value> inputs;
    InlinedVector<Value> outputs;
    InlinedVector<AttributeMatcher, 1> attributes;
  };
  const auto& Nodes() const { return nodes_; }
  size_t ValueCount() const { return value_count_; }
  bool IsInput(Value value) const;
  const auto& Constants() const { return constants_; }
  const auto& Outputs() const { return outputs_; }
  bool IsOptional(Value value) const { return optional_inputs_.count(value.handle) != 0; }
  const auto& AttributeNames() const { return attribute_names_; }
  const auto& TypeConstraints() const { return type_constraints_; }
  bool IsAttribute(Value value) const { return attribute_inputs_.count(value.handle) != 0; }

 private:
  friend class detail::PatternParser;
  friend struct Replacement;
  friend struct Match;
  friend class PatternRewriter;
  Pattern() = default;
  Value Input(std::string_view name);
  Value Constant(float value);
  Value Constant(int64_t value);
  Value Constant(Literal value);
  void Name(Value value, std::string_view name);
  void Attributes(InlinedVector<AttributeMatcher, 1> matchers);
  InlinedVector<Value> MultiOp(Operator op, gsl::span<const Value> inputs, size_t output_count);

  size_t value_count_ = 0;
  InlinedVector<NodePattern, 1> nodes_;
  InlinedVector<Value> values_;
  InlinedHashMap<int, Literal> constants_;
  InlinedVector<Value> outputs_;
  InlinedHashSet<int> optional_inputs_;
  InlinedHashSet<std::string> attribute_names_;
  InlinedHashMap<int, InlinedVector<TypeConstraint, 1>> type_constraints_;
  InlinedHashSet<int> attribute_inputs_;
  Value AddValue(std::string_view name);
};

struct Match {
  InlinedVector<const Node*> nodes;
  InlinedVector<const NodeArg*> values;
  const NodeArg* valueOf(std::string_view name) const;
  void update(std::string_view name, ONNX_NAMESPACE::AttributeProto value);
  const ONNX_NAMESPACE::AttributeProto& attributeOf(std::string_view name) const;
  const Dimension& dimensionOf(std::string_view name) const;
  std::optional<int32_t> elementTypeOf(std::string_view name) const;
  void ClearUpdates() {
    extra_attributes_.clear();
    rejection_reason_.clear();
  }
  bool BindAttribute(std::string_view name, const ONNX_NAMESPACE::AttributeProto& value);
  bool BindDimension(std::string_view name, Dimension value);
  bool BindElementType(std::string_view name, int32_t value);
  bool Reject(std::string_view reason) {
    rejection_reason_ = reason;
    return false;
  }
  const std::string& RejectionReason() const { return rejection_reason_; }
  const Pattern* pattern = nullptr;

 private:
  NodeAttributes captured_attributes_;
  InlinedHashMap<std::string, Dimension> dimensions_;
  InlinedHashMap<std::string, int32_t> element_types_;
  NodeAttributes extra_attributes_;
  std::string rejection_reason_;
};

struct Replacement {
  struct Output {
    size_t node;
    size_t slot = 0;
  };
  struct Scalar {
    Literal value;
  };
  using Input = std::variant<Value, Output, std::string, Scalar, std::monostate>;
  struct NodeRecipe {
    std::string op;
    std::string domain;
    InlinedVector<Input> inputs;
    NodeAttributes attributes;
    size_t output_count = 1;
    InlinedHashMap<std::string, std::string> attribute_bindings;
  };
  explicit Replacement(std::string_view text);
  // A parsed replacement is a reusable recipe callback; copying it shares the immutable plan.
  Replacement operator()(const Graph&, const Match&) const { return *this; }
  const auto& Plan() const { return plan_; }
  const auto& Outputs() const { return outputs_; }

 private:
  friend class PatternRewriter;
  std::shared_ptr<const InlinedVector<NodeRecipe, 1>> plan_;
  InlinedVector<Output> outputs_;
};

struct Rule {
  using CheckAndBind = std::function<bool(const Graph&, Match&)>;
  using ReplacementFunction = std::function<Replacement(const Graph&, const Match&)>;

  Rule(std::string name, Pattern pattern, ReplacementFunction replacement, CheckAndBind check_and_bind = {},
       InlinedVector<OpsetRequirement, 1> opsets = {});
  Rule(std::string name, std::string_view pattern, ReplacementFunction replacement, CheckAndBind check_and_bind = {},
       InlinedVector<OpsetRequirement, 1> opsets = {});
  Rule(std::string name, Pattern pattern, std::string_view replacement, CheckAndBind check_and_bind = {},
       InlinedVector<OpsetRequirement, 1> opsets = {});
  Rule(std::string name, std::string_view pattern, std::string_view replacement, CheckAndBind check_and_bind = {},
       InlinedVector<OpsetRequirement, 1> opsets = {});

  std::string name;
  Pattern pattern;
  // Check-and-bind callbacks may prepare match-local attributes. Neither callback may mutate the graph
  // or retain pointers into a Match; replacement callbacks receive a read-only match.
  ReplacementFunction replacement;
  CheckAndBind check_and_bind;  // Empty accepts a structurally valid match without additional checks.
  InlinedVector<OpsetRequirement, 1> opsets;
};

using Rules = InlinedVector<Rule, 1>;

struct Statistics {
  size_t visited = 0;
  size_t candidates = 0;
  size_t structural_checks = 0;
  size_t check_and_bind_calls = 0;
  size_t matches = 0;
  size_t rewrites = 0;
  size_t iterations = 0;
  bool iteration_limit_reached = false;
};

enum class FailureStage { Opset,
                          Structure,
                          TypeShape,
                          Constant,
                          Attribute,
                          Boundary,
                          ExecutionProvider,
                          CheckAndBind,
                          Replacement };

struct MatchFailure {
  std::string rule;
  NodeIndex root;
  FailureStage stage;
  std::string reason;
};

struct Options {
  bool match_only = false;
  bool indexed = true;  // false is the rule-by-rule reference implementation.
  // Unassigned nodes are always eligible. Assigned nodes must share an allowed provider.
  InlinedHashSet<std::string> compatible_execution_providers{kCpuExecutionProvider};
  // Called synchronously. Records own their strings and contain no graph/match pointers.
  std::function<void(const MatchFailure&)> on_failure;
  size_t max_iterations = 10;  // Positive upper bound on candidate batches; match-only uses one batch.
};

// Immutable compiled rules; statistics belong to a call, not this transformer.
// Not registered among ORT's default optimizers. Call Apply or Run explicitly.
class PatternRewriter final : public GraphTransformer {
 public:
  explicit PatternRewriter(Rules rules, Options options = {});
  Status Run(Graph& graph, Statistics& statistics) const;

 private:
  struct Group {
    Pattern pattern;
    InlinedVector<size_t> rules;
  };
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  Rules rules_;
  InlinedVector<Group, 1> groups_;
  InlinedHashMap<std::string, InlinedVector<size_t>> roots_;
  size_t invalidation_depth_ = 0;
  Options options_;
};

// Float32 examples adapted from genuine ONNXScript rules; EP eligibility is controlled by Options.
Rules MakeExampleRules();

}  // namespace rewrite
}  // namespace onnxruntime

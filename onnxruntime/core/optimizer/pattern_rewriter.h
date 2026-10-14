// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {
namespace rewrite {

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

struct OpsetRequirement {
  std::string domain;
  int min_version;
  int max_version;  // Inclusive model/function import range, not schema since-versions.
};

// A straight-line definition describes a DAG, not necessarily a chain.
// Reusing a Value requires exactly the same NodeArg, including for input variables.
class Pattern {
 public:
  Pattern() = default;
  explicit Pattern(std::string_view text);
  Value Input(std::string_view name = "");
  Value Constant(float value);
  Value Constant(int64_t value);
  Value Constant(int value) { return Constant(int64_t{value}); }
  Value Constant(double value) { return Constant(static_cast<float>(value)); }
  Value NamedValue(std::string_view name) const;
  void Name(Value value, std::string_view name);
  void Attributes(std::initializer_list<AttributeMatcher> matchers);
  void Attributes(InlinedVector<AttributeMatcher, 1> matchers);
  Value Op(Operator op, std::initializer_list<Value> inputs);
  template <typename Name, std::enable_if_t<std::is_convertible_v<Name, std::string_view>, int> = 0>
  Value Op(Name&& name, std::initializer_list<Value> inputs, std::string_view domain = "") {
    return Op(Operator{std::string(std::string_view(name)), std::string(domain)}, inputs);
  }
  InlinedVector<Value> MultiOp(Operator op, std::initializer_list<Value> inputs, size_t output_count);
  InlinedVector<Value> MultiOp(Operator op, gsl::span<const Value> inputs, size_t output_count);
  template <typename Name, std::enable_if_t<std::is_convertible_v<Name, std::string_view>, int> = 0>
  InlinedVector<Value> MultiOp(Name&& name, std::initializer_list<Value> inputs, size_t output_count,
                               std::string_view domain = "") {
    return MultiOp(Operator{std::string(std::string_view(name)), std::string(domain)}, inputs, output_count);
  }

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

 private:
  friend struct Match;
  friend class PatternRewriter;
  size_t value_count_ = 0;
  InlinedVector<NodePattern, 1> nodes_;
  InlinedVector<Value> values_;
  InlinedHashMap<int, std::variant<float, int64_t>> constants_;
  Value AddValue(std::string_view name);
};

struct Match {
  InlinedVector<const Node*> nodes;
  InlinedVector<const NodeArg*> values;
  const NodeArg* valueOf(std::string_view name) const;
  void update(std::string_view name, ONNX_NAMESPACE::AttributeProto value) const;
  const ONNX_NAMESPACE::AttributeProto& attributeOf(std::string_view name) const;
  void ClearUpdates() const { extra_attributes_.clear(); }
  const Pattern* pattern = nullptr;

 private:
  mutable NodeAttributes extra_attributes_;
};

struct Replacement {
  struct Output {
    size_t node;
    size_t slot = 0;
  };
  struct Scalar {
    std::variant<float, int64_t> value;
  };
  using Input = std::variant<Value, const NodeArg*, Output, std::string, Scalar>;
  struct NodeRecipe {
    std::string op;
    std::string domain;
    InlinedVector<Input> inputs;
    NodeAttributes attributes;
    size_t output_count = 1;
    InlinedHashMap<std::string, std::string> attribute_bindings;
  };
  Replacement() = default;
  Replacement(std::string op_name, std::string op_domain, InlinedVector<Input> arguments, NodeAttributes attrs)
      : op(std::move(op_name)),
        domain(std::move(op_domain)),
        inputs(std::move(arguments)),
        attributes(std::move(attrs)) {}
  explicit Replacement(std::string_view text);
  explicit Replacement(InlinedVector<NodeRecipe, 1> nodes);
  // A parsed replacement is a reusable recipe callback; copying it shares the immutable plan.
  Replacement operator()(const Graph&, const Match&) const { return *this; }
  const auto& Plan() const { return plan_; }
  std::string op;
  std::string domain;
  InlinedVector<Input> inputs;
  NodeAttributes attributes;

 private:
  friend class PatternRewriter;
  std::shared_ptr<const InlinedVector<NodeRecipe, 1>> plan_;
};

struct Rule {
  std::string name;
  Pattern pattern;
  InlinedVector<OpsetRequirement, 1> opsets;
  // Guards must be pure. Recipes may set extra attribute bindings, but must not mutate the graph
  // or retain pointers into a Match.
  std::function<bool(const Graph&, const Match&)> guard;
  std::function<Replacement(const Graph&, const Match&)> replacement;
};

using Rules = InlinedVector<Rule, 1>;

struct Statistics {
  size_t visited = 0;
  size_t candidates = 0;
  size_t structural_checks = 0;
  size_t guards = 0;
  size_t matches = 0;
  size_t rewrites = 0;
};

struct Options {
  bool match_only = false;
  bool indexed = true;  // false is the rule-by-rule reference implementation.
  // Unassigned nodes are always eligible. Assigned nodes must share an allowed provider.
  InlinedHashSet<std::string> compatible_execution_providers{kCpuExecutionProvider};
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

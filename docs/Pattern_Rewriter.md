# Writing C++ graph rewrite rules

The experimental `onnxruntime::rewrite::PatternRewriter` replaces connected
operator patterns with smaller, semantically equivalent graphs. Include
`core/optimizer/pattern_rewriter.h`. Construct `Pattern` and `Replacement` from
text; programmatic builders and mutable replacement recipes are not public APIs.
Guards and replacement callbacks remain ordinary C++ functions.

A `Rule` combines a diagnostic name, a parsed pattern, domain/opset requirements,
a semantic guard, and a parsed replacement or callback returning one. The matcher
establishes structural and deletion safety. **The rule author must establish
semantic equivalence, supported types, and kernel compatibility.**

## Example: HardSigmoid followed by Mul

This complete rule recognizes the HardSwish decomposition with the exact
HardSigmoid attributes shown:

```cpp
#include "core/optimizer/pattern_rewriter.h"

using namespace onnxruntime;
using namespace onnxruntime::rewrite;

bool IsFloat32(const NodeArg& value) {
  const auto* type = value.TypeAsProto();
  return type && type->has_tensor_type() &&
         type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}

Rule MakeHardSwishRule() {
  Pattern pattern(R"(
    (x) =>
      sigmoid = HardSigmoid(x, alpha=0.16666667, beta=0.5)
      return Mul(sigmoid, x)
  )");
  auto guard = [](const Graph&, const Match& match) {
    return IsFloat32(*match.valueOf("x"));
  };
  Replacement replacement("(x) => HardSwish(x)");
  return Rule{"HardSigmoidMul", std::move(pattern),
              {{kOnnxDomain, 14, 21}}, guard, replacement};
}
```

Both occurrences of `x` must bind the **same `NodeArg`**, not just tensors with
equal shapes or contents. The guard checks the relevant data binding `x`;
resolved schemas establish the corresponding intermediate and output types.
It does not require every matched value to have the same element type.

These opset bounds are a conservative supported range for this example, not an
automatic statement about every schema version. Floating-point rewrites can
change evaluation order; validate equivalence within the intended tolerance.

## Pattern and replacement syntax

Definitions start with named inputs and `=>`. Assignments create named operator
outputs; `return` identifies the final operator, called the root. Whitespace
separates calls, and semicolons are optional separators. Comments are not supported
inside definitions.

```cpp
Pattern pattern(R"(
  (x) =>
    add = Add(x, 3.0)
    clip = Clip(add, 0.0, 6.0)
    product = Mul(clip, x)
    return result = Div(product, 6.0)
)");
Replacement replacement("(x) => HardSwish(x)");
```

The example has one root output, named `result`. A final call without an
assignment, such as `return Div(product, 6.0)`, also describes one output,
but does not give that output a lookup name. Replacements additionally accept
a final call without the `return` keyword, as above.

Use `domain::Operator(...)` for qualified operators:

```cpp
Pattern pattern("(x,y) => mm = com.microsoft::FusedMatMul(x,y); return Identity(mm)");
Replacement replacement("(x,y) => com.microsoft::FusedMatMul(x,y)");
```

An unqualified operator uses the standard ONNX domain. `ai.onnx::Identity`
is equivalent to `Identity`. Replacements also accept
`com.microsoft.FusedMatMul(...)`; patterns use the `::` form.

Names must be unique and defined before use. Every declared pattern input must
participate, and every operator must reach the root. Inputs to different named
variables may alias; repeating one variable requires identical binding.
`match.valueOf("x")` and `match.valueOf("clip")` return bound `const NodeArg*`
values. Unknown names and unbound values raise explicit errors. Read-only
`Pattern::NamedValue`, `Nodes`, `Constants`, `ValueCount`, and `IsInput`
are available for introspection.

### Constants and attributes

Positional numeric literals in patterns match **rank-zero constant initializers**
of the exact value and element type: `3.0`, `3e0`, and `0.5` are float32;
`3`, `-1`, and `0` are int64. Thus `2` and `2.0` are distinct constraints.
A singleton vector is not a scalar. External tensor data and overridable
initializers do not satisfy these constraints. Constant operators require
an earlier constant-folding pass.

Keyword arguments constrain attributes, not tensor inputs:

```cpp
Pattern pattern("(x) => h = HardSigmoid(x, alpha=0.25, beta=0.5); return Mul(h,x)");
```

Attribute equality compares the protobuf attribute type as well as its value.
For example, `axis=-1` is an integer attribute, while `epsilon=1e-5` is a float.
Single- or double-quoted string attributes are supported; strings may escape
quotes and backslashes. Other attributes are unconstrained. A listed attribute
must be present in the resolved node's attribute map.

Pattern syntax supports scalar integer, float, and string attribute equality,
not attribute lists or custom predicate syntax. Inspect `match.nodes` in the C++
guard for ranges, list-valued attributes, shapes, or other semantic conditions.
Positional inputs must precede keyword attributes.

Numeric literals in replacements synthesize rank-zero float32 or int64
initializers. Replacement keyword literals set attributes on the new node.

### Branched patterns and multiple outputs

Assignments can describe a DAG rather than only a chain:

```cpp
Pattern pattern(R"(
  (x,y) =>
    sum = Add(x,y)
    product = Mul(sum,x)
    return Add(sum,product)
)");
```

Here both uses of `sum` must have the same producer. The matcher does not search
for equivalent computations or reorder commutative inputs.

Comma-separated assignment names declare the exact ordered output count:

```cpp
Pattern pattern("(x) => first,second = Split(x,axis=0); return Concat(first,second,axis=0)");
Replacement replacement("(x) => Identity(x)");
```

For a multi-output root, put the names on the final return assignment:

```cpp
Pattern pattern("(x) => copied = Identity(x); return left,right = Split(copied,axis=0)");
Replacement replacement("(x) => return left,right = Split(x,axis=0)");
```

This replacement preserves the root's two original output `NodeArg`s in order,
including their names, types, shapes, graph-output roles, and consumers.
The replacement names are local labels, not new externally visible graph names.
`return Split(x)` describes only one output and cannot replace a two-output root.
Returning an existing value or a subset/reordering of an earlier operator's
outputs is not supported; the return must define the final operator.

Intermediate operators may have unused outputs, but **none** of their outputs
may be exposed outside the matched deletion set.

## Computed attributes with a replacement callback

Reuse a parsed replacement plan when an attribute depends on each match.
Declare the attribute binding in the replacement's input list, set it with
`match.update`, and reference it as a keyword argument.

The following rule folds division by a finite float32 scalar into FusedMatMul's
`alpha` attribute. It requires the Microsoft-domain opset to be imported already.

```cpp
#include <cmath>
#include <optional>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

std::optional<float> FloatScalar(const Graph& graph, const NodeArg& value) {
  const auto* tensor = graph_utils::GetConstantInitializer(graph, value.Name());
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL ||
      tensor->dims_size() != 0)
    return std::nullopt;
  Initializer initializer(graph, *tensor, graph.ModelPath());
  if (initializer.size() != 1) return std::nullopt;
  float scalar = *initializer.data<float>();
  return std::isfinite(scalar) ? std::optional<float>(scalar) : std::nullopt;
}

Rule MakeMatMulDivRule() {
  Pattern pattern("(x,y,divisor) => mm = MatMul(x,y); return Div(mm,divisor)");
  auto guard = [](const Graph& graph, const Match& match) {
    auto divisor = FloatScalar(graph, *match.valueOf("divisor"));
    return IsFloat32(*match.valueOf("x")) && IsFloat32(*match.valueOf("y")) &&
           divisor && *divisor != 0.0f && std::isfinite(1.0f / *divisor);
  };
  Replacement plan("(x,y,alpha) => com.microsoft::FusedMatMul(x,y,alpha=alpha)");
  auto replacement = [plan](const Graph& graph, const Match& match) {
    ONNX_NAMESPACE::AttributeProto alpha;
    alpha.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    alpha.set_f(1.0f / *FloatScalar(graph, *match.valueOf("divisor")));
    match.update("alpha", std::move(alpha));
    return plan;
  };
  return Rule{"MatMulDiv", std::move(pattern),
              {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}, guard, replacement};
}
```

This example uses `IsFloat32` from the first example. The callback is called only
after its guard succeeds. Guards must be pure; neither guard nor callback may
mutate the graph, retain match pointers, or rely on mutable global graph state.
Use local bindings, attributes, and constants to establish applicability.

`update` accepts a concrete `AttributeProto`, including list-valued attributes.
It cannot overwrite a named pattern value or store a referenced/undefined
attribute. Extra bindings are cleared before each replacement callback;
`attributeOf` retrieves one and errors if it is missing. A replacement header
name cannot be used as both a tensor and an attribute. Attribute-only names need
not appear in the pattern, but must be supplied by the callback. Reusing a parsed
plan does not reuse a previous match's attributes.

Replacement tensor inputs may reference only pattern input variables, literal
constants, or earlier outputs in the replacement itself. A matched intermediate
is about to be deleted and cannot be a replacement input. Replacement sequences
must be connected to their final operator.

## When a match succeeds or fails

Before calling a rule's guard, the matcher checks:

- Exact operator name/domain, ordered input/output counts, and repeated binding
  identity. Missing optional slots and variadic patterns are not supported.
- Literal and listed attribute constraints. It does not otherwise infer semantic
  equivalence or check arbitrary shapes/types for the author.
- Distinct matched nodes with the same layer annotation. Matched operators cannot
  own subgraphs or have implicit inputs.
- Pattern input variables are outside the matched deletion set.
- Every non-root output stays internal. A graph output, an external consumer, or
  an implicit capture by an external subgraph prevents the rewrite.
- Execution-provider eligibility as described below.

The root may have graph outputs or external consumers, because its complete
output list is preserved. Even match-only mode applies deletion-boundary checks:
it reports safe rewrite eligibility, not arbitrary structural occurrences.

### Types in the supplied examples

`MakeExampleRules()` supplies six rules when contrib operators are enabled:

| Rule | Explicit float32 checks |
|---|---|
| HardSwish and HardSigmoidMul | `x` |
| MatMulDiv and FusedMatMulDiv | `x`, `y`; separate finite scalar initializer check on `divisor` |
| TransposeMatMul | `x`, `y` |
| RmsNormalization | `x`, `scale`; separate positive finite scalar initializer check on `epsilon` |

There is no match-wide float32 restriction. Independently typed operands such
as shape/index tensors or a Pow exponent require their own semantic constraints.
The supplied RMS pattern specifically matches `Pow(x,2.0)`; the float32 exponent
is a literal constraint, not a blanket guard check.

The RMS example also requires last-axis reduction, keepdims, known positive
width and a matching one-dimensional scale, with pre-opset-18 attribute axes.
TransposeMatMul requires a known rank of at least two and exactly the last-two-axis
swap. Division examples require a finite scaling result. Consult the rule
definitions and correctness tests before extending their supported ranges.
Contrib-disabled builds supply only the two HardSwish rules.

## Opset and execution-provider applicability

Each `OpsetRequirement` is `{domain, min_version, max_version}`, with positive,
inclusive bounds. Declare every domain used by a pattern or replacement;
additional requirements are allowed. `""` and `"ai.onnx"` denote the same domain.
Duplicate domains, reversed ranges, and undeclared pattern domains are errors.

All requirements are evaluated against each graph's `DomainToVersionMap()`,
including independently for subgraphs. Missing imports or out-of-range versions
make the rule inapplicable **before its guard runs**. The engine does not add
imports, compare individual node schema since-versions, or rewrite model-local
function bodies. Authors must establish correctness across their declared range.

All matched nodes must have the same execution-provider assignment. Unassigned
nodes are eligible; assigned nodes must belong to
`Options::compatible_execution_providers`, which defaults to CPU only.
Mixed assignments are rejected. This policy applies before the rule guard and
also in match-only mode. To opt into another provider:

```cpp
Options options;
options.compatible_execution_providers.insert(kCudaExecutionProvider);
PatternRewriter rewriter(Rules{MakeHardSwishRule()}, options);
```

Allowing a provider is the caller's assertion of kernel and precision
compatibility, not automatic kernel discovery. Clearing the allowed set permits
only unassigned nodes. Replacements inherit the root's provider and layer
annotation.

## Applying rules, priority, and errors

Register the transformer through the normal session pipeline:

```cpp
auto transformer = std::make_unique<PatternRewriter>(Rules{MakeHardSwishRule()});
ORT_RETURN_IF_ERROR(session.RegisterGraphTransformer(
    std::move(transformer), TransformerLevel::Level1));
```

For one already-resolved graph, use `Run` and explicitly resolve afterwards:

```cpp
PatternRewriter rewriter(Rules{MakeHardSwishRule()});
Statistics statistics;
ORT_RETURN_IF_ERROR(rewriter.Run(graph, statistics));
ORT_RETURN_IF_ERROR(graph.Resolve());
```

`Run` handles one graph. The transformer's `Apply` path recurses into subgraphs
and performs normal graph resolution. Callers must not concurrently mutate a
graph. Options and compiled rules are immutable; match state and statistics are
per call.

Rules have deterministic **registration-order priority**, including across
different structural groups. At each eligible root, rewriting chooses the first
passing guard. With `Options{.match_only=true}`, all passing guards are counted
and replacements are not called. `indexed=false` selects the reference matching
mode with the same eligibility and priority semantics.

Every pattern must contain at least two nodes. A replacement must have a
nonempty node sequence with **strictly fewer nodes** than the pattern, and its
final output count must equal the root's output count. Expansion and one-node
rewrites are rejected. Each successful rewrite reduces live node count, so a
graph with N nodes admits at most N-1 rewrites. This establishes termination,
not a time bound for arbitrary callbacks or graph fanout.

After a rewrite, the engine updates edges and producer/consumer indexes,
notifies node replacement, and requeues replacement nodes and affected
descendants through a bounded invalidation walk. This permits newly enabled
matches without retaining pointers to deleted nodes.

Malformed definitions throw `OnnxRuntimeException` with parse-location
diagnostics; exception-disabled builds follow ORT's abort behavior. Invalid rule
configuration raises an explicit error when the rewriter is constructed.
Direct parsed replacements are checked against their rule during construction;
callback-generated replacements also undergo runtime validation. Guard false
is an ordinary nonmatch.

Replacement schemas and attributes are validated before matched nodes are
deleted. Failed staging rolls back the staged nodes and literal initializers;
there is no whole-call transaction undoing earlier successful rewrites.
Errors propagate as statuses or ORT exceptions rather than being treated as
nonmatches. Final graph resolution checks types and inferred shapes; a rule
author must not rely on schema staging alone to prove type correctness.

`Statistics` exposes live roots visited (including revisits), enumerated
candidates, structural checks, guard calls, successful matches, and completed
rewrites. Match-only counts all passing guards; rewrite mode counts selected
winners. These counters do not alter rule semantics.

## Limits and unsupported forms

Definitions are limited to 64 nodes, 256 values, and 64 inputs/outputs per
operator. They are connected acyclic graphs rooted at the final operator.
There is no alternation, implicit commutation, optional/variadic matching,
general tensor synthesis, domain-import editing, or model-local function
rewriting. Control-flow operators are not matched, although their subgraphs can
be rewritten through `Apply`.

Tests in `onnxruntime/test/optimizer/pattern_rewriter_test.cc` cover parsing,
guards, boundaries, multi-output rewrites, rule priority, and numerical CPU
equivalence, including small decoder prefill/decode graphs. Add structural and
executed equivalence tests when writing or extending a rule.

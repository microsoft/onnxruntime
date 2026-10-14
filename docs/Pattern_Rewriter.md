# Experimental indexed C++ pattern rewriter

`onnxruntime/core/optimizer/pattern_rewriter.h` defines an opt-in `GraphTransformer`.
It is **not registered in the default optimization pipeline**, and introduces no
session configuration switch or public C ABI. C++ callers may register it using
`InferenceSession::RegisterGraphTransformer`, just like existing custom transformers.
The example predicates are provider-independent and restrict types to float32.
The engine's generic partition policy defaults to CPU or unassigned graphs;
allowing other EPs requires the caller to verify replacement kernel compatibility.

## Source investigation and supported examples

The ONNXScript reference revision is
[`9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36`](https://github.com/microsoft/onnxscript/tree/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36).
The following files were examined, rather than assuming that Python syntax defines
the right architecture for ORT:

| ONNXScript source at that revision | C++ adaptation |
|---|---|
| [`rewriter/_matcher.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/_matcher.py) | Backward DAG binding, repeated-value equality, distinct matched nodes, and removal boundary checks |
| [`rewriter/_rewrite_rule.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/_rewrite_rule.py) | Separate structural patterns, semantic guards, replacement recipes, ordered priority, recursive subgraph handling |
| [`rules/common/_fuse_hardswish.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/rules/common/_fuse_hardswish.py) | `Div(Mul(Clip(Add(x,3),0,6),x),6)` and `Mul(HardSigmoid(x),x)` to HardSwish |
| [`ort_fusions/fused_matmul_rule_sets.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/ort_fusions/fused_matmul_rule_sets.py) | MatMul/scalar Div, FusedMatMul/scalar Div (preserving attributes), and last-two-dimension Transpose/MatMul |
| [`ort_fusions/rms_normalization.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/ort_fusions/rms_normalization.py) | Float32 Pow/ReduceMean/Add/Sqrt/Reciprocal/Mul/Mul to SimplifiedLayerNormalization |
| [`ort_fusions/softmax.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/ort_fusions/softmax.py) | Studied but deliberately not enabled: fp16 cast removal needs a separately verified precision/EP policy |
| [`rules/fusion/_layer_norm.py`, `LayerNormBiasFusion`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/rules/fusion/_layer_norm.py) | Test-only adaptation matches all three LayerNormalization outputs, fuses a trailing bias Add, and rejects observable statistics |

Existing ORT `SelectorActionRegistry` already dispatches selectors by root op/domain
and version. `QuickGeluFusion` illustrates repeated inputs and provider restrictions,
while `graph_utils::FinalizeNodeFusion` illustrates edge maintenance. The new
transformer uses ORT's graph/node/edge APIs directly; it does not copy the graph to
a parallel IR. Generic backward bindings and sharing identical structural plans
are the additional capabilities over hand-written selectors.

## Defining a rule

The small DSL uses `Value {std::string name; int handle;}`. A name helps diagnose
patterns; only the handle determines binding identity. An input introduces a variable;
`Op` introduces one output, while `MultiOp` introduces an ordered output list.
Reusing a handle requires the identical
`NodeArg`, not merely the same shape or equivalent contents. Pattern operators
specify only name and domain. **Model-imported opset ranges belong to the rule**,
not to individual nodes or their schema since-versions.

```cpp
using namespace onnxruntime::rewrite;
Pattern p;
auto x = p.Input("x");
auto h = p.Op("HardSigmoid", {x});
p.Name(h, "sigmoid");
p.Attributes({AttributeMatcher::Equal("alpha", 1.0f / 6.0f),
              AttributeMatcher::Equal("beta", 0.5f)});
p.Op("Mul", {h, x});
auto guard = [](const Graph&, const Match& m) {
  for (const NodeArg* value : m.values) {
    const auto* type = value->TypeAsProto();
    if (!type || !type->has_tensor_type() ||
        type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      return false;
  }
  return true;
};
Replacement replacement("(x) => HardSwish(x)");
Rule rule{
    "HardSigmoidMul", p,
    {{kOnnxDomain, 14, 21}},  // inclusive imported opset range
    guard, replacement};
auto transformer = std::make_unique<PatternRewriter>(Rules{rule});
ORT_RETURN_IF_ERROR(session.RegisterGraphTransformer(std::move(transformer),
                                                    TransformerLevel::Level1));
```

`pattern_rewriter_rules.cc` contains complete readable guard/recipe examples.
Each example defines its pattern, guard and replacement separately, assembles a
named `Rule rule`, and registers it with `rules.push_back(std::move(rule))`.
`Op(name, inputs)` defaults to the standard ONNX domain; the optional third
argument selects another domain, e.g. `p.Op("FusedMatMul", {x, y}, kMSDomain)`.
The original `Op(Operator{name, domain}, inputs)` overload remains available.
Name overloads accept literals, `std::string` and `std::string_view`; template
deduction keeps the original braced operator form unambiguous.

`p.NamedValue("sigmoid")` retrieves a named handle. Nonempty names must be unique.
`p.Name(value, name)` updates the canonical name and existing node definitions;
subsequent operators also use that canonical name. Previously returned `Value`
copies are not renamed in place. `Value` converts to `int` for indexing
`Match::values`; an unnamed value has an empty name, and a default value has
invalid handle -1.

In guards and replacement recipes, `m.valueOf("x")` directly returns the matched
`const NodeArg*`; there is no need to capture `p.NamedValue("x")` and index
`m.values` separately. Lookup also supports named intermediate outputs.
Unknown/empty names or unbound values throw an explicit error rather than
returning null. Names come from the current rule, even when its structural plan
is shared with a differently named pattern. Lookup scans the bounded pattern
value table and does not allocate a name map for each candidate.

`Replacement::inputs` accepts either `Value` handles or matched `const NodeArg*`
bindings, so recipes can use `{m.valueOf("x"), m.valueOf("y")}` directly.
Pointer inputs are validated against the match's external input bindings:
intermediates, root outputs, nulls and unrelated graph values are rejected before
mutation. Matches reference immutable compiled patterns; do not retain them or
their pointers beyond a callback.

### Attribute matchers and literal constants

`p.Attributes(...)` attaches matchers to the most recently defined node.
Each `AttributeMatcher` contains an attribute name and a boolean predicate on
`const AttributeProto&`. Missing attributes fail the match; schema defaults are
not synthesized. Equality factories check both attribute type and value:
`Equal("axis", 1)` requires INT, `Equal("alpha", 0.5f)` requires FLOAT,
and `Equal("mode", std::string("linear"))` requires STRING. Extra unmatched
attributes are permitted. Arbitrary predicates are also supported:

```cpp
p.Attributes({{"axis", [](const ONNX_NAMESPACE::AttributeProto& value) {
  return value.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT &&
         value.i() < 0;
}}});
```

Predicates are trusted, pure C++ callbacks, with the same locality and lifetime
requirements as rule guards. Duplicate/empty attribute names, empty callbacks,
or attachment before any node are errors.

`p.Constant(3.0f)` introduces a float32 scalar initializer constraint;
`p.Constant(int64_t{3})` introduces an int64 constraint. Integer literals in C++
select int64; `double` arguments are converted to float32. Float constants must
be finite. Matching requires an immutable rank-zero initializer of the exact
type and value; a one-element vector, an overridable initializer, or external-data
initializer does not match. These leaf handles can be used as replacement inputs,
but matching never creates tensors.

### Textual patterns

The explicit `Pattern(std::string_view text)` constructor parses a textual pattern.
The default constructor remains available for the programmatic builder.
Malformed text throws `OnnxRuntimeException` with a byte-offset diagnostic
(or aborts in exception-disabled builds, following ORT's throwing macros).
The requested branched pattern can be written directly:

```cpp
Pattern p(R"(
  (x, low, high, divisor) =>
    add = Add(x, 3.0)
    clip = Clip(add, low, high)
    mul = Mul(clip, x)
    return Div(mul, divisor)
)");
```

The header declares named inputs. Assignment names label operator outputs.
Reusing `x` enforces exact binding identity. Decimal/scientific literals become
float32 scalar constants, while integer spellings become int64 constants.
This text describes structure, **not a semantically complete HardSwish rule**:
the rule guard must additionally establish float32, low=0, high=6 and divisor=6.
The numerical test retains those checks before executing the replacement.

The actual example rule eliminates those fixed bindings entirely:

```cpp
Pattern p(R"(
  (x) =>
    add = Add(x, 3.0)
    clip = Clip(add, 0.0, 6.0)
    mul = Mul(clip, x)
    return Div(mul, 6.0)
)");
```

Its guard only checks float32, since the pattern checks every fixed scalar.
RMSNorm similarly uses `Pow(x, 2.0)` directly; epsilon remains a named binding
because its value varies and is copied to the fused operator's attributes.
MatMul division factors also remain variable for computed scaling attributes.

Attributes use Python keyword syntax, after positional inputs:
`h = HardSigmoid(x, alpha = 0.16666667, beta = 0.5)`.
Signed/scientific numbers and single/double-quoted string attributes are supported.
Only escaped quotes and backslashes are supported in strings.
The parsed spelling determines the attribute type: `axis=1` differs from `axis=1.0`.
An empty quoted string is valid.

Multiple-output assignments and qualified domains are supported:

```text
(x) => a, b = Split(x, axis=0); return Concat(a, b, axis=0)
(x, y) => mm = com.microsoft::FusedMatMul(x, y, alpha=0.5); return Identity(mm)
```

Unqualified operators use the standard ONNX domain. Whitespace/newlines separate
tokens; semicolons may separate assignments. Values are ASCII identifiers.
Definitions must precede use; every input and node must reach the final operator.
There must be a final `return Op(...)`, with no subsequent statements.
All six example rules use the string constructor and retrieve bindings with
`Match::valueOf` for their guards and dynamic replacement recipes; static
replacement plans bind inputs once during compilation. Tests use the constructor for parsed
patterns and test-only Identity, Split/Concat and LayerNorm-bias rules; builder
tests and multi-output-root definitions retain the programmatic API.

The parser checks duplicate names/attributes, undefined references, numeric range,
disconnected definitions and the same 64-node/256-value/64-input-output limits as
the C++ builder. Errors include a byte offset; `p = Pattern(text)` leaves `p`
unchanged if construction fails.
A one-node parsed pattern is valid, but cannot form a node-count-decreasing rule.

The language intentionally excludes nested calls, lists/tensor literals, optional
placeholders, booleans, comments, `return` of an existing value and multi-output
`return` calls (use the C++ API for multi-output roots). It describes only matching:
opset requirements and semantic guards remain explicit C++.

### Reusable replacement plans

`Replacement(std::string_view)` parses a reusable replacement definition, once:

```cpp
Replacement replacement("(x) => HardSwish(x)");
Rule rule{"HardSwish", pattern, {{kOnnxDomain, 14, 21}}, guard, replacement};
```

The final operator call may omit `return`. Multi-node definitions use the same
assignment, domain and keyword-attribute syntax as patterns:

```cpp
Replacement replacement(R"(
  (x) =>
    h = HardSigmoid(x, alpha=0.16666667, beta=0.5)
    return Mul(h, x)
)");
```

This defines a two-node replacement, not a callback that parses a string on every
match. The constructor builds an immutable plan containing operator/domain names,
literal attributes, external input names and indexed earlier-output references.
Copies share that plan. When the plan is supplied directly as `Rule::replacement`,
transformer construction binds its external names to the rule's input handles,
and checks declared domains and node-count reduction. Binding is per rule, so
reusing a plan across differently ordered input headers is safe.

At runtime the callback returns a shared plan reference. There is no parsing,
attribute-literal construction, or name lookup for these prebound inputs.
The engine binds the external `NodeArg`s, allocates fresh intermediate values,
creates nodes in definition order, and applies the stored attributes.
Callbacks that need computed attributes can use extra bindings with a captured
parsed plan (below), or return programmatic recipes. Do not construct
`Replacement(text)` inside a per-match callback.

Replacement headers may select a subset of the matched pattern's inputs.
Tensor parameters cannot refer to matched intermediates. All declared replacement
parameters must be used as tensor inputs or attribute bindings, and all nodes
must reach the final operator. A multi-output assignment such as
`a, b = Split(x, axis=0)` declares the exact internal arity.
The final operator inherits **all matched root outputs in order**, including a
multi-output root; its schema must accept that count. Intermediate outputs cannot
be mapped independently to observable root slots yet.

Numeric positional literals are synthesized as immutable rank-zero float32 or
int64 initializers, using the same literal typing as patterns:
`Replacement("(x) => Add(x, 0.0)")`. No Constant kernel or subsequent
constant-folding pass is required. There is no general tensor/list synthesis.

Programmatic sequences use `NodeRecipe` and `Output{node_index, output_slot}`:

```cpp
Replacement replacement(InlinedVector<Replacement::NodeRecipe, 1>{
    {"HardSigmoid", kOnnxDomain, {std::string("x")}, sigmoid_attributes, 1},
    {"Mul", kOnnxDomain, {Replacement::Output{0, 0}, std::string("x")}, {}, 1}});
```

`NodeRecipe` contains op, domain, inputs, attributes and internal output count.
It also has `attribute_bindings`, a map from target attribute names to extra
binding names.
Its inputs may be names, existing `Value` handles, matched `NodeArg*` bindings,
earlier-output references or `Replacement::Scalar{...}` literals.
The final node's output count is inherited from the root rather than this field.
The original four-argument single-node constructor remains supported, including
computed-attribute callbacks. Its public single-node fields cannot be combined
with an immutable parsed/programmatic plan; mixed representations are rejected.

#### Computed attributes through extra bindings

Replacement keyword values may refer to names declared in its header.
Tensor arguments come from the pattern match; extra attribute arguments come
from `Match::update`, which accepts a concrete `AttributeProto`:

```cpp
Replacement plan("(x, y, alpha) => com.microsoft.FusedMatMul(x, y, alpha=alpha)");
auto replacement = [plan](const Graph& graph, const Match& m) {
  m.update("alpha", FloatAttribute("alpha",
      1.0f / *Scalar(graph, *m.valueOf("divisor"))));
  return plan;
};
Rule rule{"MatMulDiv", pattern, opsets, guard, replacement};
```

Here `FloatAttribute` and `Scalar` are the helpers used by the example rules;
the guard establishes float32 and a finite, nonzero scalar divisor.
The plan is parsed **once when defining the rule**, not once per successful
match. Capturing/returning it shares the immutable plan. Computed attributes
are looked up and copied into the new nodes at rewrite time. Captured callback
plans still resolve tensor names at runtime; only directly supplied plans have
their tensor names prebound by transformer construction.

`com.microsoft.FusedMatMul` is accepted in replacement text, as is the existing
`com.microsoft::FusedMatMul` spelling. Literal attributes and bound attributes
can be mixed, e.g. `axis=-1, epsilon=fused_epsilon, stash_type=1`.
Multiple nodes can share a binding, and target attribute names can differ from
binding names. Attribute proto names are normalized to the target names.
The proto carries the type, including list/tensor/string types not representable
as text literals; imported operator schemas validate the resolved attributes.

Extra bindings are invocation-local state, separate from immutable matched
graph values. `update` is callable on the callback's `const Match&`, because it
changes only this extra state, not the graph or matched values. Repeated updates
to an extra name replace its value. Guards must remain pure; the engine clears
extra state between candidates and before invoking the winning recipe. No binding
is retained between rewrites, graph runs or rules; match-only mode never invokes
recipes. `attributeOf(name)` retrieves a previously set extra attribute.

Empty names, undefined attribute types, reference attributes, missing bindings
and attempts to shadow any named pattern value are explicit errors.
A header name cannot be used both as a tensor argument and an attribute binding;
attribute references must name header parameters, not intermediate outputs.
All required extras are checked before graph mutation. Schema validation failures
use the same staged-node rollback as literal attributes. As with other recipes,
callbacks must establish semantic/type compatibility; extra bindings are not
an automatic semantic guard or a general graph transaction.

The MatMulDiv and RMSNorm examples use captured plans with computed bindings.
TransposeMatMul uses a literal parsed replacement. FusedMatMulDiv also captures
a parsed plan, computing `alpha` and binding `transA`, `transB`, `transBatchA`
and `transBatchB` from the original node. Missing transpose attributes use the
imported schema's defaults; the replacement emits all five version-1 attributes
explicitly, preserving their effective values. All six examples therefore use
parsed replacements. Extra bindings describe individual attributes, not an
attribute-map spread; a rule for a schema with other attributes must explicitly
preserve those too.

Every replacement must contain at least one node and **strictly fewer nodes
than the matched pattern**. Equal-size rewrites and expansions are rejected,
including dynamic recipes, retaining the node-count termination policy.
All internal references must point backward to valid output slots; disconnected
nodes and undeclared domains are errors. Limits remain 64 nodes and 64 inputs/
outputs per node. Pattern parsing additionally bounds named values to 256.

All replacement nodes are staged and their imported schemas/attributes checked
before deleting matched nodes or rewiring edges. A schema failure removes every
staged node and synthesized initializer, leaving the original live graph intact;
unused staging `NodeArg`s and consumed unique-name IDs may remain. Final Resolve
is still responsible for type/shape inference and validation; arbitrary computed
recipes must establish semantic/type compatibility, and post-commit Resolve
failure is not a general graph transaction rollback.
All added nodes inherit root provider/layer metadata. Producer/consumer maps,
internal edges and external root edges are updated; every added node seeds the
invalidation worklist. Replacement notification attributes the removed pattern
to the final node.

### Multiple outputs

`MultiOp(name, inputs, output_count, domain)` returns an ordered vector of distinct
value handles. `domain` is optional, and an `Operator` overload is also available.
For example:

```cpp
Pattern p;
auto x = p.Input();
auto pieces = p.MultiOp("Split", {x}, 2);
p.Op("Concat", {pieces[0], pieces[1]});
```

The output count must match the node exactly. Matching can find a producer
through any output, including its second or later output, and checks every
bound output against its exact output slot. Unused outputs of a connected node
are allowed in the definition, but cannot be graph outputs or have external
consumers if the node is deleted. The tests include Split/Concat and a genuine
ONNXScript LayerNorm-bias adaptation with unused mean/inverse-standard-deviation
outputs; exposing either statistic prevents its deletion.

A multi-output root is also supported. The final replacement node inherits
**all root outputs in the original order**, including their names, types, shapes,
graph-output status, and consumers. The recipe does not need an output list.
Its schema must permit that output count. Identity/Split to Split tests execute
both root outputs and downstream fanout, including repeated uses of output 1.
This does not yet support mapping observable outputs from multiple matched nodes
to arbitrary replacement slots, or matching omitted optional output placeholders.

### Guards, applicability and integration

Guards and recipes are trusted C++ code: they must be pure, guards must depend on
local bindings/attributes/constants rather than mutable global graph conditions,
they must not retain match
pointers, and must establish semantic equivalence and replacement type
compatibility. The caller must establish kernel compatibility for its EP policy.
The engine provides structural and deletion safety; it cannot prove
an arbitrary callback's mathematical correctness. Guard false means an ordinary
nonmatch; invalid patterns or recipes report explicit errors.

Each `OpsetRequirement` is `{domain, min_version, max_version}`, with inclusive,
positive bounds. A rule must declare every domain used by its pattern or
replacement; additional domain requirements are allowed. All requirements must
be satisfied by the graph's import map. Missing imports or out-of-range versions
make the whole rule inapplicable, before matching or guards. Duplicate domains
(including `""`/`"ai.onnx"` aliases), reversed ranges and undeclared pattern
domains are rejected when compiling rules.

For example, MatMul/Div to FusedMatMul declares
`{{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}`. The standard-domain nodes use the
model's ONNX import; the replacement uses the existing Microsoft-domain import.
Rule authors are responsible for equivalence throughout the declared ranges.
The engine neither derives nor compares node schema since-versions. Replacement
schemas are resolved normally from imports and validated before deletion.

`Apply` recurses into subgraphs and resolves the graph through the normal ORT
transformer wrapper. For per-call statistics and explicit graph resolution:

```cpp
PatternRewriter rewriter(MakeExampleRules());
Statistics stats;
ORT_RETURN_IF_ERROR(rewriter.Run(graph, stats));  // already-resolved graph, one graph only
ORT_RETURN_IF_ERROR(graph.Resolve());            // required after rewriting
```

Callers must not concurrently mutate a graph. Compiled rules and options are
immutable and per-call state/statistics are local.

### Generic execution-provider policy

Rules do not inspect EP assignments. After structural matching and before rule
guards, the engine applies `CanRewritePartition`: all matched nodes must have
the same assignment, and an assigned EP must belong to
`Options::compatible_execution_providers`. Unassigned nodes are always eligible.
The default allowed set contains only CPU, preserving the verified execution scope.
Callers can opt into an EP (or clear the set for unassigned-only rewrites):

```cpp
Options options;
options.compatible_execution_providers.insert(kCudaExecutionProvider);
PatternRewriter rewriter(MakeExampleRules(), options);
```

The replacement inherits the root's assignment. Allowing an EP is a caller assertion
of kernel/precision compatibility, not automatic kernel discovery or evidence of
GPU execution. Mixed assignments are rejected even if both EPs are allowed. This
generic policy also applies in match-only mode, which measures safe rewrite
eligibility rather than partition-independent structural matches.

## Compilation, matching, mutation, and termination

Construction validates connected acyclic definitions, used inputs, rule-level
domain/opset requirements, and limits of 64 nodes, 256 values and 64 inputs/outputs per operator.
It buckets groups by root op/domain and interns identical structural signatures,
including ordered output handles, binding identities and scalar literal type/value
constraints, but not diagnostic names or rule-level import ranges.
Rules with identical topology and literal constraints can share one structural
match even when their guards or applicability ranges differ. Patterns containing
attribute predicates currently receive separate plans, including equality predicates:
opaque `std::function` callbacks cannot safely be compared. This conservative
choice preserves correctness at the cost of repeated matching for attribute-heavy
rule sets; arbitrary callbacks also have no engine-imposed time bound.
There is **no arbitrary-prefix trie**: different structural
plans still match independently. This is intentionally a simpler, measurable
starting point than a decision DAG compiler.

Each call filters rules once against `Graph::DomainToVersionMap()`, retaining
registration order within each structural group. No applicability state is cached
across graphs; recursive subgraph calls use their own graph import context.
Filtering costs the sum of rule requirement counts, outside the node traversal.
If no rules apply, the call returns without traversing nodes. Model-local function
rewriting remains unsupported; future function rewriting must use that function's
imports, not the enclosing model's.

The call then builds a producer index and topological worklist once. Matching walks
the compiled node definitions backward without recursive search, alternation,
commutative backtracking, or unbounded neighborhood exploration. Each node checks
op/domain, exact input/output counts, repeated bindings and layer annotation
agreement, plus any attribute predicates and scalar literal constraints.
Provider eligibility is a separate generic check. All non-root outputs
must be internal: graph outputs,
external consumers (including implicit subgraph captures), and pattern inputs
produced inside the deletion set prevent replacement.

Registration order defines priority even across different structural groups.
Match-only mode reports all passing rule guards; rewrite mode selects only the
first passing rule. Replacements may consume **only matched input variables,
new scalar literals or earlier replacement outputs**, retain the
root's output `NodeArg`s (name/type/shape and external users), and inherit its
execution provider and layer annotation. The replacement domain must be declared
in the rule requirements and already imported for every added node. Replacement schema/attribute checks
precede deletion. ORT edges and producer/consumer indexes are updated; removed
nodes are reported via `NotifyNodeReplacement`. Final Resolve validates types and
inferred shapes, rather than silently ignoring graph-validation errors.

After mutation, all replacement nodes and surviving input producers seed a bounded
forward invalidation walk. Descendant roots up to the maximum pattern size are
requeued, since producer changes or reduced use counts can unlock deeper patterns.
Queue deduplication avoids redundant pending entries; removed IDs are skipped.
Traversal and rule priority are deterministic.

Every rule replaces at least two nodes with a smaller nonempty node sequence. Thus every
successful rewrite strictly decreases live node count, and a graph with N nodes
admits at most N-1 rewrites. This is a termination policy, **not a runtime bound**:
guards, fanout, structural checks, and worklist invalidation still cost time.
Expansion rules and one-node rewrites are deliberately rejected.

For match-only workloads, let U be the number of distinct structural plans in a
root bucket and P the maximum pattern size. Structure work is bounded by the
sum over visited roots of U times pattern/boundary checking cost, plus passing
plan guard costs. Boundary checks include matched-node fanout; they are not
constant-time. The reference mode performs the same checks per rule instead,
with cheap root op/domain rejection before allocating binding buffers.
Rewrite workloads additionally pay edge updates and bounded descendant walks;
large fanout can make these expensive. Arbitrary trusted guards have no imposed
time bound. These structural bounds do not provide mathematical worst-case runtime
guarantees for arbitrary rules or models.

`visited` counts live worklist roots, including revisits. `candidates` counts
enumerated rule IDs (all rules in the reference mode, root-indexed rules in the
indexed mode); `structural_checks` counts plan attempts, including cheap root
rejections in the reference. `guards` counts semantic callback evaluations after
structural and deletion-boundary checks. `matches` counts all passing guards in
match-only mode, or selected winners in rewrite mode; `rewrites` counts completed
mutations. Match-only therefore measures safe rewrite eligibility, not a search
that ignores external consumers.

## Intentional limitations and precision

Matched and replacement nodes may have multiple outputs, but the final replacement
node must preserve the root's complete ordered output list. Optional/missing
inputs or outputs, variadic patterns,
model-local function rewriting, general tensor synthesis, domain-import edits, alternative
patterns and implicit commutation are unsupported. Control-flow nodes are not
matched, but `Apply` rewrites inside their subgraphs. External tensor data and
overridable initializers do not satisfy example constant guards; scalar constants
must be finite float32 rank-zero initializers. Singleton vectors are deliberately
not treated as scalars, since broadcasting can otherwise change the output rank.
Constant nodes require an earlier constant-folding pass.

The examples support a conservative subset, not every ONNXScript variant:
RMSNorm requires the last axis, keepdims, positive finite epsilon, float32,
known positive-width matching one-dimensional scale, and pre-opset-18 attribute
axes. Transpose requires known rank >=2 and exactly the last-two-axis swap.
RMSNorm declares ONNX opsets 7 through 17. MatMul Div requires a finite nonzero
scalar and finite reciprocal. The FusedMatMul examples declare ONNX opsets 9
through 21 and require an existing `com.microsoft` opset-1 import; they do not add
one. The upper bounds are conservative supported ranges, not inferred schema
version boundaries; extending them requires reviewing and testing semantics.
HardSwish requires exact float32 constants/attributes and model opset 14 through
21. The tested CPU build executes HardSwish by expanding its ONNX function body
to HardSigmoid/Mul; RMSNorm and FusedMatMul execute native CPU kernels.
Floating-point fusions change evaluation order; equivalence means tested
tolerance, not bitwise identity. NaNs/infinities and every extreme finite input
are not claimed to have bitwise-equivalent behavior.

## Correctness tests

The normal tests live in `onnxruntime/test/optimizer/pattern_rewriter_test.cc`.
They include executed CPU equivalence and structural assertions, not only match
counts. Use an isolated build directory and virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe tools\ci_build\build.py --build_dir build\rewriter --config Release --update --parallel 4 --skip_tests --cmake_generator "Visual Studio 17 2022" --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
.\.venv\Scripts\python.exe tools\ci_build\build.py --build_dir build\rewriter --config Release --build --parallel 4 --target onnxruntime_test_all --skip_tests
Set-Location build\rewriter\Release\Release
.\onnxruntime_test_all.exe --gtest_filter="PatternRewriterTests.*"
```

Ensure CMake is on PATH (for example, Visual Studio's bundled CMake).
`FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER` avoids accidentally importing an
incompatible installed dependency, such as a Debug-only protobuf installation
while building Release. Dependencies are built inside this isolated
build tree instead of modifying the shared installation.

## Decoder correctness topology

`BuildDecoder` is a **hand-constructed, model-derived eager Llama-style topology**,
not an exported pretrained model or a claim of identical exporter node order.
Sources at ORT base revision
[`42518d89d7c5393baf97d69f8c2670668c40decf`](https://github.com/microsoft/onnxruntime/tree/42518d89d7c5393baf97d69f8c2670668c40decf)
are `onnxruntime/python/tools/transformers/models/llama/convert_to_onnx.py`
(eager decoder, dynamic past/present KV exports),
`models/llama/llama_inputs.py` (cache layout),
`onnxruntime/test/optimizer/group_query_attention_pre_norm_fusion_test.cc`
(small Q/K/V/head/cache geometry), and ONNXScript's RMSNorm source above.

Each block has two decomposed RMSNorms, Q/K/V projections, reshape/transposes,
split-half RoPE via Slice/Neg/Concat and gathered sine/cosine position caches,
past/current KV concatenation with per-layer graph outputs, grouped-head expansion,
QK transpose/MatMul/scaling/causal-mask/Softmax/AV attention, output projection,
two residual adds, and a SiLU-gated up/gate/down MLP. B=1, query heads=4,
KV heads=2, head size=4, hidden width=16, MLP width=64, past length=3.
Sine/cosine caches use standard paired frequencies; weights are tiny random
float32 tensors. No tokenizer, embedding lookup, LM head, dynamic position/cache
control flow, mixed precision, quantization or production tensor sizes are modeled.
Both sequence=1 decode and sequence=4 causal prefill have executed equivalence tests.

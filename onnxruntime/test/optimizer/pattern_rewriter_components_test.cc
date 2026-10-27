// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <limits>
#include <sstream>

#include "core/optimizer/pattern_rewriter.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {
using namespace rewrite;

struct TestGraph {
  TestGraph()
      : model{"rewrite-components", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}}, {}, DefaultLoggingManager().DefaultLogger()} {}
  Model model;
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder{graph};
};

size_t Count(const Graph& graph, const std::string& op) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == op;
  return count;
}

bool ComposeTransposePermutations(const Graph&, Match& match) {
  const auto* shape = match.valueOf("x")->Shape();
  if (!shape) return match.Reject("Transpose composition requires known rank.");
  const int rank = shape->dim_size();
  const auto& p = match.attributeOf("p");
  const auto& q = match.attributeOf("q");
  auto valid = [rank](const ONNX_NAMESPACE::AttributeProto& permutation) {
    if (permutation.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INTS ||
        permutation.ints_size() != rank)
      return false;
    InlinedHashSet<int64_t> seen;
    for (int64_t axis : permutation.ints())
      if (axis < 0 || axis >= rank || !seen.insert(axis).second) return false;
    return true;
  };
  if (!valid(p) || !valid(q)) return match.Reject("Invalid transpose permutation.");
  ONNX_NAMESPACE::AttributeProto composed;
  composed.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  for (int64_t axis : q.ints()) composed.add_ints(p.ints(static_cast<int>(axis)));
  match.update("composed", std::move(composed));
  return true;
}

Pattern TransposePairPattern() {
  return Pattern("(x,p:ints,q:ints)=>{a=Transpose(x,perm=p);return Transpose(a,perm=q)}");
}

Rule ComposeTransposeRule() {
  return Rule{"ComposeTranspose", TransposePairPattern(), "(x,composed)=>Transpose(x,perm=composed)", ComposeTransposePermutations, {{kOnnxDomain, 13, 25}}};
}
}  // namespace

TEST(PatternComponentTests, AnchoredMatchingDoesNotRewriteOrRequireARule) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* y = b.MakeIntermediate();
  b.AddNode("Identity", {x}, {y});
  auto& root = b.AddNode("Sigmoid", {y}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  const auto before = test.model.ToProto().SerializeAsString();
  int calls = 0;
  ASSERT_STATUS_OK(Pattern("(x:float[2])=>{a=Identity(x);return Sigmoid(a)}")
                       .VisitMatches(test.graph, root.Index(), [&](Match& match) {
                         ++calls;
                         EXPECT_EQ(match.valueOf("x"), x);
                         EXPECT_EQ(match.valueOf("a"), y);
                         return true;
                       }));
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(test.model.ToProto().SerializeAsString(), before);
  ASSERT_STATUS_OK(Pattern("(x:int64[2])=>return Sigmoid(x)")
                       .VisitMatches(test.graph, root.Index(), [&](Match&) { ADD_FAILURE(); return true; }));
}

TEST(PatternComponentTests, OrderedCandidatesOwnPlansAndIsolatePreparedState) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* y = b.MakeIntermediate();
  b.AddNode("Identity", {x}, {y});
  auto& root = b.AddNode("Sigmoid", {y}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  Match saved;
  int calls = 0;
  Rule::CheckAndBind check = [&](const Graph&, Match& match) {
    ++calls;
    if (calls == 1) {
      ONNX_NAMESPACE::AttributeProto value;
      value.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
      value.set_i(1);
      match.update("rejected", value);
      return match.Reject("try next");
    }
    EXPECT_THROW(match.attributeOf("rejected"), OnnxRuntimeException);
    EXPECT_TRUE(match.RejectionReason().empty());
    ONNX_NAMESPACE::AttributeProto accepted;
    accepted.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    accepted.set_i(2);
    match.update("accepted", accepted);
    return true;
  };
  ASSERT_STATUS_OK(PatternLibrary(R"(
    pattern E(x){v=Identity(x)|x|x;return Sigmoid(v)}
  )")
                       .GetPattern("E")
                       .VisitMatches(test.graph, root.Index(), [&](Match& match) {
                         if (!check(test.graph, match)) return true;
                         saved = match;
                         return false;
                       }));
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(saved.valueOf("x"), y);
  EXPECT_EQ(saved.attributeOf("accepted").i(), 2);
  EXPECT_EQ(saved.instance("v").alternative(), "x");
  Match copied = saved;
  saved = {};
  EXPECT_EQ(copied.valueOf("x"), y);
  EXPECT_EQ(copied.attributeOf("accepted").i(), 2);
}

TEST(PatternComponentTests, InvalidContextAndCallbackErrorsAreNotNonmatches) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto& root = b.AddNode("Identity", {x}, {b.MakeOutput()});
  b.SetGraphOutputs();
  Pattern pattern("(x)=>return Identity(x)");
  auto visitor = [](Match&) { return true; };
  EXPECT_FALSE(pattern.VisitMatches(test.graph, root.Index(), visitor).IsOK());
  ASSERT_STATUS_OK(test.graph.Resolve());
  EXPECT_FALSE(pattern.VisitMatches(test.graph, std::numeric_limits<NodeIndex>::max(), visitor).IsOK());
  EXPECT_FALSE(pattern.VisitMatches(test.graph, root.Index(), {}).IsOK());
  EXPECT_THROW((void)pattern.VisitMatches(test.graph, root.Index(), [](Match&) -> bool {
    ORT_THROW("callback error");
  }),
               OnnxRuntimeException);
}

TEST(PatternComponentTests, ValueAliasesSurviveSymbolTableGrowth) {
  TestGraph test;
  auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  auto& root = test.builder.AddNode("Sigmoid", {x}, {test.builder.MakeOutput()});
  test.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  for (size_t count : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
    SCOPED_TRACE(count);
    std::ostringstream text;
    text << "pattern E(x:float[2]) {";
    for (size_t i = 0; i < count; ++i) text << "a" << i << "=x;";
    text << "return Sigmoid(a" << count - 1 << ")}";
    std::optional<Match> result;
    ASSERT_STATUS_OK(PatternLibrary(text.str()).GetPattern("E").MatchAt(test.graph, root.Index(), result));
    ASSERT_TRUE(result);
    for (size_t i = 0; i < count; ++i) EXPECT_EQ(result->valueOf("a" + std::to_string(i)), x);
  }
}

TEST(PatternComponentTests, MatchAtReturnsFirstCandidateAndClearsOrdinaryNonmatch) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* y = b.MakeIntermediate();
  auto& inner = b.AddNode("Identity", {x}, {y});
  auto& root = b.AddNode("Sigmoid", {y}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  const auto before = test.model.ToProto().SerializeAsString();
  std::optional<Match> result;
  ASSERT_STATUS_OK(PatternLibrary("pattern E(x){v=Identity(x)|x;return Sigmoid(v)}")
                       .GetPattern("E")
                       .MatchAt(test.graph, root.Index(), result));
  ASSERT_TRUE(result);
  EXPECT_EQ(result->valueOf("x"), x);
  EXPECT_EQ(result->nodes.size(), 2u);
  Match copy = *result;
  EXPECT_FALSE(Pattern("(x)=>return Identity(x)").MatchAt(test.graph, std::numeric_limits<NodeIndex>::max(), result).IsOK());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->valueOf("x"), x);
  EXPECT_THROW((void)Pattern("(x)=>return x").MatchAt(test.graph, root.Index(), result), OnnxRuntimeException);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->valueOf("x"), x);
  ASSERT_STATUS_OK(Pattern("(x)=>return Identity(x)").MatchAt(test.graph, root.Index(), result));
  EXPECT_FALSE(result);
  EXPECT_EQ(copy.valueOf("x"), x);
  ASSERT_STATUS_OK(Pattern("(x)=>return Identity(x)").MatchAt(test.graph, inner.Index(), result));
  ASSERT_TRUE(result);
  EXPECT_EQ(result->valueOf("x"), x);
  EXPECT_EQ(test.model.ToProto().SerializeAsString(), before);
  test.graph.SetGraphResolveNeeded();
  EXPECT_FALSE(Pattern("(x)=>return Identity(x)").MatchAt(test.graph, inner.Index(), result).IsOK());
  ASSERT_TRUE(result);
}

TEST(PatternComponentTests, OptionalAttributeCapturesDistinguishAbsenceFromUnknownNames) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
  auto& root = b.AddNode("Transpose", {x}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  std::optional<Match> result;
  ASSERT_STATUS_OK(Pattern("(x,p:ints)=>return Transpose(x,perm=p)").MatchAt(test.graph, root.Index(), result));
  EXPECT_FALSE(result);
  ASSERT_STATUS_OK(Pattern("(x)=>return Transpose(x)").MatchAt(test.graph, root.Index(), result));
  ASSERT_TRUE(result);
  EXPECT_EQ(result->nodes[0]->GetAttributes().find("perm"), result->nodes[0]->GetAttributes().end());
  EXPECT_THROW(result->attributeOf("p"), OnnxRuntimeException);
  ASSERT_STATUS_OK(Pattern("(x,p:ints|None)=>return Transpose(x,perm=p)").MatchAt(test.graph, root.Index(), result));
  ASSERT_TRUE(result);
  EXPECT_EQ(result->optionalAttributeOf("p"), nullptr);
  EXPECT_EQ(result->bindings().optionalAttributeOf("p"), nullptr);
  EXPECT_THROW(result->attributeOf("p"), OnnxRuntimeException);
  EXPECT_THROW(result->optionalAttributeOf("missing"), OnnxRuntimeException);
  EXPECT_THROW(result->optionalAttributeOf("x"), OnnxRuntimeException);
  EXPECT_THROW(result->valueOf("p"), OnnxRuntimeException);
  EXPECT_THROW(result->optionalValueOf("p"), OnnxRuntimeException);
  Match copy = *result;
  ONNX_NAMESPACE::AttributeProto prepared;
  prepared.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  prepared.set_i(1);
  EXPECT_THROW(result->update("p", prepared), OnnxRuntimeException);
  result->update("prepared", prepared);
  EXPECT_EQ(result->optionalAttributeOf("prepared")->i(), 1);
  result->Reject("rejected");
  result->ClearUpdates();
  EXPECT_EQ(result->optionalAttributeOf("p"), nullptr);
  EXPECT_EQ(copy.optionalAttributeOf("p"), nullptr);
  EXPECT_THROW(result->optionalAttributeOf("prepared"), OnnxRuntimeException);
}

TEST(PatternComponentTests, OptionalAttributeTypesAndResolvedDefaults) {
  for (const char* type : {"int", "ints", "float", "floats", "string", "strings", "tensor"}) {
    const auto text = MakeString("(x,p:", type, "|None)=>return Identity(x,unused=p)");
    EXPECT_NO_THROW((void)Pattern(text));
  }
  for (const char* text : {
           "(x,p:None)=>return Identity(x,unused=p)",
           "(x,p:ints|float|None)=>return Identity(x,unused=p)",
           "(x,p:const int64[1]|None)=>return Identity(x,unused=p)",
           "(x,p:float[1]|None)=>return Identity(x,unused=p)",
           "(x,p:ints|None)=>return Add(x,p,unused=p)"}) {
    EXPECT_THROW((void)Pattern(text), OnnxRuntimeException) << text;
  }
  for (bool present : {false, true}) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({2, 3}, -1.0f, 1.0f);
    auto& root = test.builder.AddNode("Transpose", {x}, {test.builder.MakeOutput()});
    if (present) root.AddAttribute("perm", std::vector<int64_t>{1, 0});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    for (const char* annotation : {"ints|None", "None|ints", "float|None"}) {
      std::optional<Match> match;
      ASSERT_STATUS_OK(Pattern(MakeString("(x,p:", annotation, ")=>return Transpose(x,perm=p)"))
                           .MatchAt(test.graph, root.Index(), match));
      EXPECT_EQ(match.has_value(), !present || std::string_view(annotation) != "float|None");
      if (match) {
        const auto* attribute = match->optionalAttributeOf("p");
        EXPECT_EQ(attribute != nullptr, present);
        if (attribute) EXPECT_EQ(attribute->ints_size(), 2);
      }
    }
  }
  TestGraph defaults;
  auto* x = defaults.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  auto& root = defaults.builder.AddNode("LeakyRelu", {x}, {defaults.builder.MakeOutput()});
  defaults.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(defaults.graph.Resolve());
  std::optional<Match> match;
  ASSERT_STATUS_OK(Pattern("(x,p:float|None)=>return LeakyRelu(x,alpha=p)")
                       .MatchAt(defaults.graph, root.Index(), match));
  ASSERT_TRUE(match);
  ASSERT_NE(match->optionalAttributeOf("p"), nullptr);
  EXPECT_FLOAT_EQ(match->attributeOf("p").f(), 0.01f);
  TestGraph integers;
  auto* input = integers.builder.MakeInput<float>({2, 3}, -1.0f, 1.0f);
  auto& softmax = integers.builder.AddNode("Softmax", {input}, {integers.builder.MakeOutput()});
  integers.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(integers.graph.Resolve());
  ASSERT_STATUS_OK(PatternLibrary("pattern E(x,p:int|None)->(y){y=Softmax(x,axis=p);return y}")
                       .GetPattern("E")
                       .MatchAt(integers.graph, softmax.Index(), match));
  ASSERT_TRUE(match);
  EXPECT_EQ(match->attributeOf("p").i(), -1);
}

TEST(PatternComponentTests, RepeatedOptionalAttributesRequireEqualPresenceAndPayload) {
  for (int inner = 0; inner < 3; ++inner) {
    for (int outer = 0; outer < 3; ++outer) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto* y = b.MakeIntermediate();
      auto& a = b.AddNode("Transpose", {x}, {y});
      auto& root = b.AddNode("Transpose", {y}, {b.MakeOutput()});
      if (inner) a.AddAttribute("perm", inner == 1 ? std::vector<int64_t>{1, 0} : std::vector<int64_t>{0, 1});
      if (outer) root.AddAttribute("perm", outer == 1 ? std::vector<int64_t>{1, 0} : std::vector<int64_t>{0, 1});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      std::optional<Match> match;
      ASSERT_STATUS_OK(Pattern("(x,p:ints|None)=>{a=Transpose(x,perm=p);return Transpose(a,perm=p)}")
                           .MatchAt(test.graph, root.Index(), match));
      EXPECT_EQ(match.has_value(), inner == outer) << inner << "," << outer;
    }
  }
  TestGraph scalar;
  auto* x = scalar.builder.MakeInput<float>({}, -1.0f, 1.0f);
  auto& root = scalar.builder.AddNode("Transpose", {x}, {scalar.builder.MakeOutput()});
  root.AddAttribute("perm", std::vector<int64_t>{});
  scalar.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(scalar.graph.Resolve());
  std::optional<Match> match;
  ASSERT_STATUS_OK(Pattern("(x,p:ints|None)=>return Transpose(x,perm=p)")
                       .MatchAt(scalar.graph, root.Index(), match));
  ASSERT_TRUE(match);
  ASSERT_NE(match->optionalAttributeOf("p"), nullptr);
  EXPECT_EQ(match->attributeOf("p").ints_size(), 0);
}

TEST(PatternComponentTests, OptionalAttributeCompositionAndCheckerRetryAreIsolated) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
  auto* y = b.MakeIntermediate();
  b.AddNode("Transpose", {x}, {y}).AddAttribute("perm", std::vector<int64_t>{1, 0});
  auto& root = b.AddNode("Transpose", {y}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  PatternLibrary library(R"(
    pattern Short(x,p:ints|None) { return Transpose(x,perm=p) }
    pattern Long(x,p:ints|None) { a=Transpose(x,perm=p); return Transpose(a) }
    pattern Nested(x,p:ints|None) { a=Short(x,p); return a }
    pattern E(x,p:ints|None) { a=Nested(x,p)|Long(x,p); return a }
  )");
  int calls = 0;
  Match saved;
  auto visitor = [&](Match& match) {
    if (++calls == 1) {
      EXPECT_EQ(match.optionalAttributeOf("p"), nullptr);
      EXPECT_EQ(match.instance("a").optionalAttributeOf("p"), nullptr);
      return true;
    }
    EXPECT_NE(match.optionalAttributeOf("p"), nullptr);
    EXPECT_EQ(match.attributeOf("p").ints(0), 1);
    saved = match;
    return false;
  };
  ASSERT_STATUS_OK(library.GetPattern("E").VisitMatches(test.graph, root.Index(), visitor));
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(saved.attributeOf("p").ints_size(), 2);
  EXPECT_EQ(saved.instance("a").optionalAttributeOf("p"), &saved.attributeOf("p"));
}

TEST(ReplacementComponentTests, ExplicitAbsentAttributesOmitKeywordsAndPreserveMissingErrors) {
  TestGraph test;
  auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  Replacement::Instantiation result;
  ASSERT_STATUS_OK(Replacement("(x,p)=>LeakyRelu(x,alpha=p)")
                       .Instantiate(test.graph, {{"x", x}}, {}, result, nullptr, {"p", "unused"}));
  EXPECT_FLOAT_EQ(test.graph.GetNode(result.nodes.front())->GetAttributes().at("alpha").f(), 0.01f);
  Replacement::Instantiation transpose;
  ASSERT_STATUS_OK(Replacement("(x,p)=>Transpose(x,perm=p)")
                       .Instantiate(test.graph, {{"x", x}}, {}, transpose, nullptr, {"p"}));
  EXPECT_TRUE(test.graph.GetNode(transpose.nodes.front())->GetAttributes().empty());
  const auto count = test.graph.NumberOfNodes();
  const auto initializers = test.graph.GetAllInitializedTensors().size();
  for (const char* text : {"(x,p)=>LeakyRelu(x,alpha=p)", "(x,p)=>{a=Add(x,1.0);return Cast(a,to=p)}"}) {
    Replacement::Instantiation unchanged = result;
    const bool required = std::string_view(text).find("Cast") != std::string_view::npos;
    const InlinedHashSet<std::string> absent = required ? InlinedHashSet<std::string>{"p"} : InlinedHashSet<std::string>{"typo"};
    EXPECT_FALSE(Replacement(text).Instantiate(test.graph, {{"x", x}}, {}, unchanged, nullptr, absent).IsOK());
    EXPECT_EQ(unchanged.nodes, result.nodes);
    EXPECT_EQ(unchanged.outputs, result.outputs);
    EXPECT_EQ(test.graph.NumberOfNodes(), count);
    EXPECT_EQ(test.graph.GetAllInitializedTensors().size(), initializers);
  }
  ONNX_NAMESPACE::AttributeProto attribute;
  attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  attribute.set_f(0.5f);
  EXPECT_FALSE(Replacement("(x,p)=>LeakyRelu(x,alpha=p)")
                   .Instantiate(test.graph, {{"x", x}}, {{"p", attribute}}, result, nullptr, {"p"})
                   .IsOK());
  EXPECT_FALSE(Replacement("(x)=>Identity(x)")
                   .Instantiate(test.graph, {{"x", x}}, {}, result, nullptr, {"x"})
                   .IsOK());
}

TEST(PatternRewriterTests, OptionalAttributeReplacementAndRequiredRuleIsolation) {
  for (bool indexed : {false, true}) {
    for (bool present : {false, true}) {
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto& root = test.builder.AddNode("Transpose", {x}, {test.builder.MakeOutput()});
      if (present) root.AddAttribute("perm", std::vector<int64_t>{1, 0});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      int required_calls = 0, optional_calls = 0;
      Rules rules{
          Rule{"Required", "(x,p:ints)=>return Transpose(x,perm=p)", "(x,p)=>Transpose(x,perm=p)", [&](const Graph&, Match&) { ++required_calls; return false; }, {{kOnnxDomain, 13, 25}}},
          Rule{"Optional", "(x,p:ints|None)=>return Transpose(x,perm=p)", "(x,p)=>Transpose(x,perm=p)", [&](const Graph&, Match& match) {
                 ++optional_calls;
                 EXPECT_EQ(match.optionalAttributeOf("p") != nullptr, present);
                 return true;
               },
               {{kOnnxDomain, 13, 25}}}};
      Options options;
      options.indexed = indexed;
      options.max_iterations = 1;
      PatternRewriter rewriter(std::move(rules), options);
      Statistics statistics;
      ASSERT_STATUS_OK(rewriter.Run(test.graph, statistics));
      EXPECT_EQ(statistics.rewrites, 1u);
      EXPECT_EQ(required_calls, present ? 1 : 0);
      EXPECT_EQ(optional_calls, 1);
      for (const auto& node : test.graph.Nodes()) {
        EXPECT_EQ(node.GetAttributes().count("perm"), present ? 1u : 0u);
      }
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, OptionalAttributeCheckerRejectionRetriesAndRequiredOmissionRollsBack) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
    auto* y = b.MakeIntermediate();
    b.AddNode("Transpose", {x}, {y}).AddAttribute("perm", std::vector<int64_t>{1, 0});
    b.AddNode("Transpose", {y}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    PatternLibrary library(R"(
      pattern Short(x,p:ints|None) { return Transpose(x,perm=p) }
      pattern Long(x,p:ints|None) { a=Transpose(x,perm=p);return Transpose(a) }
      pattern E(x,p:ints|None) { a=Short(x,p)|Long(x,p);return a }
    )");
    int rejected = 0, accepted = 0;
    Rule rule{"OptionalRetry", library.GetPattern("E"), "(x,p)=>Transpose(x,perm=p)", [&](const Graph&, Match& match) {
                if (!match.optionalAttributeOf("p")) {
                  ++rejected;
                  return match.Reject("Need explicit perm");
                }
                ++accepted;
                return true;
              },
              {{kOnnxDomain, 13, 25}}};
    Options options;
    options.indexed = indexed;
    options.match_only = true;
    Statistics statistics;
    ASSERT_STATUS_OK(PatternRewriter({std::move(rule)}, options).Run(test.graph, statistics));
    EXPECT_EQ(rejected, 1);
    EXPECT_EQ(accepted, 2);  // The inner short match and the outer long match.

    const auto before = test.model.ToProto().SerializeAsString();
    Rule invalid{"MissingRequired", "(x,p:ints|None)=>return Transpose(x,perm=p)", "(x,p)=>{a=Add(x,1.0);return Cast(a,to=p)}", [](const Graph&, Match& match) { return !match.optionalAttributeOf("p"); }, {{kOnnxDomain, 13, 25}}};
    options.match_only = false;
    EXPECT_FALSE(PatternRewriter({std::move(invalid)}, options).Run(test.graph, statistics).IsOK());
    EXPECT_EQ(test.model.ToProto().SerializeAsString(), before);
    EXPECT_EQ(test.graph.NumberOfNodes(), 2);
    EXPECT_TRUE(test.graph.GetAllInitializedTensors().empty());
  }
}

TEST(PatternComponentTests, DeletionBoundaryButNotProviderPolicyIsIncluded) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* y = scenario == 1 ? b.MakeOutput() : b.MakeIntermediate();
    auto& inner = b.AddNode("Identity", {x}, {y});
    auto& root = b.AddNode("Sigmoid", {y}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    inner.SetExecutionProviderType(kCudaExecutionProvider);
    root.SetExecutionProviderType(kCpuExecutionProvider);
    if (scenario == 2) inner.SetLayeringAnnotation("different-layer");
    int calls = 0;
    ASSERT_STATUS_OK(Pattern("(x)=>{a=Identity(x);return Sigmoid(a)}")
                         .VisitMatches(test.graph, root.Index(), [&](Match&) { ++calls; return true; }));
    EXPECT_EQ(calls, scenario == 0 ? 1 : 0);
  }
}

TEST(ReplacementComponentTests, InstantiateFreshFragmentWithoutPatternOrMatch) {
  TestGraph test;
  auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  Replacement recipe("(x)=>{a=Add(x,1.0);return LeakyRelu(a)}");
  Replacement::Instantiation result;
  ONNX_NAMESPACE::AttributeProto unused;
  unused.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  unused.set_i(1);
  ASSERT_STATUS_OK(recipe.Instantiate(test.graph, {{"x", x}, {"unused", x}}, {{"unused_attribute", unused}}, result));
  ASSERT_EQ(result.nodes.size(), 2u);
  ASSERT_EQ(result.outputs.size(), 1u);
  EXPECT_TRUE(test.graph.GraphResolveNeeded());
  EXPECT_TRUE(test.graph.GetOutputs().empty());
  EXPECT_TRUE(test.graph.GetNode(result.nodes.back())->GetExecutionProviderType().empty());
  EXPECT_EQ(test.graph.GetNode(result.nodes.back())->GetAttributes().at("alpha").f(), 0.01f);
  test.graph.SetOutputs({result.outputs.front()});
  ASSERT_STATUS_OK(test.graph.Resolve());
  EXPECT_EQ(result.outputs.front()->Shape()->dim(0).dim_value(), 2);
  EXPECT_EQ(test.graph.GetNode(result.nodes.back())->GetInputEdgesCount(), 1u);
}

TEST(ReplacementComponentTests, PositionalOutputsAttributesOptionalInputsAndSource) {
  TestGraph test;
  auto* x = test.builder.MakeInput<float>({4}, -1.0f, 1.0f);
  auto& source = test.builder.AddNode("Identity", {x}, {test.builder.MakeOutput()});
  source.SetExecutionProviderType(kCpuExecutionProvider);
  source.SetLayeringAnnotation("layer0");
  test.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  ONNX_NAMESPACE::AttributeProto axis;
  axis.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  axis.set_i(0);
  Replacement::Instantiation result;
  ASSERT_STATUS_OK(Replacement("(x,axis)=>{a,b=Split(x,axis=axis);c=Concat(a,b,axis=axis);return b,c}")
                       .Instantiate(test.graph, {{"x", x}}, {{"axis", axis}}, result, &source));
  ASSERT_EQ(result.outputs.size(), 2u);
  EXPECT_NE(result.outputs[0], result.outputs[1]);
  EXPECT_EQ(test.graph.GetNode(result.nodes.front())->GetExecutionProviderType(), kCpuExecutionProvider);
  EXPECT_EQ(test.graph.GetNode(result.nodes.front())->GetLayeringAnnotation(), "layer0");
  EXPECT_EQ(test.graph.GetOutputs().front(), source.OutputDefs()[0]);
  test.graph.SetOutputs({result.outputs[0], result.outputs[1]});
  ASSERT_STATUS_OK(test.graph.Resolve());
  EXPECT_EQ(result.outputs[0]->Shape()->dim(0).dim_value(), 2);
  Replacement::Instantiation clip;
  ASSERT_STATUS_OK(Replacement("(x,min)=>Clip(x,min)").Instantiate(test.graph, {{"x", x}, {"min", nullptr}}, {}, clip));
  EXPECT_FALSE(test.graph.GetNode(clip.nodes.front())->InputDefs()[1]->Exists());
  Replacement::Instantiation another_clip;
  ASSERT_STATUS_OK(Replacement("(x)=>Clip(x)").Instantiate(test.graph, {{"x", x}}, {}, another_clip));
  EXPECT_NE(clip.outputs.front(), another_clip.outputs.front());
  EXPECT_EQ(test.graph.GetOutputs()[0], result.outputs[0]);
}

TEST(ReplacementComponentTests, SchemaVerificationIsNotShapeInferenceOrResolveTransaction) {
  TestGraph test;
  auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  Replacement::Instantiation result;
  ASSERT_STATUS_OK(Replacement("(x)=>Transpose(x,perm=[0,1])").Instantiate(test.graph, {{"x", x}}, {}, result));
  test.graph.SetOutputs({result.outputs.front()});
  EXPECT_FALSE(test.graph.Resolve().IsOK());
  EXPECT_NE(test.graph.GetNode(result.nodes.front()), nullptr);

  TestGraph wrong_attribute;
  auto* input = wrong_attribute.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  ONNX_NAMESPACE::AttributeProto permutation;
  permutation.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  permutation.set_i(0);
  Replacement::Instantiation unchanged;
  EXPECT_FALSE(Replacement("(x,p)=>Transpose(x,perm=p)")
                   .Instantiate(wrong_attribute.graph, {{"x", input}}, {{"p", permutation}}, unchanged)
                   .IsOK());
  EXPECT_EQ(wrong_attribute.graph.NumberOfNodes(), 0);
  EXPECT_TRUE(unchanged.nodes.empty());
}

TEST(ReplacementComponentTests, ValidationRollsBackAndPreservesCallerResult) {
  for (const char* text : {
           "(x)=>{a=Add(x,1.0);return Identity(a,unexpected=1)}",
           "(x)=>{a=Add(x,1.0);return missing.Op(a)}",
           "(x)=>{a=Add(x,1.0);return MissingStandard(a)}",
           "(x)=>Identity(x)", "(a,missing)=>Transpose(a,perm=missing)"}) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
    Replacement::Instantiation result;
    result.nodes.push_back(123);
    result.outputs.push_back(x);
    const auto initializers = test.graph.GetAllInitializedTensors().size();
    Replacement::TensorBindings bindings;
    if (std::string_view(text).find("unexpected") != std::string_view::npos ||
        std::string_view(text).find("missing.Op") != std::string_view::npos ||
        std::string_view(text).find("MissingStandard") != std::string_view::npos)
      bindings.emplace("x", x);
    if (std::string_view(text).find("perm=missing") != std::string_view::npos) bindings.emplace("a", x);
    EXPECT_FALSE(Replacement(text).Instantiate(test.graph, bindings, {}, result).IsOK());
    EXPECT_EQ(test.graph.NumberOfNodes(), 0);
    EXPECT_EQ(test.graph.GetAllInitializedTensors().size(), initializers);
    EXPECT_EQ(result.nodes, (InlinedVector<NodeIndex>{123}));
    EXPECT_EQ(result.outputs, (InlinedVector<NodeArg*>{x}));
  }
  TestGraph test;
  Replacement::Instantiation result;
  EXPECT_FALSE(Replacement("(x)=>Identity(x)").Instantiate(test.graph, {{"x", nullptr}}, {}, result).IsOK());
  EXPECT_TRUE(result.nodes.empty());
}

TEST(ReplacementComponentTests, CrossGraphBindingsAndAnnotationSourcesAreRejected) {
  TestGraph test, other;
  auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* foreign = other.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  auto& foreign_source = other.builder.AddNode("Identity", {foreign}, {other.builder.MakeOutput()});
  Replacement recipe("(x)=>Identity(x)");
  Replacement::Instantiation result;
  EXPECT_FALSE(recipe.Instantiate(test.graph, {{"x", foreign}}, {}, result).IsOK());
  EXPECT_FALSE(recipe.Instantiate(test.graph, {{"x", x}}, {}, result, &foreign_source).IsOK());
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr.set_i(1);
  EXPECT_FALSE(recipe.Instantiate(test.graph, {{"x", x}}, {{"x", attr}}, result).IsOK());
  EXPECT_EQ(test.graph.NumberOfNodes(), 0);
  auto& local_source = test.builder.AddNode("Identity", {x}, {test.builder.MakeOutput()});
  ASSERT_EQ(local_source.Index(), foreign_source.Index());
  EXPECT_FALSE(recipe.Instantiate(test.graph, {{"x", x}}, {}, result, &foreign_source).IsOK());
  EXPECT_EQ(test.graph.NumberOfNodes(), 1);
  EXPECT_TRUE(result.nodes.empty());
  EXPECT_TRUE(result.outputs.empty());
}

TEST(MatchComponentTests, CapturedPreparedBindingsAndCopyAreSeparate) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto& root = b.AddNode("LeakyRelu", {x}, {b.MakeOutput()});
  root.AddAttribute("alpha", 0.25f);
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  bool visited = false;
  auto visitor = [&](Match& match) {
    visited = true;
    auto prepared = match.attributeOf("a");
    prepared.set_f(0.5f);
    EXPECT_THROW(match.update("a", prepared), OnnxRuntimeException);
    EXPECT_THROW(match.update("x", prepared), OnnxRuntimeException);
    match.update("prepared", prepared);
    Match copy = match;
    prepared.set_f(0.75f);
    match.update("prepared", prepared);
    EXPECT_FLOAT_EQ(match.attributeOf("prepared").f(), 0.75f);
    EXPECT_FLOAT_EQ(copy.attributeOf("prepared").f(), 0.5f);
    EXPECT_FLOAT_EQ(copy.attributeOf("a").f(), 0.25f);
    EXPECT_FALSE(match.Reject("unsupported"));
    match.ClearUpdates();
    EXPECT_TRUE(match.RejectionReason().empty());
    EXPECT_FLOAT_EQ(match.attributeOf("a").f(), 0.25f);
    EXPECT_THROW(match.attributeOf("prepared"), OnnxRuntimeException);
    EXPECT_EQ(std::get<int64_t>(match.dimensionOf("N")), 2);
    EXPECT_THROW(match.optionalValueOf("missing"), OnnxRuntimeException);
    EXPECT_THROW(match.valueOf("a"), OnnxRuntimeException);
    return false;
  };
  ASSERT_STATUS_OK(Pattern("(x:float[N],a:float)=>return LeakyRelu(x,alpha=a)")
                       .VisitMatches(test.graph, root.Index(), visitor));
  EXPECT_TRUE(visited);
  Match retained;
  {
    Rule rule{"Evidence", "(x,a:float)=>return LeakyRelu(x,alpha=a)", "(x)=>Identity(x)", [&](const Graph&, Match& match) { retained = match; return true; }, {{kOnnxDomain, 13, 25}}};
    PatternRewriter rewriter({std::move(rule)}, Options{.match_only = true});
    Statistics statistics;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, statistics));
    EXPECT_EQ(statistics.matches, 1u);
  }
  EXPECT_EQ(retained.valueOf("x"), x);
  EXPECT_FLOAT_EQ(retained.attributeOf("a").f(), 0.25f);
}

TEST(CheckAndBindComponentTests, TransposeCheckerChecksAndBindsWithoutEngine) {
  for (int scenario = 0; scenario < 6; ++scenario) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>(scenario == 4 ? std::nullopt
                                               : std::optional<std::vector<int64_t>>{{scenario == 5 ? -1 : 2, 3, 4}});
    auto* intermediate = b.MakeIntermediate();
    b.AddNode("Transpose", {x}, {intermediate}).AddAttribute("perm", std::vector<int64_t>{1, 0, 2});
    auto& root = b.AddNode("Transpose", {intermediate}, {b.MakeOutput()});
    root.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    bool visited = false;
    ASSERT_STATUS_OK(Pattern("(x)=>{a=Transpose(x);return Transpose(a)}")
                         .VisitMatches(test.graph, root.Index(), [&](Match& match) {
                           visited = true;
                           auto p = match.nodes[0]->GetAttributes().at("perm");
                           auto q = match.nodes[1]->GetAttributes().at("perm");
                           if (scenario == 1) p.set_ints(1, 1);
                           if (scenario == 2) q.set_ints(1, 3);
                           if (scenario == 3) q.mutable_ints()->RemoveLast();
                           match.update("p", p);
                           match.update("q", q);
                           const bool valid = scenario == 0 || scenario == 5;
                           EXPECT_EQ(ComposeTransposePermutations(test.graph, match), valid);
                           if (valid) {
                             const auto& perm = match.attributeOf("composed");
                             EXPECT_EQ(perm.ints(0), 1);
                             EXPECT_EQ(perm.ints(1), 2);
                             EXPECT_EQ(perm.ints(2), 0);
                           } else {
                             EXPECT_FALSE(match.RejectionReason().empty());
                             EXPECT_THROW(match.attributeOf("composed"), OnnxRuntimeException);
                           }
                           return false;
                         }));
    EXPECT_TRUE(visited);
  }
}

TEST(PatternRewriterTests, RunAndTransformerApplyHaveDifferentResolveBoundaries) {
  for (bool use_apply : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
    auto* y = b.MakeIntermediate();
    b.AddNode("Transpose", {x}, {y}).AddAttribute("perm", std::vector<int64_t>{1, 0});
    b.AddNode("Transpose", {y}, {b.MakeOutput()}).AddAttribute("perm", std::vector<int64_t>{1, 0});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    PatternRewriter rewriter({ComposeTransposeRule()});
    if (use_apply) {
      bool modified = false;
      ASSERT_STATUS_OK(rewriter.Apply(test.graph, modified, DefaultLoggingManager().DefaultLogger()));
      EXPECT_TRUE(modified);
      EXPECT_FALSE(test.graph.GraphResolveNeeded());
    } else {
      Statistics statistics;
      ASSERT_STATUS_OK(rewriter.Run(test.graph, statistics));
      EXPECT_EQ(statistics.rewrites, 1u);
      EXPECT_TRUE(test.graph.GraphResolveNeeded());
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    EXPECT_EQ(Count(test.graph, "Transpose"), 1u);
  }
}

TEST(PatternRewriterTests, ComposeTransposeRuleIntegrationAndNumerics) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2, 3, 4}, -1.0f, 1.0f);
      auto* middle = scenario == 1 ? b.MakeOutput() : b.MakeIntermediate();
      b.AddNode("Transpose", {x}, {middle}).AddAttribute("perm", std::vector<int64_t>{1, 0, 2});
      auto& root = b.AddNode("Transpose", {middle}, {b.MakeOutput()});
      if (scenario != 3) root.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      if (scenario == 2) b.AddNode("Identity", {middle}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      const auto original = test.model.ToProto().SerializeAsString();
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({ComposeTransposeRule()}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
      if (scenario != 0) continue;
      EXPECT_EQ(Count(test.graph, "Transpose"), 1u);
      const auto rewritten = test.model.ToProto().SerializeAsString();
      auto execute = [&](const std::string& serialized, std::vector<OrtValue>& outputs) {
        SessionOptions options;
        options.graph_optimization_level = TransformerLevel::Default;
        options.intra_op_param.thread_pool_size = 1;
        InferenceSessionWrapper session{options, GetEnvironment()};
        std::istringstream stream(serialized);
        ORT_RETURN_IF_ERROR(session.Load(stream));
        ORT_RETURN_IF_ERROR(session.Initialize());
        return session.Run(RunOptions{}, b.feeds_, b.output_names_, &outputs);
      };
      std::vector<OrtValue> before, after;
      ASSERT_STATUS_OK(execute(original, before));
      ASSERT_STATUS_OK(execute(rewritten, after));
      ASSERT_EQ(before.size(), 1u);
      ASSERT_EQ(after.size(), 1u);
      const auto& expected = before[0].Get<Tensor>();
      const auto& actual = after[0].Get<Tensor>();
      EXPECT_EQ(actual.Shape(), expected.Shape());
      ASSERT_EQ(actual.Shape().Size(), expected.Shape().Size());
      for (int64_t i = 0; i < expected.Shape().Size(); ++i)
        EXPECT_EQ(actual.Data<float>()[i], expected.Data<float>()[i]);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

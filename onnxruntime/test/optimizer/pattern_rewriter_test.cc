// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#include "core/optimizer/pattern_rewriter.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {
using namespace rewrite;

struct TestGraph {
  explicit TestGraph(const std::unordered_map<std::string, int>& imports = {{kOnnxDomain, 17}, {kMSDomain, 1}})
      : model{"rewrite",
              false,
              ModelMetaData(),
              PathString(),
              IOnnxRuntimeOpSchemaRegistryList(),
              imports,
              {},
              DefaultLoggingManager().DefaultLogger()} {}
  Model model;
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder{graph};
};

void BuildHardSwish(ModelTestBuilder& b, bool mismatch = false, bool intermediate_output = false, float divisor = 6.0f,
                    bool external_consumer = false) {
  auto* x = b.MakeInput<float>({2, 4}, -5.0f, 5.0f);
  auto* y = mismatch ? b.MakeInput<float>({2, 4}, -5.0f, 5.0f) : x;
  auto* add = b.MakeIntermediate();
  auto* clip = intermediate_output ? b.MakeOutput() : b.MakeIntermediate();
  auto* mul = b.MakeIntermediate();
  b.AddNode("Add", {x, b.MakeScalarInitializer<float>(3.0f)}, {add});
  b.AddNode("Clip", {add, b.MakeScalarInitializer<float>(0.0f), b.MakeScalarInitializer<float>(6.0f)}, {clip});
  b.AddNode("Mul", {clip, y}, {mul});
  b.AddNode("Div", {mul, b.MakeScalarInitializer<float>(divisor)}, {b.MakeOutput()});
  if (external_consumer) b.AddNode("Identity", {clip}, {b.MakeOutput()});
}

void BuildRms(ModelTestBuilder& b, NodeArg* x, NodeArg* output, int64_t width, float epsilon = 1e-5f,
              bool wrong_axis = false) {
  auto* square = b.MakeIntermediate();
  auto* mean = b.MakeIntermediate();
  auto* add = b.MakeIntermediate();
  auto* sqrt = b.MakeIntermediate();
  auto* reciprocal = b.MakeIntermediate();
  auto* normalized = b.MakeIntermediate();
  b.AddNode("Pow", {x, b.MakeScalarInitializer<float>(2.0f)}, {square});
  auto& reduce = b.AddNode("ReduceMean", {square}, {mean});
  reduce.AddAttribute("axes", std::vector<int64_t>{wrong_axis ? -2 : -1});
  reduce.AddAttribute("keepdims", int64_t{1});
  b.AddNode("Add", {mean, b.MakeScalarInitializer<float>(epsilon)}, {add});
  b.AddNode("Sqrt", {add}, {sqrt});
  b.AddNode("Reciprocal", {sqrt}, {reciprocal});
  b.AddNode("Mul", {x, reciprocal}, {normalized});
  b.AddNode("Mul", {normalized, b.MakeInitializer<float>({width}, 0.8f, 1.2f)}, {output});
}

size_t Count(const Graph& graph, const std::string& op) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == op;
  return count;
}

Rule IdentityPair(const char* name = "identity", bool accept = true) {
  const auto* pattern = "(x) => a = Identity(x); return Identity(a)";
  auto check_and_bind = [accept](const Graph&, Match&) { return accept; };
  const auto* replacement = "(x) => Identity(x)";
  Rule rule{name, pattern, replacement, check_and_bind, {{kOnnxDomain, 1, 21}}};
  return rule;
}

Rule IdentitySplit() {
  const auto* pattern = "(x) => { first = Identity(x); a, b = Split(first); return a, b; }";
  auto check_and_bind = [](const Graph&, Match& match) {
    const auto& node = *match.nodes.back();
    auto axis = node.GetAttributes().find("axis");
    match.update("axis", axis == node.GetAttributes().end() ? node.Op()->attributes().at("axis").default_value
                                                            : axis->second);
    return true;
  };
  const auto* replacement = "(x, axis) => { a, b = Split(x, axis=axis); return a, b; }";
  Rule rule{"IdentitySplit", pattern, replacement, check_and_bind, {{kOnnxDomain, 13, 17}}};
  return rule;
}

Rule SplitConcat() {
  const auto* pattern = "(x) => a, b = Split(x); return Concat(a, b)";
  auto check_and_bind = [](const Graph&, Match& match) {
    const auto& split_attributes = match.nodes[0]->GetAttributes();
    auto axis = split_attributes.find("axis");
    return (axis == split_attributes.end() || axis->second.i() == 0) &&
           match.nodes[1]->GetAttributes().at("axis").i() == 0;
  };
  const auto* replacement = "(x) => Identity(x)";
  Rule rule{"SplitConcat", pattern, replacement, check_and_bind, {{kOnnxDomain, 13, 17}}};
  return rule;
}

// Multi-output adaptation of ONNXScript's LayerNormBiasFusion; unused statistics may be removed.
Rule LayerNormBias() {
  const auto* pattern = "(x, scale, bias) => y, mean, inv_std = LayerNormalization(x, scale); return Add(y, bias)";
  auto check_and_bind = [](const Graph&, Match& match) {
    const auto* shape = match.valueOf("x")->Shape();
    const auto* scale_shape = match.valueOf("scale")->Shape();
    const auto* bias_shape = match.valueOf("bias")->Shape();
    if (!shape || shape->dim_size() < 1 || !scale_shape || !bias_shape || scale_shape->dim_size() != 1 ||
        bias_shape->dim_size() != 1)
      return false;
    const auto& width = shape->dim(shape->dim_size() - 1);
    const auto& scale_width = scale_shape->dim(0);
    const auto& bias_width = bias_shape->dim(0);
    auto axis = match.nodes[0]->GetAttributes().find("axis");
    if (!width.has_dim_value() || width.dim_value() <= 0 || !scale_width.has_dim_value() ||
        !bias_width.has_dim_value() || width.dim_value() != scale_width.dim_value() ||
        width.dim_value() != bias_width.dim_value() ||
        (axis != match.nodes[0]->GetAttributes().end() && axis->second.i() != -1))
      return false;
    const auto& node = *match.nodes[0];
    for (const char* name : {"axis", "epsilon", "stash_type"}) {
      auto attribute = node.GetAttributes().find(name);
      match.update(name, attribute == node.GetAttributes().end() ? node.Op()->attributes().at(name).default_value
                                                                 : attribute->second);
    }
    return true;
  };
  const auto* replacement = R"(
    (x, scale, bias, axis, epsilon, stash_type) =>
      LayerNormalization(x, scale, bias, axis=axis, epsilon=epsilon, stash_type=stash_type)
  )";
  Rule rule{"LayerNormBias", pattern, replacement, check_and_bind, {{kOnnxDomain, 17, 17}}};
  return rule;
}

// Small-weight eager decoder: B=1, Hq=4, Hkv=2, D=4. Cache length is 3.
void BuildDecoder(ModelTestBuilder& b, size_t layers, int64_t sequence = 1) {
  constexpr int64_t width = 16, kv_width = 8, head = 4, past = 3;
  auto* x = b.MakeInput<float>({1, sequence, width}, -0.5f, 0.5f);
  auto unary = [&](const char* op, NodeArg* input) {
    auto* output = b.MakeIntermediate();
    b.AddNode(op, {input}, {output});
    return output;
  };
  auto binary = [&](const char* op, NodeArg* a, NodeArg* c) {
    auto* output = b.MakeIntermediate();
    b.AddNode(op, {a, c}, {output});
    return output;
  };
  auto reshape = [&](NodeArg* input, std::vector<int64_t> shape) {
    auto* output = b.MakeIntermediate();
    auto rank = static_cast<int64_t>(shape.size());
    b.AddNode("Reshape", {input, b.MakeInitializer<int64_t>({rank}, shape)}, {output});
    return output;
  };
  auto transpose = [&](NodeArg* input, std::vector<int64_t> perm) {
    auto* output = b.MakeIntermediate();
    b.AddNode("Transpose", {input}, {output}).AddAttribute("perm", perm);
    return output;
  };
  auto projection = [&](NodeArg* input, int64_t in, int64_t out) {
    return binary("MatMul", input, b.MakeInitializer<float>({in, out}, -0.08f, 0.08f));
  };
  auto slice = [&](NodeArg* input, int64_t start, int64_t end) {
    auto* output = b.MakeIntermediate();
    b.AddNode("Slice",
              {input, b.MakeInitializer<int64_t>({1}, {start}), b.MakeInitializer<int64_t>({1}, {end}),
               b.MakeInitializer<int64_t>({1}, {3})},
              {output});
    return output;
  };
  auto concat = [&](NodeArg* a, NodeArg* c, int64_t axis, NodeArg* output) {
    b.AddNode("Concat", {a, c}, {output}).AddAttribute("axis", axis);
    return output;
  };
  auto rope = [&](NodeArg* input, int64_t heads) {
    auto* shaped = transpose(reshape(input, {1, sequence, heads, head}), {0, 2, 1, 3});
    auto* first = slice(shaped, 0, head / 2);
    auto* second = slice(shaped, head / 2, head);
    auto* rotated = concat(unary("Neg", second), first, 3, b.MakeIntermediate());
    // Nontrivial position-dependent cache entries, gathered at decode positions.
    std::vector<int64_t> positions;
    for (int64_t s = 0; s < sequence; ++s) positions.push_back(past + s);
    auto* indices = b.MakeInitializer<int64_t>({sequence}, positions);
    std::vector<float> cosine(32 * head), sine(32 * head);
    for (int64_t position = 0; position < 32; ++position) {
      for (int64_t d = 0; d < head; ++d) {
        float angle = static_cast<float>(position) * std::pow(10000.0f, -2.0f * (d % (head / 2)) / head);
        cosine[static_cast<size_t>(position * head + d)] = std::cos(angle);
        sine[static_cast<size_t>(position * head + d)] = std::sin(angle);
      }
    }
    auto* cos = binary("Gather", b.MakeInitializer<float>({32, head}, cosine), indices);
    auto* sin = binary("Gather", b.MakeInitializer<float>({32, head}, sine), indices);
    return binary("Add", binary("Mul", shaped, reshape(cos, {1, 1, sequence, head})),
                  binary("Mul", rotated, reshape(sin, {1, 1, sequence, head})));
  };
  auto repeat_kv = [&](NodeArg* input) {
    auto* expanded = b.MakeIntermediate();
    b.AddNode("Expand",
              {reshape(input, {1, 2, 1, past + sequence, head}),
               b.MakeInitializer<int64_t>({5}, {1, 2, 2, past + sequence, head})},
              {expanded});
    return reshape(expanded, {1, 4, past + sequence, head});
  };
  for (size_t layer = 0; layer < layers; ++layer) {
    auto* normalized = b.MakeIntermediate();
    BuildRms(b, x, normalized, width);
    auto* q = rope(projection(normalized, width, width), 4);
    auto* k = rope(projection(normalized, width, kv_width), 2);
    auto* v = transpose(reshape(projection(normalized, width, kv_width), {1, sequence, 2, head}), {0, 2, 1, 3});
    auto* present_k = concat(b.MakeInput<float>({1, 2, past, head}, -0.1f, 0.1f), k, 2, b.MakeOutput());
    auto* present_v = concat(b.MakeInput<float>({1, 2, past, head}, -0.1f, 0.1f), v, 2, b.MakeOutput());
    auto* scores = binary("MatMul", q, transpose(repeat_kv(present_k), {0, 1, 3, 2}));
    auto* scaled = binary("Div", scores, b.MakeScalarInitializer<float>(2.0f));
    std::vector<float> mask(static_cast<size_t>(sequence * (past + sequence)), 0.0f);
    for (int64_t s = 0; s < sequence; ++s) {
      for (int64_t t = past + s + 1; t < past + sequence; ++t) {
        mask[static_cast<size_t>(s * (past + sequence) + t)] = -10000.0f;
      }
    }
    auto* masked = binary("Add", scaled, b.MakeInitializer<float>({1, 1, sequence, past + sequence}, mask));
    auto* probabilities = b.MakeIntermediate();
    b.AddNode("Softmax", {masked}, {probabilities}).AddAttribute("axis", int64_t{-1});
    auto* attention = binary("MatMul", probabilities, repeat_kv(present_v));
    auto* context = reshape(transpose(attention, {0, 2, 1, 3}), {1, sequence, width});
    auto* residual = binary("Add", x, projection(context, width, width));
    auto* norm2 = b.MakeIntermediate();
    BuildRms(b, residual, norm2, width);
    auto* gate = projection(norm2, width, width * 4);
    auto* silu = binary("Mul", gate, unary("Sigmoid", gate));
    auto* mlp = binary("Mul", silu, projection(norm2, width, width * 4));
    x = binary("Add", residual, projection(mlp, width * 4, width));
  }
  b.AddNode("Identity", {x}, {b.MakeOutput()});
}
}  // namespace

TEST(PatternRewriterTests, HardSwishPositiveAndBoundaries) {
  for (int scenario = 0; scenario < 6; ++scenario) {
    TestGraph test;
    BuildHardSwish(test.builder, scenario == 1, scenario == 2, scenario == 3 ? 7.0f : 6.0f, scenario == 4);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    if (scenario == 5) test.graph.Nodes().begin()->SetExecutionProviderType(kCudaExecutionProvider);
    PatternRewriter rewriter(MakeExampleRules());
    Statistics statistics;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, statistics));
    EXPECT_EQ(statistics.rewrites, scenario == 0 ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    EXPECT_EQ(Count(test.graph, "HardSwish"), scenario == 0 ? 1u : 0u);
  }
}

TEST(PatternRewriterTests, NumericalHardSwish) {
  TransformerTester([](ModelTestBuilder& b) { BuildHardSwish(b); },
                    [](InferenceSessionWrapper& s) {
                      // CPU executes HardSwish's ONNX function body (HardSigmoid + Mul).
                      EXPECT_EQ(Count(s.GetGraph(), "HardSigmoid"), 1u);
                      EXPECT_EQ(Count(s.GetGraph(), "Clip"), 0u);
                    },
                    TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
                    std::make_unique<PatternRewriter>(MakeExampleRules()));
}

TEST(PatternRewriterTests, HardSigmoidAttributeGuard) {
  for (bool correct : {true, false}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2, 4}, -5.0f, 5.0f);
    auto* h = b.MakeIntermediate();
    auto& sigmoid = b.AddNode("HardSigmoid", {x}, {h});
    sigmoid.AddAttribute("alpha", correct ? 1.0f / 6.0f : 0.2f);
    sigmoid.AddAttribute("beta", 0.5f);
    b.AddNode("Mul", {h, x}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, correct ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, NamedValuesAndAttributeGuards) {
  Pattern p("(x) => sigmoid = HardSigmoid(x, alpha=0.25); return Mul(sigmoid, x)");
  auto x = p.NamedValue("x");
  auto first = p.NamedValue("sigmoid");
  EXPECT_EQ(x.name, "x");
  EXPECT_EQ(x.handle, 0);
  EXPECT_EQ(p.NamedValue("sigmoid").handle, first.handle);
  EXPECT_EQ(p.Nodes()[0].outputs[0].name, "sigmoid");
  EXPECT_EQ(p.Nodes()[1].inputs[0].name, "sigmoid");
  EXPECT_THROW(Pattern("(x,x) => return Identity(x)"), OnnxRuntimeException);
  EXPECT_THROW(p.NamedValue("missing"), OnnxRuntimeException);
  EXPECT_THROW(Pattern("(x) => x = Identity(x); return Identity(x)"), OnnxRuntimeException);
  EXPECT_THROW(Pattern("(x) => return Identity(missing)"), OnnxRuntimeException);
  auto integer = AttributeMatcher::Equal("axis", int64_t{1});
  auto string = AttributeMatcher::Equal("mode", std::string("linear"));
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr.set_i(1);
  EXPECT_TRUE(integer.Matches(attr));
  EXPECT_FALSE(string.Matches(attr));
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  attr.set_s("linear");
  EXPECT_TRUE(string.Matches(attr));
  EXPECT_FALSE(integer.Matches(attr));
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      auto* input = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* output = test.builder.MakeIntermediate();
      auto& sigmoid = test.builder.AddNode("HardSigmoid", {input}, {output});
      if (scenario != 3) sigmoid.AddAttribute("alpha", scenario == 1 ? 0.2f : 0.25f);
      sigmoid.AddAttribute("beta", scenario == 2 ? 0.3f : 0.5f);
      test.builder.AddNode("Mul", {output, input}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = p;
      rule.check_and_bind = [](const Graph&, Match& match) {
        const auto& beta = match.nodes[0]->GetAttributes().at("beta");
        return beta.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT && beta.f() > 0.4f;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, scenario == 0 ? 1u : 0u);
    }
  }
}

TEST(PatternRewriterTests, NamedMatchLookupAndReplacementSafety) {
  Match empty;
  EXPECT_THROW(empty.valueOf("x"), OnnxRuntimeException);
  Pattern incomplete("(x) => return Identity(x)");
  empty.pattern = &incomplete;
  EXPECT_THROW(empty.valueOf("x"), OnnxRuntimeException);
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* intermediate = test.builder.MakeIntermediate();
      auto* output = test.builder.MakeOutput();
      test.builder.AddNode("Identity", {x}, {intermediate});
      test.builder.AddNode("Identity", {intermediate}, {output});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto first = IdentityPair(), second = IdentityPair();
      second.pattern = Pattern("(input) => { middle = Identity(input); root = Identity(middle); return root; }");
      first.check_and_bind = [=](const Graph&, Match& m) {
        EXPECT_EQ(m.valueOf("x"), x);
        EXPECT_EQ(m.valueOf("a"), intermediate);
        EXPECT_THROW(m.valueOf("input"), OnnxRuntimeException);
        EXPECT_THROW(m.valueOf(""), OnnxRuntimeException);
        return false;
      };
      second.check_and_bind = [=](const Graph&, Match& m) {
        EXPECT_EQ(m.valueOf("input"), x);
        EXPECT_EQ(m.valueOf("middle"), intermediate);
        EXPECT_THROW(m.valueOf("x"), OnnxRuntimeException);
        return true;
      };
      second.replacement = [scenario](const Graph&, const Match&) {
        if (scenario == 1) return Replacement("(middle) => Identity(middle)");
        if (scenario == 2) return Replacement("(root) => Identity(root)");
        if (scenario == 3) return Replacement("(input) => Identity(input, invalid=0)");
        return Replacement("(input) => Identity(input)");
      };
      PatternRewriter rewriter({first, second}, {.indexed = indexed});
      Statistics stats;
      auto status = rewriter.Run(test.graph, stats);
      EXPECT_EQ(status.IsOK(), scenario == 0);
      EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
      EXPECT_EQ(test.graph.NumberOfNodes(), scenario == 0 ? 1u : 2u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, ParsedReplacementSequenceAndNumerics) {
  Replacement replacement(R"(
    (x) =>
      h = HardSigmoid(x, alpha=0.16666667, beta=0.5)
      return Mul(h, x)
  )");
  const auto* plan = replacement.Plan().get();
  auto copy = replacement;
  EXPECT_EQ(copy.Plan().get(), plan);
  EXPECT_EQ(replacement.Plan()->size(), 2u);
  EXPECT_FLOAT_EQ(replacement.Plan()->front().attributes.at("alpha").f(), 1.0f / 6.0f);
  for (bool indexed : {false, true}) {
    TestGraph test;
    EXPECT_EQ(replacement(test.graph, {}).Plan().get(), plan);
    BuildHardSwish(test.builder);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto outputs = test.graph.GetOutputs();
    auto rule = MakeExampleRules()[0];
    rule.replacement = replacement;
    PatternRewriter rewriter({rule}, {.indexed = indexed});
    Statistics stats;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    EXPECT_EQ(test.graph.NumberOfNodes(), 2u);
    EXPECT_EQ(test.graph.GetOutputs(), outputs);
    ASSERT_STATUS_OK(test.graph.Resolve());
    EXPECT_EQ(Count(test.graph, "HardSigmoid"), 1u);
    EXPECT_EQ(Count(test.graph, "Mul"), 1u);
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
  }
  auto rule = MakeExampleRules()[0];
  rule.replacement = replacement;
  TransformerTester([](ModelTestBuilder& b) { BuildHardSwish(b); },
                    [](InferenceSessionWrapper& session) {
                      EXPECT_EQ(Count(session.GetGraph(), "HardSigmoid"), 1u);
                      EXPECT_EQ(Count(session.GetGraph(), "Mul"), 1u);
                      EXPECT_EQ(Count(session.GetGraph(), "Clip"), 0u);
                    },
                    TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
                    std::make_unique<PatternRewriter>(Rules{rule}));
}

TEST(PatternRewriterTests, ReplacementBranchesAndMultiOutputRoot) {
  auto branched = IdentityPair();
  branched.pattern = Pattern(R"(
    (x) => a = Identity(x); b = Identity(a); c = Identity(b); return Identity(c)
  )");
  branched.replacement = Replacement("(x) => a, b = Split(x, axis=0); return Concat(a, b, axis=0)");
  auto build_branch = [](ModelTestBuilder& b) {
    auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
    auto* upstream = b.MakeIntermediate();
    b.AddNode("Add", {x, x}, {upstream});
    auto* current = upstream;
    for (int i = 0; i < 4; ++i) {
      auto* next = i == 3 ? b.MakeOutput() : b.MakeIntermediate();
      b.AddNode("Identity", {current}, {next});
      current = next;
    }
    b.AddNode("Add", {current, current}, {b.MakeOutput()});
  };
  for (bool indexed : {false, true}) {
    TestGraph test;
    build_branch(test.builder);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    for (auto& node : test.graph.Nodes()) node.SetExecutionProviderType(kCpuExecutionProvider);
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({branched}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    EXPECT_EQ(test.graph.NumberOfNodes(), 4u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    for (const auto& node : test.graph.Nodes()) {
      EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider);
      if (node.OpType() == "Concat") EXPECT_EQ(node.GetOutputEdgesCount(), 2u);
      if (node.OpType() == "Split") {
        EXPECT_EQ(node.OutputDefs().size(), 2u);
        EXPECT_EQ(node.GetOutputEdgesCount(), 2u);
      }
    }
  }
  TransformerTester(
      build_branch,
      [](InferenceSessionWrapper& session) {
        EXPECT_EQ(Count(session.GetGraph(), "Split"), 1u);
        EXPECT_EQ(Count(session.GetGraph(), "Concat"), 1u);
        EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
      },
      TransformerLevel::Default, TransformerLevel::Level1, 17, 0, 0,
      std::make_unique<PatternRewriter>(Rules{branched}));

  auto split = IdentitySplit();
  split.pattern = Pattern("(x) => { a = Identity(x); b = Identity(a); left, right = Split(b); return left, right; }");
  split.replacement = Replacement("(x) => { a=Identity(x); left, right = Split(a, axis=0); return left, right; }");
  auto build_split = [](ModelTestBuilder& b) {
    auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
    auto* a = b.MakeIntermediate();
    auto* intermediate = b.MakeIntermediate();
    b.AddNode("Identity", {x}, {a});
    b.AddNode("Identity", {a}, {intermediate});
    b.AddNode("Split", {intermediate}, {b.MakeOutput(), b.MakeOutput()}).AddAttribute("axis", int64_t{0});
  };
  TransformerTester(
      build_split,
      [](InferenceSessionWrapper& session) {
        EXPECT_EQ(Count(session.GetGraph(), "Identity"), 1u);
        EXPECT_EQ(Count(session.GetGraph(), "Split"), 1u);
      },
      TransformerLevel::Default, TransformerLevel::Level1, 17, 0, 0, std::make_unique<PatternRewriter>(Rules{split}));
}

TEST(PatternRewriterTests, ReplacementParsingCompilationAndRollback) {
  for (auto text : {"(x)=>Unknown(y)", "(x)=>a=Identity(x); Identity(x)", "(x)=>Add(x,1e999)",
                    "(x)=>a=Identity(x);return Identity(a,axis=[0,])", "(x)=>Identity(x);more",
                    "(x)=>Identity(x,axis=0,axis=1)", "(x)=>Identity(x,mode='bad)",
                    "(x)=>a=Identity(a);return Identity(a)", "(x)=>a=Split(x);return Identity(b)",
                    "(x)=>a=Identity(x);return Identity(x)", "(x)=>return = Identity(x)",
                    "(x)=>return a,a = Split(x)"})
    EXPECT_THROW(Replacement{text}, OnnxRuntimeException) << text;
  auto invalid = IdentityPair();
  invalid.replacement = Replacement("(missing)=>Identity(missing)");
  EXPECT_THROW(PatternRewriter({invalid}), OnnxRuntimeException);
  invalid.replacement = Replacement("(a)=>Identity(a)");
  EXPECT_THROW(PatternRewriter({invalid}), OnnxRuntimeException);
  invalid.replacement = Replacement("(x)=>a=Identity(x);return Identity(a)");
  EXPECT_NO_THROW(PatternRewriter({invalid}));
  invalid.replacement = Replacement("(x)=>com.microsoft::FusedMatMul(x,x)");
  EXPECT_THROW(PatternRewriter({invalid}), OnnxRuntimeException);
  EXPECT_NO_THROW(Replacement("(Identity)=>Identity(Identity)"));
  for (bool indexed : {false, true}) {
    for (auto text : {"(x)=>a=Identity(x);return NoSuchReplacementOperator(a)",
                      "(x)=>a=Identity(x);return Softmax(a,axis='wrong')",
                      "(x)=>a=Identity(x);return com.microsoft::HardSwish(a)",
                      "(x)=>a=Add(x,0.0);return NoSuchReplacementOperator(a)",
                      "(x)=>HardSwish(x,invalid=0)"}) {
      SCOPED_TRACE(text);
      TestGraph test;
      BuildHardSwish(test.builder);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto initializer_count = test.graph.GetAllInitializedTensors().size();
      auto rule = MakeExampleRules()[0];
      rule.replacement = [recipe = Replacement(text)](const Graph&, const Match&) { return recipe; };
      Statistics stats;
      ASSERT_FALSE(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats).IsOK());
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_EQ(test.graph.NumberOfNodes(), 4u);
      EXPECT_EQ(test.graph.GetAllInitializedTensors().size(), initializer_count);
      ASSERT_STATUS_OK(test.graph.Resolve());
      rule.replacement = Replacement("(x)=>a=Identity(x);return HardSwish(a)");
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, ReplacementScalarConstantsAndParsedSequence) {
  auto rule = IdentityPair();
  rule.pattern = Pattern("(x)=>a=Identity(x);b=Identity(a);return Identity(b)");
  for (bool integer : {false, true}) {
    rule.replacement = Replacement(integer ? "(x)=>Add(x,0)" : "(x)=>Add(x,0.0)");
    auto* compiled = rule.replacement.target<Replacement>();
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(compiled->Plan()->size(), 1u);
    const auto& scalar = std::get<Replacement::Scalar>(compiled->Plan()->front().inputs[1]);
    EXPECT_EQ(std::holds_alternative<int64_t>(scalar.value), integer);
    auto build = [integer](ModelTestBuilder& b) {
      auto* x = integer ? b.MakeInput<int64_t>({4}, -5, 5) : b.MakeInput<float>({4}, -1.0f, 1.0f);
      auto* current = x;
      for (int i = 0; i < 3; ++i) {
        auto* next = i == 2 ? b.MakeOutput() : b.MakeIntermediate();
        b.AddNode("Identity", {current}, {next});
        current = next;
      }
    };
    TransformerTester(
        build,
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "Add"), 1u);
          EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0, 0, std::make_unique<PatternRewriter>(Rules{rule}));
  }
  rule.replacement = Replacement("(x)=>a=Identity(x);return Identity(a)");
  TestGraph test;
  auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* current = x;
  for (int i = 0; i < 3; ++i) {
    auto* next = i == 2 ? test.builder.MakeOutput() : test.builder.MakeIntermediate();
    test.builder.AddNode("Identity", {current}, {next});
    current = next;
  }
  test.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter({rule}).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 1u);
  EXPECT_EQ(test.graph.NumberOfNodes(), 2u);
  ASSERT_STATUS_OK(test.graph.Resolve());
}

TEST(PatternRewriterTests, ReplacementPlanBindingAndSequenceInvalidation) {
  Replacement shared("(x,y)=>Sub(x,y)");
  for (auto text : {"(x,y)=>a=Identity(x);b=Identity(y);return Sub(a,b)",
                    "(y,x)=>a=Identity(x);b=Identity(y);return Sub(a,b)"}) {
    auto rule = IdentityPair();
    rule.pattern = Pattern(text);
    rule.replacement = shared;
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2}, 1.0f, 2.0f);
          auto* y = b.MakeInput<float>({2}, -2.0f, -1.0f);
          auto* a = b.MakeIntermediate();
          auto* intermediate = b.MakeIntermediate();
          b.AddNode("Identity", {x}, {a});
          b.AddNode("Identity", {y}, {intermediate});
          b.AddNode("Sub", {a, intermediate}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
          EXPECT_EQ(Count(session.GetGraph(), "Sub"), 1u);
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0, 0,
        std::make_unique<PatternRewriter>(Rules{rule}));
  }
  for (bool indexed : {false, true}) {
    auto first = IdentityPair("first", false);
    first.pattern = Pattern("(first)=>a=Identity(first);b=Identity(a);c=Identity(b);return Identity(c)");
    first.replacement = Replacement("(first)=>Identity(first)");
    auto second = first;
    second.pattern = Pattern("(second)=>a=Identity(second);b=Identity(a);c=Identity(b);return Identity(c)");
    second.check_and_bind = [](const Graph&, Match&) { return true; };
    second.replacement = Replacement("(second)=>a=Identity(second);return Identity(a)");
    auto followup = IdentityPair();
    followup.check_and_bind = [](const Graph&, Match& m) { return m.nodes.back()->Name().find("rewrite") == 0; };
    PatternRewriter rewriter({first, second, followup}, {.indexed = indexed});
    for (int repeat = 0; repeat < 2; ++repeat) {
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* current = x;
      for (int i = 0; i < 4; ++i) {
        auto* next = i == 3 ? test.builder.MakeOutput() : test.builder.MakeIntermediate();
        test.builder.AddNode("Identity", {current}, {next});
        current = next;
      }
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 2u);
      EXPECT_EQ(test.graph.NumberOfNodes(), 1u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    auto invalid = second;
    invalid.replacement = [](const Graph&, const Match&) {
      return Replacement("(second)=>a=Identity(second);b=Identity(a);c=Identity(b);return Identity(c,invalid=1)");
    };
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* current = x;
    for (int i = 0; i < 4; ++i) {
      auto* next = i == 3 ? test.builder.MakeOutput() : test.builder.MakeIntermediate();
      test.builder.AddNode("Identity", {current}, {next});
      current = next;
    }
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    EXPECT_FALSE(PatternRewriter({invalid}, {.indexed = indexed}).Run(test.graph, stats).IsOK());
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(test.graph.NumberOfNodes(), 4u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, ReplacementExtraAttributeBindings) {
  static_assert(std::is_invocable_v<decltype(&Match::update), Match&, std::string_view,
                                    ONNX_NAMESPACE::AttributeProto>);
  static_assert(!std::is_invocable_v<decltype(&Match::update), const Match&, std::string_view,
                                     ONNX_NAMESPACE::AttributeProto>);
  static_assert(!std::is_invocable_v<decltype(&Match::ClearUpdates), const Match&>);
  static_assert(!std::is_invocable_v<decltype(&Match::Reject), const Match&, std::string_view>);
  static_assert(!std::is_invocable_v<Rule::CheckAndBind, const Graph&, const Match&>);
  static_assert(std::is_invocable_v<Rule::ReplacementFunction, const Graph&, const Match&>);
  auto make_attribute = [](ONNX_NAMESPACE::AttributeProto_AttributeType type) {
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name("ignored_source_name");
    attr.set_type(type);
    return attr;
  };
  Match empty;
  EXPECT_THROW(empty.attributeOf("missing"), OnnxRuntimeException);
  auto alpha = make_attribute(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  alpha.set_f(0.25f);
  EXPECT_THROW(empty.update("", alpha), OnnxRuntimeException);
  EXPECT_THROW(empty.update("alpha", {}), OnnxRuntimeException);
  auto reference = alpha;
  reference.set_ref_attr_name("other");
  EXPECT_THROW(empty.update("alpha", reference), OnnxRuntimeException);
  empty.update("alpha", alpha);
  EXPECT_FLOAT_EQ(empty.attributeOf("alpha").f(), 0.25f);
  alpha.set_f(0.5f);
  empty.update("alpha", alpha);
  EXPECT_FLOAT_EQ(empty.attributeOf("alpha").f(), 0.5f);
  auto text_attribute = make_attribute(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  text_attribute.set_s("none");
  empty.update("approximate", text_attribute);
  EXPECT_EQ(empty.attributeOf("approximate").s(), "none");
  empty.ClearUpdates();
  EXPECT_THROW(empty.attributeOf("alpha"), OnnxRuntimeException);

  for (auto text : {"(x)=>HardSigmoid(x,alpha=unknown)",
                    "(x,a)=>HardSigmoid(x,alpha=x)",
                    "(x)=>a=Identity(x);return HardSigmoid(a,alpha=a)",
                    "(x,a)=>HardSigmoid(x,alpha=a,alpha=0.5)"})
    EXPECT_THROW(Replacement{text}, OnnxRuntimeException);
  EXPECT_NO_THROW(Pattern{"(x,a)=>return HardSigmoid(x,alpha=a)"});
  Replacement dotted("(x,alpha)=>com.microsoft.FusedMatMul(x,x,alpha=alpha)");
  EXPECT_EQ(dotted.Plan()->front().domain, kMSDomain);
  EXPECT_EQ(dotted.Plan()->front().op, "FusedMatMul");
  EXPECT_EQ(dotted.Plan()->front().attribute_bindings.at("alpha"), "alpha");

  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 5; ++scenario) {
      TestGraph test;
      BuildHardSwish(test.builder);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto first = MakeExampleRules()[0];
      auto second = first;
      // Rejected rules cannot leak prepared bindings into the next rule.
      first.check_and_bind = [alpha](const Graph&, Match& m) {
        m.update("alpha", alpha);
        return false;
      };
      second.replacement = Replacement(scenario == 4
                                           ? "(x,alpha,beta)=>h=HardSigmoid(x,alpha=alpha,beta=beta,extra=0.0);return Mul(h,x)"
                                           : "(x,alpha,beta)=>h=HardSigmoid(x,alpha=alpha,beta=beta);return Mul(h,x)");
      second.check_and_bind = [scenario, make_attribute](const Graph&, Match& m) {
        auto beta = make_attribute(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
        beta.set_f(0.5f);
        m.update("beta", beta);
        if (scenario == 1) return true;  // alpha is missing; must not leak from the rejected rule.
        if (scenario == 2) m.update("x", beta);
        auto alpha_value = make_attribute(scenario == 3 ? ONNX_NAMESPACE::AttributeProto_AttributeType_INT
                                                        : ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
        alpha_value.set_f(1.0f / 6.0f);
        m.update("alpha", alpha_value);
        return true;
      };
      PatternRewriter rewriter({first, second}, {.indexed = indexed});
      Statistics stats;
      if (scenario == 1 || scenario == 2) {
        EXPECT_THROW((void)rewriter.Run(test.graph, stats), OnnxRuntimeException);
      } else {
        EXPECT_EQ(rewriter.Run(test.graph, stats).IsOK(), scenario == 0);
      }
      EXPECT_EQ(test.graph.NumberOfNodes(), scenario == 0 ? 2u : 4u);
      ASSERT_STATUS_OK(test.graph.Resolve());
      if (scenario == 0) {
        for (const auto& node : test.graph.Nodes()) {
          if (node.OpType() != "HardSigmoid") continue;
          EXPECT_EQ(node.GetAttributes().at("alpha").name(), "alpha");
          EXPECT_FLOAT_EQ(node.GetAttributes().at("alpha").f(), 1.0f / 6.0f);
          EXPECT_FLOAT_EQ(node.GetAttributes().at("beta").f(), 0.5f);
        }
      }
    }
  }
}

TEST(PatternRewriterTests, ReplacementExtraBindingsVaryPerMatchAndAttributeTypes) {
  for (bool indexed : {false, true}) {
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x)=>a=Identity(x);b=Identity(a);return Identity(b)");
    rule.replacement = Replacement("(x,axis)=>Softmax(x,axis=axis)");
    int calls = 0;
    rule.check_and_bind = [&calls](const Graph&, Match& m) {
      ONNX_NAMESPACE::AttributeProto axis;
      axis.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
      axis.set_i((calls++ / 2) % 2);
      m.update("axis", axis);
      return true;
    };
    PatternRewriter matcher({rule}, {.match_only = true, .indexed = indexed});
    PatternRewriter rewriter({rule}, {.indexed = indexed});
    for (int repeat = 0; repeat < 2; ++repeat) {
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({2, 2}, -1.0f, 1.0f);
      auto* a = test.builder.MakeIntermediate();
      auto* b = test.builder.MakeIntermediate();
      test.builder.AddNode("Identity", {x}, {a});
      test.builder.AddNode("Identity", {a}, {b});
      test.builder.AddNode("Identity", {b}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(matcher.Run(test.graph, stats));
      EXPECT_EQ(calls, 2 * repeat + 1);
      EXPECT_EQ(test.graph.NumberOfNodes(), 3u);
      EXPECT_EQ(stats.matches, 1u);
      ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
      EXPECT_EQ(calls, 2 * repeat + 2);
      EXPECT_EQ(stats.rewrites, 1u);
      EXPECT_EQ(test.graph.Nodes().begin()->GetAttributes().at("axis").i(), repeat);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }

  auto rule = IdentityPair();
  rule.pattern = Pattern("(x)=>a=Identity(x);b=Identity(a);return Identity(b)");
  rule.replacement = Replacement("(x,perm)=>Transpose(x,perm=perm)");
  auto check_and_bind = [](const Graph&, Match& m) {
    ONNX_NAMESPACE::AttributeProto perm;
    perm.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    perm.add_ints(0);
    perm.add_ints(1);
    m.update("perm", perm);
    return true;
  };
  rule.check_and_bind = check_and_bind;
  TransformerTester(
      [](ModelTestBuilder& b) {
        auto* x = b.MakeInput<float>({2, 2}, -1.0f, 1.0f);
        auto* a = b.MakeIntermediate();
        auto* intermediate = b.MakeIntermediate();
        b.AddNode("Identity", {x}, {a});
        b.AddNode("Identity", {a}, {intermediate});
        b.AddNode("Identity", {intermediate}, {b.MakeOutput()});
      },
      [](InferenceSessionWrapper& session) {
        EXPECT_EQ(Count(session.GetGraph(), "Transpose"), 1u);
        EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
      },
      TransformerLevel::Default, TransformerLevel::Level1, 17, 0, 0,
      std::make_unique<PatternRewriter>(Rules{rule}));
}

TEST(PatternRewriterTests, CheckAndBindTensorAttributeToTextReplacement) {
  for (bool indexed : {false, true}) {
    auto check_and_bind = [](const Graph&, Match& match) {
      ONNX_NAMESPACE::AttributeProto zero;
      zero.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
      auto* tensor = zero.mutable_t();
      tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      tensor->add_dims(1);
      tensor->add_float_data(0.0f);
      match.update("zero", std::move(zero));
      return true;
    };
    Rule rule{"prepared_tensor",
              "(x:float[2,3])=>{a=Identity(x);return Identity(a);}",
              "(x,zero)=>{z=ConstantOfShape([2,3],value=zero);return Add(x,z);}",
              check_and_bind,
              {{kOnnxDomain, 13, 17}}};
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
          auto* a = b.MakeIntermediate();
          b.AddNode("Identity", {x}, {a});
          b.AddNode("Identity", {a}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& session) {
          const auto& graph = session.GetGraph();
          EXPECT_EQ(Count(graph, "Identity"), 0u);
          EXPECT_EQ(Count(graph, "Add"), 1u);
          EXPECT_EQ(Count(graph, "ConstantOfShape"), 1u);
          for (const auto& node : graph.Nodes()) {
            if (node.OpType() == "ConstantOfShape") {
              const auto& tensor = node.GetAttributes().at("value").t();
              EXPECT_EQ(tensor.data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
              ASSERT_EQ(tensor.dims_size(), 1);
              EXPECT_EQ(tensor.dims(0), 1);
              ASSERT_EQ(tensor.float_data_size(), 1);
              EXPECT_FLOAT_EQ(tensor.float_data(0), 0.0f);
            }
          }
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
        std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, CheckAndBindRejectedStateIsolation) {
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      for (int i = 0; i < 2; ++i) {
        auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
        auto* a = b.MakeIntermediate();
        b.AddNode("Identity", {x}, {a});
        b.AddNode("Identity", {a}, {b.MakeOutput()});
      }
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rejected = IdentityPair("rejected");
      rejected.check_and_bind = [](const Graph&, Match& match) {
        ONNX_NAMESPACE::AttributeProto value;
        value.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
        value.set_f(0.25f);
        match.update("prepared", value);
        EXPECT_TRUE(match.BindAttribute("captured", value));
        EXPECT_TRUE(match.BindDimension("scratch_dimension", int64_t{2}));
        EXPECT_TRUE(match.BindElementType("scratch_type", ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
        return match.Reject("rejected after preparation");
      };
      auto accepted = IdentityPair("accepted");
      size_t calls = 0;
      accepted.check_and_bind = [&calls](const Graph&, Match& match) {
        EXPECT_THROW(match.attributeOf("prepared"), OnnxRuntimeException);
        EXPECT_THROW(match.attributeOf("captured"), OnnxRuntimeException);
        EXPECT_THROW(match.dimensionOf("scratch_dimension"), OnnxRuntimeException);
        EXPECT_THROW(match.elementTypeOf("scratch_type"), OnnxRuntimeException);
        EXPECT_TRUE(match.RejectionReason().empty());
        ++calls;
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rejected, accepted}, {.match_only = match_only, .indexed = indexed})
                           .Run(test.graph, stats));
      EXPECT_EQ(calls, 2u);
      EXPECT_EQ(stats.check_and_bind_calls, 4u);
      EXPECT_EQ(stats.matches, 2u);
      EXPECT_EQ(stats.rewrites, match_only ? 0u : 2u);
      EXPECT_EQ(test.graph.NumberOfNodes(), match_only ? 4 : 2);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, ParsedHardSwishConstantsAndExecution) {
  constexpr auto text = R"(
    (x, low, high, divisor) =>
      add = Add(x, 3.0)
      clip = Clip(add, low, high)
      mul = Mul(clip, x)
      return Div(mul, divisor)
  )";
  Pattern parsed(text);
  EXPECT_EQ(parsed.NamedValue("x").name, "x");
  EXPECT_EQ(parsed.NamedValue("low").handle, 1);
  EXPECT_EQ(parsed.NamedValue("divisor").handle, 3);
  EXPECT_EQ(parsed.Nodes()[0].outputs[0].name, "add");
  EXPECT_EQ(parsed.Constants().size(), 1u);
  auto semantic_guard = [type_guard = MakeExampleRules()[0].check_and_bind](const Graph& graph, Match& match) {
    if (!type_guard(graph, match)) return false;
    for (const auto& [name, expected] :
         {std::pair{"low", 0.0f}, std::pair{"high", 6.0f}, std::pair{"divisor", 6.0f}}) {
      const auto* tensor = graph_utils::GetConstantInitializer(graph, match.valueOf(name)->Name());
      if (!tensor || tensor->dims_size() != 0 || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
          tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL)
        return false;
      Initializer initializer(graph, *tensor, graph.ModelPath());
      if (initializer.size() != 1 || *initializer.data<float>() != expected) return false;
    }
    return true;
  };
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      BuildHardSwish(test.builder);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      NodeArg* bias = nullptr;
      for (auto& node : test.graph.Nodes()) {
        if (node.OpType() == "Add") bias = node.MutableInputDefs()[1];
      }
      ASSERT_NE(bias, nullptr);
      if (scenario == 1) {
        const auto* tensor = test.graph.GetInitializer(bias->Name(), false);
        ASSERT_NE(tensor, nullptr);
        auto updated = *tensor;
        updated.clear_raw_data();
        updated.clear_float_data();
        updated.add_float_data(4.0f);
        test.graph.RemoveInitializedTensor(bias->Name());
        test.graph.AddInitializedTensor(updated);
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
      if (scenario == 2) {
        auto inputs = test.graph.GetInputsIncludingInitializers();
        inputs.push_back(bias);
        test.graph.SetInputs(inputs);
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
      if (scenario == 3) {
        for (auto& node : test.graph.Nodes()) {
          if (node.OpType() == "Mul") node.MutableInputDefs()[1] = node.MutableInputDefs()[0];
        }
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
      auto rule = MakeExampleRules()[0];
      rule.pattern = parsed;
      rule.check_and_bind = semantic_guard;
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
  auto rule = MakeExampleRules()[0];
  rule.pattern = parsed;
  rule.check_and_bind = semantic_guard;
  TransformerTester([](ModelTestBuilder& b) { BuildHardSwish(b); },
                    [](InferenceSessionWrapper& session) {
                      EXPECT_EQ(Count(session.GetGraph(), "HardSigmoid"), 1u);
                      EXPECT_EQ(Count(session.GetGraph(), "Clip"), 0u);
                    },
                    TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
                    std::make_unique<PatternRewriter>(Rules{rule}));
}

TEST(PatternRewriterTests, ExampleFixedConstantsArePatternLiterals) {
  const auto rule = MakeExampleRules()[0];
  EXPECT_EQ(rule.pattern.Constants().size(), 4u);
  EXPECT_EQ(rule.pattern.NamedValue("x").handle, 0);
  for (auto name : {"bias", "low", "high", "divisor"})
    EXPECT_THROW(rule.pattern.NamedValue(name), OnnxRuntimeException);
  for (bool indexed : {false, true}) {
    for (auto op : {"Add", "Clip", "Div"}) {
      for (int slot = op == std::string_view("Clip") ? 1 : 0;
           slot < (op == std::string_view("Clip") ? 3 : 1); ++slot) {
        for (int scenario = 0; scenario < 4; ++scenario) {
          TestGraph test;
          BuildHardSwish(test.builder);
          test.builder.SetGraphOutputs();
          ASSERT_STATUS_OK(test.graph.Resolve());
          const NodeArg* argument = nullptr;
          for (const auto& node : test.graph.Nodes()) {
            if (node.OpType() == op) argument = node.InputDefs()[op == std::string_view("Clip") ? slot : 1];
          }
          ASSERT_NE(argument, nullptr);
          const auto* original = test.graph.GetInitializer(argument->Name(), false);
          ASSERT_NE(original, nullptr);
          auto tensor = *original;
          if (scenario == 0) {
            tensor.clear_raw_data();
            tensor.clear_float_data();
            tensor.add_float_data(9.0f);
          }
          if (scenario == 1) {
            tensor.add_dims(1);
            test.graph.GetNodeArg(argument->Name())->ClearShape();
          }
          if (scenario == 2) {
            tensor.clear_raw_data();
            tensor.clear_float_data();
            tensor.add_float_data(std::numeric_limits<float>::infinity());
          }
          if (scenario != 3) {
            test.graph.RemoveInitializedTensor(argument->Name());
            test.graph.AddInitializedTensor(tensor);
          } else {
            auto inputs = test.graph.GetInputsIncludingInitializers();
            inputs.push_back(argument);
            test.graph.SetInputs(inputs);
          }
          ASSERT_STATUS_OK(test.graph.Resolve());
          Statistics stats;
          ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
          EXPECT_EQ(stats.rewrites, 0u) << op << " input " << slot << " scenario " << scenario;
          EXPECT_EQ(test.graph.NumberOfNodes(), 4u);
        }
      }
    }
  }
#ifndef DISABLE_CONTRIB_OPS
  const auto rms = MakeExampleRules().back();
  ASSERT_EQ(rms.pattern.Constants().size(), 1u);
  EXPECT_FLOAT_EQ(std::get<float>(rms.pattern.Constants().begin()->second), 2.0f);
  EXPECT_THROW(rms.pattern.NamedValue("exponent"), OnnxRuntimeException);
  EXPECT_NO_THROW(rms.pattern.NamedValue("epsilon"));
#endif
}

TEST(PatternRewriterTests, ParsedAttributesMultiOutputsAndDomains) {
  for (bool indexed : {false, true}) {
    Pattern split("(x) => a, b = Split(x, axis = 0); return Concat(a, b, axis = 0)");
    auto rule = SplitConcat();
    rule.pattern = split;
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({4}, -1.0f, 1.0f);
    auto* a = test.builder.MakeIntermediate();
    auto* b = test.builder.MakeIntermediate();
    test.builder.AddNode("Split", {x}, {a, b}).AddAttribute("axis", int64_t{0});
    test.builder.AddNode("Concat", {a, b}, {test.builder.MakeOutput()}).AddAttribute("axis", int64_t{0});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
  Pattern p("(x, y) => a = com.microsoft::FusedMatMul(x, y, alpha=+1.0e-1); return Identity(a)");
  EXPECT_EQ(p.Nodes()[0].op.domain, kMSDomain);
  EXPECT_EQ(p.Nodes()[0].attributes.size(), 1u);
  p = Pattern("(x) => a = Identity(x, mode='linear'); return Identity(a)");
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  attr.set_s("linear");
  EXPECT_TRUE(p.Nodes()[0].attributes[0].Matches(attr));
  Pattern original("(x) => return Identity(x)");
  for (auto text :
       {"(x,x)=>return Identity(x)", "(x)=>return Identity(y)", "(x)=>a=Identity(x)",
        "(x)=>a=Identity(x);return Identity(a);extra", "(x)=>a=Identity(x);return Add(a, 1e999)",
        "(x)=>a=Identity(x);return Add(a, 9223372036854775808)", "(x)=>a=Identity(x);return Add(a, 1.2.3)",
        "(x)=>a=Identity(x);return Identity(a, mode='bad)", "(x)=>a=Identity(x);return Identity(a, axis=1, axis=2)",
        "(x)=>a=Identity(x);return Add(axis=1, a)", "(x)=>a=Identity(x);return Identity(a,) ",
        "(x)=>a=Identity(x);return Identity(a, axis=[1,])", "(x,y)=>return Identity(x)",
        "(x)=>a=Identity(x);return Identity(x)", "(x)=>x=Identity(x);return Identity(x)",
        "(x)=>a=Identity(x);return Identity(a) ''", "(x)=>'return' Identity(x)", "(x)=>return Identity(x ')'"}) {
    try {
      original = Pattern(text);
      FAIL() << "Expected malformed pattern to throw: " << text;
    } catch (const OnnxRuntimeException& error) {
      EXPECT_NE(std::string(error.what()).find("byte"), std::string::npos);
    }
    EXPECT_EQ(original.Nodes().size(), 1u);
    EXPECT_EQ(original.NamedValue("x").handle, 0);
  }
  p = Pattern("(x)=>a=Identity(x, mode='');return Identity(a)");
  attr.set_s("");
  EXPECT_TRUE(p.Nodes()[0].attributes[0].Matches(attr));
  std::string excessive = "(x)=>";
  for (int i = 0; i < 65; ++i)
    excessive += "v" + std::to_string(i) + "=Identity(" + (i == 0 ? "x" : "v" + std::to_string(i - 1)) + ");";
  excessive += "return Identity(v64)";
  EXPECT_THROW(original = Pattern(excessive), OnnxRuntimeException);
}

TEST(PatternRewriterTests, LiteralTypeAndRankMatching) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 3; ++scenario) {
      TestGraph test;
      auto* x = test.builder.MakeInput<int64_t>({2}, 0, 5);
      auto* intermediate = test.builder.MakeIntermediate();
      auto* literal = scenario == 1 ? test.builder.MakeInitializer<int64_t>({1}, {3})
                                    : test.builder.MakeScalarInitializer<int64_t>(scenario == 2 ? 4 : 3);
      test.builder.AddNode("Add", {x, literal}, {intermediate});
      test.builder.AddNode("Identity", {intermediate}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x)=>a=Add(x,3);return Identity(a)");
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, scenario == 0 ? 1u : 0u);
      rule.pattern = Pattern("(x)=>a=Add(x,3.0);return Identity(a)");
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, 0u);
    }
  }
  Pattern p("(x)=>a=Add(x,3);return Add(a,3.0)");
  auto integer = p.Nodes()[0].inputs[1];
  auto floating = p.Nodes()[1].inputs[1];
  EXPECT_TRUE(std::holds_alternative<int64_t>(p.Constants().at(integer)));
  EXPECT_TRUE(std::holds_alternative<float>(p.Constants().at(floating)));
}

TEST(PatternRewriterTests, VectorLiteralParsingAndErrors) {
  Pattern pattern("(x)=>{a=Add(x,[1,2]);return Identity(a);}");
  const auto& integers = std::get<InlinedVector<int64_t>>(pattern.Constants().begin()->second);
  ASSERT_EQ(integers.size(), 2u);
  EXPECT_EQ(integers[0], 1);
  EXPECT_EQ(integers[1], 2);
  Replacement plan("(x)=>Add(x,[1.0,2e0])");
  const auto& values = std::get<InlinedVector<float>>(
      std::get<Replacement::Scalar>(plan.Plan()->front().inputs[1]).value);
  ASSERT_EQ(values.size(), 2u);
  EXPECT_FLOAT_EQ(values[1], 2.0f);
  for (auto text : {"(x)=>return Add(x,[1,2.0])", "(x)=>return Add(x,[[1]])",
                    "(x)=>return Add(x,[1,])", "(x)=>return Add(x,[1)", "(x)=>return Add(x,[1e999])",
                    "(x)=>return Add(x,[9223372036854775808])", "(x)=>return Add(x,[1 2])",
                    "(x)=>return Identity(x,axes=[1,2.0])", "(x)=>return Identity(x,axes=[1,])"}) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)Pattern{text}, OnnxRuntimeException);
    EXPECT_THROW((void)Replacement{text}, OnnxRuntimeException);
  }
  Pattern empty("(x)=>{a=Add(x,[]);return Identity(a);}");
  EXPECT_TRUE(std::get<InlinedVector<int64_t>>(empty.Constants().begin()->second).empty());
}

TEST(PatternRewriterTests, VectorLiteralMatchingAndNumerics) {
  for (bool indexed : {false, true}) {
    SCOPED_TRACE(indexed);
    for (int scenario = 0; scenario < 5; ++scenario) {
      SCOPED_TRACE(scenario);
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<int64_t>({2}, 0, 5);
      NodeArg* literal = scenario == 1   ? b.MakeInitializer<int64_t>({1, 2}, {1, 2})
                         : scenario == 2 ? b.MakeInitializer<float>({2}, {1.0f, 2.0f})
                                         : b.MakeInitializer<int64_t>({2}, {1, scenario == 3 ? 3 : 2});
      if (scenario == 2) {
        x = b.MakeInput<float>({2}, 0.0f, 5.0f);
      }
      auto* a = b.MakeIntermediate();
      b.AddNode("Add", {x, literal}, {a});
      b.AddNode("Identity", {a}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      if (scenario == 4) {
        auto inputs = test.graph.GetInputsIncludingInitializers();
        inputs.push_back(literal);
        test.graph.SetInputs(inputs);
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x)=>{a=Add(x,[1,2]);return Identity(a);}");
      rule.replacement = Replacement("(x)=>Add(x,[1,2])");
      auto rejected = rule;
      rejected.pattern = Pattern("(x)=>{a=Add(x,[4,5]);return Identity(a);}");
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rejected, rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    for (bool floating : {false, true}) {
      auto rule = IdentityPair();
      rule.replacement = Replacement(floating ? "(x)=>Add(x,[0.0,0.0])" : "(x)=>Add(x,[0,0])");
      TransformerTester(
          [floating](ModelTestBuilder& b) {
            auto* x = floating ? b.MakeInput<float>({2}, -1.0f, 1.0f) : b.MakeInput<int64_t>({2}, -5, 5);
            auto* a = b.MakeIntermediate();
            b.AddNode("Identity", {x}, {a});
            b.AddNode("Identity", {a}, {b.MakeOutput()});
          },
          [](InferenceSessionWrapper& session) { EXPECT_EQ(Count(session.GetGraph(), "Add"), 1u); },
          TransformerLevel::Default, TransformerLevel::Level1, 17, 0, 0,
          std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
    }
    for (auto text : {"(x)=>{a=Add(x,[0.0,0.0]);return Identity(a);}",
                      "(x)=>{a=Add(x,[]);return Identity(a);}"}) {
      TestGraph test;
      const bool empty = std::string_view(text).find("[]") != std::string_view::npos;
      auto* x = empty ? test.builder.MakeInput<int64_t>({0}, std::vector<int64_t>{})
                      : test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* literal = empty ? test.builder.MakeInitializer<int64_t>({0}, std::vector<int64_t>{})
                            : test.builder.MakeInitializer<float>({2}, {0.0f, 0.0f});
      auto* a = test.builder.MakeIntermediate();
      test.builder.AddNode("Add", {x, literal}, {a});
      test.builder.AddNode("Identity", {a}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern(text);
      rule.replacement = Replacement(empty ? "(x)=>Add(x,[])" : "(x)=>Add(x,[0.0,0.0])");
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    for (auto text : {"(x)=>{a=Add(x,[0,0]);return NoSuchOperator(a);}",
                      "(x)=>{a=Add(x,[0.0,0.0]);return NoSuchOperator(a);}"}) {
      TestGraph test;
      BuildHardSwish(test.builder);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto count = test.graph.GetAllInitializedTensors().size();
      auto rule = MakeExampleRules()[0];
      rule.replacement = Replacement(text);
      Statistics stats;
      EXPECT_FALSE(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats).IsOK());
      EXPECT_EQ(test.graph.NumberOfNodes(), 4u);
      EXPECT_EQ(test.graph.GetAllInitializedTensors().size(), count);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, ListAttributesAndDottedOperators) {
  for (auto text : {"(x,y)=>{a=com.microsoft.FusedMatMul(x,y);return Identity(a);}",
                    "(x,y)=>{a=com.microsoft::FusedMatMul(x,y);return Identity(a);}"}) {
    Pattern pattern(text);
    EXPECT_EQ(pattern.Nodes()[0].op.domain, kMSDomain);
    EXPECT_EQ(pattern.Nodes()[0].op.name, "FusedMatMul");
  }
  for (auto text : {"(x)=>return ai.onnx.Identity(x)", "(x)=>return ai.onnx::Identity(x)"}) {
    EXPECT_EQ(Pattern(text).Nodes()[0].op.domain, kOnnxDomain);
    EXPECT_EQ(Replacement(text).Plan()->front().domain, kOnnxDomain);
  }
  for (bool indexed : {false, true}) {
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x)=>{a=ai.onnx.Transpose(x,perm=[1,0]);return Identity(a);}");
    rule.replacement = Replacement("(x)=>ai.onnx.Transpose(x,perm=[1,0])");
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
          auto* a = b.MakeIntermediate();
          b.AddNode("Transpose", {x}, {a}).AddAttribute("perm", std::vector<int64_t>{1, 0});
          b.AddNode("Identity", {a}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& session) { EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u); },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0, 0,
        std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
    for (auto text : {"(x)=>{a=Transpose(x,perm=[0,1]);return Identity(a);}",
                      "(x)=>{a=Transpose(x,perm=[1]);return Identity(a);}",
                      "(x)=>{a=Transpose(x,perm=[1.0,0.0]);return Identity(a);}"}) {
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto* a = test.builder.MakeIntermediate();
      test.builder.AddNode("Transpose", {x}, {a}).AddAttribute("perm", std::vector<int64_t>{1, 0});
      test.builder.AddNode("Identity", {a}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      rule.pattern = Pattern(text);
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, 0u);
    }
  }
  Pattern floats("(x)=>{a=Identity(x,coefficients=[1.0,2.0]);return Identity(a);}");
  ONNX_NAMESPACE::AttributeProto attribute;
  attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
  attribute.add_floats(1.0f);
  attribute.add_floats(2.0f);
  EXPECT_TRUE(floats.Nodes()[0].attributes[0].Matches(attribute));
  attribute.set_floats(1, 3.0f);
  EXPECT_FALSE(floats.Nodes()[0].attributes[0].Matches(attribute));
  Replacement float_plan("(x)=>Identity(x,coefficients=[1.0,2.0])");
  EXPECT_EQ(float_plan.Plan()->front().attributes.at("coefficients").type(),
            ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
}

TEST(PatternRewriterTests, AttributeAndConstantPlansCannotAlias) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* a = test.builder.MakeIntermediate();
    test.builder.AddNode("HardSigmoid", {x}, {a}).AddAttribute("alpha", 0.25f);
    test.builder.AddNode("Mul", {a, x}, {test.builder.MakeOutput()});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto rejected = IdentityPair(), accepted = IdentityPair();
    rejected.pattern = Pattern("(x)=>a=HardSigmoid(x, alpha=0.2);return Mul(a,x)");
    accepted.pattern = Pattern("(x)=>a=HardSigmoid(x, alpha=0.25);return Mul(a,x)");
    Statistics stats;
    ASSERT_STATUS_OK(
        PatternRewriter({rejected, accepted}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
    Pattern first("(x)=>a=Add(x,3.0);return Identity(a)");
    Pattern second("(x)=>a=Add(x,4.0);return Identity(a)");
    TestGraph constants;
    auto* input = constants.builder.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* intermediate = constants.builder.MakeIntermediate();
    constants.builder.AddNode("Add", {input, constants.builder.MakeScalarInitializer<float>(4.0f)}, {intermediate});
    constants.builder.AddNode("Identity", {intermediate}, {constants.builder.MakeOutput()});
    constants.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(constants.graph.Resolve());
    rejected.pattern = first;
    accepted.pattern = second;
    ASSERT_STATUS_OK(
        PatternRewriter({rejected, accepted}, {.match_only = true, .indexed = indexed}).Run(constants.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
  }
}

TEST(PatternRewriterTests, GenericExecutionProviderPolicy) {
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      for (int scenario = 0; scenario < 5; ++scenario) {
        TestGraph test;
        auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
        auto* first = test.builder.MakeIntermediate();
        auto& a = test.builder.AddNode("Identity", {x}, {first});
        auto& b = test.builder.AddNode("Identity", {first}, {test.builder.MakeOutput()});
        test.builder.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        if (scenario == 1) {
          a.SetExecutionProviderType(kCpuExecutionProvider);
          b.SetExecutionProviderType(kCpuExecutionProvider);
        }
        if (scenario >= 2) {
          a.SetExecutionProviderType(scenario == 4 ? kCpuExecutionProvider : kCudaExecutionProvider);
          b.SetExecutionProviderType(kCudaExecutionProvider);
        }
        Options options{.match_only = match_only, .indexed = indexed};
        if (scenario >= 3) options.compatible_execution_providers.insert(kCudaExecutionProvider);
        auto rule = IdentityPair();
        int guards = 0;
        rule.check_and_bind = [&guards](const Graph&, Match&) {
          ++guards;
          return true;
        };
        const auto expected_provider = b.GetExecutionProviderType();
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
        const bool supported = scenario == 0 || scenario == 1 || scenario == 3;
        EXPECT_EQ(guards, supported ? 1 : 0);
        EXPECT_EQ(stats.matches, supported ? 1u : 0u);
        EXPECT_EQ(stats.rewrites, supported && !match_only ? 1u : 0u);
        ASSERT_STATUS_OK(test.graph.Resolve());
        if (stats.rewrites) {
          EXPECT_EQ(test.graph.Nodes().begin()->GetExecutionProviderType(), expected_provider);
        }
      }
    }
  }
}

TEST(PatternRewriterTests, ExampleRulesDoNotCheckExecutionProvider) {
  TestGraph test;
  BuildHardSwish(test.builder);
  test.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  for (auto& node : test.graph.Nodes()) node.SetExecutionProviderType(kCudaExecutionProvider);
  Options options;
  options.compatible_execution_providers.insert(kCudaExecutionProvider);
  Statistics stats;
  // Graph-only validation: permitting a provider is not evidence of GPU kernel execution.
  ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules(), options).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 1u);
  ASSERT_STATUS_OK(test.graph.Resolve());
  EXPECT_EQ(test.graph.Nodes().begin()->GetExecutionProviderType(), kCudaExecutionProvider);
}

TEST(PatternRewriterTests, PriorityOverlapTerminationAndMatchOnly) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    for (int i = 0; i < 8; ++i) {
      auto* output = i == 7 ? b.MakeOutput() : b.MakeIntermediate();
      b.AddNode("Identity", {x}, {output});
      x = output;
    }
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    int first = 0, second = 0;
    Rule a = IdentityPair("first"), c = IdentityPair("second");
    a.check_and_bind = [&first](const Graph&, Match&) {
      ++first;
      return true;
    };
    c.check_and_bind = [&second](const Graph&, Match&) {
      ++second;
      return true;
    };
    PatternRewriter rewriter({a, c}, {.indexed = indexed});
    Statistics match_stats;
    ASSERT_STATUS_OK(PatternRewriter({a, c}, {.match_only = true, .indexed = indexed}).Run(test.graph, match_stats));
    EXPECT_EQ(test.graph.NumberOfNodes(), 8);
    EXPECT_EQ(match_stats.matches, 14u);
    first = second = 0;
    Statistics stats;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 7u);
    EXPECT_EQ(first, 7);
    EXPECT_EQ(second, 0);
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics repeated;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, repeated));
    EXPECT_EQ(repeated.rewrites, 0u);
  }
}

TEST(PatternRewriterTests, SingleNodeSameSizeAndExpansionNumerics) {
  for (bool indexed : {false, true}) {
    for (bool expand : {false, true}) {
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x)=>return Neg(x)");
      rule.replacement = Replacement(expand ? "(x)=>{a=Identity(x);return Neg(a);}" : "(x)=>Neg(x)");
      auto numerical_rule = rule;
      if (!expand) numerical_rule.replacement = Replacement("(x)=>Mul(x,-1.0)");
      numerical_rule.check_and_bind = [](const Graph& graph, Match& match) {
        const Node* producer = graph.GetProducerNode(match.valueOf("x")->Name());
        return !producer || producer->OpType() != "Identity";
      };
      TransformerTester(
          [](ModelTestBuilder& b) {
            auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
            b.AddNode("Neg", {x}, {b.MakeOutput()});
          },
          [expand](InferenceSessionWrapper& session) {
            EXPECT_EQ(Count(session.GetGraph(), "Neg"), expand ? 1u : 0u);
            EXPECT_EQ(Count(session.GetGraph(), "Mul"), expand ? 0u : 1u);
            EXPECT_EQ(Count(session.GetGraph(), "Identity"), expand ? 1u : 0u);
          },
          TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
          std::make_unique<PatternRewriter>(Rules{numerical_rule}, Options{.indexed = indexed, .max_iterations = 1}));
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({4}, -1.0f, 1.0f);
      auto* output = test.builder.MakeOutput();
      test.builder.AddNode("Neg", {x}, {output});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed, .max_iterations = 3}).Run(test.graph, stats));
      EXPECT_EQ(stats.iterations, 3u);
      EXPECT_TRUE(stats.iteration_limit_reached);
      EXPECT_EQ(stats.rewrites, 3u);
      EXPECT_EQ(Count(test.graph, "Identity"), expand ? 3u : 0u);
      EXPECT_EQ(test.graph.GetProducerNode(output->Name())->OpType(), "Neg");
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
  EXPECT_THROW((void)Replacement{"(x)=>return x"}, OnnxRuntimeException);
}

TEST(PatternRewriterTests, BoundedCyclesPriorityAndMatchOnly) {
  for (bool indexed : {false, true}) {
    for (size_t limit : {1u, 2u, 5u}) {
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      test.builder.AddNode("Identity", {x}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto forward = IdentityPair("forward");
      forward.pattern = Pattern("(x)=>return Identity(x)");
      forward.replacement = Replacement("(x)=>Transpose(x,perm=[0])");
      auto backward = IdentityPair("backward");
      backward.pattern = Pattern("(x)=>return Transpose(x,perm=[0])");
      backward.replacement = Replacement("(x)=>Identity(x)");
      auto later = forward;
      later.name = "later";
      later.check_and_bind = [](const Graph&, Match&) {
        ADD_FAILURE() << "Lower priority rule selected.";
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({forward, later, backward},
                                       {.indexed = indexed, .max_iterations = limit})
                           .Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, limit);
      EXPECT_EQ(stats.iterations, limit);
      EXPECT_TRUE(stats.iteration_limit_reached);
      EXPECT_EQ(Count(test.graph, "Transpose"), limit % 2);
      ASSERT_STATUS_OK(test.graph.Resolve());
      ASSERT_STATUS_OK(PatternRewriter({forward, backward},
                                       {.match_only = true, .indexed = indexed, .max_iterations = limit})
                           .Run(test.graph, stats));
      EXPECT_EQ(stats.matches, 1u);
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_EQ(stats.iterations, 1u);
      EXPECT_FALSE(stats.iteration_limit_reached);
    }
  }
}

TEST(PatternRewriterTests, BoundedIterationsStopOnConvergence) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
    test.builder.AddNode("Identity", {x}, {test.builder.MakeOutput()});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x)=>return Identity(x)");
    rule.replacement = Replacement("(x)=>Transpose(x,perm=[0])");
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed, .max_iterations = 5}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    EXPECT_EQ(stats.iterations, 2u);
    EXPECT_FALSE(stats.iteration_limit_reached);
    ASSERT_STATUS_OK(test.graph.Resolve());
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(stats.iterations, 1u);
    EXPECT_FALSE(stats.iteration_limit_reached);
  }
}

TEST(PatternRewriterTests, InvalidRuleAndRecipe) {
  EXPECT_THROW(Pattern("(x)=>a=Identity(x);return Identity(x)"), OnnxRuntimeException);
  auto rule = IdentityPair();
  rule.pattern = Pattern("(x)=>return Identity(x)");
  EXPECT_NO_THROW(PatternRewriter({rule}));
  EXPECT_THROW((PatternRewriter({rule}, {.max_iterations = 0})), OnnxRuntimeException);
  TestGraph test;
  auto* input = test.builder.MakeInput<float>({1}, -1.0f, 1.0f);
  auto* a = test.builder.MakeIntermediate();
  test.builder.AddNode("Identity", {input}, {a});
  test.builder.AddNode("Identity", {a}, {test.builder.MakeOutput()});
  test.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  rule = IdentityPair();
  for (auto text : {"(x)=>NoSuchReplacementOperator(x)", "(x)=>not.imported::Identity(x)",
                    "(a)=>Identity(a)", "(x)=>Identity(x,invalid=1)", "(x)=>com.microsoft::Identity(x)"}) {
    rule.replacement = [recipe = Replacement(text)](const Graph&, const Match&) { return recipe; };
    Statistics stats;
    EXPECT_FALSE(PatternRewriter({rule}).Run(test.graph, stats).IsOK());
    EXPECT_EQ(test.graph.NumberOfNodes(), 2);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, InvalidOpsetRequirements) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 5; ++scenario) {
      auto rule = IdentityPair();
      if (scenario == 0) rule.opsets.clear();
      if (scenario == 1) rule.opsets = {{kOnnxDomain, 0, 17}};
      if (scenario == 2) rule.opsets = {{kOnnxDomain, 18, 17}};
      if (scenario == 3) rule.opsets = {{kOnnxDomain, 1, 17}, {kOnnxDomainAlias, 1, 17}};
      if (scenario == 4) rule.opsets = {{kMSDomain, 1, 1}};
      EXPECT_THROW((PatternRewriter({rule}, {.indexed = indexed})), OnnxRuntimeException);
    }
  }
}

TEST(PatternRewriterTests, RuleImportRangesAndAliases) {
  for (bool indexed : {false, true}) {
    auto rule = IdentityPair();
    rule.opsets = {{kOnnxDomainAlias, 14, 17}};
    rule.pattern = Pattern("(x)=>a=ai.onnx::Identity(x);return ai.onnx::Identity(a)");
    int guards = 0;
    rule.check_and_bind = [&guards](const Graph&, Match&) {
      ++guards;
      return true;
    };
    rule.replacement = Replacement("(x)=>ai.onnx::Identity(x)");
    // Reusing one compiled rule across models must not retain the first model's imports.
    PatternRewriter rewriter({rule}, {.indexed = indexed});
    for (int opset : {13, 14, 15, 16, 17, 18}) {
      TestGraph test({{kOnnxDomain, opset}});
      auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* middle = test.builder.MakeIntermediate();
      test.builder.AddNode("Identity", {x}, {middle});
      test.builder.AddNode("Identity", {middle}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      const bool supported = opset >= 14 && opset <= 17;
      guards = 0;
      Statistics stats;
      ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
      EXPECT_EQ(guards, supported ? 1 : 0);
      EXPECT_EQ(stats.rewrites, supported ? 1u : 0u);
      if (!supported) {
        EXPECT_EQ(stats.visited, 0u);
        EXPECT_EQ(stats.candidates, 0u);
        EXPECT_EQ(stats.structural_checks, 0u);
      }
      ASSERT_STATUS_OK(test.graph.Resolve());
      EXPECT_EQ(test.graph.NumberOfNodes(), supported ? 1 : 2);
    }
  }
}

TEST(PatternRewriterTests, ImportFilteringWithinSharedStructuralGroups) {
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* middle = test.builder.MakeIntermediate();
      test.builder.AddNode("Identity", {x}, {middle});
      test.builder.AddNode("Identity", {middle}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto inactive = IdentityPair("inactive"), active = IdentityPair("active"), later = IdentityPair("later");
      inactive.opsets = {{kOnnxDomain, 18, 21}};
      int inactive_guards = 0, active_guards = 0, later_guards = 0;
      inactive.check_and_bind = [&inactive_guards](const Graph&, Match&) {
        ++inactive_guards;
        return true;
      };
      active.check_and_bind = [&active_guards](const Graph&, Match&) {
        ++active_guards;
        return true;
      };
      later.check_and_bind = [&later_guards](const Graph&, Match&) {
        ++later_guards;
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({inactive, active, later}, {.match_only = match_only, .indexed = indexed})
                           .Run(test.graph, stats));
      EXPECT_EQ(inactive_guards, 0);
      EXPECT_EQ(active_guards, 1);
      EXPECT_EQ(later_guards, match_only ? 1 : 0);
      EXPECT_EQ(stats.matches, match_only ? 2u : 1u);
      EXPECT_EQ(stats.rewrites, match_only ? 0u : 1u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, MultipleDomainImportRequirements) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 3; ++scenario) {
      TestGraph test(scenario == 0 ? std::unordered_map<std::string, int>{{kOnnxDomain, 17}}
                                   : std::unordered_map<std::string, int>{{kOnnxDomain, 17}, {kMSDomain, 1}});
      auto* x = test.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* middle = test.builder.MakeIntermediate();
      test.builder.AddNode("Identity", {x}, {middle});
      test.builder.AddNode("Identity", {middle}, {test.builder.MakeOutput()});
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.opsets.push_back({kMSDomain, scenario == 1 ? 2 : 1, 2});
      int guards = 0;
      rule.check_and_bind = [&guards](const Graph&, Match&) {
        ++guards;
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(guards, scenario == 2 ? 1 : 0);
      EXPECT_EQ(stats.rewrites, scenario == 2 ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, ExampleRuleImportBoundaries) {
  for (bool indexed : {false, true}) {
    PatternRewriter rewriter(MakeExampleRules(), {.indexed = indexed});
    for (int opset : {13, 14, 17, 21, 22}) {
      TestGraph test({{kOnnxDomain, opset}});
      BuildHardSwish(test.builder);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
      const bool supported = opset >= 14 && opset <= 21;
      EXPECT_EQ(stats.rewrites, supported ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
      EXPECT_EQ(Count(test.graph, "HardSwish"), supported ? 1u : 0u);
    }
  }
}

TEST(PatternRewriterTests, RuleConstructorTextAndParsedForms) {
  static_assert(std::is_copy_constructible_v<Rule> && std::is_copy_assignable_v<Rule>);
  static_assert(std::is_move_constructible_v<Rule> && std::is_move_assignable_v<Rule>);
  const auto* pattern_text = "(x)=>{a=Identity(x);return Identity(a);}";
  const auto* replacement_text = "(x)=>Identity(x)";
  auto check = [](const auto& pattern_text, const auto& replacement_text) {
    Pattern parsed_pattern(pattern_text);
    Replacement parsed_replacement(replacement_text);
    auto check_and_bind = [](const Graph&, Match&) { return true; };
    size_t calls = 0;
    auto replacement = [parsed_replacement, &calls](const Graph&, const Match&) {
      ++calls;
      return parsed_replacement;
    };
    Rules rules{
        {"literal", "(x)=>{a=Identity(x);return Identity(a);}", "(x)=>Identity(x)", check_and_bind, {{kOnnxDomain, 1, 21}}},
        {"text", pattern_text, replacement_text, check_and_bind, {{kOnnxDomain, 1, 21}}},
        {"parsed", parsed_pattern, parsed_replacement, check_and_bind, {{kOnnxDomain, 1, 21}}},
        {"parsed_pattern", parsed_pattern, replacement_text, check_and_bind, {{kOnnxDomain, 1, 21}}},
        {"parsed_replacement", pattern_text, parsed_replacement, check_and_bind, {{kOnnxDomain, 1, 21}}},
        {"callback", pattern_text, replacement, check_and_bind, {{kOnnxDomain, 1, 21}}},
        {"parsed_callback", parsed_pattern, replacement, check_and_bind, {{kOnnxDomain, 1, 21}}}};
    for (bool indexed : {false, true}) {
      for (const auto& rule : rules) {
        TestGraph test;
        auto& b = test.builder;
        auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
        auto* middle = b.MakeIntermediate();
        b.AddNode("Identity", {x}, {middle});
        b.AddNode("Identity", {middle}, {b.MakeOutput()});
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, 1u);
        EXPECT_EQ(test.graph.NumberOfNodes(), 1);
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
    }
    EXPECT_EQ(calls, 4u);
  };
  check(pattern_text, replacement_text);
  check(std::string(pattern_text), std::string(replacement_text));
  check(std::string_view(pattern_text), std::string_view(replacement_text));
}

TEST(PatternRewriterTests, RuleConstructorOwnsParsedTextAndReusesPlans) {
  std::string pattern = "(x)=>{a=Identity(x);return Identity(a);}";
  std::string replacement = "(x)=>Identity(x)";
  auto check_and_bind = [](const Graph&, Match&) { return true; };
  Rule rule{"owned", pattern, replacement, check_and_bind, {{kOnnxDomain, 1, 21}}};
  pattern.assign("not a pattern");
  replacement.assign("not a replacement");
  EXPECT_EQ(rule.pattern.Nodes().size(), 2u);
  const auto* compiled = rule.replacement.target<Replacement>();
  ASSERT_NE(compiled, nullptr);
  const auto* plan = compiled->Plan().get();
  auto copied = rule;
  auto moved = std::move(copied);
  copied = moved;
  copied = std::move(moved);
  TestGraph test;
  Match match;
  for (int i = 0; i < 3; ++i) {
    auto result = copied.replacement(test.graph, match);
    EXPECT_EQ(result.Plan().get(), plan);
  }
  size_t calls = 0;
  auto callback = [recipe = *compiled, &calls](const Graph&, const Match&) {
    ++calls;
    return recipe;
  };
  Rule computed{"computed", rule.pattern, callback, check_and_bind, rule.opsets};
  EXPECT_EQ(calls, 0u);
  for (int i = 0; i < 3; ++i) {
    auto result = computed.replacement(test.graph, match);
    EXPECT_EQ(result.Plan().get(), plan);
  }
  EXPECT_EQ(calls, 3u);
}

TEST(PatternRewriterTests, DefaultCheckAndBindAcceptsAnnotatedPatterns) {
  const auto* pattern = "(x:float[])=>{a=Identity(x);return Identity(a);}";
  const auto* replacement = "(x)=>Identity(x)";
  Pattern parsed_pattern(pattern);
  Replacement parsed_replacement(replacement);
  Rules rules{
      {"text", pattern, replacement},
      {"parsed", parsed_pattern, parsed_replacement},
      {"parsed_pattern", parsed_pattern, replacement},
      {"parsed_replacement", pattern, parsed_replacement},
      {"explicit_empty", pattern, replacement, {}, {{kOnnxDomain, 13, 17}}}};
  for (auto& rule : rules) {
    EXPECT_FALSE(rule.check_and_bind);
    if (rule.opsets.empty()) {
      EXPECT_THROW((void)PatternRewriter{Rules{rule}}, OnnxRuntimeException);
      rule.opsets = {{kOnnxDomain, 13, 17}};
    }
    for (bool indexed : {false, true}) {
      for (bool match_only : {false, true}) {
        for (bool floating : {false, true}) {
          TestGraph test;
          auto& b = test.builder;
          auto* x = floating ? b.MakeInput<float>({2, 3}, -1.0f, 1.0f)
                             : b.MakeInput<int64_t>({2, 3}, -1, 1);
          auto* a = b.MakeIntermediate();
          b.AddNode("Identity", {x}, {a});
          b.AddNode("Identity", {a}, {b.MakeOutput()});
          b.SetGraphOutputs();
          ASSERT_STATUS_OK(test.graph.Resolve());
          Statistics stats;
          ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = match_only, .indexed = indexed})
                               .Run(test.graph, stats));
          EXPECT_EQ(stats.matches, floating ? 1u : 0u);
          EXPECT_EQ(stats.rewrites, floating && !match_only ? 1u : 0u);
          EXPECT_EQ(stats.check_and_bind_calls, 0u);
          ASSERT_STATUS_OK(test.graph.Resolve());
        }
      }
    }
  }
  TransformerTester(
      [](ModelTestBuilder& b) {
        auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
        auto* a = b.MakeIntermediate();
        b.AddNode("Identity", {x}, {a});
        b.AddNode("Identity", {a}, {b.MakeOutput()});
      },
      [](InferenceSessionWrapper& session) { EXPECT_EQ(Count(session.GetGraph(), "Identity"), 1u); },
      TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
      std::make_unique<PatternRewriter>(Rules{rules.front()}));
}

TEST(PatternRewriterTests, RuleConstructorRejectsMalformedText) {
  const auto* pattern = "(x)=>{a=Identity(x);return Identity(a);}";
  const auto* replacement = "(x)=>Identity(x)";
  auto check_and_bind = [](const Graph&, Match&) { return true; };
  auto make_rule = [&](auto pattern, auto replacement) {
    return Rule{"bad", pattern, replacement, check_and_bind, {{kOnnxDomain, 1, 21}}};
  };
  EXPECT_THROW(make_rule("not a pattern", replacement), OnnxRuntimeException);
  EXPECT_THROW(make_rule(pattern, "not a replacement"), OnnxRuntimeException);
  EXPECT_THROW(make_rule(Pattern(pattern), "not a replacement"), OnnxRuntimeException);
  EXPECT_THROW(make_rule("not a pattern", Replacement(replacement)), OnnxRuntimeException);
  auto callback = [plan = Replacement(replacement)](const Graph&, const Match&) { return plan; };
  EXPECT_THROW(make_rule("not a pattern", callback), OnnxRuntimeException);
}

TEST(PatternRewriterTests, PriorityAcrossInterleavedStructuralGroups) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    for (int i = 0; i < 3; ++i) {
      auto* next = i == 2 ? b.MakeOutput() : b.MakeIntermediate();
      b.AddNode("Identity", {x}, {next});
      x = next;
    }
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto rejected = IdentityPair("rejected", false);
    auto lower_priority = IdentityPair("lower_priority");
    lower_priority.check_and_bind = [](const Graph& graph, Match& m) {
      return graph.NodeProducesGraphOutput(*m.nodes.back());
    };
    const auto* pattern = "(x)=>a=Identity(x);b=Identity(a);return Identity(b)";
    auto check_and_bind = [](const Graph&, Match&) { return true; };
    const auto* replacement = "(x)=>Identity(x)";
    Rule higher_priority{"higher_priority", pattern, replacement, check_and_bind, {{kOnnxDomain, 1, 21}}};
    Statistics stats;
    ASSERT_STATUS_OK(
        PatternRewriter({rejected, higher_priority, lower_priority}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
    EXPECT_EQ(stats.rewrites, 1u);
    EXPECT_EQ(test.graph.NumberOfNodes(), 1);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, LayerAnnotationsAndRootConsumers) {
  for (bool same_layer : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* first = b.MakeIntermediate();
    auto* root = b.MakeOutput();
    auto& a = b.AddNode("Identity", {x}, {first});
    auto& c = b.AddNode("Identity", {first}, {root});
    a.SetLayeringAnnotation(same_layer ? "layer0" : "layer1");
    c.SetLayeringAnnotation("layer0");
    b.AddNode("Neg", {root}, {b.MakeOutput()});
    b.AddNode("Abs", {root}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({IdentityPair()}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, same_layer ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    const Node* producer = test.graph.GetProducerNode(root->Name());
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->GetLayeringAnnotation(), "layer0");
    EXPECT_EQ(producer->GetOutputEdgesCount(), 2);
  }
}

TEST(PatternRewriterTests, OutputCountMismatch) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* a = b.MakeIntermediate();
  b.AddNode("Identity", {x}, {a});
  b.AddNode("Dropout", {a}, {b.MakeOutput(), b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  Pattern p("(x)=>a=Identity(x);return Dropout(a)");
  auto rule = IdentityPair();
  rule.pattern = p;
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter({rule}).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
}

TEST(PatternRewriterTests, LineCommentParsingAndQuotedHashes) {
  for (const auto* newline : {"\n", "\r\n", "\r"}) {
    SCOPED_TRACE(newline);
    std::string text = "# before header";
    for (const auto* line : {
             "(# input", "x# name", ")# header", "=># arrow", "{# body",
             "a# output", "=# assignment", "Identity# operator", "(# argument", "x# input", ")# call",
             "return# result", "Identity(a)# final call", "}# EOF comment"}) {
      text += newline;
      text += line;
    }
    Pattern pattern(text);
    Replacement replacement(text);
    EXPECT_EQ(pattern.Nodes().size(), 2u);
    ASSERT_TRUE(replacement.Plan());
    EXPECT_EQ(replacement.Plan()->size(), 2u);
  }
  Pattern annotated("(x:# type\nfloat[# dimension\nM] |# alternative\nfloat[M,N])=>return Identity(x)#EOF");
  EXPECT_EQ(annotated.TypeConstraints().at(0).size(), 2u);
  Replacement lists("(x)=>Transpose(x,perm=[1,# next element\n0])#EOF");
  EXPECT_EQ(lists.Plan()->front().attributes.at("perm").ints(1), 0);
  for (auto text : {R"((x)=>return Identity(x,label='left#right')# trailing)",
                    R"((x)=>return Identity(x,label="left#right")# trailing)",
                    R"((x)=>return Identity(x,label="left\"#right")# trailing)"}) {
    Replacement replacement(text);
    const auto& value = replacement.Plan()->front().attributes.at("label");
    EXPECT_NE(value.s().find('#'), std::string::npos);
    Pattern pattern(text);
    ASSERT_EQ(pattern.Nodes().front().attributes.size(), 1u);
    EXPECT_TRUE(pattern.Nodes().front().attributes.front().Matches(value));
    auto different = value;
    different.set_s("left");
    EXPECT_FALSE(pattern.Nodes().front().attributes.front().Matches(different));
  }
}

TEST(PatternRewriterTests, LineCommentsDoNotRepairMalformedTokens) {
  for (auto text : {"# no definition",
                    "(x)=>return Iden# split identifier\ntity(x)",
                    "(x)=# split arrow\n>return Identity(x)",
                    "(x)=>return Add(x,1# split number\n.0)",
                    "(x)=>return Add(x,1e# split exponent\n2)",
                    "(x)=>return com.microsoft:# split qualifier\n:FusedMatMul(x,x)",
                    "(x)=>return Identity(x# closing delimiter swallowed)",
                    "(x)=>{return Identity(x)# closing brace swallowed}",
                    "(x)=>return Identity(x,label='unterminated# not a comment)",
                    "(x)=>return Identity(x)# comment\nunexpected"}) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)Pattern{text}, OnnxRuntimeException);
    EXPECT_THROW((void)Replacement{text}, OnnxRuntimeException);
  }
}

TEST(PatternRewriterTests, CommentedRuleNumerics) {
  for (bool indexed : {false, true}) {
    auto check_and_bind = [](const Graph&, Match&) { return true; };
    Rule rule{"commented",
              R"(# pattern
                (x:float[# dimensions
                  M,N]) => { a=Identity(x) # intermediate
                  return Identity(a) # root
                } # end)",
              R"(# replacement
                (x) => { return Identity(# argument
                  x) } # end)",
              check_and_bind,
              {{kOnnxDomain, 13, 17}}};
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
          auto* a = b.MakeIntermediate();
          b.AddNode("Identity", {x}, {a});
          b.AddNode("Identity", {a}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& session) { EXPECT_EQ(Count(session.GetGraph(), "Identity"), 1u); },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
        std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, ParserOnlyConstruction) {
  static_assert(!std::is_default_constructible_v<Pattern>);
  static_assert(!std::is_default_constructible_v<Replacement>);
  static_assert(!std::is_constructible_v<Replacement, InlinedVector<Replacement::NodeRecipe, 1>>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<Pattern>().Nodes())>>);
  EXPECT_NO_THROW(Pattern("(x)=>{a=Identity(x);result=Identity(a);return result;}"));
  EXPECT_NO_THROW(Replacement("(x)=>{result=Identity(x);return result;}"));
}

TEST(PatternRewriterTests, BracedBodiesAndNamedReturns) {
  for (auto text : {"(x)=>{a=Identity(x);return a;}", "(x)=>{a=Identity(x);return a}",
                    "(x)=>a=Identity(x);return a", "(x)=>{return Identity(x);}",
                    "(x)=>return Identity(x)", " ( x ) => { \n a = Identity(x); \n return a; \n } \n",
                    "(x)=>{left,right=Split(x);return left,right;}",
                    "(x)=>left,right=Split(x);return left,right",
                    "(x)=>{left,right=ai.onnx::Split(x);return left,right;}",
                    "(x)=>{a=Identity(x);b=Identity(a);return a,b;}",
                    "(x)=>{a=Identity(x);b=Identity(a);return a;}",
                    "(x)=>{left,right=Split(x);return left;}",
                    "(x)=>{left,right=Split(x);return right,left;}"}) {
    SCOPED_TRACE(text);
    EXPECT_NO_THROW((void)Pattern{text});
    EXPECT_NO_THROW((void)Replacement{text});
  }
  EXPECT_NO_THROW(Replacement("(x)=>{Identity(x);}"));
  EXPECT_NO_THROW(Replacement("(x)=>Identity(x)"));
  for (auto text : {"(x)=>{a=Identity(x);return a", "(x)=>a=Identity(x);return a}",
                    "(x)=>{{return Identity(x);}}", "(x)=>{return Identity(x);}}",
                    "(x)=>{return Identity(x);} extra", "(x)=>{return Identity(x);Identity(x);}",
                    "(x)=>{}", "(x)=>{a=Identity(x);}", "(x)=>{return x;}",
                    "(x)=>{a=Identity(x);return missing;}",
                    "(x)=>{a=Identity(x);b=Identity(x);return a,b;}",
                    "(x)=>{left,right=Split(x);return left,left;}",
                    "(x)=>{left,right=Split(x);return left,right,missing;}",
                    "(x)=>{return a=Identity(x);}", "(x)=>return left,right=Split(x)",
                    "(x)=>{a=Identity(x);return a;;}", "(x)=>{return Identity(x);};"}) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)Pattern{text}, OnnxRuntimeException);
    EXPECT_THROW((void)Replacement{text}, OnnxRuntimeException);
  }
  Pattern direct("(x)=>{a=Identity(x);return Identity(a);}");
  Pattern named("(x)=>{a=Identity(x);result=Identity(a);return result;}");
  EXPECT_EQ(direct.ValueCount(), named.ValueCount());
  EXPECT_EQ(direct.Nodes().size(), named.Nodes().size());
  EXPECT_EQ(named.Nodes().back().outputs[0].name, "result");
}

TEST(PatternRewriterTests, ExplicitBoundaryOutputsAndNumerics) {
  for (bool indexed : {false, true}) {
    for (bool reversed : {false, true}) {
      auto rule = IdentityPair("boundary");
      rule.pattern = Pattern(reversed
                                 ? "(x)=>{a=Identity(x);y=Add(a,a);z=Mul(y,y);return z,y;}"
                                 : "(x)=>{a=Identity(x);y=Add(a,a);z=Mul(y,y);return y,z;}");
      rule.replacement = Replacement(reversed
                                         ? "(x)=>{y=Add(x,x);z=Mul(y,y);return z,y;}"
                                         : "(x)=>{y=Add(x,x);z=Mul(y,y);return y,z;}");
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* a = b.MakeIntermediate();
      auto* y = b.MakeOutput();
      auto* z = b.MakeOutput();
      b.AddNode("Identity", {x}, {a});
      b.AddNode("Add", {a, a}, {y});
      b.AddNode("Mul", {y, y}, {z});
      b.AddNode("Sub", {y, z}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      EXPECT_EQ(test.graph.NumberOfNodes(), 3);
      const Node* sum = test.graph.GetProducerNode(y->Name());
      const Node* product = test.graph.GetProducerNode(z->Name());
      ASSERT_NE(sum, nullptr);
      ASSERT_NE(product, nullptr);
      EXPECT_EQ(sum->OpType(), "Add");
      EXPECT_EQ(product->OpType(), "Mul");
      EXPECT_EQ(sum->GetOutputEdgesCount(), 3);
      EXPECT_EQ(product->GetOutputEdgesCount(), 1);
      EXPECT_EQ(product->InputDefs()[0], y);
      EXPECT_EQ(product->InputDefs()[1], y);
      ASSERT_STATUS_OK(test.graph.Resolve());
      TransformerTester(
          [](ModelTestBuilder& builder) {
            auto* input = builder.MakeInput<float>({2}, -1.0f, 1.0f);
            auto* copied = builder.MakeIntermediate();
            auto* sum_output = builder.MakeOutput();
            auto* product_output = builder.MakeOutput();
            builder.AddNode("Identity", {input}, {copied});
            builder.AddNode("Add", {copied, copied}, {sum_output});
            builder.AddNode("Mul", {sum_output, sum_output}, {product_output});
            builder.AddNode("Sub", {sum_output, product_output}, {builder.MakeOutput()});
          },
          [](InferenceSessionWrapper& session) {
            EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
            EXPECT_EQ(Count(session.GetGraph(), "Add"), 1u);
            EXPECT_EQ(Count(session.GetGraph(), "Mul"), 1u);
          },
          TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
          std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
    }
  }
}

TEST(PatternRewriterTests, ReturnedOutputSelectionAndBoundaries) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      SCOPED_TRACE(scenario);
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x)=>{a=Identity(x);left,right=Split(a,axis=0);return right;}");
      rule.replacement = Replacement("(x)=>{left,right=Split(x,axis=0);return right;}");
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
      auto* a = scenario == 3 ? b.MakeOutput() : b.MakeIntermediate();
      auto* left = scenario == 1 ? b.MakeOutput() : b.MakeIntermediate();
      auto* right = b.MakeOutput();
      b.AddNode("Identity", {x}, {a});
      b.AddNode("Split", {a}, {left, right}).AddAttribute("axis", int64_t{0});
      if (scenario == 2) b.AddNode("Neg", {left}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
      if (scenario == 0) {
        const Node* producer = test.graph.GetProducerNode(right->Name());
        ASSERT_NE(producer, nullptr);
        EXPECT_EQ(producer->OutputDefs()[1], right);
        EXPECT_NE(producer->OutputDefs()[0], left);
      }
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x)=>{a=Identity(x);left,right=Split(a,axis=0);return right,left;}");
    rule.replacement = Replacement("(x)=>{left,right=Split(x,axis=0);return right,left;}");
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
          auto* copied = b.MakeIntermediate();
          b.AddNode("Identity", {x}, {copied});
          b.AddNode("Split", {copied}, {b.MakeOutput(), b.MakeOutput()}).AddAttribute("axis", int64_t{0});
        },
        [](InferenceSessionWrapper& session) { EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u); },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
        std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, BoundaryOutputMappingCycleChecks) {
  for (bool indexed : {false, true}) {
    for (bool valid : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
      auto* a = b.MakeIntermediate();
      auto* y = b.MakeOutput();
      auto* external = b.MakeIntermediate();
      b.AddNode("Identity", {x}, {a});
      b.AddNode("Add", {a, a}, {y});
      b.AddNode("Neg", {y}, {external});
      b.AddNode("Mul", {y, external}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x,e)=>{a=Identity(x);y=Add(a,a);z=Mul(y,e);return y,z;}");
      rule.replacement = Replacement(valid
                                         ? "(x,e)=>{y=Add(x,x);z=Mul(y,e);return y,z;}"
                                         : "(e)=>{left,right=Split(e,axis=0);return left,right;}");
      Statistics stats;
      auto status = PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats);
      EXPECT_EQ(status.IsOK(), valid);
      EXPECT_EQ(stats.rewrites, valid ? 1u : 0u);
      if (!valid) EXPECT_NE(status.ErrorMessage().find("cycle"), std::string::npos);
      EXPECT_EQ(test.graph.NumberOfNodes(), valid ? 3 : 4);
      EXPECT_EQ(test.graph.GetProducerNode(y->Name())->OpType(), "Add");
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, ReturnedIntermediateImplicitCapture) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
    auto* copied = b.MakeIntermediate();
    auto* left = b.MakeIntermediate();
    auto* right = b.MakeIntermediate();
    b.AddNode("Identity", {x}, {copied});
    b.AddNode("Split", {copied}, {left, right}).AddAttribute("axis", int64_t{0});
    b.AddNode("Concat", {left, right}, {b.MakeOutput()}).AddAttribute("axis", int64_t{0});
    ONNX_NAMESPACE::GraphProto branch;
    branch.set_name("returned_intermediate");
    auto* identity = branch.add_node();
    identity->set_op_type("Identity");
    identity->add_input(right->Name());
    identity->add_output("branch_out");
    auto* output = branch.add_output();
    output->set_name("branch_out");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    auto& if_node = b.AddNode("If", {b.MakeInput<bool>({}, std::vector<bool>{true})}, {b.MakeOutput()});
    if_node.AddAttribute("then_branch", branch);
    if_node.AddAttribute("else_branch", branch);
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x)=>{a=Identity(x);l,r=Split(a,axis=0);y=Concat(l,r,axis=0);return r,y;}");
    rule.replacement = Replacement("(x)=>{l,r=Split(x,axis=0);y=Concat(l,r,axis=0);return r,y;}");
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    const Node* producer = test.graph.GetProducerNode(right->Name());
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->OpType(), "Split");
    EXPECT_EQ(producer->OutputDefs()[1], right);
    EXPECT_EQ(Count(test.graph, "Identity"), 0u);
  }
}

TEST(PatternRewriterTests, NamedAttributeCaptures) {
  for (auto text : {"(x)=>{t=Transpose(x,perm=p);return Identity(t);}",
                    "(x)=>{t=Transpose(x,perm=p);return Transpose(t,perm=p);}",
                    "(x)=>{h=HardSigmoid(x,alpha=a,beta=b);return Mul(h,x);}"})
    EXPECT_NO_THROW((void)Pattern{text});
  for (auto text : {"(x)=>{t=Transpose(x,perm=x);return Identity(t);}",
                    "(x)=>{t=Transpose(x,perm=p);p=Identity(t);return p;}",
                    "(x)=>{t=Transpose(x,perm=None);return Identity(t);}"})
    EXPECT_THROW((void)Pattern{text}, OnnxRuntimeException);
  for (bool indexed : {false, true}) {
    for (bool equal : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2, 2}, -1.0f, 1.0f);
      auto* t = b.MakeIntermediate();
      auto& first = b.AddNode("Transpose", {x}, {t});
      first.AddAttribute("perm", std::vector<int64_t>{1, 0});
      auto& second = b.AddNode("Transpose", {t}, {b.MakeOutput()});
      second.AddAttribute("perm", equal ? std::vector<int64_t>{1, 0} : std::vector<int64_t>{0, 1});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x)=>{t=Transpose(x,perm=p);return Transpose(t,perm=p);}");
      int guards = 0;
      rule.check_and_bind = [&guards](const Graph&, Match& match) {
        ++guards;
        EXPECT_EQ(match.attributeOf("p").type(), ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
        EXPECT_EQ(match.attributeOf("p").ints(0), 1);
        EXPECT_THROW(match.update("p", match.attributeOf("p")), OnnxRuntimeException);
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(guards, equal ? 1 : 0);
      EXPECT_EQ(stats.rewrites, equal ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x)=>{t=Transpose(x,perm=axes);return Identity(t);}");
    rule.replacement = Replacement("(x,axes)=>Transpose(x,perm=axes)");
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
          auto* t = b.MakeIntermediate();
          b.AddNode("Transpose", {x}, {t}).AddAttribute("perm", std::vector<int64_t>{1, 0});
          b.AddNode("Identity", {t}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
          EXPECT_EQ(session.GetGraph().Nodes().begin()->GetAttributes().at("perm").ints(0), 1);
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
        std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, OptionalInputSlotsAndNumerics) {
  for (auto text : {"(x,low:tensor[...] | None,high:tensor[...] | None)=>{a=Identity(x);return Clip(a,low,high);}",
                    "(x,high)=>{a=Identity(x);return Clip(a,None,high);}",
                    "(x)=>{a=Identity(x);return Clip(a,None,None);}"}) {
    EXPECT_NO_THROW((void)Pattern{text});
    if (std::string_view(text).find(':') != std::string_view::npos)
      EXPECT_THROW((void)Replacement{text}, OnnxRuntimeException);
    else
      EXPECT_NO_THROW((void)Replacement{text});
  }
  for (auto text : {"(None)=>return Identity(None)", "(x??)=>return Identity(x)",
                    "(x)=>{a=Identity(x);return Clip(a,missing);}",
                    "(x?)=>return x", "(x)=>{a=Identity(x);return Clip(a,None=0);}"})
    EXPECT_THROW((void)Pattern{text}, OnnxRuntimeException);
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 5; ++scenario) {
      SCOPED_TRACE(scenario);
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x,low:tensor[...] | None,high:tensor[...] | None)=>{a=Identity(x);return Clip(a,low,high);}");
      rule.replacement = Replacement("(x,low,high)=>Clip(x,low,high)");
      auto build = [scenario](ModelTestBuilder& b) {
        auto* x = b.MakeInput<float>({4}, -2.0f, 2.0f);
        auto* a = b.MakeIntermediate();
        b.AddNode("Identity", {x}, {a});
        std::vector<NodeArg*> inputs{x};
        inputs[0] = a;
        if (scenario > 0) {
          auto* empty = b.MakeOptionalTensor();
          inputs.push_back(scenario == 2 ? empty : b.MakeScalarInitializer<float>(-1.0f));
          if (scenario >= 2)
            inputs.push_back(scenario == 4 ? empty : b.MakeScalarInitializer<float>(1.0f));
        }
        b.AddNode("Clip", inputs, {b.MakeOutput()});
      };
      TestGraph test;
      build(test.builder);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      if (scenario == 0) {
        auto invalid = rule;
        invalid.replacement = Replacement("(low)=>Identity(low)");
        Statistics rejected;
        EXPECT_FALSE(PatternRewriter({invalid}, {.indexed = indexed}).Run(test.graph, rejected).IsOK());
        EXPECT_EQ(test.graph.NumberOfNodes(), 2);
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
      int guards = 0;
      rule.check_and_bind = [scenario, &guards](const Graph&, Match& match) {
        ++guards;
        EXPECT_EQ(match.valueOf("low") == nullptr, scenario == 0 || scenario == 2);
        EXPECT_EQ(match.valueOf("high") == nullptr, scenario < 2 || scenario == 4);
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      EXPECT_EQ(guards, 1);
      ASSERT_STATUS_OK(test.graph.Resolve());
      // Numerical check_and_bind must not retain references to a previous graph's match.
      rule.check_and_bind = [](const Graph&, Match&) { return true; };
      TransformerTester(
          build,
          [](InferenceSessionWrapper& session) {
            EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
            EXPECT_EQ(Count(session.GetGraph(), "Clip"), 1u);
          },
          TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
          std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
    }
  }
}

TEST(PatternRewriterTests, CapturedAttributesVaryAcrossMatches) {
  for (bool indexed : {false, true}) {
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x)=>{h=HardSigmoid(x,alpha=slope,beta=bias);return Identity(h);}");
    rule.replacement = Replacement("(x,slope,bias)=>HardSigmoid(x,alpha=slope,beta=bias)");
    auto build = [](ModelTestBuilder& b) {
      auto* x = b.MakeInput<float>({4}, -2.0f, 2.0f);
      for (float alpha : {0.25f, 0.5f}) {
        auto* h = b.MakeIntermediate();
        auto& sigmoid = b.AddNode("HardSigmoid", {x}, {h});
        sigmoid.AddAttribute("alpha", alpha);
        sigmoid.AddAttribute("beta", 0.5f);
        b.AddNode("Identity", {h}, {b.MakeOutput()});
      }
    };
    TestGraph test;
    build(test.builder);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    int guards = 0;
    rule.check_and_bind = [&guards](const Graph&, Match& match) {
      ++guards;
      EXPECT_EQ(match.attributeOf("slope").type(), ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
      EXPECT_FLOAT_EQ(match.attributeOf("bias").f(), 0.5f);
      return true;
    };
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 2u);
    EXPECT_EQ(guards, 2);
    ASSERT_STATUS_OK(test.graph.Resolve());
    InlinedHashSet<float> slopes;
    for (const auto& node : test.graph.Nodes()) slopes.insert(node.GetAttributes().at("alpha").f());
    EXPECT_EQ(slopes.size(), 2u);
    rule.check_and_bind = [](const Graph&, Match&) { return true; };
    TransformerTester(
        build,
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "HardSigmoid"), 2u);
          EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
        std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, OptionalInputConstraintsAndAbsentRepetition) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* a = b.MakeIntermediate();
      b.AddNode("Identity", {x}, {a});
      auto* empty = &test.graph.GetOrCreateNodeArg("", nullptr);
      auto* low = scenario == 0 ? empty : b.MakeScalarInitializer<float>(0.0f);
      auto* high = scenario < 2 ? empty : (scenario == 2 ? low : b.MakeScalarInitializer<float>(1.0f));
      b.AddNode("Clip", {a, low, high}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      for (int constraint = 0; constraint < 3; ++constraint) {
        auto rule = IdentityPair();
        rule.pattern = Pattern(constraint == 0   ? "(x,v:tensor[...] | None)=>{a=Identity(x);return Clip(a,v,v);}"
                               : constraint == 1 ? "(x)=>{a=Identity(x);return Clip(a,None,None);}"
                                                 : "(x,lo,hi)=>{a=Identity(x);return Clip(a,lo,hi);}");
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
        const bool expected = constraint == 0   ? scenario == 0 || scenario == 2
                              : constraint == 1 ? scenario == 0
                                                : scenario >= 2;
        EXPECT_EQ(stats.matches, expected ? 1u : 0u);
      }
    }
  }
}

TEST(PatternRewriterTests, StructuredFailureDiagnostics) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 6; ++scenario) {
      SCOPED_TRACE(scenario);
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* a = scenario == 1 ? b.MakeOutput() : b.MakeIntermediate();
      auto& first = b.AddNode("Identity", {x}, {a});
      auto& root = b.AddNode("Identity", {a}, {b.MakeOutput()});
      const NodeIndex root_index = root.Index();
      if (scenario == 2) first.SetExecutionProviderType(kCudaExecutionProvider);
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair("diagnostic");
      if (scenario == 0) rule.check_and_bind = [](const Graph&, Match& match) { return match.Reject("unsupported dtype"); };
      if (scenario == 3) rule.opsets = {{kOnnxDomain, 18, 21}};
      if (scenario == 4) rule.pattern = Pattern("(x)=>{a=Identity(x,absent=captured);return Identity(a);}");
      if (scenario == 5) {
        rule.replacement = [](const Graph&, const Match&) { return Replacement("(x)=>Identity(x,invalid=1)"); };
      }
      InlinedVector<MatchFailure> failures;
      Options options{.indexed = indexed};
      options.on_failure = [&failures](const MatchFailure& failure) { failures.push_back(failure); };
      Statistics stats;
      auto status = PatternRewriter({rule}, options).Run(test.graph, stats);
      EXPECT_EQ(status.IsOK(), scenario != 5);
      EXPECT_EQ(stats.rewrites, 0u);
      const auto expected_stage = scenario == 0   ? FailureStage::CheckAndBind
                                  : scenario == 1 ? FailureStage::Boundary
                                  : scenario == 2 ? FailureStage::ExecutionProvider
                                  : scenario == 3 ? FailureStage::Opset
                                  : scenario == 4 ? FailureStage::Attribute
                                                  : FailureStage::Replacement;
      auto found = std::find_if(failures.begin(), failures.end(),
                                [expected_stage](const auto& failure) { return failure.stage == expected_stage; });
      ASSERT_NE(found, failures.end());
      EXPECT_EQ(found->rule, "diagnostic");
      EXPECT_EQ(found->root, scenario == 3 ? std::numeric_limits<NodeIndex>::max() : root_index);
      EXPECT_FALSE(found->reason.empty());
      if (scenario == 0) EXPECT_EQ(found->reason, "unsupported dtype");
      ASSERT_STATUS_OK(test.graph.Resolve());
      EXPECT_EQ(test.graph.NumberOfNodes(), 2);
    }
  }
}

TEST(PatternRewriterTests, TypeShapeAnnotationParsing) {
  for (auto text : {
           "(x:float[M,N])=>return Identity(x)",
           "(x:float[])=>return Identity(x)",
           "(x:float[...])=>return Identity(x)",
           "(x:float)=>return Identity(x)",
           "(x:tensor[M,0])=>return Identity(x)",
           "(x:tensor<T>[M,...],y:tensor<T>[M,_])=>return CastLike(x,y)",
           "(x:int64[...] | None)=>return Identity(x)",
           "(x:float[M,N])=>{y:float[M,N]=Identity(x);return y;}",
           "(x:float[M,N])=>{a:float[M,N],b:float[M,N]=Split(x);return a,b;}",
           "(x:float[],axis:int)=>return Softmax(x,axis=axis)",
           "(x:float[],p:ints)=>return Transpose(x,perm=p)",
           "(x,a:float,b:float)=>return HardSigmoid(x,alpha=a,beta=b)",
           "(x,s:string)=>return Identity(x,mode=s)",
           "(x,t:tensor)=>return Identity(x,value=t)",
           "(x,f:floats,s:strings)=>return Identity(x,f=f,s=s)"}) {
    SCOPED_TRACE(text);
    EXPECT_NO_THROW((void)Pattern{text});
    EXPECT_THROW((void)Replacement{text}, OnnxRuntimeException);
  }
  for (auto text : {
           "(x:bogus)=>return Identity(x)", "(x:undefined)=>return Identity(x)",
           "(x:float[-1])=>return Identity(x)", "(x:float[1.0])=>return Identity(x)",
           "(x:float[M,])=>return Identity(x)", "(x:float[...,M])=>return Identity(x)",
           "(x:float[M,...,N])=>return Identity(x)", "(x:float[M)=>return Identity(x)",
           "(x:float<T>[M])=>return Identity(x)", "(x:tensor<>[M])=>return Identity(x)",
           "(x:tensor<T[M])=>return Identity(x)", "(x:float[M,N])=>return x",
           "(M,x:float[M])=>return Identity(x)", "(x:float[x])=>return Identity(x)",
           "(x:tensor<x>[M])=>return Identity(x)", "(x:tensor<T>[T])=>return Identity(x)",
           "(x:float[M],M:int)=>return Softmax(x,axis=M)",
           "(x:float[...],p:float[M])=>return Transpose(x,perm=p)",
           "(x:int)=>return Identity(x)", "(x,axis?:int)=>return Softmax(x,axis=axis)",
           "(x,axis:int)=>return Add(x,axis)", "(x,axis:int)=>return Identity(x)",
           "(x)=>{y:float=Identity(x);return y:float;}"}) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)Pattern{text}, OnnxRuntimeException);
  }
  Match empty;
  EXPECT_THROW(empty.dimensionOf("M"), OnnxRuntimeException);
  EXPECT_THROW(empty.elementTypeOf("T"), OnnxRuntimeException);
}

TEST(PatternRewriterTests, TypeShapeAnnotationRankAndDtype) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 5; ++scenario) {
      SCOPED_TRACE(scenario);
      TestGraph test;
      auto& b = test.builder;
      auto* x = scenario == 0   ? b.MakeInput<float>({}, -1.0f, 1.0f)
                : scenario == 1 ? b.MakeInput<float>({2}, -1.0f, 1.0f)
                : scenario == 2 ? b.MakeInput<float>({2, 3}, -1.0f, 1.0f)
                : scenario == 3 ? b.MakeInput<int64_t>({2, 3}, -1, 1)
                                : b.MakeInput<float>({2, 3, 4}, -1.0f, 1.0f);
      auto* output = b.MakeOutput();
      b.AddNode("Identity", {x}, {output});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      const std::pair<const char*, bool> cases[] = {
          {"float", scenario == 0},
          {"float[]", scenario != 3},
          {"float[...]", scenario != 3},
          {"float[M,N]", scenario == 2},
          {"float[2,3]", scenario == 2},
          {"float[M,...]", scenario != 0 && scenario != 3},
          {"tensor[M,N]", scenario == 2 || scenario == 3},
          {"tensor<T>[...]", true},
          {"int64[2,3]", scenario == 3},
          {"float[_,_]", scenario == 2}};
      for (const auto& [type, expected] : cases) {
        SCOPED_TRACE(type);
        auto rule = IdentityPair();
        rule.pattern = Pattern("(x:" + std::string(type) + ")=>return Identity(x)");
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
        EXPECT_EQ(stats.matches, expected ? 1u : 0u);
      }
      x->ClearShape();
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x:float[])=>return Identity(x)");
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, scenario == 3 ? 0u : 1u);
      rule.pattern = Pattern("(x:float[M,...])=>return Identity(x)");
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, 0u);
    }
  }
}

TEST(PatternRewriterTests, SharedDimensionAndElementBindings) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 7; ++scenario) {
      SCOPED_TRACE(scenario);
      TestGraph test;
      auto& b = test.builder;
      using Shape = std::vector<std::variant<int64_t, std::string>>;
      Shape sx{int64_t{2}, int64_t{3}}, sy = sx;
      if (scenario == 1) sy[0] = int64_t{4};
      if (scenario >= 2 && scenario <= 4) {
        sx[0] = std::string("batch");
        sy[0] = scenario == 3 ? std::string("other") : std::string("batch");
        if (scenario == 4) sy[0] = int64_t{2};
      }
      if (scenario == 5) sx[0] = sy[0] = std::string{};
      auto* x = b.MakeSymbolicInput<float>(sx);
      auto* y = scenario == 6 ? b.MakeSymbolicInput<int64_t>(sy) : b.MakeSymbolicInput<float>(sy);
      auto* a = b.MakeIntermediate();
      b.AddNode("Identity", {x}, {a});
      b.AddNode("CastLike", {a, y}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x:tensor<T>[M,N],y:tensor<T>[M,N])=>{a=Identity(x);return CastLike(a,y);}");
      int guards = 0;
      rule.check_and_bind = [scenario, &guards](const Graph&, Match& match) {
        ++guards;
        EXPECT_EQ(match.elementTypeOf("T"), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        EXPECT_EQ(std::get<int64_t>(match.dimensionOf("N")), 3);
        if (scenario == 0) EXPECT_EQ(std::get<int64_t>(match.dimensionOf("M")), 2);
        if (scenario == 2) EXPECT_EQ(std::get<std::string>(match.dimensionOf("M")), "batch");
        if (scenario == 5) EXPECT_TRUE(std::holds_alternative<std::monostate>(match.dimensionOf("M")));
        EXPECT_THROW(match.dimensionOf("missing"), OnnxRuntimeException);
        EXPECT_THROW(match.elementTypeOf("missing"), OnnxRuntimeException);
        ONNX_NAMESPACE::AttributeProto attribute;
        attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        attribute.set_i(1);
        EXPECT_THROW(match.update("M", attribute), OnnxRuntimeException);
        EXPECT_THROW(match.update("T", attribute), OnnxRuntimeException);
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      const bool expected = scenario == 0 || scenario == 2 || scenario == 5;
      EXPECT_EQ(stats.matches, expected ? 1u : 0u);
      EXPECT_EQ(guards, expected ? 1 : 0);
    }
  }
}

TEST(PatternRewriterTests, TypedAttributeCapturesAndNumerics) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 3; ++scenario) {
      const bool accepted = scenario == 0;
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto* t = b.MakeIntermediate();
      b.AddNode("Transpose", {x}, {t}).AddAttribute("perm", std::vector<int64_t>{1, 0});
      b.AddNode("Identity", {t}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern(accepted
                                 ? "(x:float[M,N],axes:ints)=>{t:float[N,M]=Transpose(x,perm=axes);return Identity(t);}"
                             : scenario == 1
                                 ? "(x:float[M,N],axes:floats)=>{t=Transpose(x,perm=axes);return Identity(t);}"
                                 : "(x:float[M,N],axes:ints)=>{t:float[M,N]=Transpose(x,perm=axes);return Identity(t);}");
      rule.replacement = Replacement("(x,axes)=>Transpose(x,perm=axes)");
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, accepted ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x:float[M,N],axes:ints)=>{t:float[N,M]=Transpose(x,perm=axes);return Identity(t);}");
    rule.replacement = Replacement("(x,axes)=>Transpose(x,perm=axes)");
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
          auto* t = b.MakeIntermediate();
          b.AddNode("Transpose", {x}, {t}).AddAttribute("perm", std::vector<int64_t>{1, 0});
          b.AddNode("Identity", {t}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
          EXPECT_EQ(Count(session.GetGraph(), "Transpose"), 1u);
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
        std::make_unique<PatternRewriter>(Rules{rule}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, AnnotatedOptionalValuesAndRuleIsolation) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 3; ++scenario) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto* clipped = b.MakeIntermediate();
      b.AddNode("Clip", {x}, {clipped});
      b.AddNode("Identity", {clipped}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto first = IdentityPair("reject"), second = IdentityPair("accept");
      first.pattern = Pattern("(x:float[2,4],lo:float | None,hi:float | None)=>{a=Clip(x,lo,hi);return Identity(a);}");
      second.pattern = Pattern(
          "(input:tensor<T>[M,N],low:tensor<U> | None,high:float | None)=>"
          "{a:tensor<T>[M,N]=Clip(input,low,high);return Identity(a);}");
      second.replacement = Replacement("(input)=>Clip(input)");
      second.check_and_bind = [](const Graph&, Match& match) {
        EXPECT_EQ(match.valueOf("low"), nullptr);
        EXPECT_FALSE(match.elementTypeOf("U").has_value());
        EXPECT_EQ(std::get<int64_t>(match.dimensionOf("M")), 2);
        return true;
      };
      if (scenario == 1) first.pattern = Pattern(
                             "(x:float[2,3],lo:float | None,hi:float | None)=>"
                             "{a=Clip(x,lo,hi);return Identity(a);}");
      if (scenario == 1) first.check_and_bind = [](const Graph&, Match&) { return false; };
      if (scenario == 2) second.pattern = Pattern(
                             "(input:tensor<T>[M,M],low:tensor<U> | None,high:float | None)=>"
                             "{a=Clip(input,low,high);return Identity(a);}");
      InlinedVector<MatchFailure> failures;
      Options options{.indexed = indexed};
      options.on_failure = [&failures](const MatchFailure& failure) { failures.push_back(failure); };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({first, second}, options).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 2 ? 0u : 1u);
      EXPECT_TRUE(std::any_of(failures.begin(), failures.end(),
                              [](const auto& failure) { return failure.stage == FailureStage::TypeShape; }) ||
                  scenario == 1);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, TypeUnionParsingAndBounds) {
  for (auto text : {"(x:float[M] | float[M,N])=>return Identity(x)",
                    "(x:float[M] | None)=>return Identity(x)",
                    "(x:None | tensor<T>[...])=>return Identity(x)",
                    "(x)=>{y:float[M] | float[M,N]=Identity(x);return y;}"}) {
    EXPECT_NO_THROW((void)Pattern{text});
    EXPECT_THROW((void)Replacement{text}, OnnxRuntimeException);
  }
  for (auto text : {"(x:|float[M])=>return Identity(x)", "(x:float[M]|)=>return Identity(x)",
                    "(x:float[M]||None)=>return Identity(x)", "(x:None)=>return Identity(x)",
                    "(x:None|None)=>return Identity(x)", "(x:float[M]|float[M])=>return Identity(x)",
                    "(x:float[]|float[...])=>return Identity(x)",
                    "(x?)=>return Identity(x)", "(x?:float[])=>return Identity(x)",
                    "(x:float[]?)=>return Identity(x)",
                    "(x,p:ints|floats)=>return Identity(x,perm=p)",
                    "(x,p:int|None)=>return Identity(x,axis=p)",
                    "(x)=>{y:float[]|None=Identity(x);return y;}"}) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)Pattern{text}, OnnxRuntimeException);
  }
  auto make_text = [](int count) {
    std::string text = "(";
    for (int i = 0; i < count; ++i) {
      if (i) text += ',';
      text += "x" + std::to_string(i) + ":float[]|int64[]";
    }
    text += ")=>return Concat(";
    for (int i = 0; i < count; ++i) {
      if (i) text += ',';
      text += "x" + std::to_string(i);
    }
    return text + ",axis=0)";
  };
  EXPECT_NO_THROW((void)Pattern{make_text(6)});
  EXPECT_THROW((void)Pattern{make_text(7)}, OnnxRuntimeException);
}

TEST(PatternRewriterTests, TypeUnionRanksAndNone) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 5; ++scenario) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = scenario == 0   ? b.MakeInput<float>({2}, -1.0f, 1.0f)
                : scenario == 1 ? b.MakeInput<float>({2, 3}, -1.0f, 1.0f)
                : scenario == 2 ? b.MakeInput<float>({}, -1.0f, 1.0f)
                : scenario == 3 ? b.MakeInput<float>({2, 3, 4}, -1.0f, 1.0f)
                                : b.MakeInput<int64_t>({2}, -1, 1);
      b.AddNode("Identity", {x}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x:float[M] | float[M,N])=>return Identity(x)");
      rule.check_and_bind = [scenario](const Graph&, Match& match) {
        EXPECT_EQ(std::get<int64_t>(match.dimensionOf("M")), 2);
        if (scenario == 0)
          EXPECT_TRUE(std::holds_alternative<std::monostate>(match.dimensionOf("N")));
        else
          EXPECT_EQ(std::get<int64_t>(match.dimensionOf("N")), 3);
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, scenario < 2 ? 1u : 0u);
    }
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      std::vector<NodeArg*> inputs{x};
      if (scenario == 1) inputs.push_back(b.MakeOptionalTensor());
      if (scenario == 2) inputs.push_back(b.MakeScalarInitializer<float>(0.0f));
      if (scenario == 3) inputs.push_back(b.MakeInput<float>({2}, -1.0f, 1.0f));
      b.AddNode("Clip", inputs, {b.MakeOutput()});
      b.SetGraphOutputs();
      // A vector bound is schema-valid but cannot satisfy the scalar annotation.
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x:float[M],low:float | None)=>return Clip(x,low)");
      rule.check_and_bind = [scenario](const Graph&, Match& match) {
        EXPECT_EQ(match.valueOf("low") == nullptr, scenario < 2);
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, scenario < 3 ? 1u : 0u);
    }
  }
}

TEST(PatternRewriterTests, TypeUnionGlobalBindingsAndRollback) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto* y = b.MakeInput<float>({scenario == 2 ? 4 : 3}, -1.0f, 1.0f);
      b.AddNode("CastLike", {x, y}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern(scenario == 0
                                 ? "(x:float[M,9] | float[2,M],y:float[M])=>return CastLike(x,y)"
                             : scenario == 1 || scenario == 2
                                 ? "(x:float[M,_] | float[_,M],y:float[M])=>return CastLike(x,y)"
                                 : "(x:tensor<T>[M,_] | tensor<U>[_,M],y:tensor<U>[M])=>return CastLike(x,y)");
      rule.check_and_bind = [scenario](const Graph&, Match& match) {
        EXPECT_EQ(std::get<int64_t>(match.dimensionOf("M")), 3);
        if (scenario == 3) {
          EXPECT_FALSE(match.elementTypeOf("T").has_value());
          EXPECT_EQ(match.elementTypeOf("U"), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        }
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, scenario == 2 ? 0u : 1u);
    }
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    b.AddNode("Clip", {x}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x,lo:tensor<T>[D] | None)=>return Clip(x,lo)");
    rule.check_and_bind = [](const Graph&, Match& match) {
      EXPECT_EQ(match.valueOf("lo"), nullptr);
      EXPECT_TRUE(std::holds_alternative<std::monostate>(match.dimensionOf("D")));
      EXPECT_FALSE(match.elementTypeOf("T").has_value());
      return true;
    };
    rule.replacement = Replacement("(x,lo)=>Clip(x,lo)");
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed, .max_iterations = 1}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, ParsedMultiOutputDefinitionsAndLimits) {
  Pattern p("(x)=>{a=Identity(x);b=ai.onnx::Identity(a);left,right=ai.onnx::Split(b);return left,right;}");
  auto x = p.NamedValue("x");
  const auto& outputs = p.Nodes().back().outputs;
  EXPECT_EQ(outputs.size(), 2u);
  EXPECT_NE(outputs[0], outputs[1]);
  EXPECT_TRUE(p.IsInput(x));
  EXPECT_FALSE(p.IsInput(outputs[0]));
  EXPECT_FALSE(p.IsInput(outputs[1]));
  EXPECT_EQ(p.Nodes().back().op.domain, kOnnxDomain);
  const auto count = p.ValueCount();
  EXPECT_THROW(p = Pattern("(x)=>return = Split(x)"), OnnxRuntimeException);
  EXPECT_THROW(p = Pattern("(x)=>a,b=Split(missing);return a,b"), OnnxRuntimeException);
  EXPECT_THROW(p = Pattern("(x)=>left,left=Split(x);return left,left"), OnnxRuntimeException);
  EXPECT_THROW(p = Pattern("(x)=>x,right=Split(x);return x,right"), OnnxRuntimeException);
  EXPECT_NO_THROW(Pattern("(x)=>a,b=Split(x);return a,b"));
  EXPECT_EQ(p.ValueCount(), count);
  auto outputs_text = [](int node, int count) {
    std::string text;
    for (int slot = 0; slot < count; ++slot) {
      if (slot) text += ',';
      text += "v" + std::to_string(node) + "_" + std::to_string(slot);
    }
    return text;
  };
  EXPECT_THROW(Pattern("(x)=>{" + outputs_text(0, 65) + "=Split(x);return " + outputs_text(0, 65) + ";}"),
               OnnxRuntimeException);
  std::string many_values = "(x)=>";
  for (int node = 0; node < 3; ++node) {
    const auto input = node == 0 ? "x" : "v" + std::to_string(node - 1) + "_0";
    many_values += outputs_text(node, 64) + "=Split(" + input + ");";
  }
  EXPECT_NO_THROW(Pattern(many_values + "return Identity(v2_0)"));
  EXPECT_THROW(Pattern(many_values + outputs_text(3, 64) + "=Split(v2_0);return " + outputs_text(3, 64)),
               OnnxRuntimeException);
}

TEST(PatternRewriterTests, MultiOutputIntermediateBoundariesAndBindings) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 5; ++scenario) {
      TestGraph test;
      auto* x = test.builder.MakeInput<float>({4}, -1.0f, 1.0f);
      auto* a = test.builder.MakeIntermediate();
      auto* b = scenario == 1 ? test.builder.MakeOutput() : test.builder.MakeIntermediate();
      test.builder.AddNode("Split", {x}, {a, b});
      auto& concat = test.builder.AddNode("Concat", {a, scenario == 3 ? a : b}, {test.builder.MakeOutput()});
      concat.AddAttribute("axis", int64_t{0});
      if (scenario == 2) test.builder.AddNode("Identity", {b}, {test.builder.MakeOutput()});
      if (scenario == 4) {
        // A repeated output binding must refer to the same node, not a second Split.
        auto* c = test.builder.MakeIntermediate();
        auto* d = test.builder.MakeIntermediate();
        test.builder.AddNode("Split", {x}, {c, d});
        concat.MutableInputDefs()[1] = d;
      }
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({SplitConcat()}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
      if (scenario == 0) EXPECT_EQ(test.graph.NumberOfNodes(), 1);
    }
  }
}

TEST(PatternRewriterTests, MultiOutputProducerReachedThroughSecondOutput) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({4}, -1.0f, 1.0f);
    auto* a = test.builder.MakeIntermediate();
    auto* b = test.builder.MakeIntermediate();
    test.builder.AddNode("Split", {x}, {a, b});
    test.builder.AddNode("Identity", {b}, {test.builder.MakeOutput()});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Pattern pattern("(x)=>a,b=Split(x);return Identity(b)");
    auto rule = SplitConcat();
    rule.pattern = pattern;
    rule.check_and_bind = [](const Graph&, Match&) { return true; };
    Statistics stats;
    // Only match: an Identity replacement would not preserve slicing semantics.
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
    EXPECT_EQ(stats.rewrites, 0u);
  }
}

TEST(PatternRewriterTests, MultiOutputRootEdgesAndInvalidRecipe) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({4}, -1.0f, 1.0f);
    auto* middle = test.builder.MakeIntermediate();
    auto* a = test.builder.MakeOutput();
    auto* b = test.builder.MakeOutput();
    test.builder.AddNode("Identity", {x}, {middle});
    test.builder.AddNode("Split", {middle}, {a, b});
    test.builder.AddNode("Neg", {a}, {test.builder.MakeOutput()});
    test.builder.AddNode("Add", {b, b}, {test.builder.MakeOutput()});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    for (auto text : {"(x)=>Identity(x)", "(x)=>Split(x)",
                      "(x)=>{left,middle,right=Split(x);return left,middle,right;}"}) {
      auto invalid = IdentitySplit();
      invalid.replacement = Replacement(text);
      EXPECT_FALSE(PatternRewriter({invalid}, {.indexed = indexed}).Run(test.graph, stats).IsOK());
      EXPECT_EQ(test.graph.NumberOfNodes(), 4);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    auto wrong_count = IdentityPair("single_output_split");
    wrong_count.pattern = Pattern("(x)=>a=Identity(x);return Split(a)");
    auto valid = IdentitySplit();
    valid.check_and_bind = [a, b, prepare = valid.check_and_bind](const Graph& graph, Match& match) {
      EXPECT_EQ(match.valueOf("a"), a);
      EXPECT_EQ(match.valueOf("b"), b);
      return prepare(graph, match);
    };
    ASSERT_STATUS_OK(PatternRewriter({wrong_count, valid}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    const Node* producer = test.graph.GetProducerNode(a->Name());
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer, test.graph.GetProducerNode(b->Name()));
    EXPECT_EQ(producer->OutputDefs()[0], a);
    EXPECT_EQ(producer->OutputDefs()[1], b);
    EXPECT_EQ(producer->GetOutputEdgesCount(), 3);
    size_t slot0 = 0, slot1 = 0;
    for (auto edge = producer->OutputEdgesBegin(); edge != producer->OutputEdgesEnd(); ++edge) {
      if (edge->GetSrcArgIndex() == 0) ++slot0;
      if (edge->GetSrcArgIndex() == 1) ++slot1;
    }
    EXPECT_EQ(slot0, 1u);
    EXPECT_EQ(slot1, 2u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    ASSERT_STATUS_OK(PatternRewriter({IdentitySplit()}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
  }
}

TEST(PatternRewriterTests, MultiOutputImplicitCaptureBoundary) {
  for (bool indexed : {false, true}) {
    for (bool root : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
      auto* split_input = x;
      if (root) {
        split_input = b.MakeIntermediate();
        b.AddNode("Identity", {x}, {split_input});
      }
      auto* a = b.MakeIntermediate();
      auto* c = b.MakeIntermediate();
      b.AddNode("Split", {split_input}, {a, c});
      b.AddNode("Concat", {a, c}, {b.MakeOutput()}).AddAttribute("axis", int64_t{0});
      ONNX_NAMESPACE::GraphProto branch;
      branch.set_name("captured_split_output");
      auto* identity = branch.add_node();
      identity->set_op_type("Identity");
      identity->add_input(c->Name());
      identity->add_output("branch_out");
      auto* output = branch.add_output();
      output->set_name("branch_out");
      output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
      auto& if_node = b.AddNode("If", {b.MakeInput<bool>({}, std::vector<bool>{true})}, {b.MakeOutput()});
      if_node.AddAttribute("then_branch", branch);
      if_node.AddAttribute("else_branch", branch);
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({IdentitySplit(), SplitConcat()}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, root ? 1u : 0u);
      const Node* producer = test.graph.GetProducerNode(c->Name());
      ASSERT_NE(producer, nullptr);
      EXPECT_EQ(producer->OutputDefs()[1], c);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, NumericalMultiOutputPatterns) {
  for (bool indexed : {false, true}) {
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
          auto* a = b.MakeIntermediate();
          auto* c = b.MakeIntermediate();
          b.AddNode("Split", {x}, {a, c});
          b.AddNode("Concat", {a, c}, {b.MakeOutput()}).AddAttribute("axis", int64_t{0});
        },
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "Split"), 0u);
          EXPECT_EQ(Count(session.GetGraph(), "Concat"), 0u);
          EXPECT_EQ(Count(session.GetGraph(), "Identity"), 1u);
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
        std::make_unique<PatternRewriter>(Rules{SplitConcat()}, Options{.indexed = indexed}));
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
          auto* middle = b.MakeIntermediate();
          auto* a = b.MakeOutput();
          auto* c = b.MakeOutput();
          b.AddNode("Identity", {x}, {middle});
          b.AddNode("Split", {middle}, {a, c});
          b.AddNode("Neg", {a}, {b.MakeOutput()});
          b.AddNode("Add", {c, c}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "Identity"), 0u);
          EXPECT_EQ(Count(session.GetGraph(), "Split"), 1u);
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
        std::make_unique<PatternRewriter>(Rules{IdentitySplit()}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, OnnxscriptMultiOutputLayerNormBias) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 3; ++scenario) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
      auto* y = b.MakeIntermediate();
      auto* mean = scenario == 1 ? b.MakeOutput() : b.MakeIntermediate();
      auto* inverse_std = b.MakeIntermediate();
      b.AddNode("LayerNormalization", {x, b.MakeInitializer<float>({4}, 0.8f, 1.2f)}, {y, mean, inverse_std});
      b.AddNode("Add", {y, b.MakeInitializer<float>({4}, -0.1f, 0.1f)}, {b.MakeOutput()});
      if (scenario == 2) b.AddNode("Identity", {inverse_std}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({LayerNormBias()}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
          auto* y = b.MakeIntermediate();
          b.AddNode("LayerNormalization", {x, b.MakeInitializer<float>({4}, 0.8f, 1.2f)},
                    {y, b.MakeIntermediate(), b.MakeIntermediate()});
          b.AddNode("Add", {y, b.MakeInitializer<float>({4}, -0.1f, 0.1f)}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "LayerNormalization"), 1u);
          EXPECT_EQ(Count(session.GetGraph(), "Add"), 0u);
          for (const auto& node : session.GetGraph().Nodes()) {
            if (node.OpType() == "LayerNormalization") {
              EXPECT_EQ(node.InputDefs().size(), 3u);
              EXPECT_EQ(node.OutputDefs().size(), 1u);
            }
          }
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
        std::make_unique<PatternRewriter>(Rules{LayerNormBias()}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, NumericalMultiOutputLeafPreservation) {
  TransformerTester(
      [](ModelTestBuilder& b) {
        auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
        auto* dropout = b.MakeIntermediate();
        auto* first = b.MakeIntermediate();
        b.AddNode("Dropout", {x}, {dropout, b.MakeOutput()});
        b.AddNode("Identity", {dropout}, {first});
        b.AddNode("Identity", {first}, {b.MakeOutput()});
      },
      [](InferenceSessionWrapper& session) {
        EXPECT_EQ(Count(session.GetGraph(), "Dropout"), 1u);
        EXPECT_EQ(Count(session.GetGraph(), "Identity"), 1u);
      },
      TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
      std::make_unique<PatternRewriter>(Rules{IdentityPair()}));
}

TEST(PatternRewriterTests, DoubleTypeRejection) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<double>({2}, -1.0, 1.0);
  auto* h = b.MakeIntermediate();
  auto& sigmoid = b.AddNode("HardSigmoid", {x}, {h});
  sigmoid.AddAttribute("alpha", 1.0f / 6.0f);
  sigmoid.AddAttribute("beta", 0.5f);
  b.AddNode("Mul", {h, x}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
}

TEST(PatternRewriterTests, ExampleGuardsCheckRelevantBindings) {
  auto examples = MakeExampleRules();
  for (size_t index = 0; index < examples.size(); ++index) {
    SCOPED_TRACE(examples[index].name);
    ASSERT_NE(examples[index].replacement.target<Replacement>(), nullptr);
    TestGraph test;
    auto& b = test.builder;
    auto* non_float = b.MakeScalarInitializer<double>(2.0);
    auto* integer = b.MakeScalarInitializer<int64_t>(2);
    // Resolve removes unused untyped graph NodeArgs.
    NodeArg unknown{"guard_unknown", nullptr};
    b.AddNode("Identity", {non_float}, {b.MakeOutput()});
    b.AddNode("Identity", {integer}, {b.MakeOutput()});
    InlinedVector<std::string_view> bindings{"x"};
    if (index == 0) {
      BuildHardSwish(b);
    } else if (index == 1) {
      auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
      auto* sigmoid = b.MakeIntermediate();
      auto& node = b.AddNode("HardSigmoid", {x}, {sigmoid});
      node.AddAttribute("alpha", 1.0f / 6.0f);
      node.AddAttribute("beta", 0.5f);
      b.AddNode("Mul", {sigmoid, x}, {b.MakeOutput()});
    } else if (index == 5) {
      auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
      BuildRms(b, x, b.MakeOutput(), 4);
      bindings.push_back("scale");
      bindings.push_back("epsilon");
    } else {
      auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
      auto* y = b.MakeInput<float>(index == 3 ? std::vector<int64_t>{3, 4}
                                              : std::vector<int64_t>{4, 3},
                                   -1.0f, 1.0f);
      bindings.push_back("y");
      if (index == 3) {
        auto* transpose = b.MakeIntermediate();
        b.AddNode("Transpose", {y}, {transpose}).AddAttribute("perm", std::vector<int64_t>{1, 0});
        b.AddNode("MatMul", {x, transpose}, {b.MakeOutput()});
      } else {
        auto* mm = b.MakeIntermediate();
        b.AddNode(index == 4 ? "FusedMatMul" : "MatMul", {x, y}, {mm},
                  index == 4 ? kMSDomain : kOnnxDomain);
        b.AddNode("Div", {mm, b.MakeScalarInitializer<float>(2.0f)}, {b.MakeOutput()});
        bindings.push_back("divisor");
      }
    }
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto rule = examples[index];
    auto check_and_bind = rule.check_and_bind;
    size_t checked = 0;
    rule.check_and_bind = [&](const Graph& graph, Match& match) {
      EXPECT_TRUE(check_and_bind(graph, match));
      for (auto binding : bindings) {
        SCOPED_TRACE(binding);
        // Exercise the check_and_bind independently without making the resolved graph ill-typed.
        for (const NodeArg* value : {non_float, integer, &unknown}) {
          Match changed = match;
          changed.values[match.pattern->NamedValue(binding).handle] = value;
          EXPECT_FALSE(check_and_bind(graph, changed));
        }
      }
      ++checked;
      return check_and_bind(graph, match);
    };
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({std::move(rule)}).Run(test.graph, stats));
    EXPECT_EQ(checked, 1u);
    EXPECT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, BranchedSharedNodeBindings) {
  for (bool external : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* y = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* sum = b.MakeIntermediate();
    auto* product = b.MakeIntermediate();
    b.AddNode("Add", {x, y}, {sum});
    b.AddNode("Mul", {sum, x}, {product});
    b.AddNode("Add", {sum, product}, {b.MakeOutput()});
    if (external) b.AddNode("Identity", {sum}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    const auto* pattern = "(x,y)=>sum=Add(x,y);product=Mul(sum,x);return Add(sum,product)";
    auto check_and_bind = [](const Graph&, Match&) { return true; };
    const auto* replacement = "(x)=>Identity(x)";
    Rule rule{"diamond", pattern, replacement, check_and_bind, {{kOnnxDomain, 7, 21}}};
    Statistics stats;
    // Structural-only test: this deliberately has no semantically equivalent replacement.
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true}).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, external ? 0u : 1u);
    EXPECT_EQ(stats.rewrites, 0u);
  }
}

TEST(PatternRewriterTests, SubgraphsAndImplicitCaptureBoundary) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* captured = b.MakeIntermediate();
  b.AddNode("Identity", {x}, {captured});
  b.AddNode("Identity", {captured}, {b.MakeOutput()});
  ONNX_NAMESPACE::GraphProto branch;
  branch.set_name("branch");
  auto* first = branch.add_node();
  first->set_op_type("Identity");
  first->add_input(captured->Name());
  first->add_output("branch_mid");
  auto* second = branch.add_node();
  second->set_op_type("Identity");
  second->add_input("branch_mid");
  second->add_output("branch_out");
  auto* output = branch.add_output();
  output->set_name("branch_out");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  auto* condition = b.MakeInput<bool>({}, std::vector<bool>{true});
  auto& if_node = b.AddNode("If", {condition}, {b.MakeOutput()});
  if_node.AddAttribute("then_branch", branch);
  if_node.AddAttribute("else_branch", branch);
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  bool modified = false;
  ASSERT_STATUS_OK(
      PatternRewriter({IdentityPair()}).Apply(test.graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_TRUE(modified);
  // The captured intermediate keeps both outer Identity nodes alive.
  EXPECT_EQ(Count(test.graph, "Identity"), 2u);
  for (const auto& entry : if_node.GetAttributeNameToSubgraphMap()) {
    EXPECT_EQ(Count(*entry.second, "Identity"), 1u);
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(PatternRewriterTests, RmsGuardAllowsIntegerExponent) {
  for (bool indexed : {false, true}) {
    Rule rule = MakeExampleRules()[5];
    rule.pattern = Pattern(R"(
      (x, scale, epsilon) =>
        square = Pow(x, 2)
        mean = ReduceMean(square)
        add = Add(mean, epsilon)
        sqrt = Sqrt(add)
        reciprocal = Reciprocal(sqrt)
        normalize = Mul(x, reciprocal)
        return Mul(normalize, scale)
    )");
    TransformerTester(
        [](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
          BuildRms(b, x, b.MakeOutput(), 4);
          auto* exponent = b.MakeScalarInitializer<int64_t>(2);
          for (auto& node : b.graph_.Nodes()) {
            if (node.OpType() == "Pow") node.MutableInputDefs()[1] = exponent;
          }
        },
        [](InferenceSessionWrapper& session) {
          EXPECT_EQ(Count(session.GetGraph(), "SimplifiedLayerNormalization"), 1u);
          EXPECT_EQ(Count(session.GetGraph(), "Pow"), 0u);
        },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
        std::make_unique<PatternRewriter>(Rules{std::move(rule)}, Options{.indexed = indexed}));
  }
}

TEST(PatternRewriterTests, MatMulConstantGuardsAndBroadcastRank) {
  for (int scenario = 0; scenario < 6; ++scenario) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* y = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* mm = b.MakeIntermediate();
    b.AddNode("MatMul", {x, y}, {mm});
    float value = 2.0f;
    if (scenario == 1) value = 0.0f;
    if (scenario == 2) value = std::numeric_limits<float>::infinity();
    NodeArg* divisor;
    if (scenario == 3) {
      divisor = b.MakeInitializer<float>({1}, {value});
    } else if (scenario == 4) {
      divisor = b.MakeInput<float>({}, std::vector<float>{value});
    } else {
      divisor = b.MakeScalarInitializer<float>(value);
    }
    b.AddNode("Div", {mm, divisor}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    if (scenario == 5) {
      auto inputs = test.graph.GetInputsIncludingInitializers();
      inputs.push_back(divisor);
      test.graph.SetInputs(inputs);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, NumericalMatMulDivAndTranspose) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    bool transpose = scenario != 0;
    bool divide = scenario != 1;
    TransformerTester(
        [transpose, divide](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
          auto* y =
              b.MakeInput<float>(transpose ? std::vector<int64_t>{4, 3} : std::vector<int64_t>{3, 4}, -1.0f, 1.0f);
          auto* mm = divide ? b.MakeIntermediate() : b.MakeOutput();
          if (transpose) {
            auto* t = b.MakeIntermediate();
            b.AddNode("Transpose", {y}, {t}).AddAttribute("perm", std::vector<int64_t>{1, 0});
            y = t;
          }
          b.AddNode("MatMul", {x, y}, {mm});
          if (divide) b.AddNode("Div", {mm, b.MakeScalarInitializer<float>(2.0f)}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& s) { EXPECT_EQ(Count(s.GetGraph(), "FusedMatMul"), 1u); },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
        std::make_unique<PatternRewriter>(MakeExampleRules()));
  }
}

TEST(PatternRewriterTests, NumericalFusedMatMulDivPreservesAttributes) {
  for (bool indexed : {false, true}) {
    for (int scenario = -1; scenario < 16; ++scenario) {
      SCOPED_TRACE(scenario);
      const int flags = std::max(scenario, 0);
      auto build = [scenario, flags](ModelTestBuilder& b) {
        auto input_shape = [flags](std::vector<int64_t> shape, int transpose, int transpose_batch) {
          if (flags & transpose) std::swap(shape[1], shape[2]);
          if (flags & transpose_batch) std::swap(shape[0], shape[1]);
          return shape;
        };
        auto* x = b.MakeInput<float>(input_shape({2, 3, 4}, 1, 4), -1.0f, 1.0f);
        auto* y = b.MakeInput<float>(input_shape({2, 4, 5}, 2, 8), -1.0f, 1.0f);
        auto* mm = b.MakeIntermediate();
        auto& node = b.AddNode("FusedMatMul", {x, y}, {mm}, kMSDomain);
        if (scenario >= 0) {
          node.AddAttribute("alpha", -1.5f);
          node.AddAttribute("transA", int64_t{(flags & 1) != 0});
          node.AddAttribute("transB", int64_t{(flags & 2) != 0});
          node.AddAttribute("transBatchA", int64_t{(flags & 4) != 0});
          node.AddAttribute("transBatchB", int64_t{(flags & 8) != 0});
        }
        b.AddNode("Div", {mm, b.MakeScalarInitializer<float>(2.0f)}, {b.MakeOutput()});
      };
      auto check = [scenario, flags](InferenceSessionWrapper& session) {
        EXPECT_EQ(Count(session.GetGraph(), "FusedMatMul"), 1u);
        EXPECT_EQ(Count(session.GetGraph(), "Div"), 0u);
        for (const auto& node : session.GetGraph().Nodes()) {
          if (node.OpType() != "FusedMatMul") continue;
          const auto& attributes = node.GetAttributes();
          EXPECT_FLOAT_EQ(attributes.at("alpha").f(), scenario < 0 ? 0.5f : -0.75f);
          EXPECT_EQ(attributes.at("transA").i(), (flags & 1) != 0);
          EXPECT_EQ(attributes.at("transB").i(), (flags & 2) != 0);
          EXPECT_EQ(attributes.at("transBatchA").i(), (flags & 4) != 0);
          EXPECT_EQ(attributes.at("transBatchB").i(), (flags & 8) != 0);
        }
      };
      Rule rule = MakeExampleRules()[4];
      rule.pattern = Pattern("(x,y,divisor)=>{mm=com.microsoft.FusedMatMul(x,y);return Div(mm,divisor);}");
      TransformerTester(build, check, TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
                        std::make_unique<PatternRewriter>(Rules{std::move(rule)}, Options{.indexed = indexed}));
    }
  }
}

TEST(PatternRewriterTests, RmsConstantAndAttributeGuards) {
  for (int scenario = 0; scenario < 4; ++scenario) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({2, 4}, -1.0f, 1.0f);
    BuildRms(test.builder, x, test.builder.MakeOutput(), 4, scenario == 1 ? -1e-5f : 1e-5f, scenario == 2);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    if (scenario == 3) {
      for (auto& node : test.graph.Nodes()) {
        if (node.OpType() == "Pow") node.MutableInputDefs()[1] = x;
      }
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, NumericalDecoderPrefillAndDecode) {
  for (int64_t sequence : {1, 4}) {
    TestGraph test;
    BuildDecoder(test.builder, 2, sequence);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    std::string original;
    test.model.ToProto().SerializeToString(&original);
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
    ASSERT_STATUS_OK(test.graph.Resolve());
    EXPECT_EQ(Count(test.graph, "SimplifiedLayerNormalization"), 4u);
    EXPECT_EQ(Count(test.graph, "FusedMatMul"), 2u);
    EXPECT_EQ(stats.rewrites, 8u);
    std::string rewritten;
    test.model.ToProto().SerializeToString(&rewritten);
    auto execute = [&](const std::string& serialized, std::vector<OrtValue>& outputs) {
      SessionOptions options;
      options.graph_optimization_level = TransformerLevel::Default;
      options.intra_op_param.thread_pool_size = 1;
      InferenceSessionWrapper session{options, GetEnvironment()};
      std::istringstream stream(serialized);
      ORT_RETURN_IF_ERROR(session.Load(stream));
      ORT_RETURN_IF_ERROR(session.Initialize());
      return session.Run(RunOptions{}, test.builder.feeds_, test.builder.output_names_, &outputs);
    };
    std::vector<OrtValue> baseline, fused;
    ASSERT_STATUS_OK(execute(original, baseline));
    ASSERT_STATUS_OK(execute(rewritten, fused));
    ASSERT_EQ(baseline.size(), fused.size());
    float max_absolute = 0.0f, max_relative = 0.0f;
    for (size_t output = 0; output < baseline.size(); ++output) {
      const auto& a = baseline[output].Get<Tensor>();
      const auto& c = fused[output].Get<Tensor>();
      ASSERT_EQ(a.Shape(), c.Shape());
      for (int64_t i = 0; i < a.Shape().Size(); ++i) {
        float difference = std::abs(a.Data<float>()[i] - c.Data<float>()[i]);
        float magnitude = std::abs(a.Data<float>()[i]);
        max_absolute = std::max(max_absolute, difference);
        max_relative = std::max(max_relative, difference / std::max(magnitude, 1e-6f));
        ASSERT_LE(difference, 2e-5f + 2e-4f * magnitude);
      }
    }
    std::cout << "decoder sequence=" << sequence << " outputs=" << fused.size() << " rewrites=" << stats.rewrites
              << " max_absolute=" << max_absolute << " max_relative_floor_1e-6=" << max_relative << '\n';
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
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
  Pattern p("(x) => a = Identity(x); return Identity(a)");
  auto guard = [accept](const Graph&, const Match&) { return accept; };
  auto replacement = [](const Graph&, const Match& m) {
    return Replacement{"Identity", kOnnxDomain, {m.valueOf("x")}, {}};
  };
  Rule rule{name, p, {{kOnnxDomain, 1, 21}}, guard, replacement};
  return rule;
}

Rule IdentitySplit() {
  Pattern p;
  auto x = p.Input();
  auto first = p.Op("Identity", {x});
  p.MultiOp("Split", {first}, 2);
  auto guard = [](const Graph&, const Match&) { return true; };
  auto replacement = [x](const Graph&, const Match& match) {
    return Replacement{"Split", kOnnxDomain, {x}, match.nodes.back()->GetAttributes()};
  };
  Rule rule{"IdentitySplit", p, {{kOnnxDomain, 13, 17}}, guard, replacement};
  return rule;
}

Rule SplitConcat() {
  Pattern p("(x) => a, b = Split(x); return Concat(a, b)");
  auto guard = [](const Graph&, const Match& match) {
    const auto& split_attributes = match.nodes[0]->GetAttributes();
    auto axis = split_attributes.find("axis");
    return (axis == split_attributes.end() || axis->second.i() == 0) &&
           match.nodes[1]->GetAttributes().at("axis").i() == 0;
  };
  auto replacement = [](const Graph&, const Match& m) {
    return Replacement{"Identity", kOnnxDomain, {m.valueOf("x")}, {}};
  };
  Rule rule{"SplitConcat", p, {{kOnnxDomain, 13, 17}}, guard, replacement};
  return rule;
}

// Multi-output adaptation of ONNXScript's LayerNormBiasFusion; unused statistics may be removed.
Rule LayerNormBias() {
  Pattern p("(x, scale, bias) => y, mean, inv_std = LayerNormalization(x, scale); return Add(y, bias)");
  auto guard = [](const Graph&, const Match& match) {
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
    return width.has_dim_value() && width.dim_value() > 0 && scale_width.has_dim_value() &&
           bias_width.has_dim_value() && width.dim_value() == scale_width.dim_value() &&
           width.dim_value() == bias_width.dim_value() &&
           (axis == match.nodes[0]->GetAttributes().end() || axis->second.i() == -1);
  };
  auto replacement = [](const Graph&, const Match& match) {
    return Replacement{"LayerNormalization",
                       "",
                       {match.valueOf("x"), match.valueOf("scale"), match.valueOf("bias")},
                       match.nodes[0]->GetAttributes()};
  };
  Rule rule{"LayerNormBias", p, {{kOnnxDomain, 17, 17}}, guard, replacement};
  return rule;
}

// Small-weight eager decoder: B=1, Hq=4, Hkv=2, D=4. Cache length is 3.
// See the design document for the model/export sources and deliberate simplifications.
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

TEST(PatternRewriterTests, NamedValuesAndAttributePredicates) {
  Pattern p;
  auto x = p.Input("x");
  auto first = p.Op("HardSigmoid", {x});
  p.Name(first, "sigmoid");
  p.Attributes({AttributeMatcher::Equal("alpha", 0.25f),
                {"beta", [](const auto& attr) {
                   return attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT && attr.f() > 0.4f;
                 }}});
  p.Op("Mul", {first, x});
  EXPECT_EQ(x.name, "x");
  EXPECT_EQ(x.handle, 0);
  EXPECT_EQ(p.NamedValue("sigmoid").handle, first.handle);
  EXPECT_EQ(p.Nodes()[0].outputs[0].name, "sigmoid");
  EXPECT_EQ(p.Nodes()[1].inputs[0].name, "sigmoid");
  EXPECT_THROW(p.Input("x"), OnnxRuntimeException);
  EXPECT_THROW(p.NamedValue("missing"), OnnxRuntimeException);
  EXPECT_THROW(p.Name(x, "sigmoid"), OnnxRuntimeException);
  EXPECT_THROW(p.Op("Identity", {Value{-1}}), OnnxRuntimeException);
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
      second.pattern = Pattern("(input) => middle = Identity(input); return Identity(middle)");
      first.guard = [=](const Graph&, const Match& m) {
        EXPECT_EQ(m.valueOf("x"), x);
        EXPECT_EQ(m.valueOf("a"), intermediate);
        EXPECT_THROW(m.valueOf("input"), OnnxRuntimeException);
        EXPECT_THROW(m.valueOf(""), OnnxRuntimeException);
        return false;
      };
      second.guard = [=](const Graph&, const Match& m) {
        EXPECT_EQ(m.valueOf("input"), x);
        EXPECT_EQ(m.valueOf("middle"), intermediate);
        EXPECT_THROW(m.valueOf("x"), OnnxRuntimeException);
        return true;
      };
      second.replacement = [scenario, output](const Graph&, const Match& m) {
        const NodeArg* input = scenario == 0   ? m.valueOf("input")
                               : scenario == 1 ? m.valueOf("middle")
                               : scenario == 2 ? output
                                               : nullptr;
        return Replacement{"Identity", "", {input}, {}};
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
  Pattern p;
  auto x = p.Input("x");
  auto a = p.Op("Identity", {x});
  auto b = p.Op("Identity", {a});
  p.MultiOp("Split", {b}, 2);
  split.pattern = p;
  split.replacement = Replacement("(x) => a=Identity(x); return Split(a, axis=0)");
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
                    "(x)=>a=Identity(x);return Identity(a,axis=[0])", "(x)=>Identity(x);more",
                    "(x)=>Identity(x,axis=0,axis=1)", "(x)=>Identity(x,mode='bad)"})
    EXPECT_THROW(Replacement{text}, OnnxRuntimeException) << text;
  auto invalid = IdentityPair();
  invalid.replacement = Replacement("(missing)=>Identity(missing)");
  EXPECT_THROW(PatternRewriter({invalid}), OnnxRuntimeException);
  invalid.replacement = Replacement("(a)=>Identity(a)");
  EXPECT_THROW(PatternRewriter({invalid}), OnnxRuntimeException);
  invalid.replacement = Replacement("(x)=>a=Identity(x);return Identity(a)");
  EXPECT_THROW(PatternRewriter({invalid}), OnnxRuntimeException);
  invalid.replacement = Replacement("(x)=>com.microsoft::FusedMatMul(x,x)");
  EXPECT_THROW(PatternRewriter({invalid}), OnnxRuntimeException);
  EXPECT_NO_THROW(Replacement("(Identity)=>Identity(Identity)"));
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 9; ++scenario) {
      TestGraph test;
      BuildHardSwish(test.builder);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto initializer_count = test.graph.GetAllInitializedTensors().size();
      auto rule = MakeExampleRules()[0];
      InlinedVector<Replacement::NodeRecipe, 1> nodes{{"Identity", "", {std::string("x")}, {}, 1},
                                                      {"HardSwish", "", {Replacement::Output{0}}, {}, 1}};
      if (scenario == 0) nodes[1].op = "NoSuchReplacementOperator";
      if (scenario == 1) nodes[1].inputs = {Replacement::Output{1}};
      if (scenario == 2) nodes[1].inputs = {Replacement::Output{0, 1}};
      if (scenario == 3) nodes[1].inputs = {std::string("x")};
      if (scenario == 4) {
        ONNX_NAMESPACE::AttributeProto attr;
        attr.set_name("axis");
        attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
        attr.set_s("wrong");
        nodes[1].op = "Softmax";
        nodes[1].attributes.emplace("axis", attr);
      }
      if (scenario == 5) nodes[1].domain = kMSDomain;
      if (scenario == 6) nodes[0].output_count = 0;
      if (scenario == 7) {
        nodes[0].op = "Add";
        nodes[0].inputs.push_back(Replacement::Scalar{0.0f});
        nodes[1].op = "NoSuchReplacementOperator";
      }
      rule.replacement = [recipe = Replacement(std::move(nodes))](const Graph&, const Match&) { return recipe; };
      if (scenario == 8) {
        rule.replacement = [](const Graph&, const Match&) {
          Replacement recipe("(x)=>HardSwish(x)");
          recipe.op = "Identity";
          return recipe;
        };
      }
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

TEST(PatternRewriterTests, ReplacementScalarConstantsAndProgrammaticSequence) {
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
  rule.replacement = Replacement(InlinedVector<Replacement::NodeRecipe, 1>{
      {"Identity", "", {std::string("x")}, {}, 1}, {"Identity", "", {Replacement::Output{0}}, {}, 1}});
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
    second.guard = [](const Graph&, const Match&) { return true; };
    second.replacement = Replacement("(second)=>a=Identity(second);return Identity(a)");
    auto followup = IdentityPair();
    followup.guard = [](const Graph&, const Match& m) { return m.nodes.back()->Name().find("rewrite") == 0; };
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
      return Replacement("(second)=>a=Identity(second);b=Identity(a);c=Identity(b);return Identity(c)");
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
    EXPECT_EQ(test.graph.NumberOfNodes(), 4u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, ReplacementExtraAttributeBindings) {
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
  EXPECT_THROW(Pattern{"(x,a)=>return HardSigmoid(x,alpha=a)"}, OnnxRuntimeException);
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
      // Deliberate guard misuse verifies that rejected candidates cannot leak updates.
      first.guard = [alpha](const Graph&, const Match& m) {
        m.update("alpha", alpha);
        return false;
      };
      Replacement plan("(x,alpha,beta)=>h=HardSigmoid(x,alpha=alpha,beta=beta);return Mul(h,x)");
      second.replacement = [plan, scenario, make_attribute](const Graph&, const Match& m) {
        auto beta = make_attribute(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
        beta.set_f(0.5f);
        m.update("beta", beta);
        if (scenario == 1) return plan;  // alpha is missing; must not leak from the rejected rule.
        if (scenario == 2) m.update("x", beta);
        auto alpha_value = make_attribute(scenario == 3 ? ONNX_NAMESPACE::AttributeProto_AttributeType_INT
                                                        : ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
        alpha_value.set_f(1.0f / 6.0f);
        m.update("alpha", alpha_value);
        if (scenario == 4) {
          auto malformed = plan;
          malformed.attributes.emplace("extra", beta);
          return malformed;
        }
        return plan;
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
    Replacement plan("(x,axis)=>Softmax(x,axis=axis)");
    int calls = 0;
    rule.replacement = [plan, &calls](const Graph&, const Match& m) {
      ONNX_NAMESPACE::AttributeProto axis;
      axis.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
      axis.set_i(calls++ == 0 ? 0 : 1);
      m.update("axis", axis);
      return plan;
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
      EXPECT_EQ(calls, repeat);
      EXPECT_EQ(test.graph.NumberOfNodes(), 3u);
      EXPECT_EQ(stats.matches, 1u);
      ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      EXPECT_EQ(test.graph.Nodes().begin()->GetAttributes().at("axis").i(), repeat);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }

  auto rule = IdentityPair();
  rule.pattern = Pattern("(x)=>a=Identity(x);b=Identity(a);return Identity(b)");
  Replacement plan("(x,perm)=>Transpose(x,perm=perm)");
  auto replacement = [plan](const Graph&, const Match& m) {
    ONNX_NAMESPACE::AttributeProto perm;
    perm.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    perm.add_ints(0);
    perm.add_ints(1);
    m.update("perm", perm);
    return plan;
  };
  rule.replacement = replacement;
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
  auto semantic_guard = [type_guard = MakeExampleRules()[0].guard](const Graph& graph, const Match& match) {
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
      rule.guard = semantic_guard;
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
  auto rule = MakeExampleRules()[0];
  rule.pattern = parsed;
  rule.guard = semantic_guard;
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
        "(x)=>a=Identity(x);return Identity(a, axis=[1])", "(x,y)=>return Identity(x)",
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
  Pattern p;
  auto integer = p.Constant(3);
  auto floating = p.Constant(3.0);
  EXPECT_TRUE(std::holds_alternative<int64_t>(p.Constants().at(integer)));
  EXPECT_TRUE(std::holds_alternative<float>(p.Constants().at(floating)));
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
        rule.guard = [&guards](const Graph&, const Match&) {
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
    a.guard = [&first](const Graph&, const Match&) {
      ++first;
      return true;
    };
    c.guard = [&second](const Graph&, const Match&) {
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

TEST(PatternRewriterTests, InvalidRuleAndRecipe) {
  Pattern disconnected;
  auto x = disconnected.Input();
  disconnected.Op({"Identity", ""}, {x});
  disconnected.Op({"Identity", ""}, {x});
  auto rule = IdentityPair();
  rule.pattern = disconnected;
  EXPECT_THROW(PatternRewriter({rule}), OnnxRuntimeException);
  TestGraph test;
  auto* input = test.builder.MakeInput<float>({1}, -1.0f, 1.0f);
  auto* a = test.builder.MakeIntermediate();
  test.builder.AddNode("Identity", {input}, {a});
  test.builder.AddNode("Identity", {a}, {test.builder.MakeOutput()});
  test.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  rule = IdentityPair();
  for (int scenario = 0; scenario < 5; ++scenario) {
    rule.replacement = [scenario](const Graph&, const Match&) {
      Replacement recipe{"Identity", "", {0}, {}};
      if (scenario == 0) recipe.op = "NoSuchReplacementOperator";
      if (scenario == 1) recipe.domain = "not.imported";
      if (scenario == 2) recipe.inputs = {1};  // Deleted intermediate, not an external input.
      if (scenario == 3) {
        ONNX_NAMESPACE::AttributeProto attr;
        attr.set_name("invalid");
        attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        attr.set_i(1);
        recipe.attributes.emplace("invalid", attr);
      }
      if (scenario == 4) recipe.domain = kMSDomain;  // Imported, but not declared by this rule.
      return recipe;
    };
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
    Pattern pattern;
    auto input = pattern.Input();
    auto first = pattern.Op({"Identity", kOnnxDomainAlias}, {input});
    pattern.Op({"Identity", kOnnxDomainAlias}, {first});
    rule.pattern = std::move(pattern);
    int guards = 0;
    rule.guard = [&guards](const Graph&, const Match&) {
      ++guards;
      return true;
    };
    rule.replacement = [](const Graph&, const Match&) { return Replacement{"Identity", kOnnxDomainAlias, {0}, {}}; };
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
      inactive.guard = [&inactive_guards](const Graph&, const Match&) {
        ++inactive_guards;
        return true;
      };
      active.guard = [&active_guards](const Graph&, const Match&) {
        ++active_guards;
        return true;
      };
      later.guard = [&later_guards](const Graph&, const Match&) {
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
      rule.guard = [&guards](const Graph&, const Match&) {
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
    lower_priority.guard = [](const Graph& graph, const Match& m) {
      return graph.NodeProducesGraphOutput(*m.nodes.back());
    };
    Pattern p;
    auto input = p.Input();
    auto first = p.Op({"Identity", ""}, {input});
    auto second = p.Op({"Identity", ""}, {first});
    p.Op({"Identity", ""}, {second});
    auto guard = [](const Graph&, const Match&) { return true; };
    auto replacement = [input](const Graph&, const Match&) { return Replacement{"Identity", "", {input}, {}}; };
    Rule higher_priority{"higher_priority", p, {{kOnnxDomain, 1, 21}}, guard, replacement};
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
  Pattern p;
  auto input = p.Input();
  auto identity = p.Op({"Identity", ""}, {input});
  p.Op({"Dropout", ""}, {identity});
  auto rule = IdentityPair();
  rule.pattern = p;
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter({rule}).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
}

TEST(PatternRewriterTests, MultiOpDefinitionsAndOverloads) {
  Pattern p;
  auto x = p.Input();
  std::string identity = "Identity";
  auto a = p.Op(identity, {x});
  auto b = p.Op(std::string_view(identity), {a}, kOnnxDomainAlias);
  auto outputs = p.MultiOp({"Split", kOnnxDomainAlias}, {b}, 2);
  EXPECT_EQ(outputs.size(), 2u);
  EXPECT_NE(outputs[0], outputs[1]);
  EXPECT_TRUE(p.IsInput(x));
  EXPECT_FALSE(p.IsInput(outputs[0]));
  EXPECT_FALSE(p.IsInput(outputs[1]));
  EXPECT_EQ(p.Nodes().back().op.domain, kOnnxDomain);
  const auto count = p.ValueCount();
  EXPECT_THROW(p.MultiOp("Split", {b}, 0), OnnxRuntimeException);
  EXPECT_THROW(p.MultiOp("Split", {b}, 65), OnnxRuntimeException);
  EXPECT_THROW(p.MultiOp("Split", {999}, 2), OnnxRuntimeException);
  EXPECT_EQ(p.ValueCount(), count);
  for (int i = 0; i < 3; ++i) p.MultiOp("Split", {b}, 64);
  EXPECT_THROW(p.MultiOp("Split", {b}, 64), OnnxRuntimeException);
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
    Pattern pattern;
    auto input = pattern.Input();
    auto outputs = pattern.MultiOp("Split", {input}, 2);
    pattern.Op("Identity", {outputs[1]});
    auto rule = SplitConcat();
    rule.pattern = pattern;
    rule.guard = [](const Graph&, const Match&) { return true; };
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
    auto invalid = IdentitySplit();
    invalid.replacement = [](const Graph&, const Match&) { return Replacement{"Identity", "", {0}, {}}; };
    Statistics stats;
    EXPECT_FALSE(PatternRewriter({invalid}, {.indexed = indexed}).Run(test.graph, stats).IsOK());
    EXPECT_EQ(test.graph.NumberOfNodes(), 4);
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto wrong_count = IdentityPair("single_output_split");
    Pattern single_output;
    auto input = single_output.Input();
    auto first = single_output.Op("Identity", {input});
    single_output.Op("Split", {first});
    wrong_count.pattern = single_output;
    ASSERT_STATUS_OK(PatternRewriter({wrong_count, IdentitySplit()}, {.indexed = indexed}).Run(test.graph, stats));
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
    TestGraph test;
    auto& b = test.builder;
    auto* non_float = b.MakeScalarInitializer<double>(2.0);
    auto* integer = b.MakeScalarInitializer<int64_t>(2);
    auto* unknown = b.MakeIntermediate();
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
    auto guard = rule.guard;
    size_t checked = 0;
    rule.guard = [&](const Graph& graph, const Match& match) {
      EXPECT_TRUE(guard(graph, match));
      for (auto binding : bindings) {
        SCOPED_TRACE(binding);
        // Exercise the guard independently without making the resolved graph ill-typed.
        for (const NodeArg* value : {non_float, integer, unknown}) {
          Match changed = match;
          changed.values[match.pattern->NamedValue(binding).handle] = value;
          EXPECT_FALSE(guard(graph, changed));
        }
      }
      ++checked;
      return guard(graph, match);
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
    Pattern p;
    auto px = p.Input(), py = p.Input();
    auto psum = p.Op({"Add", ""}, {px, py});
    auto pproduct = p.Op({"Mul", ""}, {psum, px});
    p.Op({"Add", ""}, {psum, pproduct});
    auto guard = [](const Graph&, const Match&) { return true; };
    auto replacement = [](const Graph&, const Match&) { return Replacement{"Identity", "", {0}, {}}; };
    Rule rule{"diamond", p, {{kOnnxDomain, 7, 21}}, guard, replacement};
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

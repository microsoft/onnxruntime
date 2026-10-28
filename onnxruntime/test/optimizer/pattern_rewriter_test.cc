// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
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

#ifndef DISABLE_CONTRIB_OPS
ONNX_NAMESPACE::AttributeProto FloatAttribute(const char* name, float value) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  attr.set_f(value);
  return attr;
}
#endif

bool IsFloat32(const NodeArg& value) {
  const auto* type = value.TypeAsProto();
  return type && type->has_tensor_type() &&
         type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}

#ifndef DISABLE_CONTRIB_OPS
std::optional<float> Scalar(const Graph& graph, const NodeArg& value) {
  const auto* tensor = graph_utils::GetConstantInitializer(graph, value.Name());
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL || tensor->dims_size() != 0)
    return std::nullopt;
  Initializer initializer(graph, *tensor, graph.ModelPath());
  if (initializer.size() != 1) return std::nullopt;
  float scalar = *initializer.data<float>();
  if (!std::isfinite(scalar)) return std::nullopt;
  return scalar;
}

bool Integer(const Node& node, const char* name, int64_t value) {
  auto it = node.GetAttributes().find(name);
  return it != node.GetAttributes().end() && it->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT &&
         it->second.i() == value;
}
#endif

Rules MakeExampleRules() {
  Rules rules;
  {
    const auto* pattern = R"(
      (x) => {
        add = Add(x, 3.0)
        clip = Clip(add, 0.0, 6.0)
        mul = Mul(clip, x)
        return Div(mul, 6.0)
      }
    )";
    auto check_and_bind = [](const Graph& graph, Match& m) {
      if (!IsFloat32(*m.valueOf("x"))) return false;
      const auto* producer = graph.GetProducerNode(m.valueOf("x")->Name());
      if (!producer || producer->OpType() != "Clip" || producer->Domain() != kOnnxDomain ||
          producer->InputDefs().size() != 3)
        return m.Reject("The approximate Clip-chain example requires an upstream Clip to [-6,6].");
      for (size_t i = 1; i < 3; ++i) {
        const auto* bound = producer->InputDefs()[i];
        if (!bound || !bound->Shape() || bound->Shape()->dim_size() != 0) return false;
        InlinedVector<float> data;
        bool available = false;
        ORT_THROW_IF_ERROR(ReadConstantTensor(graph, bound, data, available));
        if (!available || data.size() != 1 || data[0] != (i == 1 ? -6.0f : 6.0f)) return false;
      }
      return true;
    };
    const auto* replacement = "(x) => HardSwish(x)";
    Rule rule{"HardSwish", pattern, replacement, check_and_bind, {{kOnnxDomain, 14, 21}}};
    rules.push_back(std::move(rule));
  }
  {
    const auto* pattern = R"(
      (x) => {
        sigmoid = HardSigmoid(x, alpha=0.16666667, beta=0.5)
        return Mul(sigmoid, x)
      }
    )";
    auto check_and_bind = [](const Graph&, Match& m) { return IsFloat32(*m.valueOf("x")); };
    const auto* replacement = "(x) => HardSwish(x)";
    Rule rule{"HardSigmoidMul", pattern, replacement, check_and_bind, {{kOnnxDomain, 14, 21}}};
    rules.push_back(std::move(rule));
  }
#ifndef DISABLE_CONTRIB_OPS
  {
    const auto* pattern = "(x, y, divisor) => { mm = MatMul(x, y); return Div(mm, divisor); }";
    auto check_and_bind = [](const Graph& graph, Match& m) {
      auto scalar = Scalar(graph, *m.valueOf("divisor"));
      if (!IsFloat32(*m.valueOf("x")) || !IsFloat32(*m.valueOf("y")) ||
          !scalar || *scalar == 0.0f || !std::isfinite(1.0f / *scalar))
        return false;
      m.update("alpha", FloatAttribute("alpha", 1.0f / *scalar));
      return true;
    };
    const auto* replacement = "(x, y, alpha) => com.microsoft.FusedMatMul(x, y, alpha=alpha)";
    Rule rule{"MatMulDiv", pattern, replacement, check_and_bind, {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}};
    rules.push_back(std::move(rule));
  }
  {
    const auto* pattern = "(x, y) => { transpose = Transpose(y); return MatMul(x, transpose); }";
    auto check_and_bind = [](const Graph&, Match& m) {
      if (!IsFloat32(*m.valueOf("x")) || !IsFloat32(*m.valueOf("y"))) return false;
      const auto* shape = m.valueOf("y")->Shape();
      if (!shape || shape->dim_size() < 2) return false;
      auto it = m.nodes[0]->GetAttributes().find("perm");
      if (it == m.nodes[0]->GetAttributes().end()) return shape->dim_size() == 2;
      const auto& perm = it->second;
      if (perm.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INTS ||
          perm.ints_size() != shape->dim_size())
        return false;
      int rank = shape->dim_size();
      for (int i = 0; i < rank; ++i) {
        int expected = i < rank - 2 ? i : (i == rank - 2 ? rank - 1 : rank - 2);
        if (perm.ints(i) != expected) return false;
      }
      return true;
    };
    const auto* replacement = "(x,y) => com.microsoft.FusedMatMul(x,y,transB=1)";
    Rule rule{"TransposeMatMul", pattern, replacement, check_and_bind, {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}};
    rules.push_back(std::move(rule));
  }
  {
    const auto* pattern = "(x, y, divisor) => { mm = com.microsoft::FusedMatMul(x, y); return Div(mm, divisor); }";
    auto check_and_bind = [](const Graph& graph, Match& m) {
      auto scalar = Scalar(graph, *m.valueOf("divisor"));
      const auto& node = *m.nodes[0];
      const auto& attributes = node.GetAttributes();
      auto it = attributes.find("alpha");
      float alpha = it == attributes.end() ? 1.0f : it->second.f();
      if (!IsFloat32(*m.valueOf("x")) || !IsFloat32(*m.valueOf("y")) ||
          !scalar || *scalar == 0.0f || !std::isfinite(alpha / *scalar))
        return false;
      m.update("alpha", FloatAttribute("alpha", alpha / *scalar));
      for (const char* name : {"transA", "transB", "transBatchA", "transBatchB"}) {
        auto attribute = attributes.find(name);
        m.update(name, attribute == attributes.end() ? node.Op()->attributes().at(name).default_value
                                                     : attribute->second);
      }
      return true;
    };
    const auto* replacement = R"(
      (x, y, alpha, transA, transB, transBatchA, transBatchB) =>
        com.microsoft.FusedMatMul(x, y, alpha=alpha, transA=transA, transB=transB,
                                 transBatchA=transBatchA, transBatchB=transBatchB)
    )";
    Rule rule{"FusedMatMulDiv", pattern, replacement, check_and_bind, {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}};
    rules.push_back(std::move(rule));
  }
  {
    const auto* pattern = R"(
      (x, scale, epsilon) => {
        square = Pow(x, 2.0)
        mean = ReduceMean(square)
        add = Add(mean, epsilon)
        sqrt = Sqrt(add)
        reciprocal = Reciprocal(sqrt)
        normalize = Mul(x, reciprocal)
        return Mul(normalize, scale)
      }
    )";
    auto check_and_bind = [](const Graph& graph, Match& m) {
      if (!IsFloat32(*m.valueOf("x")) || !IsFloat32(*m.valueOf("scale"))) return false;
      auto eps = Scalar(graph, *m.valueOf("epsilon"));
      const auto* shape = m.valueOf("x")->Shape();
      const auto* scale_shape = m.valueOf("scale")->Shape();
      if (!eps || *eps <= 0 || !shape || shape->dim_size() < 1 || !scale_shape || scale_shape->dim_size() != 1)
        return false;
      const auto& dim = shape->dim(shape->dim_size() - 1);
      const auto& scale_dim = scale_shape->dim(0);
      if (!dim.has_dim_value() || !scale_dim.has_dim_value() || dim.dim_value() <= 0 ||
          dim.dim_value() != scale_dim.dim_value())
        return false;
      const auto& mean_node = *m.nodes[1];
      auto axes = mean_node.GetAttributes().find("axes");
      if (!Integer(mean_node, "keepdims", 1) || axes == mean_node.GetAttributes().end() ||
          axes->second.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INTS ||
          axes->second.ints_size() != 1 || axes->second.ints(0) != -1)
        return false;
      m.update("fused_epsilon", FloatAttribute("epsilon", *eps));
      return true;
    };
    const auto* replacement = R"(
      (x, scale, fused_epsilon) =>
        SimplifiedLayerNormalization(x, scale, axis=-1, epsilon=fused_epsilon, stash_type=1)
    )";
    Rule rule{"RmsNormalization", pattern, replacement, check_and_bind, {{kOnnxDomain, 7, 17}}};
    rules.push_back(std::move(rule));
  }
#endif
  return rules;
}

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
                    bool external_consumer = false, bool bounded = true) {
  auto* input = b.MakeInput<float>({2, 4}, -5.0f, 5.0f);
  auto* x = input;
  if (bounded) {
    x = b.MakeIntermediate();
    b.AddNode("Clip", {input, b.MakeScalarInitializer<float>(-6.0f), b.MakeScalarInitializer<float>(6.0f)}, {x});
  }
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
    if (scenario == 5)
      for (auto& node : test.graph.Nodes())
        if (node.OpType() == "Add") node.SetExecutionProviderType(kCudaExecutionProvider);
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
                      EXPECT_EQ(Count(s.GetGraph(), "Clip"), 1u);
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

TEST(PatternRewriterTests, ClipChainHardSwishRequiresRangeProofAndPreservesOverflowWithoutIt) {
  const std::vector<float> values{
      0x1p126f, -0x1p126f, std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
      -6.0f, std::nextafter(-3.0f, -4.0f), -3.0f, std::nextafter(-3.0f, 0.0f),
      -1.0f, -0.0f, 0.0f, 1.0f, std::nextafter(3.0f, 0.0f), 3.0f,
      std::nextafter(3.0f, 4.0f), 6.0f};
  for (bool indexed : {false, true}) {
    for (bool bounded : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* input = b.MakeInput<float>({static_cast<int64_t>(values.size())}, values);
      auto* x = input;
      if (bounded) {
        x = b.MakeIntermediate();
        b.AddNode("Clip", {input, b.MakeScalarInitializer<float>(-6.0f), b.MakeScalarInitializer<float>(6.0f)}, {x});
      }
      auto* add = b.MakeIntermediate();
      auto* clip = b.MakeIntermediate();
      auto* mul = b.MakeIntermediate();
      b.AddNode("Add", {x, b.MakeScalarInitializer<float>(3.0f)}, {add});
      b.AddNode("Clip", {add, b.MakeScalarInitializer<float>(0.0f), b.MakeScalarInitializer<float>(6.0f)}, {clip});
      b.AddNode("Mul", {clip, x}, {mul});
      b.AddNode("Div", {mul, b.MakeScalarInitializer<float>(6.0f)}, {b.MakeOutput()});
      b.AddNode("HardSwish", {x}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      const auto original = test.model.ToProto().SerializeAsString();
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({MakeExampleRules()[0]}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, bounded ? 1u : 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
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
      ASSERT_EQ(before.size(), 2u);
      ASSERT_EQ(after.size(), 2u);
      const auto* source = before[0].Get<Tensor>().Data<float>();
      const auto* target = after[0].Get<Tensor>().Data<float>();
      const auto* hard_swish = before[1].Get<Tensor>().Data<float>();
      if (!bounded) {
        EXPECT_TRUE(std::isinf(source[0]) && source[0] > 0);
        EXPECT_EQ(hard_swish[0], values[0]);
        EXPECT_TRUE(std::isfinite(hard_swish[0]));
      }
      for (size_t i = 0; i < values.size(); ++i) {
        if (std::isinf(source[i])) {
          EXPECT_EQ(target[i], source[i]);
        } else {
          EXPECT_NEAR(target[i], source[i], 1e-6f + 1e-5f * std::abs(source[i]));
          if (bounded) EXPECT_NEAR(target[i], hard_swish[i], 1e-6f + 1e-5f * std::abs(hard_swish[i]));
        }
      }
      if (!bounded) EXPECT_EQ(rewritten, original);
    }
  }
}

TEST(PatternRewriterTests, ClipChainRangeProofRejectsUnboundedOrUnreliableBounds) {
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      BuildHardSwish(test.builder, false, false, 6.0f, false, scenario != 0);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      if (scenario != 0) {
        auto& bound_clip = *test.graph.Nodes().begin();
        auto* upper = bound_clip.MutableInputDefs()[2];
        if (scenario == 1) {
          auto tensor = *test.graph.GetInitializer(upper->Name(), false);
          tensor.clear_raw_data();
          tensor.clear_float_data();
          tensor.add_float_data(7.0f);
          test.graph.RemoveInitializedTensor(upper->Name());
          test.graph.AddInitializedTensor(tensor);
        } else if (scenario == 2) {
          auto inputs = test.graph.GetInputsIncludingInitializers();
          inputs.push_back(upper);
          test.graph.SetInputs(inputs);
        } else {
          bound_clip.MutableInputDefs().resize(1);
        }
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
      const auto before = test.model.ToProto().SerializeAsString();
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({MakeExampleRules()[0]}, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_EQ(test.model.ToProto().SerializeAsString(), before);
    }
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
    EXPECT_EQ(test.graph.NumberOfNodes(), 3u);
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
                      EXPECT_EQ(Count(session.GetGraph(), "Clip"), 1u);
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
      EXPECT_EQ(test.graph.NumberOfNodes(), 5u);
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
    auto* compiled = rule.replacement.Parsed();
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
      EXPECT_EQ(test.graph.NumberOfNodes(), scenario == 0 ? 3u : 5u);
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
                      EXPECT_EQ(Count(session.GetGraph(), "Clip"), 1u);
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
          EXPECT_EQ(test.graph.NumberOfNodes(), 5u);
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
      EXPECT_EQ(test.graph.NumberOfNodes(), 5u);
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
  const auto* compiled = rule.replacement.Parsed();
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
  for (auto text : {"(None)=>return Identity(None)",
                    "(x?"
                    "?)=>return Identity(x)",
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
      const bool expected = scenario == 0 || scenario == 2;
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
    ASSERT_NE(examples[index].replacement.Parsed(), nullptr);
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

#ifndef DISABLE_CONTRIB_OPS
namespace {

const char* ActivationLibrary = R"(
  interface GatedActivation(g:float[M,N], u:float[M,N]) -> (product:float[M,N])
  pattern plain implements GatedActivation {
    s = Sigmoid(g)
    a = Mul(g,s)
    p = Mul(a,u)
    return (product=p)
  }
  pattern scaled implements GatedActivation: (alpha:float[...]) -> {
    scaled = Mul(g,alpha)
    s = Sigmoid(scaled)
    a = Mul(g,s)
    p = Mul(a,u)
    return (product=p)
  }
  pattern Forms = plain | scaled
  pattern wrapped implements GatedActivation {
    act = Forms(g,u)
    return (product=act.product)
  }
  pattern Fuse(g,u) {
    outer = wrapped(g,u)
    return Identity(outer.product)
  }
)";

// Handwritten views illustrate a future generator's common/concrete/nested API.
struct GatedView {
  const NodeArg* g;
  const NodeArg* u;
  const NodeArg* product;
  explicit GatedView(BindingView view)
      : g(view.valueOf("g")), u(view.valueOf("u")), product(view.valueOf("product")) {
    ORT_ENFORCE(view.interfaceName() == "GatedActivation");
  }
};

struct PlainView {
  GatedView ports;
  const NodeArg* sigmoid;
  explicit PlainView(BindingView view) : ports(view), sigmoid(view.valueOf("s")) {}
};

struct ScaledView {
  GatedView ports;
  const NodeArg* alpha;
  const NodeArg* scaled;
  explicit ScaledView(BindingView view) : ports(view), alpha(view.valueOf("alpha")), scaled(view.valueOf("scaled")) {}
};

struct ActivationView {
  GatedView ports;
  std::variant<PlainView, ScaledView> alternative;
  static std::variant<PlainView, ScaledView> Select(BindingView view) {
    if (view.alternative() == "plain") return PlainView(view.alternativeView());
    ORT_ENFORCE(view.alternative() == "scaled", "Unexpected activation member.");
    return ScaledView(view.alternativeView());
  }
  explicit ActivationView(BindingView view) : ports(view), alternative(Select(view)) {}
};

struct FusionView {
  GatedView outer;
  ActivationView act;
  explicit FusionView(const Match& match)
      : outer(match.instance("outer")), act(match.instance("outer").instance("act")) {}
};

bool CheckCommon(const GatedView& view) {
  if (!view.g || !view.u || !view.product || !view.g->Shape() || !view.u->Shape() ||
      view.g->Shape()->dim_size() != 2 || view.u->Shape()->dim_size() != 2)
    return false;
  for (int i = 0; i < 2; ++i) {
    const auto& g = view.g->Shape()->dim(i);
    const auto& u = view.u->Shape()->dim(i);
    if (!g.has_dim_value() || !u.has_dim_value() || g.dim_value() <= 0 || g.dim_value() != u.dim_value()) return false;
  }
  return true;
}

Rule ActivationRule(const PatternLibrary& library, std::string_view entry = "Fuse") {
  auto check = [](const Graph& graph, Match& match) {
    const FusionView view(match);
    if (!CheckCommon(view.outer) || !CheckCommon(view.act.ports)) return match.Reject("Incompatible gating shapes.");
    float alpha = 1.0f;
    if (const auto* scaled = std::get_if<ScaledView>(&view.act.alternative)) {
      const auto* shape = scaled->alpha->Shape();
      if (!shape || !(shape->dim_size() == 0 ||
                      (shape->dim_size() == 1 && shape->dim(0).has_dim_value() && shape->dim(0).dim_value() == 1)))
        return match.Reject("Alpha must have scalar or [1] shape.");
      InlinedVector<float> values;
      bool available;
      ORT_THROW_IF_ERROR(ReadConstantTensor(graph, scaled->alpha, values, available));
      if (!available || values.size() != 1 || !std::isfinite(values[0]) || values[0] <= 0)
        return match.Reject("Alpha must be a supported immutable float.");
      alpha = values[0];
    }
    ONNX_NAMESPACE::AttributeProto attribute;
    attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    attribute.set_f(alpha);
    match.update("activation_alpha", std::move(attribute));
    return true;
  };
  Replacement replacement(R"(
    (g,u,activation_alpha) => {
      a = com.microsoft.QuickGelu(g,alpha=activation_alpha)
      return Mul(a,u)
    }
  )");
  return Rule{"PatternCompositionActivation", library.GetPattern(entry), replacement, check, {{kOnnxDomain, 17, 17}, {kMSDomain, 1, 1}}};
}

}  // namespace

TEST(PatternRewriterTests, PatternCompositionManualViewsAndSyntheticFusion) {
  const PatternLibrary library(ActivationLibrary);
  for (bool indexed : {false, true}) {
    for (int variant = 0; variant < 5; ++variant) {
      TestGraph test;
      auto& b = test.builder;
      auto* g = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
      auto* u = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
      NodeArg* scaled = g;
      if (variant != 0) {
        auto* alpha = variant == 1   ? b.MakeScalarInitializer<float>(1.5f)
                      : variant == 2 ? b.MakeInitializer<float>({1}, {0.5f})
                      : variant == 3 ? b.MakeInput<float>({1}, 1.0f, 2.0f)
                                     : b.MakeInitializer<float>({2, 4}, 0.5f, 1.0f);
        scaled = b.MakeIntermediate();
        b.AddNode("Mul", {g, alpha}, {scaled});
      }
      auto* sigmoid = b.MakeIntermediate();
      auto* activated = b.MakeIntermediate();
      auto* product = b.MakeIntermediate();
      b.AddNode("Sigmoid", {scaled}, {sigmoid});
      b.AddNode("Mul", {g, sigmoid}, {activated});
      b.AddNode("Mul", {activated, u}, {product});
      b.AddNode("Identity", {product}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      std::string original;
      test.model.ToProto().SerializeToString(&original);
      const bool accepted = variant < 3;
      Options options;
      options.indexed = indexed;
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({ActivationRule(library)}, options).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, accepted ? 1u : 0u);
      EXPECT_EQ(Count(test.graph, "QuickGelu"), accepted ? 1u : 0u);
      if (accepted) {
        for (const auto& node : test.graph.Nodes())
          if (node.OpType() == "QuickGelu")
            EXPECT_EQ(node.GetAttributes().at("alpha").f(), variant == 0 ? 1.0f : variant == 1 ? 1.5f
                                                                                               : 0.5f);
      }
      ASSERT_STATUS_OK(test.graph.Resolve());
      if (accepted) {
        std::string rewritten;
        test.model.ToProto().SerializeToString(&rewritten);
        auto execute = [&](const std::string& serialized, std::vector<OrtValue>& outputs) {
          SessionOptions session_options;
          session_options.graph_optimization_level = TransformerLevel::Default;
          session_options.intra_op_param.thread_pool_size = 1;
          InferenceSessionWrapper session{session_options, GetEnvironment()};
          std::istringstream stream(serialized);
          ORT_RETURN_IF_ERROR(session.Load(stream));
          ORT_RETURN_IF_ERROR(session.Initialize());
          return session.Run(RunOptions{}, b.feeds_, b.output_names_, &outputs);
        };
        std::vector<OrtValue> baseline, fused;
        ASSERT_STATUS_OK(execute(original, baseline));
        ASSERT_STATUS_OK(execute(rewritten, fused));
        ASSERT_EQ(baseline.size(), 1u);
        ASSERT_EQ(fused.size(), 1u);
        const auto& expected = baseline[0].Get<Tensor>();
        const auto& actual = fused[0].Get<Tensor>();
        ASSERT_EQ(expected.Shape(), actual.Shape());
        for (int64_t i = 0; i < expected.Shape().Size(); ++i)
          EXPECT_NEAR(expected.Data<float>()[i], actual.Data<float>()[i], 1e-6f);
      }
    }
  }
}
#endif

TEST(PatternRewriterTests, PatternCompositionNamedAndAnonymousAlternativesAreOccurrenceLocal) {
  for (const char* call : {"Forms", "(first | second)"}) {
    const std::string text = std::string(R"(
      interface Unary(x:tensor<T>[N]) -> (y:tensor<T>[N])
      pattern first implements Unary { local=Identity(x); return (y=local) }
      pattern second implements Unary { local=Identity(x); return (y=local) }
      pattern Forms = first | second
      pattern P(x,z) { a=)") +
                             call + "(x); b=" + call + R"((z); return Add(a.y,b.y) }
    )";
    const PatternLibrary library(text);
    for (bool indexed : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, 0.0f, 1.0f);
      auto* z = b.MakeInput<float>({2}, 0.0f, 1.0f);
      auto* a = b.MakeIntermediate();
      auto* c = b.MakeIntermediate();
      b.AddNode("Identity", {x}, {a});
      b.AddNode("Identity", {z}, {c});
      b.AddNode("Add", {a, c}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      InlinedVector<std::string> attempts;
      auto check = [&](const Graph&, Match& match) {
        auto av = match.instance("a"), bv = match.instance("b");
        attempts.push_back(std::string(av.alternative()) + "/" + std::string(bv.alternative()));
        EXPECT_EQ(av.valueOf("x"), x);
        EXPECT_EQ(bv.valueOf("x"), z);
        EXPECT_EQ(av.valueOf("y"), av.valueOf("local"));
        EXPECT_EQ(bv.valueOf("y"), bv.valueOf("local"));
        EXPECT_NE(av.valueOf("local"), bv.valueOf("local"));
        EXPECT_EQ(std::get<int64_t>(av.dimensionOf("N")), 2);
        EXPECT_EQ(av.elementTypeOf("T"), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        EXPECT_THROW(match.attributeOf("leaked"), OnnxRuntimeException);
        EXPECT_THROW(match.instance("absent"), OnnxRuntimeException);
        if (av.alternative() == "first" && bv.alternative() == "second") return true;
        ONNX_NAMESPACE::AttributeProto leaked;
        leaked.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        leaked.set_i(99);
        match.update("leaked", std::move(leaked));
        return match.Reject("Try the next complete candidate.");
      };
      Options options;
      options.indexed = indexed;
      Statistics stats;
      Rule rule{"Alternatives", library.GetPattern("P"), "(x,z) => Add(x,z)", check, {{kOnnxDomain, 17, 17}}};
      ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      EXPECT_EQ(attempts, (InlinedVector<std::string>{"first/first", "first/second"}));
      EXPECT_EQ(Count(test.graph, "Identity"), 0u);
    }
  }
}

TEST(PatternRewriterTests, PatternCompositionHygienicDimensionsAttributesAndCommonOutputs) {
  PatternLibrary library(R"(
    interface Permute(x:float[N,M]) -> (y:float[M,N])
    pattern transpose implements Permute: (axes:ints) -> {
      local=Transpose(x,perm=axes)
      return (y=local)
    }
    pattern P(x,z) {
      a=transpose(x)
      b=transpose(z)
      return MatMul(a.y,b.y)
    }
  )");
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({4, 3}, 0.0f, 1.0f);
  auto* z = b.MakeInput<float>({4, 4}, 0.0f, 1.0f);
  auto* a = b.MakeIntermediate();
  auto* c = b.MakeIntermediate();
  b.AddNode("Transpose", {x}, {a}).AddAttribute("perm", std::vector<int64_t>{1, 0});
  b.AddNode("Transpose", {z}, {c}).AddAttribute("perm", std::vector<int64_t>{0, 1});
  b.AddNode("MatMul", {a, c}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  auto check = [&](const Graph&, Match& match) {
    EXPECT_EQ(std::get<int64_t>(match.instance("a").dimensionOf("M")), 3);
    EXPECT_EQ(std::get<int64_t>(match.instance("b").dimensionOf("M")), 4);
    EXPECT_EQ(match.instance("a").attributeOf("axes").ints(0), 1);
    EXPECT_EQ(match.instance("b").attributeOf("axes").ints(1), 1);
    EXPECT_EQ(match.instance("a").valueOf("y"), a);
    EXPECT_THROW(match.update("a.y", {}), OnnxRuntimeException);
    return true;
  };
  Statistics stats;
  Options options;
  options.max_iterations = 1;
  Rule rule{"Hygiene", library.GetPattern("P"), "(x,z,a.axes,b.axes)=>{a=Transpose(x,perm=a.axes);b=Transpose(z,perm=b.axes);return MatMul(a,b)}", check, {{kOnnxDomain, 17, 17}}};
  ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
  EXPECT_EQ(stats.matches, 1u);
  EXPECT_EQ(stats.rewrites, 1u);
  ASSERT_STATUS_OK(test.graph.Resolve());
}

TEST(PatternRewriterTests, PatternCompositionIdentityPortsRemapAndInterfaceConstraintsConjoin) {
  PatternLibrary library(R"(
    interface Pass(x:float[N]) -> (y:float[N])
    pattern pass implements Pass { return (y=x) }
    pattern P(x:float[2]) { p=pass(x); return Identity(p.y) }
  )");
  for (int64_t size : {2, 3}) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({size}, 0.0f, 1.0f);
    test.builder.AddNode("Identity", {x}, {test.builder.MakeOutput()});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto check = [&](const Graph&, Match& match) {
      EXPECT_EQ(match.instance("p").valueOf("x"), x);
      EXPECT_EQ(match.instance("p").valueOf("y"), x);
      return true;
    };
    Options options;
    options.match_only = true;
    Statistics stats;
    Rule rule{"Pass", library.GetPattern("P"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
    ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, size == 2 ? 1u : 0u);
  }
}

TEST(PatternRewriterTests, PatternCompositionOutputAnnotationsAreEnforcedOnAliases) {
  const PatternLibrary library(R"(
    interface Pass(x:float[]) -> (y:float[3])
    pattern pass implements Pass { return (y=x) }
    pattern P(x) { p=pass(x); return Identity(p.y) }
  )");
  for (int64_t size : {2, 3}) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({size}, 0.0f, 1.0f);
    test.builder.AddNode("Identity", {x}, {test.builder.MakeOutput()});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Options options;
    options.match_only = true;
    Statistics stats;
    Rule rule{"OutputConstraint", library.GetPattern("P"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
    ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, size == 3 ? 1u : 0u);
  }
}

TEST(PatternRewriterTests, PatternCompositionUnionTypeAndConstantRejectionRetry) {
  const PatternLibrary library(R"(
    interface Unary(x:float[...]) -> (y:float[...])
    pattern fixed implements Unary: (a:const float[1]) -> { y=Mul(x,a); return (y=y) }
    pattern dynamic implements Unary: (a:float[]) -> { y=Mul(x,a); return (y=y) }
    pattern Forms = fixed | dynamic
    pattern P(x) { f=Forms(x); return Identity(f.y) }
  )");
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2, 4}, 0.0f, 1.0f);
  auto* a = b.MakeInput<float>({1}, 0.0f, 1.0f);
  auto* y = b.MakeIntermediate();
  b.AddNode("Mul", {x, a}, {y});
  b.AddNode("Identity", {y}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  auto check = [&](const Graph&, Match& match) {
    EXPECT_EQ(match.instance("f").alternative(), "dynamic");
    EXPECT_EQ(match.instance("f").valueOf("a"), a);
    return true;
  };
  Options options;
  options.match_only = true;
  Statistics stats;
  Rule rule{"ConstantRetry", library.GetPattern("P"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
  ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
  EXPECT_EQ(stats.matches, 1u);
  EXPECT_EQ(stats.check_and_bind_calls, 1u);
}

TEST(PatternRewriterTests, PatternCompositionDeletionProviderAndLayerBoundaries) {
  const PatternLibrary library(R"(
    interface Unary(x) -> (y)
    pattern pair implements Unary { a=Identity(x); y=Identity(a); return (y=y) }
    pattern P(x) { p=pair(x); return p.y }
  )");
  for (int scenario = 0; scenario < 6; ++scenario) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, 0.0f, 1.0f);
    auto* a = scenario == 1 ? b.MakeOutput() : b.MakeIntermediate();
    auto& first = b.AddNode("Identity", {x}, {a});
    b.AddNode("Identity", {a}, {b.MakeOutput()});
    if (scenario == 2) b.AddNode("Neg", {a}, {b.MakeOutput()});
    if (scenario == 3) first.SetExecutionProviderType(kCudaExecutionProvider);
    if (scenario == 4) first.SetLayeringAnnotation("other");
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    Rule rule{"Safety", library.GetPattern("P"), "(x)=>Identity(x)", {}, {{kOnnxDomain, scenario == 5 ? 18 : 17, 21}}};
    ASSERT_STATUS_OK(PatternRewriter({rule}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, PatternCompositionReplacementFailureDoesNotRetry) {
  PatternLibrary library(R"(
    interface Unary(x) -> (y)
    pattern first implements Unary { a=Identity(x); y=Identity(a); return (y=y) }
    pattern second implements Unary { a=Identity(x); y=Identity(a); return (y=y) }
    pattern Forms=first|second
    pattern P(x) { p=Forms(x); return p.y }
  )");
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, 0.0f, 1.0f);
  auto* a = b.MakeIntermediate();
  b.AddNode("Identity", {x}, {a});
  b.AddNode("Identity", {a}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  const auto before = test.graph.ToGraphProto().SerializeAsString();
  size_t checks = 0;
  auto check = [&](const Graph&, Match&) { ++checks; return true; };
  Rule rule{"BadRecipe", library.GetPattern("P"), "(x)=>UndefinedOperator(x)", check, {{kOnnxDomain, 17, 17}}};
  Statistics stats;
  EXPECT_FALSE(PatternRewriter({rule}).Run(test.graph, stats).IsOK());
  EXPECT_EQ(checks, 1u);
  EXPECT_EQ(stats.rewrites, 0u);
  EXPECT_EQ(test.graph.ToGraphProto().SerializeAsString(), before);
  ASSERT_STATUS_OK(test.graph.Resolve());
}

TEST(PatternRewriterTests, PatternCompositionExtraInputsAreUsableByParsedReplacement) {
  PatternLibrary library(R"(
    interface Unary(x:float[]) -> (y:float[])
    pattern scale implements Unary: (alpha:float[]) -> {
      y=Mul(x,alpha)
      return (y=y)
    }
    pattern P(x) { p=scale(x); return Identity(p.y) }
  )");
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, 0.0f, 1.0f);
  auto* alpha = b.MakeInitializer<float>({1}, {2.0f});
  auto* y = b.MakeIntermediate();
  b.AddNode("Mul", {x, alpha}, {y});
  b.AddNode("Identity", {y}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  Rule rule{"Scale", library.GetPattern("P"), "(x,p.alpha)=>Mul(x,p.alpha)", {}, {{kOnnxDomain, 17, 17}}};
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter({rule}).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 1u);
  EXPECT_EQ(Count(test.graph, "Identity"), 0u);
  ASSERT_EQ(test.graph.Nodes().begin()->InputDefs()[1], alpha);
  EXPECT_THROW((PatternRewriter({Rule{"Unsafe", library.GetPattern("P"), "(p.y)=>Identity(p.y)", {}, {{kOnnxDomain, 17, 17}}}})),
               OnnxRuntimeException);
}

TEST(PatternRewriterTests, PatternCompositionDifferentRootsAndHostRecipeSelection) {
  // The returned pattern owns its linked plans after both text and library die.
  const Pattern pattern = [] {
    std::string text = R"(
      interface Unary(x) -> (y)
      pattern positive implements Unary { y=Relu(x); return (y=y) }
      pattern negative implements Unary { y=Neg(x); return (y=y) }
      pattern Forms=positive|negative
      pattern P(x) { p=Forms(x); return p.y }
    )";
    return PatternLibrary(text).GetPattern("P");
  }();
  for (bool indexed : {false, true}) {
    for (const char* op : {"Relu", "Neg"}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      b.AddNode(op, {x}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto choose = [positive = Replacement("(x)=>Relu(x)"), negative = Replacement("(x)=>Neg(x)")](
                        const Graph&, const Match& match) {
        return match.instance("p").alternative() == "positive" ? positive : negative;
      };
      Rule rule{"SelectRecipe", pattern, choose, {}, {{kOnnxDomain, 17, 17}}};
      Options options;
      options.indexed = indexed;
      options.max_iterations = 1;
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      EXPECT_EQ(Count(test.graph, op), 1u);
      EXPECT_EQ(Count(test.graph, "If"), 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, PatternCompositionCommonMultiOutputsPreserveGraphOutputs) {
  PatternLibrary library(R"(
    interface SplitPair(x:float[4]) -> (left:float[2],right:float[2])
    pattern split implements SplitPair {
      a,b=Split(x,axis=0)
      return (right=b,left=a)
    }
    pattern P(x) { p=split(x); y=Concat(p.left,p.right,axis=0); return p.left,y }
  )");
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({4}, 0.0f, 1.0f);
  auto* a = b.MakeOutput();
  auto* c = b.MakeIntermediate();
  auto* output = b.MakeOutput();
  b.AddNode("Split", {x}, {a, c}).AddAttribute("axis", int64_t{0});
  b.AddNode("Concat", {a, c}, {output}).AddAttribute("axis", int64_t{0});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  auto check = [&](const Graph&, Match& match) {
    EXPECT_EQ(match.instance("p").valueOf("left"), a);
    EXPECT_EQ(match.instance("p").valueOf("a"), a);
    EXPECT_EQ(match.instance("p").valueOf("right"), c);
    return true;
  };
  Rule rule{"MultiOutput", library.GetPattern("P"), "(x)=>{a,b=Split(x,axis=0); y=Concat(a,b,axis=0); return a,y}", check, {{kOnnxDomain, 17, 17}}};
  Options options;
  options.max_iterations = 1;
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 1u);
  ASSERT_STATUS_OK(test.graph.Resolve());
  EXPECT_EQ(test.graph.GetOutputs()[0], a);
  EXPECT_EQ(test.graph.GetOutputs()[1], output);
}

TEST(PatternRewriterTests, PatternCompositionLibraryErrorsAndBounds) {
  const char* invalid[] = {
      "interface I(x)->(y) pattern P(x){a=I(x);return a.y}",
      "pattern f implements Missing {return (y=x)}",
      "interface I(x)->(y) pattern f implements I {a=f(x);return (y=a.y)}",
      "interface I(x)->(y) pattern f implements I {a=g(x);return (y=a.y)} "
      "pattern g implements I {a=f(x);return (y=a.y)}",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (wrong=y)}",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (y=y,y=y)}",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (y=z)}",
      "interface I(x)->(x)",
      "interface I(x:float[x])->(y)",
      "interface I(x:float[N])->(y:tensor<N>[])",
      "interface I(x)->(y) pattern f implements I:(unused:float[])->{y=Identity(x);return (y=y)}",
      "interface I(x)->(y) pattern f implements I {a=Identity(x);y=Identity(a);return (y=a)}",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (y=y)} "
      "pattern P(x){a=f(x,x);return a.y}",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (y=y)} "
      "pattern P(x){a,b=f(x);return a}",
      "interface I(x)->(y) interface J(x)->(y) "
      "pattern f implements I {y=Identity(x);return (y=y)} "
      "pattern g implements J {y=Identity(x);return (y=y)} pattern U=f|g",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (y=y)} pattern U=f|missing",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (y=y)} pattern U=f|f",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (y=y)} "
      "pattern P(x){a=(f|missing)(x);return a.y}",
      "interface I(x)->(y) pattern f implements I {y=Identity(x);return (y=y)} "
      "pattern P(x){a=f(x);a=Identity(a.y);return a}",
      "pattern P(x){return misspelled(x)}",
      "pattern P(x){return Identity(x)} pattern P(x){return Identity(x)}",
      "pattern P(x:const int){return Identity(x)}",
      "pattern P(x,a:const float){return LeakyRelu(x,alpha=a)}",
      "pattern P(x:float[N]){y:float[x]=Identity(x);return y}",
      "pattern P(x:float[]=[1.0]){return Identity(x)}",
  };
  for (const auto* text : invalid) {
    SCOPED_TRACE(text);
    EXPECT_THROW(PatternLibrary{text}, OnnxRuntimeException);
  }
  std::string too_many = R"(
    interface I(x)->(y)
    pattern f implements I {return (y=x)}
    pattern g implements I {return (y=x)}
    pattern U=f|g
    pattern P(x) {
  )";
  for (int i = 0; i < 7; ++i)
    too_many += "a" + std::to_string(i) + "=U(" + (i == 0 ? "x" : "a" + std::to_string(i - 1) + ".y") + ");";
  too_many += "return Identity(a6.y)}";
  EXPECT_THROW(PatternLibrary{too_many}, OnnxRuntimeException);
  PatternLibrary valid("pattern P(x){return Identity(x)}");
  EXPECT_THROW(valid.GetPattern("missing"), OnnxRuntimeException);
}

TEST(PatternRewriterTests, ConstTensorQualifierAndTypedReaders) {
  for (int scenario = 0; scenario < 5; ++scenario) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, 0.0f, 1.0f);
    auto* c = scenario == 1 ? b.MakeInput<float>({1}, 0.0f, 1.0f)
                            : b.MakeInitializer<float>({1}, {2.0f});
    b.AddNode("Add", {x, c}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    if (scenario == 2) {
      auto inputs = test.graph.GetInputsIncludingInitializers();
      inputs.push_back(c);
      test.graph.SetInputs(inputs);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    if (scenario == 3 || scenario == 4) {
      const auto* original = test.graph.GetInitializer(c->Name(), false);
      ASSERT_NE(original, nullptr);
      auto tensor = *original;
      if (scenario == 3) {
        tensor.clear_raw_data();
        tensor.clear_float_data();
      } else {
        tensor.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
      }
      test.graph.RemoveInitializedTensor(c->Name());
      test.graph.AddInitializedTensor(tensor);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    InlinedVector<float> values{99.0f};
    bool available = true;
    const auto status = ReadConstantTensor(test.graph, c, values, available);
    if (scenario == 3 || scenario == 4) {
      EXPECT_FALSE(status.IsOK());
      EXPECT_EQ(values, (InlinedVector<float>{99.0f}));
    } else {
      ASSERT_STATUS_OK(status);
      EXPECT_EQ(available, scenario == 0);
      EXPECT_EQ(values, (InlinedVector<float>{scenario == 0 ? 2.0f : 99.0f}));
    }
    InlinedVector<int64_t> wrong_type;
    ASSERT_STATUS_OK(ReadConstantTensor(test.graph, c, wrong_type, available));
    EXPECT_FALSE(available);
    Statistics stats;
    Options options;
    options.match_only = true;
    Rule rule{"Const", "(x,c:const float[])=>return Add(x,c)", "(x,c)=>Add(x,c)", {}, {{kOnnxDomain, 17, 17}}};
    ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, scenario == 1 || scenario == 2 ? 0u : 1u);
  }
  TestGraph test;
  auto* value = test.builder.MakeInitializer<int64_t>({2}, {-1, 8});
  InlinedVector<int64_t> values;
  bool available;
  ASSERT_STATUS_OK(ReadConstantTensor(test.graph, value, values, available));
  EXPECT_TRUE(available);
  EXPECT_EQ(values, (InlinedVector<int64_t>{-1, 8}));
  EXPECT_THROW(Pattern("(x:const const float[])=>Identity(x)"), OnnxRuntimeException);
}

TEST(PatternRewriterTests, ConstQualifierIsSpecificToItsTypeAlternative) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = scenario == 0 ? b.MakeInitializer<float>({1}, {1.0f})
                            : b.MakeInput<float>({scenario == 1 ? 1 : 2}, 0.0f, 1.0f);
    b.AddNode("Identity", {x}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Options options;
    options.match_only = true;
    Statistics stats;
    Rule rule{"ConstAlternative", "(x:const float[1] | float[2])=>return Identity(x)", "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
    ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, scenario == 1 ? 0u : 1u);
  }
  Pattern optional("(x:const float[] | None)=>return Identity(x)");
  EXPECT_TRUE(optional.IsOptional(optional.NamedValue("x")));
}

TEST(PatternRewriterTests, PatternCompositionLiteralConstantsAreRemapped) {
  const PatternLibrary library(R"(
    interface Shift(x) -> (y)
    pattern shift implements Shift { y=Add(x,3.0); return (y=y) }
    pattern P(x) { p=shift(x); return p.y }
  )");
  for (float constant : {3.0f, 4.0f}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, 0.0f, 1.0f);
    b.AddNode("Add", {x, b.MakeScalarInitializer<float>(constant)}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Options options;
    options.match_only = true;
    Statistics stats;
    Rule rule{"Literal", library.GetPattern("P"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
    ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, constant == 3.0f ? 1u : 0u);
  }
}

TEST(PatternRewriterTests, UnifiedPatternsDirectNestedCallsAndIdentityEntries) {
  const PatternLibrary library(R"(
    pattern entry(x) { c=wrapper(x); return Identity(c.y) }
    pattern wrapper(x) { p=leaf(x); return (y=p.y) }
    pattern leaf(x:float[N]) { a=Identity(x); return (y=a) }
    pattern identity(x) { return (y=x) }
    pattern mixed(x) { a=Identity(x); return (original=x,y=a) }
    pattern use_identity(x) { p=identity(x); return Identity(p.y) }
    pattern implicit(x) { return Identity(x) }
    pattern use_implicit(x) { p=implicit(x); return Identity(p.output) }
    pattern use_mixed(x) { p=mixed(x); return Add(p.original,p.y) }
  )");
  for (const char* name : {"identity", "mixed"}) {
    SCOPED_TRACE(name);
    EXPECT_THROW(library.GetPattern(name), OnnxRuntimeException);
  }
  for (const char* entry : {"entry", "use_identity", "use_implicit", "use_mixed"}) {
    SCOPED_TRACE(entry);
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* a = b.MakeIntermediate();
    if (std::string_view(entry) != "use_identity") b.AddNode("Identity", {x}, {a});
    if (std::string_view(entry) == "use_mixed")
      b.AddNode("Add", {x, a}, {b.MakeOutput()});
    else
      b.AddNode("Identity", {std::string_view(entry) == "use_identity" ? x : a}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto check = [&](const Graph&, Match& match) {
      if (std::string_view(entry) == "entry") {
        const auto wrapper = match.instance("c");
        EXPECT_EQ(wrapper.definition(), "wrapper");
        EXPECT_EQ(wrapper.instance("p").definition(), "leaf");
        EXPECT_EQ(wrapper.valueOf("x"), x);
        EXPECT_EQ(wrapper.valueOf("y"), a);
        EXPECT_EQ(std::get<int64_t>(wrapper.instance("p").dimensionOf("N")), 2);
        EXPECT_EQ(wrapper.instance("p").valueOf("y"), a);
        EXPECT_TRUE(wrapper.alternative().empty());
        EXPECT_THROW(wrapper.alternativeView(), OnnxRuntimeException);
      } else if (std::string_view(entry) == "use_identity") {
        EXPECT_EQ(match.instance("p").valueOf("y"), x);
      }
      return true;
    };
    Rule rule{"UnifiedCall", library.GetPattern(entry), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
    Statistics stats;
    Options options;
    options.max_iterations = 1;
    ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
  EXPECT_NO_THROW(library.GetPattern("leaf"));
  EXPECT_NO_THROW(library.GetPattern("wrapper"));
}

TEST(PatternRewriterTests, UnifiedPatternsIndependentOutputsAreCallableNotEntries) {
  const PatternLibrary library(R"(
    pattern branches(x,z) { a=Identity(x); b=Neg(z); return (left=a,right=b) }
    pattern parent(x,z) { p=branches(x,z); return Add(p.left,p.right) }
    pattern branch_alias=branches
    pattern aliased_parent(x,z) { p=branch_alias(x,z); return Add(p.left,p.right) }
  )");
  for (const char* entry : {"branches", "branch_alias"}) {
    try {
      library.GetPattern(entry);
      FAIL() << "An independent-output definition must not serve as a standalone entry.";
    } catch (const OnnxRuntimeException& error) {
      EXPECT_NE(std::string(error.what()).find("cannot be a rule entry"), std::string::npos);
    }
  }
  for (const char* entry : {"parent", "aliased_parent"}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, 0.0f, 1.0f);
    auto* z = b.MakeInput<float>({2}, 0.0f, 1.0f);
    auto* left = b.MakeIntermediate();
    auto* right = b.MakeIntermediate();
    b.AddNode("Identity", {x}, {left});
    b.AddNode("Neg", {z}, {right});
    b.AddNode("Add", {left, right}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto check = [&](const Graph&, Match& match) {
      EXPECT_EQ(match.instance("p").valueOf("left"), left);
      EXPECT_EQ(match.instance("p").valueOf("right"), right);
      if (std::string_view(entry) == "aliased_parent") {
        EXPECT_EQ(match.instance("p").alternative(), "branches");
        EXPECT_EQ(match.instance("p").alternativeView().definition(), "branches");
        EXPECT_TRUE(match.instance("p").alternativeView().alternative().empty());
      }
      return true;
    };
    Rule rule{"IndependentOutputs", library.GetPattern(entry), "(x,z)=>Add(x,z)", check, {{kOnnxDomain, 17, 17}}};
    Options options;
    options.match_only = true;
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
  }
}

namespace {
struct NestedSelectionView {
  std::string_view definition;
  std::string_view member;
  std::unique_ptr<NestedSelectionView> selected;
  const NodeArg* y;
  explicit NestedSelectionView(BindingView view)
      : definition(view.definition()), member(view.alternative()), selected(member.empty() ? nullptr : std::make_unique<NestedSelectionView>(view.alternativeView())), y(view.valueOf("y")) {}
};
}  // namespace

TEST(PatternRewriterTests, UnifiedPatternsNestedAlternativesTagsRetryAndIsolation) {
  for (const char* callee : {"Outer", "(Alias | Inner)"}) {
    const PatternLibrary library(std::string(R"(
      interface Unary(x:float[N]) -> (y:float[N])
      pattern first implements Unary { a=Identity(x); return (y=a) }
      pattern second implements Unary { b=Identity(x); return (y=b) }
      pattern Inner=first|second
      pattern Alias=Inner
      pattern Outer=Alias|Inner
      pattern P(x,z) { p=)") + callee +
                                 "(x); q=" + callee + "(z); return Add(p.y,q.y) }");
    for (bool indexed : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* z = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* a = b.MakeIntermediate();
      auto* c = b.MakeIntermediate();
      b.AddNode("Identity", {x}, {a});
      b.AddNode("Identity", {z}, {c});
      b.AddNode("Add", {a, c}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      InlinedVector<std::string> attempts;
      auto path = [](BindingView view) {
        std::string result;
        while (!view.alternative().empty()) {
          if (!result.empty()) result += '/';
          result += view.alternative();
          view = view.alternativeView();
        }
        return result;
      };
      auto check = [&](const Graph&, Match& match) {
        const auto p = match.instance("p"), q = match.instance("q");
        const auto pp = path(p), qp = path(q);
        attempts.push_back(pp + ":" + qp);
        const NestedSelectionView typed(p);
        EXPECT_EQ(typed.y, a);
        EXPECT_NE(typed.selected, nullptr);
        if (typed.selected) EXPECT_EQ(typed.selected->y, a);
        EXPECT_EQ(p.valueOf("x"), x);
        EXPECT_EQ(q.valueOf("x"), z);
        EXPECT_THROW(match.attributeOf("leaked"), OnnxRuntimeException);
        if (pp == "Inner/second" && qp == "Alias/Inner/first") {
          EXPECT_THROW(p.valueOf("a"), OnnxRuntimeException);
          EXPECT_EQ(p.alternativeView().alternativeView().valueOf("b"), a);
          EXPECT_EQ(q.alternativeView().alternativeView().alternativeView().valueOf("a"), c);
          return true;
        }
        ONNX_NAMESPACE::AttributeProto leaked;
        leaked.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        leaked.set_i(123);
        match.update("leaked", std::move(leaked));
        return match.Reject("Try the next nested selection.");
      };
      Rule rule{"Nested", library.GetPattern("P"), "(x,z)=>Add(x,z)", check, {{kOnnxDomain, 17, 17}}};
      Options options;
      options.indexed = indexed;
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      ASSERT_EQ(attempts.size(), 13u);
      EXPECT_EQ(attempts.front(), "Alias/Inner/first:Alias/Inner/first");
      EXPECT_EQ(attempts[1], "Alias/Inner/first:Alias/Inner/second");
      EXPECT_EQ(attempts.back(), "Inner/second:Alias/Inner/first");
    }
  }
}

TEST(PatternRewriterTests, UnifiedPatternsAlternativesCanBeEntriesOrCallPlainBodies) {
  const PatternLibrary library(R"(
    pattern positive(x:float[]) { y=Relu(x); return y }
    pattern negative(x:float[]) { y=Neg(x); return y }
    pattern Forms=positive|negative
    pattern Alias=Forms
    pattern Caller(x) { p=Alias(x); return Identity(p.y) }
  )");
  for (bool indexed : {false, true}) {
    for (const char* entry : {"Forms", "Alias", "Caller"}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      if (std::string_view(entry) == "Caller") {
        auto* y = b.MakeIntermediate();
        b.AddNode("Neg", {x}, {y});
        b.AddNode("Identity", {y}, {b.MakeOutput()});
      } else {
        b.AddNode("Neg", {x}, {b.MakeOutput()});
      }
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto check = [&](const Graph&, Match& match) {
        auto view = std::string_view(entry) == "Caller" ? match.instance("p") : match.bindings();
        if (std::string_view(entry) != "Forms") {
          EXPECT_EQ(view.alternative(), "Forms");
          view = view.alternativeView();
        }
        EXPECT_EQ(view.alternative(), "negative");
        EXPECT_EQ(view.alternativeView().definition(), "negative");
        EXPECT_TRUE(view.alternativeView().interfaceName().empty());
        return true;
      };
      Rule rule{"EntryAlternative", library.GetPattern(entry), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
      Options options;
      options.indexed = indexed;
      options.max_iterations = 1;
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
    }
  }
}

TEST(PatternRewriterTests, UnifiedPatternsEntryValidityChecksEveryExpansion) {
  const PatternLibrary library(R"(
    interface Unary(x) -> (y)
    pattern pass implements Unary { return (y=x) }
    pattern op implements Unary { y=Identity(x); return (y=y) }
    pattern Forms=op|pass
    pattern Alias=Forms
    pattern Caller(x) { p=Alias(x); return Identity(p.y) }
  )");
  EXPECT_THROW(library.GetPattern("Forms"), OnnxRuntimeException);
  EXPECT_THROW(library.GetPattern("Alias"), OnnxRuntimeException);
  EXPECT_THROW(library.GetPattern("Unary"), OnnxRuntimeException);
  EXPECT_NO_THROW(library.GetPattern("Caller"));
}

TEST(PatternRewriterTests, UnifiedPatternsPublicAttributeInputsAndScopedAliases) {
  const PatternLibrary library(R"(
    pattern transpose(x:float[N,M],axes:ints) { y=Transpose(x,perm=axes); return y }
    pattern nested(x,perm:ints) { a=transpose(x,perm); return (y=a.y) }
    pattern Caller(x,axes:ints) { p=nested(x,axes); return Identity(p.y) }
  )");
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2, 3}, 0.0f, 1.0f);
  auto* y = b.MakeIntermediate();
  b.AddNode("Transpose", {x}, {y}).AddAttribute("perm", std::vector<int64_t>{1, 0});
  b.AddNode("Identity", {y}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  auto check = [&](const Graph&, Match& match) {
    EXPECT_EQ(match.instance("p").attributeOf("perm").ints(0), 1);
    EXPECT_EQ(match.instance("p").instance("a").attributeOf("axes").ints(1), 0);
    auto attribute = match.attributeOf("axes");
    EXPECT_THROW(match.update("p.a.axes", attribute), OnnxRuntimeException);
    EXPECT_THROW(match.update("p.perm", attribute), OnnxRuntimeException);
    EXPECT_THROW(match.update("axes", attribute), OnnxRuntimeException);
    return true;
  };
  Rule rule{"AttributePort", library.GetPattern("Caller"), "(x,p.a.axes)=>Transpose(x,perm=p.a.axes)", check, {{kOnnxDomain, 17, 17}}};
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter({rule}).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 1u);
  ASSERT_STATUS_OK(test.graph.Resolve());
}

TEST(PatternRewriterTests, UnifiedPatternsRejectCyclesCompatibilityAndObsoleteDeclarations) {
  const char* invalid[] = {
      "fragment old(x) { return Identity(x) }",
      "union Old=a|b",
      "pattern A=B pattern B=A",
      "pattern A=A",
      "pattern A=B pattern B(x){p=A(x);return p.y}",
      "pattern A(x){p=B(x);return p.y} pattern B=A",
      "interface I(x)->(y) pattern A=I",
      "pattern A=missing",
      "pattern a(x){return (y=x)} pattern b(z){return (y=z)} pattern C=a|b",
      "pattern a(x){return (y=x)} pattern b(x){return (z=x)} pattern C=a|b",
      "pattern a(x:float[2]){return (y=x)} pattern b(x:float[3]){return (y=x)} pattern C=a|b",
      "pattern a(x){y:float[2]=Identity(x);return y} "
      "pattern b(x){y:float[3]=Identity(x);return y} pattern C=a|b",
      "interface I(x)->(y) pattern a implements I {return (y=x)} "
      "pattern b(x){return (y=x)} pattern C=a|b",
      "pattern a(x){return (y=x)} pattern B=a pattern C(x){p=(a|B|a)(x);return Identity(p.y)}",
      "pattern a(x){return (y=x)} pattern B=a pattern C(x){p=(a|missing)(x);return Identity(p.y)}",
      "pattern a(x){return (y=x)} pattern b(x){return (z=x)} pattern C(x){p=(a|b)(x);return Identity(p.y)}",
      "pattern a(x,axes:ints){y=Transpose(x,perm=axes);return y} "
      "pattern b(x){z=Identity(x);p=a(x,z);return p.y}",
      "pattern a(x){return (left=x,right=x)} pattern b(x){p=a(x);q=a(x);return p.left,q.left}",
  };
  for (const auto* text : invalid) {
    SCOPED_TRACE(text);
    EXPECT_THROW(PatternLibrary{text}, OnnxRuntimeException);
  }
  std::string text = "pattern base(x){return (y=x)} pattern other=base pattern A0=base|other ";
  for (int i = 1; i < 6; ++i) {
    const auto previous = "A" + std::to_string(i - 1);
    text += "pattern B" + std::to_string(i) + "=" + previous + " pattern A" + std::to_string(i) +
            "=" + previous + "|B" + std::to_string(i) + " ";
  }
  EXPECT_NO_THROW(PatternLibrary{text});
  text += "pattern too_many=A5|base";
  EXPECT_THROW(PatternLibrary{text}, OnnxRuntimeException);
  EXPECT_THROW(Pattern("(x)=>return x"), OnnxRuntimeException);
  EXPECT_THROW(Pattern("(x,z)=>{a=Identity(x);b=Identity(z);return a,b}"), OnnxRuntimeException);
}

TEST(PatternRewriterTests, UnifiedPatternsExpansionAndAnnotationThresholds) {
  EXPECT_NO_THROW(PatternLibrary(R"(
    pattern a(x:float[]) { y=Identity(x); return y }
    pattern b(x:float[...]) { y=Neg(x); return y }
    pattern Forms=a|b
    pattern P(x) { p=(Forms)(x); return Identity(p.y) }
  )")
                      .GetPattern("P"));
  std::string body = "pattern node(x){y=Identity(x);return y} pattern P(x){";
  for (int i = 0; i < 64; ++i)
    body += "a" + std::to_string(i) + "=node(" + (i == 0 ? "x" : "a" + std::to_string(i - 1) + ".y") + ");";
  EXPECT_NO_THROW(PatternLibrary(body + "return a63.y}").GetPattern("P"));
  EXPECT_THROW(PatternLibrary(body + "a64=node(a63.y);return a64.y}"), OnnxRuntimeException);
  std::string types;
  for (int i = 0; i < 64; ++i) {
    if (!types.empty()) types += '|';
    types += "float[" + std::to_string(i) + "]";
  }
  // Inferred output ports must not reapply and square the same disjunction.
  EXPECT_NO_THROW(PatternLibrary("pattern P(x){y:" + types + "=Identity(x);return y}").GetPattern("P"));
  EXPECT_THROW(PatternLibrary("pattern P(x){y:" + types + "|float[64]=Identity(x);return y}"),
               OnnxRuntimeException);
}

TEST(PatternRewriterTests, PortableFloatLiteralsValidateWholeToken) {
  for (const char* literal : {"+1.25", "-1.25", "1.25e+2", ".125", "1."}) {
    SCOPED_TRACE(literal);
    EXPECT_NO_THROW(Pattern(std::string("(x)=>return Add(x,") + literal + ")"));
    EXPECT_NO_THROW(Replacement(std::string("(x)=>LeakyRelu(x,alpha=") + literal + ")"));
  }
  for (const char* literal : {"1e999", "1e", "1.2.3", "1e+", "1.0-2.0"}) {
    SCOPED_TRACE(literal);
    EXPECT_THROW(Pattern(std::string("(x)=>return Add(x,") + literal + ")"), OnnxRuntimeException);
    EXPECT_THROW(Replacement(std::string("(x)=>LeakyRelu(x,alpha=") + literal + ")"), OnnxRuntimeException);
  }
}

TEST(PatternRewriterTests, AliasedReturnsAreCallableButNotDuplicateEntryOutputs) {
  const PatternLibrary library(R"(
    pattern IdentityAlias(x) { return (y=x) }
    pattern Duplicate(x) { y=Identity(x); return (a=y,b=y) }
    pattern NestedDuplicate(x) { y=Identity(x); p=IdentityAlias(y); return (a=y,b=p.y) }
    pattern Caller(x) { p=Duplicate(x); return Add(p.a,p.b) }
    pattern NestedCaller(x) { p=NestedDuplicate(x); return Add(p.a,p.b) }
    pattern Forms=Duplicate|NestedDuplicate
  )");
  for (const char* name : {"Duplicate", "NestedDuplicate", "Forms"}) {
    SCOPED_TRACE(name);
    try {
      library.GetPattern(name);
      FAIL() << "Aliased entry returns must not create duplicate replacement producers.";
    } catch (const OnnxRuntimeException& error) {
      EXPECT_NE(std::string(error.what()).find("duplicate returned value"), std::string::npos);
    }
  }
  for (const char* name : {"Caller", "NestedCaller"}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, 0.0f, 1.0f);
    auto* y = b.MakeIntermediate();
    b.AddNode("Identity", {x}, {y});
    b.AddNode("Add", {y, y}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Rule rule{"Aliases", library.GetPattern(name), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
    Options options;
    options.max_iterations = 1;
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, ZeroNodeCompositionHasCumulativeCallBudget) {
  std::string text = "pattern P0(x){return (y=x)} ";
  for (int i = 1; i <= 7; ++i) {
    const auto previous = "P" + std::to_string(i - 1);
    text += "pattern P" + std::to_string(i) + "(x){a=" + previous + "(x);b=" + previous + "(a.y);return (y=b.y)} ";
  }
  // P7 contains 254 expanded calls; these wrappers reach exactly 256.
  text += "pattern P8(x){a=P7(x);return (y=a.y)} pattern P9(x){a=P8(x);return (y=a.y)} ";
  EXPECT_NO_THROW(PatternLibrary{text});
  const std::string overflow = text + "pattern P10(x){a=P9(x);return (y=a.y)}";
  try {
    PatternLibrary library(overflow);
    FAIL() << "Zero-node metadata expansion must be bounded before copying.";
  } catch (const OnnxRuntimeException& error) {
    EXPECT_NE(std::string(error.what()).find("expanded call limit of 256"), std::string::npos);
  }
  EXPECT_THROW(PatternLibrary(text + "pattern P10(x){a=P7(x);b=P7(a.y);return (y=b.y)}"),
               OnnxRuntimeException);
}

TEST(PatternRewriterTests, InterfacePortsCannotBecomeAttributes) {
  for (const char* text : {
           "interface I(x,alpha:float)->(y) pattern P implements I {y=LeakyRelu(x,alpha=alpha);return y}",
           "interface I(x)->(y,z) pattern P implements I:(alpha:float)->"
           "{a=LeakyRelu(x,alpha=alpha);return (y=alpha,z=a)}",
           "pattern Attr(x,alpha:float){y=LeakyRelu(x,alpha=alpha);return y} "
           "interface I(x,alpha:float)->(y) pattern P implements I {a=Attr(x,alpha);return (y=a.y)}",
           "interface I(x,alpha:float)->(y) "
           "pattern A implements I {y=Mul(x,alpha);return y} "
           "pattern B implements I {y=LeakyRelu(x,alpha=alpha);return y} pattern Forms=A|B"}) {
    SCOPED_TRACE(text);
    try {
      PatternLibrary library(text);
      FAIL() << "Interface ports must remain tensor bindings.";
    } catch (const OnnxRuntimeException& error) {
      EXPECT_NE(std::string(error.what()).find("must remain a tensor"), std::string::npos);
    }
  }
  EXPECT_NO_THROW(PatternLibrary(R"(
    interface I(x) -> (y)
    pattern P implements I: (alpha:float) -> { y=LeakyRelu(x,alpha=alpha);return y }
    pattern C(x) { p=P(x); return Identity(p.y) }
  )")
                      .GetPattern("C"));
}

namespace {
class CommaDecimalPoint : public std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

class ScopedGlobalLocale {
 public:
  explicit ScopedGlobalLocale(const std::locale& locale) : previous_(std::locale::global(locale)) {}
  ~ScopedGlobalLocale() { std::locale::global(previous_); }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedGlobalLocale);

 private:
  std::locale previous_;
};
}  // namespace

TEST(PatternRewriterTests, GroupSignaturesAreIndependentOfGlobalLocale) {
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      for (bool second_constant : {false, true}) {
        TestGraph test;
        auto& b = test.builder;
        auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
        auto* constant = b.MakeInitializer<float>({2}, second_constant ? std::vector<float>{1.0f, 5.5f}
                                                                       : std::vector<float>{1.5f, 5.0f});
        b.AddNode("Add", {x, constant}, {b.MakeOutput()});
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        InlinedVector<int> callbacks;
        Rules rules;
        rules.emplace_back("First", "(x)=>return Add(x,[1.5,5.0])", "(x)=>Identity(x)", [&](const Graph&, Match&) {callbacks.push_back(0); return true; }, InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 17, 17}});
        rules.emplace_back("Second", "(x)=>return Add(x,[1.0,5.5])", "(x)=>Identity(x)", [&](const Graph&, Match&) {callbacks.push_back(1); return true; }, InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 17, 17}});
        Statistics stats;
        {
          const ScopedGlobalLocale locale{std::locale(std::locale::classic(), new CommaDecimalPoint)};
          ASSERT_EQ(std::use_facet<std::numpunct<char>>(std::locale()).decimal_point(), ',');
          ASSERT_STATUS_OK(PatternRewriter(std::move(rules), {.match_only = match_only, .indexed = indexed, .max_iterations = 1})
                               .Run(test.graph, stats));
        }
        ASSERT_EQ(callbacks.size(), 1u);
        EXPECT_EQ(callbacks[0], second_constant ? 1 : 0);
        EXPECT_EQ(stats.matches, 1u);
        EXPECT_EQ(stats.rewrites, match_only ? 0u : 1u);
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
    }
  }
}

TEST(PatternRewriterTests, UnannotatedInterfaceOutputAliasesMustBePresent) {
  const PatternLibrary library(R"(
    interface I(v:float | None) -> (y)
    pattern P implements I { return (y=v) }
    pattern Alias=P
    pattern Wrapped implements I { p=Alias(v);return (y=p.y) }
    pattern Entry(x,v:float | None) { p=P(v);return Clip(x,p.y) }
    pattern Nested(x,v:float | None) { p=Wrapped(v);return Clip(x,p.y) }
  )");
  for (const char* entry : {"Entry", "Nested"}) {
    for (bool indexed : {false, true}) {
      for (int scenario = 0; scenario < 3; ++scenario) {
        TestGraph test;
        auto& b = test.builder;
        auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
        std::vector<NodeArg*> inputs{x};
        const NodeArg* minimum = nullptr;
        if (scenario == 1) inputs.push_back(b.MakeOptionalTensor());
        if (scenario == 2) {
          auto* value = b.MakeScalarInitializer<float>(0.0f);
          inputs.push_back(value);
          minimum = value;
        }
        b.AddNode("Clip", inputs, {b.MakeOutput()});
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        int checks = 0;
        auto check = [&](const Graph&, Match& match) {
          ++checks;
          EXPECT_NE(match.instance("p").valueOf("y"), nullptr);
          EXPECT_EQ(match.instance("p").valueOf("y"), minimum);
          return true;
        };
        Rule rule{"RequiredOutput", library.GetPattern(entry), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
        EXPECT_EQ(checks, scenario == 2 ? 1 : 0);
        EXPECT_EQ(stats.matches, scenario == 2 ? 1u : 0u);
      }
    }
  }
}

TEST(PatternRewriterTests, InterleavedGroupsDispatchCallbacksInRegistrationOrder) {
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      for (bool throw_late : {false, true}) {
        TestGraph test;
        auto& b = test.builder;
        auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
        auto* y = b.MakeInput<float>({2}, -1.0f, 1.0f);
        auto* a = b.MakeIntermediate();
        b.AddNode("Identity", {x}, {a});
        auto& root = b.AddNode("Add", {a, y}, {b.MakeOutput()});
        const NodeIndex index = root.Index();
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        InlinedVector<int> callbacks;
        InlinedVector<std::string> rejected;
        auto reject = [&](const Graph&, Match& match) {
          callbacks.push_back(0);
          ONNX_NAMESPACE::AttributeProto extra;
          extra.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
          extra.set_i(99);
          match.update("candidate_only", std::move(extra));
          return match.Reject("first rejects");
        };
        auto accept = [&](const Graph&, Match& match) {
          callbacks.push_back(1);
          EXPECT_THROW(match.attributeOf("candidate_only"), OnnxRuntimeException);
          return true;
        };
        auto later = [&](const Graph&, Match& match) {
          callbacks.push_back(2);
          EXPECT_THROW(match.attributeOf("candidate_only"), OnnxRuntimeException);
          ORT_ENFORCE(!throw_late, "lower priority callback");
          return true;
        };
        Rule first{"First", "(x,y)=>return Add(x,y)", "(x,y)=>Add(x,y)", reject, {{kOnnxDomain, 17, 17}}};
        Rule middle{"Middle", "(x,y)=>{a=Identity(x);return Add(a,y)}", "(x,y)=>Add(x,y)", accept, {{kOnnxDomain, 17, 17}}};
        Rule last{"Last", "(x,y)=>return Add(x,y)", "(x,y)=>Add(x,y)", later, {{kOnnxDomain, 17, 17}}};
        Options options{.match_only = match_only, .indexed = indexed, .max_iterations = 1};
        options.on_failure = [&](const MatchFailure& failure) {
          if (failure.root == index && failure.stage == FailureStage::CheckAndBind) rejected.push_back(failure.rule);
        };
        Statistics stats;
        PatternRewriter rewriter({first, middle, last}, options);
        if (match_only && throw_late) {
          EXPECT_THROW((void)rewriter.Run(test.graph, stats), OnnxRuntimeException);
        } else {
          ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
          EXPECT_EQ(stats.rewrites, match_only ? 0u : 1u);
          EXPECT_EQ(stats.matches, match_only ? 2u : 1u);
        }
        ASSERT_EQ(callbacks.size(), match_only ? 3u : 2u);
        EXPECT_EQ(callbacks[0], 0);
        EXPECT_EQ(callbacks[1], 1);
        if (match_only) EXPECT_EQ(callbacks[2], 2);
        ASSERT_EQ(rejected.size(), 1u);
        EXPECT_EQ(rejected[0], "First");
        // Indexed mode shares First/Last's traversal even when other groups intervene.
        EXPECT_EQ(stats.structural_checks, indexed ? 2u : (match_only ? 6u : 5u));
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
    }
  }
}

TEST(PatternRewriterTests, AnnotationPrioritySurvivesReorderedAndAliasedCalls) {
  for (int repeat = 0; repeat < 16; ++repeat) {
    const PatternLibrary library(R"(
      pattern F(x:float[M,_] | float[_,M], z:float[_,M] | float[M,_]) {
        y=CastLike(x,z);return y
      }
      pattern Reordered(v,w) { p=F(w,v);return p.y }
      pattern Aliased(v) { p=F(v,v);return p.y }
      pattern Alias=F
      pattern Nested(v,w) { p=Alias(w,v);return p.y }
    )");
    for (const char* entry : {"F", "Reordered", "Aliased", "Nested"}) {
      for (bool indexed : {false, true}) {
        TestGraph test;
        auto& b = test.builder;
        auto* v = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
        auto* w = std::string_view(entry) == "Aliased" ? v : b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
        b.AddNode("CastLike", {w, v}, {b.MakeOutput()});
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        int checks = 0;
        auto check = [&](const Graph&, Match& match) {
          ++checks;
          const auto view = std::string_view(entry) == "F" ? match.bindings() : match.instance("p");
          EXPECT_EQ(std::get<int64_t>(view.dimensionOf("M")), 2);
          return true;
        };
        Rule rule{"Priority", library.GetPattern(entry), std::string_view(entry) == "F" ? "(x)=>Identity(x)" : "(v)=>Identity(v)", check, {{kOnnxDomain, 17, 17}}};
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
        EXPECT_EQ(stats.matches, 1u);
        EXPECT_EQ(checks, 1);
      }
    }
  }
}

TEST(PatternRewriterTests, ComposedInterfaceTensorBindingsCannotBecomeAttributes) {
  for (bool annotated : {false, true}) {
    const std::string prefix = annotated
                                   ? "interface I(a:float[])->(y:float[]) "
                                   : "interface I(a)->(y) ";
    const std::string definitions = prefix + R"(
      pattern Pass implements I { return (y=a) }
      pattern Alias=Pass
      pattern Wrapped implements I { p=Alias(a);return (y=p.y) }
      pattern Attr(x,alpha:float) { y=LeakyRelu(x,alpha=alpha);return y }
    )";
    for (const char* call : {"Pass", "Alias", "Wrapped"}) {
      for (const char* body : {
               "p=CALL(a);return LeakyRelu(x,alpha=a)",
               "p=CALL(a);q=Attr(x,p.y);return q.y",
               "p=CALL(a);q=Attr(x,p.a);return q.y",
               "q=Attr(x,a);p=CALL(a);return q.y"}) {
        std::string text = body;
        text.replace(text.find("CALL"), 4, call);
        try {
          const PatternLibrary library(definitions + "pattern Entry(x,a){" + text + "}");
          FAIL() << "Accepted interface tensor/attribute conflict: " << text;
        } catch (const OnnxRuntimeException& error) {
          EXPECT_NE(std::string(error.what()).find("Declared tensor binding"), std::string::npos) << error.what();
        }
      }
    }
  }
}

TEST(PatternRewriterTests, ReplacementDefaultsAreVisibleToChainedRules) {
  for (bool indexed : {false, true}) {
    for (bool explicit_alpha : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* a = b.MakeIntermediate();
      b.AddNode("Identity", {x}, {a});
      auto* output = b.MakeOutput();
      b.AddNode("LeakyRelu", {a}, {output});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      size_t checks = 0;
      auto check = [&](const Graph&, Match& match) {
        ++checks;
        EXPECT_FLOAT_EQ(match.attributeOf("alpha").f(), explicit_alpha ? 0.02f : 0.01f);
        EXPECT_FLOAT_EQ(match.nodes.back()->GetAttributes().at("alpha").f(), explicit_alpha ? 0.02f : 0.01f);
        return true;
      };
      Rules rules{
          {"RemoveIdentity", "(x)=>{a=Identity(x);return LeakyRelu(a,alpha=0.01)}", explicit_alpha ? "(x)=>LeakyRelu(x,alpha=0.02)" : "(x)=>LeakyRelu(x)", {}, {{kOnnxDomain, 17, 17}}},
          {"UsePRelu", "(x)=>return LeakyRelu(x,alpha=alpha)", explicit_alpha ? "(x)=>PRelu(x,0.02)" : "(x)=>PRelu(x,0.01)", check, {{kOnnxDomain, 17, 17}}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter(rules, {.indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 2u);
      EXPECT_EQ(checks, 1u);
      EXPECT_FALSE(stats.iteration_limit_reached);
      ASSERT_NE(test.graph.GetProducerNode(output->Name()), nullptr);
      EXPECT_EQ(test.graph.GetProducerNode(output->Name())->OpType(), "PRelu");
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, ForwardDefinitionNestingHasExplicitBoundary) {
  for (bool reverse : {false, true}) {
    for (bool aliases : {false, true}) {
      for (int depth : {64, 65, 256}) {
        std::string text;
        for (int index = 0; index < depth; ++index) {
          const int i = reverse ? depth - 1 - index : index;
          const std::string name = "P" + std::to_string(i);
          const std::string next = "P" + std::to_string(i + 1);
          if (i == depth - 1)
            text += "pattern " + name + "(x){return (y=x)} ";
          else if (i > 0 && aliases)
            text += "pattern " + name + "=" + next + " ";
          else
            text += "pattern " + name + "(x){p=" + next + "(x);" +
                    (i == 0 ? "y=Identity(p.y);return y" : "return (y=p.y)") + "} ";
        }
        if (depth == 64) {
          const PatternLibrary library(text);
          EXPECT_EQ(library.GetPattern("P0").Nodes().size(), 1u);
        } else {
          try {
            const PatternLibrary library(text);
            FAIL() << "Accepted depth " << depth;
          } catch (const OnnxRuntimeException& error) {
            EXPECT_NE(std::string(error.what()).find("dependency nesting limit of 64"), std::string::npos)
                << error.what();
          }
        }
      }
    }
  }
}

TEST(PatternRewriterTests, ComposedTensorAnnotationsCannotBecomeAttributes) {
  const std::string definitions = R"(
    pattern Pass(a:float) { return (y=a) }
    pattern Alias=Pass
    pattern Wrapped(a) { p=Alias(a);return (y=p.y) }
    pattern Attr(x,alpha:float) { y=LeakyRelu(x,alpha=alpha);return y }
  )";
  for (const char* call : {"Pass", "Alias", "Wrapped"}) {
    for (const char* body : {
             "p=CALL(a);return LeakyRelu(x,alpha=a)",
             "p=CALL(a);q=Attr(x,p.y);return q.y",
             "q=Attr(x,a);p=CALL(a);return q.y"}) {
      std::string text = body;
      text.replace(text.find("CALL"), 4, call);
      for (bool indexed : {false, true}) {
        SCOPED_TRACE(text);
        try {
          const PatternLibrary library(definitions + "pattern Entry(x,a){" + text + "}");
          Rule rule{"Invalid", library.GetPattern("Entry"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
          const PatternRewriter rewriter({rule}, {.match_only = true, .indexed = indexed});
          FAIL() << "Accepted tensor-to-attribute conversion";
        } catch (const OnnxRuntimeException& error) {
          EXPECT_NE(std::string(error.what()).find("Composed tensor annotation"), std::string::npos)
              << error.what();
        }
      }
    }
  }
  const PatternLibrary library(definitions + R"(
    pattern Entry(x,a:float) { p=Attr(x,a);return p.y }
    pattern Nested(x,a:float) { p=Entry(x,a);return p.y }
  )");
  for (const char* entry : {"Entry", "Nested"}) {
    for (bool indexed : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      b.AddNode("LeakyRelu", {x}, {b.MakeOutput()}).AddAttribute("alpha", 0.25f);
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      int checks = 0;
      auto check = [&](const Graph&, Match& match) {
        ++checks;
        EXPECT_FLOAT_EQ(match.attributeOf("a").f(), 0.25f);
        const auto child = match.instance("p");
        EXPECT_FLOAT_EQ(child.attributeOf(std::string_view(entry) == "Entry" ? "alpha" : "a").f(), 0.25f);
        return true;
      };
      Rule rule{"Valid", library.GetPattern(entry), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, 1u);
      EXPECT_EQ(checks, 1);
    }
  }
}

TEST(PatternRewriterTests, RepeatedDimensionsRequireKnownEqualEvidence) {
  using Dim = std::variant<int64_t, std::string>;
  struct Case {
    Dim first, second;
    bool equal;
  };
  const Case cases[] = {
      {std::string{}, std::string{}, false}, {std::string{}, int64_t{2}, false}, {int64_t{2}, std::string{}, false}, {int64_t{2}, int64_t{2}, true}, {int64_t{2}, int64_t{3}, false}, {std::string("batch"), std::string("batch"), true}, {std::string("batch"), std::string("other"), false}, {std::string("batch"), int64_t{2}, false}, {int64_t{2}, std::string("batch"), false}, {std::string{}, std::string("batch"), false}, {std::string("batch"), std::string{}, false}};
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      for (bool same_tensor : {false, true}) {
        for (const auto& c : cases) {
          TestGraph test;
          auto& b = test.builder;
          auto* x = b.MakeSymbolicInput<float>({c.first, same_tensor ? c.second : Dim{std::string{}}});
          auto* y = same_tensor ? nullptr : b.MakeSymbolicInput<float>({c.second, std::string{}});
          if (same_tensor)
            b.AddNode("Identity", {x}, {b.MakeOutput()});
          else
            b.AddNode("CastLike", {x, y}, {b.MakeOutput()});
          b.SetGraphOutputs();
          ASSERT_STATUS_OK(test.graph.Resolve());
          auto rule = IdentityPair();
          rule.pattern = Pattern(same_tensor ? "(x:float[M,M])=>return Identity(x)"
                                             : "(x:float[M,K],y:float[M,N])=>return CastLike(x,y)");
          rule.replacement = Replacement("(x)=>Identity(x)");
          int checks = 0;
          rule.check_and_bind = [&](const Graph&, Match& match) {
            ++checks;
            if (const auto* integer = std::get_if<int64_t>(&c.first))
              EXPECT_EQ(std::get<int64_t>(match.dimensionOf("M")), *integer);
            else
              EXPECT_EQ(std::get<std::string>(match.dimensionOf("M")), std::get<std::string>(c.first));
            if (!same_tensor) {
              EXPECT_TRUE(std::holds_alternative<std::monostate>(match.dimensionOf("K")));
              EXPECT_TRUE(std::holds_alternative<std::monostate>(match.dimensionOf("N")));
            }
            return true;
          };
          InlinedVector<std::string> failures;
          Options options{.match_only = match_only, .indexed = indexed, .max_iterations = 1};
          options.on_failure = [&](const MatchFailure& failure) {
            if (failure.stage == FailureStage::TypeShape) failures.emplace_back(failure.reason);
          };
          Statistics stats;
          ASSERT_STATUS_OK(PatternRewriter({rule}, options).Run(test.graph, stats));
          EXPECT_EQ(stats.matches, c.equal ? 1u : 0u);
          EXPECT_EQ(stats.rewrites, c.equal && !match_only ? 1u : 0u);
          EXPECT_EQ(checks, c.equal ? 1 : 0);
          if (!c.equal) {
            ASSERT_FALSE(failures.empty());
            EXPECT_NE(failures.front().find("dimension 'M'"), std::string::npos);
            const bool unknown = c.first == Dim{std::string{}} || c.second == Dim{std::string{}};
            EXPECT_NE(failures.front().find(unknown ? "Unknown occurrence" : "Conflicting bindings"), std::string::npos);
          }
          ASSERT_STATUS_OK(test.graph.Resolve());
        }
      }
    }
  }
}

TEST(PatternRewriterTests, SingletonAndSelectedAlternativeDimensionsAllowUnknowns) {
  for (bool indexed : {false, true}) {
    for (const char* annotation : {
             "float[M,N]", "float[_,_]", "float[M,_] | float[_,M]",
             "float[M,M] | float[M,_]", "float[M,M] | float[_,N]"}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeSymbolicInput<float>({std::string{}, std::string{}});
      b.AddNode("Identity", {x}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto rule = IdentityPair();
      rule.pattern = Pattern("(x:" + std::string(annotation) + ")=>return Identity(x)");
      int checks = 0;
      rule.check_and_bind = [&](const Graph&, Match& match) {
        ++checks;
        if (std::string_view(annotation).find('M') != std::string::npos)
          EXPECT_TRUE(std::holds_alternative<std::monostate>(match.dimensionOf("M")));
        if (std::string_view(annotation).find('N') != std::string::npos)
          EXPECT_TRUE(std::holds_alternative<std::monostate>(match.dimensionOf("N")));
        return true;
      };
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, 1u);
      EXPECT_EQ(checks, 1);
    }
    // The first alternative tentatively binds M, but fails unknown equality.
    // The successful alternative must neither inherit that binding nor its occurrence count.
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeSymbolicInput<float>({int64_t{2}, std::string{}});
    b.AddNode("Identity", {x}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto rule = IdentityPair();
    rule.pattern = Pattern("(x:float[M,M] | float[_,M])=>return Identity(x)");
    rule.check_and_bind = [](const Graph&, Match& match) {
      EXPECT_TRUE(std::holds_alternative<std::monostate>(match.dimensionOf("M")));
      return true;
    };
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
    TestGraph pair;
    auto& pair_builder = pair.builder;
    auto* unknown = pair_builder.MakeSymbolicInput<float>({std::string{}});
    auto* known = pair_builder.MakeInput<float>({2}, -1.0f, 1.0f);
    pair_builder.AddNode("CastLike", {unknown, known}, {pair_builder.MakeOutput()});
    pair_builder.SetGraphOutputs();
    ASSERT_STATUS_OK(pair.graph.Resolve());
    rule.pattern = Pattern("(x:float[M] | float[_],y:float[M])=>return CastLike(x,y)");
    rule.check_and_bind = [](const Graph&, Match& match) {
      EXPECT_EQ(std::get<int64_t>(match.dimensionOf("M")), 2);
      return true;
    };
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(pair.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
  }
}

TEST(PatternRewriterTests, ComposedDimensionOccurrencesAreHygienicAndConjunctive) {
  const PatternLibrary library(R"(
    interface Capture(a:float[M]) -> (y)
    interface Equality(a:float[M]) -> (y:float[M])
    pattern Pass implements Capture { return (y=a) }
    pattern Equal implements Equality { return (y=a) }
    pattern Alias=Pass
    pattern Wrapped(a) { p=Alias(a);return (y=p.y) }
    pattern Single(x) { p=Wrapped(x);return Identity(p.y) }
    pattern Independent(x,y) { p=Wrapped(x);q=Wrapped(y);return CastLike(p.y,q.y) }
    pattern Repeated(x) { p=Equal(x);return Identity(p.y) }
  )");
  for (bool indexed : {false, true}) {
    for (const char* entry : {"Single", "Independent", "Repeated"}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeSymbolicInput<float>({std::string{}});
      if (std::string_view(entry) == "Independent") {
        auto* y = b.MakeSymbolicInput<float>({std::string{}});
        b.AddNode("CastLike", {x, y}, {b.MakeOutput()});
      } else {
        b.AddNode("Identity", {x}, {b.MakeOutput()});
      }
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      int checks = 0;
      auto check = [&](const Graph&, Match& match) {
        ++checks;
        auto view = match.instance("p").instance("p");
        EXPECT_TRUE(std::holds_alternative<std::monostate>(view.dimensionOf("M")));
        if (std::string_view(entry) == "Independent")
          EXPECT_TRUE(std::holds_alternative<std::monostate>(match.instance("q").instance("p").dimensionOf("M")));
        return true;
      };
      Rule rule{"Composition", library.GetPattern(entry), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      const bool expected = std::string_view(entry) != "Repeated";
      EXPECT_EQ(stats.matches, expected ? 1u : 0u);
      EXPECT_EQ(checks, expected ? 1 : 0);
    }
  }
}

TEST(PatternRewriterTests, DimensionEqualityRejectionRetriesGraphAlternatives) {
  const PatternLibrary library(R"(
    interface I(x) -> (y)
    pattern Strict implements I { y:float[M,M]=Identity(x);return y }
    pattern Loose implements I { y:float[M,N]=Identity(x);return y }
    pattern Wild implements I { y:float[_,_]=Identity(x);return y }
    pattern Forms=Strict | Loose | Wild
    pattern Entry(x) { p=Forms(x);return p.y }
  )");
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeSymbolicInput<float>({std::string{}, std::string{}});
      auto* output = b.MakeOutput();
      b.AddNode("Identity", {x}, {output});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      // Identity shape inference may supply symbolic output dimensions. The
      // candidate metadata here deliberately has two anonymous dimensions.
      output->SetShape(*x->Shape());
      InlinedVector<std::string> checks;
      auto check = [&](const Graph&, Match& match) {
        const auto view = match.instance("p");
        checks.emplace_back(view.alternative());
        if (view.alternative() == "Loose") {
          EXPECT_TRUE(std::holds_alternative<std::monostate>(view.dimensionOf("M")));
          EXPECT_TRUE(std::holds_alternative<std::monostate>(view.dimensionOf("N")));
          ONNX_NAMESPACE::AttributeProto attribute;
          attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
          attribute.set_i(1);
          match.update("only_loose", attribute);
          return match.Reject("try wild");
        }
        EXPECT_EQ(view.alternative(), "Wild");
        EXPECT_THROW(match.attributeOf("only_loose"), OnnxRuntimeException);
        return true;
      };
      Rule rule{"Forms", library.GetPattern("Entry"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = match_only, .indexed = indexed, .max_iterations = 1})
                           .Run(test.graph, stats));
      ASSERT_EQ(checks.size(), 2u);
      EXPECT_EQ(checks[0], "Loose");
      EXPECT_EQ(checks[1], "Wild");
      EXPECT_EQ(stats.matches, 1u);
      EXPECT_EQ(stats.rewrites, match_only ? 0u : 1u);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, PositionalSignaturesCallsAndOptionalScaling) {
  const PatternLibrary library(R"(
    pattern plain(g:float[M,N],u:float[M,N]) -> (product:float[M,N]) {
      s=Sigmoid(g);a=Mul(g,s);product=Mul(a,u);return product
    }
    pattern scaled(g:float[M,N],u:float[M,N],alpha:float[...]) -> (product:float[M,N]) {
      v=Mul(g,alpha);s=Sigmoid(v);a=Mul(g,s);product=Mul(a,u);return product
    }
    pattern explicit(g,u,alpha:float[...] | None) {
      product=plain(g,u) | scaled(g,u,alpha);return product
    }
    pattern wildcard(g,u) {
      product=plain(g,u) | scaled(g,u,_);return product
    }
    pattern factored(g,u,alpha:float[...] | None) {
      v=(Mul(g,alpha) | g);s=Sigmoid(v);a=Mul(g,s);product=Mul(a,u);return product
    }
  )");
  for (const char* entry : {"explicit", "wildcard", "factored"}) {
    for (bool indexed : {false, true}) {
      for (bool scaled : {false, true}) {
        TestGraph test;
        auto& b = test.builder;
        auto* g = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
        auto* u = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
        auto* alpha = scaled ? b.MakeScalarInitializer<float>(0.5f) : nullptr;
        auto* v = g;
        if (scaled) {
          v = b.MakeIntermediate();
          b.AddNode("Mul", {g, alpha}, {v});
        }
        auto* s = b.MakeIntermediate();
        auto* a = b.MakeIntermediate();
        auto* output = b.MakeOutput();
        b.AddNode("Sigmoid", {v}, {s});
        b.AddNode("Mul", {g, s}, {a});
        b.AddNode("Mul", {a, u}, {output});
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        int checks = 0;
        auto check = [&](const Graph&, Match& match) {
          ++checks;
          EXPECT_EQ(match.valueOf("product"), output);
          if (std::string_view(entry) != "wildcard") {
            EXPECT_EQ(match.optionalValueOf("alpha"), alpha);
            EXPECT_EQ(match.valueOf("alpha"), alpha);
          } else {
            EXPECT_THROW(match.optionalValueOf("alpha"), OnnxRuntimeException);
            EXPECT_THROW(match.optionalValueOf("_"), OnnxRuntimeException);
          }
          EXPECT_THROW(match.optionalValueOf("unknown"), OnnxRuntimeException);
          if (std::string_view(entry) != "factored") {
            auto view = match.instance("product");
            EXPECT_EQ(view.alternative(), scaled ? "scaled" : "plain");
            EXPECT_EQ(view.alternativeView().valueOf("product"), output);
            if (scaled) EXPECT_EQ(view.valueOf("alpha"), alpha);
          } else {
            EXPECT_EQ(match.valueOf("v"), v);
            EXPECT_EQ(match.instance("v").alternative(), scaled ? "Mul" : "g");
          }
          return true;
        };
        Rule rule{"Scaled", library.GetPattern(entry), "(g,u)=>Mul(g,u)", check, {{kOnnxDomain, 17, 17}}};
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
        EXPECT_EQ(checks, 1);
        EXPECT_EQ(stats.matches, 1u);
      }
    }
  }
}

TEST(PatternRewriterTests, PositionalMultiOutputAndLegacyInstanceNamespaces) {
  const PatternLibrary library(R"(
    pattern Pair(x:float[4]) -> (first:float[2],second:float[2]) {
      left,right=Split(x,axis=0);return right,left
    }
    pattern Alias=Pair
    pattern Entry(x) { a,b=Alias(x);return Sub(a,b) }
    pattern Legacy(x) { p=Pair(x);return Sub(p.first,p.second) }
  )");
  for (const char* entry : {"Entry", "Legacy"}) {
    for (bool indexed : {false, true}) {
      for (bool match_only : {false, true}) {
        TestGraph test;
        auto& b = test.builder;
        auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
        auto* left = b.MakeIntermediate();
        auto* right = b.MakeIntermediate();
        b.AddNode("Split", {x}, {left, right}).AddAttribute("axis", int64_t{0});
        b.AddNode("Sub", {right, left}, {b.MakeOutput()});
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        int checks = 0;
        auto check = [&](const Graph&, Match& match) {
          ++checks;
          if (std::string_view(entry) == "Entry") {
            EXPECT_EQ(match.valueOf("a"), right);
            EXPECT_EQ(match.valueOf("b"), left);
            EXPECT_EQ(match.instance("a").valueOf("first"), right);
            EXPECT_THROW(match.instance("b"), OnnxRuntimeException);
          } else {
            EXPECT_THROW(match.valueOf("p"), OnnxRuntimeException);
            EXPECT_THROW(match.optionalValueOf("p"), OnnxRuntimeException);
            EXPECT_EQ(match.instance("p").valueOf("first"), right);
          }
          return true;
        };
        Rule rule{"Pair", library.GetPattern(entry), "(x)=>{a,b=Split(x,axis=0);return Sub(b,a)}", check, {{kOnnxDomain, 17, 17}}};
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = match_only, .indexed = indexed, .max_iterations = 1}).Run(test.graph, stats));
        EXPECT_EQ(checks, 1);
        EXPECT_EQ(stats.matches, 1u);
        EXPECT_EQ(stats.rewrites, match_only ? 0u : 1u);
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
    }
  }
}

TEST(PatternRewriterTests, ExpressionIdentityPriorityAndBindingIsolation) {
  const PatternLibrary library(R"(
    pattern Long(x,a) -> (y) { y=Mul(x,a);return y }
    pattern Short(x) -> (y) { return x }
    pattern Entry(x,a:float[...] | None) {
      v=Long(x,a) | Short(x);return Add(v,x)
    }
    pattern PreferShort(x,a:float[...] | None) {
      v=Short(x) | Long(x,a);return Add(v,x)
    }
    pattern Direct(x) { v=x;return Add(v,x) }
  )");
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      for (bool prefer_short : {false, true}) {
        for (bool reject_long : {false, true}) {
          TestGraph test;
          auto& b = test.builder;
          auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
          auto* alpha = b.MakeScalarInitializer<float>(0.5f);
          auto* mul = b.MakeIntermediate();
          b.AddNode("Mul", {x, alpha}, {mul});
          b.AddNode("Add", {mul, x}, {b.MakeOutput()});
          b.SetGraphOutputs();
          ASSERT_STATUS_OK(test.graph.Resolve());
          InlinedVector<std::string> calls;
          auto check = [&](const Graph&, Match& match) {
            const std::string selected(match.instance("v").alternative());
            calls.push_back(selected);
            if (selected == "Long") {
              EXPECT_EQ(match.optionalValueOf("a"), alpha);
              if (reject_long) return match.Reject("try identity");
            }
            return true;
          };
          Rule rule{"Priority", library.GetPattern(prefer_short ? "PreferShort" : "Entry"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
          Statistics stats;
          ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = match_only, .indexed = indexed, .max_iterations = 1})
                               .Run(test.graph, stats));
          // Short aliases v and x, so Add(v,x) cannot match this graph's distinct operands.
          ASSERT_EQ(calls.size(), 1u);
          EXPECT_EQ(calls.front(), "Long");
          EXPECT_EQ(stats.matches, reject_long ? 0u : 1u);
        }
      }
    }
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* other = b.MakeInput<float>({2}, -1.0f, 1.0f);
    b.AddNode("Add", {x, other}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Rule rule{"Alias", library.GetPattern("Direct"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, 0u);
  }
}

TEST(PatternRewriterTests, ExpressionAlternativesRetryOverlappingLongAndIdentity) {
  const PatternLibrary library(R"(
    pattern Scale(x,a) -> (y) { y=Mul(x,a);return y }
    pattern Pass(x) -> (y) { return x }
    pattern Entry(x,a:float[...] | None) { y=Scale(x,a) | Pass(x);return Sigmoid(y) }
    pattern ShortFirst(x,a:float[...] | None) { y=Pass(x) | Scale(x,a);return Sigmoid(y) }
  )");
  for (bool indexed : {false, true}) {
    for (bool short_first : {false, true}) {
      for (bool reject_first : {false, true}) {
        TestGraph test;
        auto& b = test.builder;
        auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
        auto* alpha = b.MakeScalarInitializer<float>(0.5f);
        auto* mul = b.MakeIntermediate();
        b.AddNode("Mul", {x, alpha}, {mul});
        auto* output = b.MakeOutput();
        b.AddNode("Sigmoid", {mul}, {output});
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        InlinedVector<std::string> calls;
        auto check = [&](const Graph&, Match& match) {
          const auto selected = match.instance("y").alternative();
          calls.emplace_back(selected);
          if (calls.size() == 1 && reject_first) {
            ONNX_NAMESPACE::AttributeProto attr;
            attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
            attr.set_i(1);
            match.update("only_first", attr);
            return match.Reject("try next");
          }
          EXPECT_THROW(match.attributeOf("only_first"), OnnxRuntimeException);
          EXPECT_EQ(match.valueOf("y"), mul);
          EXPECT_EQ(match.valueOf("x"), selected == "Scale" ? x : mul);
          EXPECT_EQ(match.optionalValueOf("a"), selected == "Scale" ? alpha : nullptr);
          EXPECT_EQ(match.instance("y").alternativeView().valueOf("y"), mul);
          return true;
        };
        Rule rule{"Overlap", library.GetPattern(short_first ? "ShortFirst" : "Entry"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed, .max_iterations = 1}).Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, 1u);
        ASSERT_EQ(calls.size(), reject_first ? 2u : 1u);
        EXPECT_EQ(calls.front(), short_first ? "Pass" : "Scale");
        if (reject_first) EXPECT_EQ(calls.back(), short_first ? "Scale" : "Pass");
        ASSERT_STATUS_OK(test.graph.Resolve());
      }
    }
  }
}

TEST(PatternRewriterTests, OptionalBranchCaptureElsewhereAndRequiredSlotSafety) {
  const PatternLibrary library(R"(
    pattern Entry(x,alpha:float[...] | None) {
      v=(Mul(x,alpha) | x);s=Sigmoid(v);return Add(s,alpha)
    }
    pattern Optional(x,alpha:float[...] | None) { y=Identity(x);return y }
    pattern Elsewhere(x,alpha:float[...] | None) {
      v=(Mul(x,alpha) | x);s=Sigmoid(v);return CastLike(s,alpha)
    }
    pattern RequiredResult(a:float | None) -> (y) { return a }
    pattern RequiredAtCall(x,a:float | None) { v=RequiredResult(a);return Clip(x,v) }
  )");
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* alpha = b.MakeScalarInitializer<float>(0.25f);
    auto* s = b.MakeIntermediate();
    b.AddNode("Sigmoid", {x}, {s});
    b.AddNode("Add", {s, alpha}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    int checks = 0;
    auto check = [&](const Graph&, Match& match) {
      ++checks;
      EXPECT_EQ(match.optionalValueOf("alpha"), alpha);
      return true;
    };
    Rule rule{"BoundElsewhere", library.GetPattern("Entry"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(checks, 1);
    TestGraph missing;
    auto& mb = missing.builder;
    auto* input = mb.MakeInput<float>({2}, -1.0f, 1.0f);
    mb.AddNode("Identity", {input}, {mb.MakeOutput()});
    mb.SetGraphOutputs();
    ASSERT_STATUS_OK(missing.graph.Resolve());
    Rule invalid{"AbsentReplacement", library.GetPattern("Optional"), "(alpha)=>Identity(alpha)", {}, {{kOnnxDomain, 17, 17}}};
    const auto status = PatternRewriter({invalid}, {.indexed = indexed}).Run(missing.graph, stats);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(missing.graph.NumberOfNodes(), 1);
    ASSERT_STATUS_OK(missing.graph.Resolve());
    for (bool present : {false, true}) {
      TestGraph clip;
      auto& cb = clip.builder;
      auto* clip_x = cb.MakeInput<float>({2}, -1.0f, 1.0f);
      if (present)
        cb.AddNode("Clip", {clip_x, cb.MakeScalarInitializer<float>(0.0f)}, {cb.MakeOutput()});
      else
        cb.AddNode("Clip", {clip_x}, {cb.MakeOutput()});
      cb.SetGraphOutputs();
      ASSERT_STATUS_OK(clip.graph.Resolve());
      Rule required_result{"RequiredResult", library.GetPattern("RequiredAtCall"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
      ASSERT_STATUS_OK(PatternRewriter({required_result}, {.match_only = true, .indexed = indexed}).Run(clip.graph, stats));
      EXPECT_EQ(stats.matches, present ? 1u : 0u);
    }
    for (bool floating : {false, true}) {
      TestGraph elsewhere;
      auto& eb = elsewhere.builder;
      auto* elsewhere_x = eb.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* a = floating ? eb.MakeScalarInitializer<float>(0.25f) : eb.MakeScalarInitializer<int64_t>(1);
      auto* sigmoid = eb.MakeIntermediate();
      eb.AddNode("Sigmoid", {elsewhere_x}, {sigmoid});
      eb.AddNode("CastLike", {sigmoid, a}, {eb.MakeOutput()});
      eb.SetGraphOutputs();
      ASSERT_STATUS_OK(elsewhere.graph.Resolve());
      Rule typed{"OptionalType", library.GetPattern("Elsewhere"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
      ASSERT_STATUS_OK(PatternRewriter({typed}, {.match_only = true, .indexed = indexed}).Run(elsewhere.graph, stats));
      EXPECT_EQ(stats.matches, floating ? 1u : 0u);
    }
  }
}

TEST(PatternRewriterTests, PositionalExpressionValidationAndLimits) {
  const std::string definitions = R"(
    pattern One(x:float[2]) -> (y:float[2]) { y=Identity(x);return y }
    pattern Different(x:float[2],a:float[...]) -> (z:int64[2]) { z=Cast(x,to=7);return z }
    pattern Two(x) -> (a,b) { a,b=Split(x,axis=0);return a,b }
    pattern Alias(x) { return x }
    pattern Attribute(x,a:float) { return LeakyRelu(x,alpha=a) }
  )";
  for (const char* entry : {
           "pattern E(x) { a,b=One(x);return Add(a,b) }",
           "pattern E(x) { a,b,c=Two(x);return Add(a,b) }",
           "pattern E(x) { a,a=Two(x);return a }",
           "pattern E(x) { a,x=Two(x);return a }",
           "pattern E(x) { p=Two(x);return Identity(p) }",
           "pattern E(x) { p=Two(x);p=Identity(x);return p }",
           "pattern E(x) { p=Two(x);v=p;return Identity(v) }",
           "pattern E(x) { a,b=Two(x);b=One(x);return Add(a,b) }",
           "pattern E(x) { p=Two(x);return LeakyRelu(x,alpha=p) }",
           "pattern E(x) { p:float[2]=Two(x);return Add(p.a,p.b) }",
           "pattern E(x,a) { y=One(x) | Different(x,a);return y }",
           "pattern E(x) { y=One(x) | Two(x);return Identity(y) }",
           "pattern E(x,a,b) { alpha=Add(a,b);v=(Mul(x,alpha)|x);return Sigmoid(v) }",
           "pattern E(x,a) { v=(Mul(x,a)|x);return Sigmoid(v) }",
           "pattern E(x) { y=One(x,x);return y }",
           "pattern E(x) { y=Attribute(x,_);return y }",
           "pattern E(x) { y=Alias(missing);return Identity(y) }",
           "pattern E(x) { y=x | missing;return Identity(y) }",
           "pattern E(x) -> (y:float[2] | None) { y=Identity(x);return y }"}) {
    SCOPED_TRACE(entry);
    EXPECT_THROW((void)PatternLibrary(definitions + entry), OnnxRuntimeException);
  }
  EXPECT_NO_THROW((void)PatternLibrary(R"(
    pattern E(x,a,b) {
      alpha=Add(a,b);v=(Mul(x,alpha)|x);s=Sigmoid(v);return Add(s,alpha)
    }
  )"));
  std::string text = "pattern E(x) {";
  for (int i = 0; i < 7; ++i) {
    const auto input = i == 0 ? std::string("x") : "v" + std::to_string(i - 1);
    text += "v" + std::to_string(i) + "=(Identity(" + input + ") | " + input + ");";
  }
  text += "return Sigmoid(v6)}";
  EXPECT_THROW((void)PatternLibrary(text), OnnxRuntimeException);
}

TEST(PatternRewriterTests, WildcardActualsAreIndependentRealConstrainedValues) {
  const PatternLibrary library(R"(
    pattern Double(x:float[2]) -> (y) { y=Add(x,x);return y }
    pattern Independent() { a=Double(_);b=Double(_);return Mul(a,b) }
    pattern Constant(x,a:const float) -> (y) { y=Mul(x,a);return y }
    pattern WithConstant(x) { y=Constant(x,_);return y }
    pattern ClipMin(x,a:float | None) -> (y) { y=Clip(x,a);return y }
    pattern WithMinimum(x) { y=ClipMin(x,_);return y }
  )");
  for (bool indexed : {false, true}) {
    for (bool same : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* z = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* a = b.MakeIntermediate();
      auto* c = b.MakeIntermediate();
      b.AddNode("Add", {x, same ? x : z}, {a});
      b.AddNode("Add", {z, z}, {c});
      b.AddNode("Mul", {a, c}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      int checks = 0;
      auto check = [&](const Graph&, Match& match) {
        ++checks;
        EXPECT_EQ(match.instance("a").valueOf("x"), x);
        EXPECT_EQ(match.instance("b").valueOf("x"), z);
        EXPECT_THROW(match.valueOf("_"), OnnxRuntimeException);
        return true;
      };
      Rule rule{"Independent", library.GetPattern("Independent"), "()=>Identity(0.0)", check, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(checks, same ? 1 : 0);
    }
    for (int scenario = 0; scenario < 4; ++scenario) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* alpha = scenario == 0   ? b.MakeScalarInitializer<float>(0.5f)
                    : scenario == 1 ? b.MakeInput<float>({}, 0.5f, 0.5f)
                    : scenario == 2 ? b.MakeInitializer<float>({1}, {0.5f})
                                    : b.MakeScalarInitializer<int64_t>(1);
      if (scenario == 3) {
        auto* integer = b.MakeInput<int64_t>({2}, 1, 2);
        b.AddNode("Mul", {integer, alpha}, {b.MakeOutput()});
      } else {
        b.AddNode("Mul", {x, alpha}, {b.MakeOutput()});
      }
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Rule rule{"Constant", library.GetPattern("WithConstant"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, scenario == 0 ? 1u : 0u);
    }
    for (bool present : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
      std::vector<NodeArg*> inputs{x};
      if (present) inputs.push_back(b.MakeScalarInitializer<float>(0.0f));
      b.AddNode("Clip", inputs, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      Rule rule{"RealWildcard", library.GetPattern("WithMinimum"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, present ? 1u : 0u);
    }
  }
}

TEST(PatternRewriterTests, NestedMultiOutputExpressionsPreservePositionalBindings) {
  const PatternLibrary library(R"(
    pattern Pair(x:float[4]) -> (first:float[2],second:float[2]) {
      a,b=Split(x,axis=0);return b,a
    }
    pattern Added(x:float[4],alpha:float) -> (p:float[2],q:float[2]) {
      z=Add(x,alpha);a,b=Pair(z);return a,b
    }
    pattern Entry(x,alpha:float | None) {
      a,b=(Pair(x) | Added(x,alpha));return Sub(a,b)
    }
  )");
  for (bool indexed : {false, true}) {
    for (bool added : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      auto* x = b.MakeInput<float>({4}, -1.0f, 1.0f);
      auto* alpha = added ? b.MakeScalarInitializer<float>(0.25f) : nullptr;
      auto* z = x;
      if (added) {
        z = b.MakeIntermediate();
        b.AddNode("Add", {x, alpha}, {z});
      }
      auto* left = b.MakeIntermediate();
      auto* right = b.MakeIntermediate();
      b.AddNode("Split", {z}, {left, right}).AddAttribute("axis", int64_t{0});
      b.AddNode("Sub", {right, left}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      InlinedVector<std::string> attempts;
      auto check = [&](const Graph&, Match& match) {
        auto view = match.instance("a");
        attempts.emplace_back(view.alternative());
        if (added && view.alternative() == "Pair") return match.Reject("Prefer the included Add.");
        EXPECT_EQ(match.valueOf("a"), right);
        EXPECT_EQ(match.valueOf("b"), left);
        EXPECT_EQ(match.optionalValueOf("alpha"), alpha);
        if (added) {
          auto nested = view.instance("a");
          EXPECT_EQ(nested.valueOf("first"), right);
          EXPECT_EQ(nested.valueOf("second"), left);
        }
        return true;
      };
      Rule rule{"NestedPair", library.GetPattern("Entry"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(stats.matches, 1u);
      EXPECT_EQ(attempts, added ? InlinedVector<std::string>({"Pair", "Added"})
                                : InlinedVector<std::string>({"Pair"}));
    }
  }
}

TEST(PatternRewriterTests, PositionalSelectedDimensionsAndTensorAliasContracts) {
  const PatternLibrary library(R"(
    pattern Repeated(x:float[M,M]) -> (y) { return x }
    pattern Single(x:float[M,N]) -> (y) { return x }
    pattern Entry(x) { y=Repeated(x) | Single(x);return Identity(y) }
    pattern Signature(x:float[M]) -> (y:float[M]) { y=Identity(x);return y }
  )");
  for (bool indexed : {false, true}) {
    for (bool known : {false, true}) {
      TestGraph test;
      auto& b = test.builder;
      using Dim = std::variant<int64_t, std::string>;
      const Dim dim = known ? Dim{int64_t{2}} : Dim{std::string{}};
      auto* x = b.MakeSymbolicInput<float>({dim, dim});
      b.AddNode("Identity", {x}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      InlinedVector<std::string> attempts;
      auto check = [&](const Graph&, Match& match) {
        const auto view = match.instance("y");
        attempts.emplace_back(view.alternative());
        if (known)
          EXPECT_EQ(std::get<int64_t>(view.dimensionOf("M")), 2);
        else
          EXPECT_TRUE(std::holds_alternative<std::monostate>(view.dimensionOf("M")));
        return true;
      };
      Rule rule{"SelectedDimensions", library.GetPattern("Entry"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true, .indexed = indexed}).Run(test.graph, stats));
      EXPECT_EQ(attempts, known ? InlinedVector<std::string>({"Repeated", "Single"})
                                : InlinedVector<std::string>({"Single"}));
      TestGraph signature;
      auto& sb = signature.builder;
      auto* input = sb.MakeSymbolicInput<float>({dim});
      sb.AddNode("Identity", {input}, {sb.MakeOutput()});
      sb.SetGraphOutputs();
      ASSERT_STATUS_OK(signature.graph.Resolve());
      Rule signed_rule{"Signature", library.GetPattern("Signature"), "(x)=>Identity(x)", {}, {{kOnnxDomain, 17, 17}}};
      ASSERT_STATUS_OK(PatternRewriter({signed_rule}, {.match_only = true, .indexed = indexed}).Run(signature.graph, stats));
      EXPECT_EQ(stats.matches, known ? 1u : 0u);
    }
  }
  for (const char* source : {
           "pattern P(x,a:float){v=a;return LeakyRelu(x,alpha=v)}",
           "pattern P(x,a:float){y=LeakyRelu(x,alpha=a);v=a;return Identity(y)}",
           "pattern F(a)->(y){return a} pattern P(x,a){p=F(a);return LeakyRelu(x,alpha=a)}"}) {
    SCOPED_TRACE(source);
    EXPECT_THROW((void)PatternLibrary(source), OnnxRuntimeException);
  }
  EXPECT_NO_THROW((void)PatternLibrary(R"(
    pattern P(x) { alias=x;return Add(alias,x) }
    pattern Strings(x) { y=test::Op(x,label=")|(");return y }
  )"));
}

TEST(PatternRewriterTests, InferredExpressionOutputsAllowUnconstrainedBranches) {
  for (bool alias_first : {false, true}) {
    const std::string text = std::string(R"(
      pattern Fixed(x:float[2]) { y=)") +
                             (alias_first ? "x | Identity(x)" : "Identity(x) | x") + R"(;return y }
      pattern Symbolic(x:float[M]) { y=)" +
                             (alias_first ? "x | Identity(x)" : "Identity(x) | x") + R"(;return y }
      pattern FixedEntry(x) { y=Fixed(x);return Sigmoid(y) }
      pattern Nested(x) { y=Symbolic(x);return y }
      pattern SymbolicEntry(x) { y=Nested(x);return Sigmoid(y) }
    )";
    const PatternLibrary library(text);
    for (bool indexed : {false, true}) {
      for (bool match_only : {false, true}) {
        for (const char* entry : {"FixedEntry", "SymbolicEntry"}) {
          TestGraph test;
          auto& b = test.builder;
          auto* x = std::string_view(entry) == "FixedEntry"
                        ? b.MakeInput<float>({2}, -1.0f, 1.0f)
                        : b.MakeSymbolicInput<float>({std::string{}});
          auto* identity = b.MakeIntermediate();
          b.AddNode("Identity", {x}, {identity});
          b.AddNode("Sigmoid", {identity}, {b.MakeOutput()});
          b.SetGraphOutputs();
          ASSERT_STATUS_OK(test.graph.Resolve());
          int checks = 0;
          auto check = [&](const Graph&, Match& match) {
            ++checks;
            EXPECT_EQ(match.valueOf("y"), identity);
            if (std::string_view(entry) == "SymbolicEntry")
              EXPECT_TRUE(std::holds_alternative<std::monostate>(
                  match.instance("y").instance("y").dimensionOf("M")));
            return true;
          };
          Rule rule{"Inferred", library.GetPattern(entry), "(x)=>Sigmoid(x)", check, {{kOnnxDomain, 17, 17}}};
          Statistics stats;
          ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = match_only, .indexed = indexed, .max_iterations = 1})
                               .Run(test.graph, stats));
          EXPECT_EQ(checks, match_only ? 2 : 1);
          EXPECT_EQ(stats.matches, match_only ? 2u : 1u);
          EXPECT_EQ(stats.rewrites, match_only ? 0u : 1u);
          ASSERT_STATUS_OK(test.graph.Resolve());
        }
      }
    }
  }
}

TEST(PatternRewriterTests, DeclaredOutputsCannotShadowAliasesOrPositionalCallResults) {
  const std::string copy = "pattern Copy(x)->(y){return x} ";
  for (const char* body : {"a=x;return Identity(a)", "a=Copy(x);return Identity(a)",
                           "a,b=Pair(x);return Identity(a)"}) {
    const std::string text = copy +
                             "pattern Pair(x)->(first,second){first=x;second=x;return first,second} "
                             "pattern E(x:float[2])->(a){" +
                             body + "}";
    try {
      const PatternLibrary library(text);
      FAIL() << "Accepted an output name that shadows an alias: " << body;
    } catch (const OnnxRuntimeException& error) {
      EXPECT_NE(std::string(error.what()).find("Output port shadows a different local value"), std::string::npos)
          << body << ": " << error.what();
    }
  }
  for (const char* body : {"a=x;return a", "a=Copy(x);return a"}) {
    const PatternLibrary library(copy + "pattern F(x:float[2])->(a){" + body +
                                 "} "
                                 "pattern E(x){a=F(x);return Identity(a)}");
    for (bool indexed : {false, true}) {
      for (bool match_only : {false, true}) {
        TestGraph test;
        auto& b = test.builder;
        auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
        b.AddNode("Identity", {x}, {b.MakeOutput()});
        b.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        auto check = [&](const Graph&, Match& match) {
          EXPECT_EQ(match.valueOf("a"), x);
          EXPECT_EQ(match.instance("a").valueOf("a"), x);
          if (std::string_view(body).find("Copy") != std::string_view::npos)
            EXPECT_EQ(match.instance("a").instance("a").valueOf("y"), x);
          return true;
        };
        Rule rule{"SameHandle", library.GetPattern("E"), "(x)=>Identity(x)", check, {{kOnnxDomain, 17, 17}}};
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = match_only, .indexed = indexed, .max_iterations = 1})
                             .Run(test.graph, stats));
        EXPECT_EQ(stats.matches, 1u);
      }
    }
  }
}

TEST(PatternRewriterTests, OrdinaryOutputSignaturesPreserveTypedAttributeInputs) {
  const PatternLibrary library(R"(
    pattern Int(x,axis:int)->(y){y=Softmax(x,axis=axis);return y}
    pattern Ints(x,axes:ints)->(y){y=Transpose(x,perm=axes);return y}
    pattern Float(x,alpha:float)->(y){y=LeakyRelu(x,alpha=alpha);return y}
    pattern NestedInt(x,axis:int)->(y){y=Int(x,axis);return y}
    pattern NestedInts(x,axes:ints)->(y){y=Ints(x,axes);return y}
    pattern NestedFloat(x,alpha:float)->(y){y=Float(x,alpha);return y}
  )");
  for (bool indexed : {false, true}) {
    for (bool match_only : {false, true}) {
      for (bool nested : {false, true}) {
        for (const char* kind : {"Int", "Ints", "Float"}) {
          TestGraph test;
          auto& b = test.builder;
          auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
          const std::string_view type(kind);
          auto& node = b.AddNode(type == "Int" ? "Softmax" : type == "Ints" ? "Transpose"
                                                                            : "LeakyRelu",
                                 {x}, {b.MakeOutput()});
          if (type == "Int")
            node.AddAttribute("axis", int64_t{1});
          else if (type == "Ints")
            node.AddAttribute("perm", std::vector<int64_t>{1, 0});
          else
            node.AddAttribute("alpha", 0.25f);
          b.SetGraphOutputs();
          ASSERT_STATUS_OK(test.graph.Resolve());
          int checks = 0;
          auto check = [&](const Graph&, Match& match) {
            ++checks;
            const auto view = nested ? match.instance("y") : match.bindings();
            if (type == "Int") {
              EXPECT_EQ(match.attributeOf("axis").i(), 1);
              EXPECT_EQ(view.attributeOf("axis").i(), 1);
            } else if (type == "Ints") {
              EXPECT_EQ(match.attributeOf("axes").ints_size(), 2);
              EXPECT_EQ(view.attributeOf("axes").ints(0), 1);
            } else {
              EXPECT_FLOAT_EQ(match.attributeOf("alpha").f(), 0.25f);
              EXPECT_FLOAT_EQ(view.attributeOf("alpha").f(), 0.25f);
            }
            return true;
          };
          const std::string entry = std::string(nested ? "Nested" : "") + kind;
          const char* recipe = type == "Int"    ? "(x,axis)=>Softmax(x,axis=axis)"
                               : type == "Ints" ? "(x,axes)=>Transpose(x,perm=axes)"
                                                : "(x,alpha)=>LeakyRelu(x,alpha=alpha)";
          Rule rule{"Attribute", library.GetPattern(entry), recipe, check, {{kOnnxDomain, 17, 17}}};
          Statistics stats;
          ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = match_only, .indexed = indexed, .max_iterations = 1})
                               .Run(test.graph, stats));
          EXPECT_EQ(checks, 1);
          EXPECT_EQ(stats.matches, 1u);
          EXPECT_EQ(stats.rewrites, match_only ? 0u : 1u);
          ASSERT_STATUS_OK(test.graph.Resolve());
        }
      }
    }
  }
  for (const char* text : {
           "interface I(x,axis:int)->(y)",
           "pattern P(x)->(y:ints){y=Identity(x);return y}",
           "pattern P(x,a:int)->(y){y=Softmax(x,axis=a);return y} "
           "pattern E(x){y=P(x,_);return y}"}) {
    SCOPED_TRACE(text);
    EXPECT_THROW((void)PatternLibrary(text), OnnxRuntimeException);
  }
}

TEST(PatternRewriterTests, FactoredOptionalScalingNumericalRewrite) {
  const PatternLibrary library(R"(
    pattern activation(g:float[M,N],u:float[M,N],alpha:float[...] | None) -> (product:float[M,N]) {
      scaled=(Mul(g,alpha) | g);s=Sigmoid(scaled);a=Mul(g,s);product=Mul(a,u);return product
    }
    pattern Fuse(g,u,alpha:float[...] | None) {
      product=activation(g,u,alpha);return Identity(product)
    }
  )");
  for (bool indexed : {false, true}) {
    for (int variant = 0; variant < 3; ++variant) {
      TestGraph test;
      auto& b = test.builder;
      auto* g = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto* u = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
      auto* alpha = variant == 0   ? nullptr
                    : variant == 1 ? b.MakeScalarInitializer<float>(0.5f)
                                   : b.MakeInitializer<float>({1}, {0.5f});
      auto* scaled = g;
      if (alpha) {
        scaled = b.MakeIntermediate();
        b.AddNode("Mul", {g, alpha}, {scaled});
      }
      auto* sigmoid = b.MakeIntermediate();
      auto* activated = b.MakeIntermediate();
      auto* product = b.MakeIntermediate();
      b.AddNode("Sigmoid", {scaled}, {sigmoid});
      b.AddNode("Mul", {g, sigmoid}, {activated});
      b.AddNode("Mul", {activated, u}, {product});
      b.AddNode("Identity", {product}, {b.MakeOutput()});
      b.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      const std::string original = test.model.ToProto().SerializeAsString();
      auto check = [alpha](const Graph&, Match& match) {
        EXPECT_EQ(match.optionalValueOf("alpha"), alpha);
        EXPECT_EQ(match.instance("product").optionalValueOf("alpha"), alpha);
        return true;
      };
      const Replacement plain(R"((g,u)=>{s=Sigmoid(g);a=Mul(g,s);return Mul(a,u)})");
      const Replacement with_alpha(R"((g,u,alpha)=>{v=Mul(g,alpha);s=Sigmoid(v);a=Mul(g,s);return Mul(a,u)})");
      auto replacement = [plain, with_alpha](const Graph&, const Match& match) {
        return match.optionalValueOf("alpha") ? with_alpha : plain;
      };
      Rule rule{"Factored", library.GetPattern("Fuse"), replacement, check, {{kOnnxDomain, 17, 17}}};
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed, .max_iterations = 1}).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      EXPECT_EQ(Count(test.graph, "Identity"), 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
      const std::string rewritten = test.model.ToProto().SerializeAsString();
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
      ASSERT_EQ(expected.Shape(), actual.Shape());
      for (int64_t i = 0; i < expected.Shape().Size(); ++i)
        EXPECT_NEAR(expected.Data<float>()[i], actual.Data<float>()[i], 1e-6f);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

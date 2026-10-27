// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/optimizer/swiglu_fusion.h"

#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

namespace {

constexpr int64_t kRows = 2;
constexpr int64_t kCols = 8;
constexpr float kAlpha = 1.5f;

struct BuildOptions {
  bool up_first = false;
  bool broadcast_up = false;
  bool extra_gelu_consumer = false;
  const char* ep = kCudaExecutionProvider;
};

// gate -> QuickGelu(alpha) -> Mul <- up
void BuildQuickGeluMul(ModelTestBuilder& builder, const BuildOptions& opts,
                       std::string& gate_name, std::string& up_name) {
  const std::vector<int64_t> shape{kRows, kCols};
  const std::vector<int64_t> up_shape = opts.broadcast_up ? std::vector<int64_t>{1, kCols} : shape;
  NodeArg* gate = builder.MakeInput<float>(shape, -3.0f, 3.0f);
  NodeArg* up = builder.MakeInput<float>(up_shape, -3.0f, 3.0f);
  NodeArg* activated = builder.MakeIntermediate<float>(shape);
  NodeArg* output = builder.MakeOutput<float>(shape);
  gate_name = gate->Name();
  up_name = up->Name();

  Node& quick_gelu = builder.AddNode("QuickGelu", {gate}, {activated}, kMSDomain);
  quick_gelu.AddAttribute("alpha", kAlpha);
  if (opts.up_first) {
    builder.AddNode("Mul", {up, activated}, {output});
  } else {
    builder.AddNode("Mul", {activated, up}, {output});
  }
  if (opts.extra_gelu_consumer) {
    builder.AddNode("Identity", {activated}, {builder.MakeOutput<float>(shape)});
  }

  for (auto& node : builder.graph_.Nodes()) {
    node.SetExecutionProviderType(opts.ep);
  }
}

std::unique_ptr<SwiGluFusion> MakeCudaTransformer() {
  return std::make_unique<SwiGluFusion>(InlinedHashSet<std::string_view>{kCudaExecutionProvider});
}

Status CheckFused(const Graph& graph, const std::string& gate_name, const std::string& up_name) {
  auto op_to_count = CountOpsInGraph(graph);
  ORT_RETURN_IF_NOT(op_to_count["com.microsoft.SwiGLU"] == 1, "Expected one SwiGLU node.");
  ORT_RETURN_IF_NOT(op_to_count["com.microsoft.QuickGelu"] == 0, "QuickGelu should be fused.");
  ORT_RETURN_IF_NOT(op_to_count["Mul"] == 0, "Mul should be fused.");

  for (const auto& node : graph.Nodes()) {
    if (node.OpType() != "SwiGLU") continue;
    ORT_RETURN_IF_NOT(node.InputDefs().size() == 2, "SwiGLU must take gate and up.");
    ORT_RETURN_IF_NOT(node.InputDefs()[0]->Name() == gate_name, "SwiGLU input 0 must be gate.");
    ORT_RETURN_IF_NOT(node.InputDefs()[1]->Name() == up_name, "SwiGLU input 1 must be up.");
    ORT_RETURN_IF_NOT(node.GetExecutionProviderType() == kCudaExecutionProvider, "SwiGLU must stay on CUDA.");
    const auto* alpha = graph_utils::GetNodeAttribute(node, "activation_alpha");
    ORT_RETURN_IF_NOT(alpha != nullptr && alpha->f() == kAlpha, "activation_alpha must carry QuickGelu alpha.");
    const auto* round_activation = graph_utils::GetNodeAttribute(node, "round_activation");
    ORT_RETURN_IF_NOT(round_activation != nullptr && round_activation->i() == 1,
                      "round_activation must keep QuickGelu's rounding to the tensor type.");
  }
  return Status::OK();
}

Status CheckUnfused(const Graph& graph) {
  auto op_to_count = CountOpsInGraph(graph);
  ORT_RETURN_IF_NOT(op_to_count["com.microsoft.SwiGLU"] == 0, "Graph was fused unexpectedly.");
  ORT_RETURN_IF_NOT(op_to_count["com.microsoft.QuickGelu"] == 1 && op_to_count["Mul"] == 1,
                    "QuickGelu and Mul should be kept.");
  return Status::OK();
}

void RunFusionTest(const BuildOptions& opts, bool expect_fused, const logging::Logger& logger) {
  std::string gate_name;
  std::string up_name;
  auto build = [&](ModelTestBuilder& builder) { BuildQuickGeluMul(builder, opts, gate_name, up_name); };
  auto check = [&](Graph& graph) {
    return expect_fused ? CheckFused(graph, gate_name, up_name) : CheckUnfused(graph);
  };
  ASSERT_STATUS_OK(TestGraphTransformer(build, /*opset_version=*/14, logger, MakeCudaTransformer(),
                                        TransformerLevel::Level2, /*steps=*/1, nullptr, check));
}

// x * Sigmoid(x) * up, which QuickGeluFusion turns into the pattern above.
template <typename T>
void BuildDecomposedSwiGlu(ModelTestBuilder& builder) {
  const std::vector<int64_t> shape{kRows, kCols};
  NodeArg* gate = builder.MakeInput<T>(shape, T(-3.0f), T(3.0f));
  NodeArg* up = builder.MakeInput<T>(shape, T(-3.0f), T(3.0f));
  NodeArg* sigmoid = builder.MakeIntermediate<T>(shape);
  NodeArg* silu = builder.MakeIntermediate<T>(shape);
  NodeArg* output = builder.MakeOutput<T>(shape);
  builder.AddNode("Sigmoid", {gate}, {sigmoid});
  builder.AddNode("Mul", {gate, sigmoid}, {silu});
  builder.AddNode("Mul", {silu, up}, {output});
}

template <typename T>
void RunCudaParityTest(double tolerance) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }

  auto check = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.SwiGLU"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.QuickGelu"], 0);
    EXPECT_EQ(op_to_count["Sigmoid"], 0);
    EXPECT_EQ(op_to_count["Mul"], 0);
  };

  TransformerTester(BuildDecomposedSwiGlu<T>, check, TransformerLevel::Level1, TransformerLevel::Level2,
                    /*opset_version=*/14, tolerance, 0.0, nullptr, {}, {}, std::move(cuda_ep));
}

constexpr int64_t kExactRows = 16;
constexpr int64_t kExactCols = 128;

// gate -> QuickGelu -> Mul <- up in T. QuickGelu rounds its output to T before the Mul, so the fused node has
// to as well. A subnormal or zero intermediate makes a missing rounding obvious, and a tolerance of zero holds
// the fusion to the unfused graph's exact result.
template <typename T>
void RunCudaQuickGeluMulExactParityTest(float gate_min, float gate_max, float up_min, float up_max) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }

  auto build = [&](ModelTestBuilder& builder) {
    const std::vector<int64_t> shape{kExactRows, kExactCols};
    NodeArg* gate = builder.MakeInput<T>(shape, T(gate_min), T(gate_max));
    NodeArg* up = builder.MakeInput<T>(shape, T(up_min), T(up_max));
    NodeArg* activated = builder.MakeIntermediate<T>(shape);
    NodeArg* output = builder.MakeOutput<T>(shape);
    Node& quick_gelu = builder.AddNode("QuickGelu", {gate}, {activated}, kMSDomain);
    quick_gelu.AddAttribute("alpha", 1.0f);
    builder.AddNode("Mul", {activated, up}, {output});
  };

  auto check = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.SwiGLU"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.QuickGelu"], 0);
    EXPECT_EQ(op_to_count["Mul"], 0);
  };

  TransformerTester(build, check, TransformerLevel::Level1, TransformerLevel::Level2,
                    /*opset_version=*/14, /*per_sample_tolerance=*/0.0, /*relative_per_sample_tolerance=*/0.0,
                    nullptr, {}, {}, std::move(cuda_ep));
}

}  // namespace

TEST_F(GraphTransformationTests, SwiGluFusionFusesQuickGeluMul) {
  RunFusionTest(BuildOptions{}, /*expect_fused=*/true, *logger_);
}

TEST_F(GraphTransformationTests, SwiGluFusionFusesWhenUpIsFirstMulInput) {
  BuildOptions opts;
  opts.up_first = true;
  RunFusionTest(opts, /*expect_fused=*/true, *logger_);
}

TEST_F(GraphTransformationTests, SwiGluFusionPreservesProducedUpEdge) {
  std::string gate_name;
  std::string up_name;
  NodeIndex producer_index = 0;
  auto build = [&](ModelTestBuilder& builder) {
    const std::vector<int64_t> shape{kRows, kCols};
    NodeArg* gate = builder.MakeInput<float>(shape, -3.0f, 3.0f);
    NodeArg* raw_up = builder.MakeInput<float>(shape, -3.0f, 3.0f);
    NodeArg* up = builder.MakeIntermediate<float>(shape);
    NodeArg* activated = builder.MakeIntermediate<float>(shape);
    Node& producer = builder.AddNode("Identity", {raw_up}, {up});
    producer_index = producer.Index();
    Node& quick_gelu = builder.AddNode("QuickGelu", {gate}, {activated}, kMSDomain);
    quick_gelu.AddAttribute("alpha", kAlpha);
    builder.AddNode("Mul", {activated, up}, {builder.MakeOutput<float>(shape)});
    gate_name = gate->Name();
    up_name = up->Name();
    for (auto& node : builder.graph_.Nodes()) {
      node.SetExecutionProviderType(kCudaExecutionProvider);
    }
  };
  auto before = [&](Graph& graph) {
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() != "Mul") continue;
      const auto* edge = graph_utils::GetInputEdge(node, 1);
      ORT_RETURN_IF_NOT(edge != nullptr && edge->GetNode().Index() == producer_index,
                        "The up producer edge must exist before fusion.");
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expected a Mul node.");
  };
  auto after = [&](Graph& graph) {
    ORT_RETURN_IF_ERROR(CheckFused(graph, gate_name, up_name));
    const Node* producer = graph.GetNode(producer_index);
    ORT_RETURN_IF_NOT(producer != nullptr && producer->OpType() == "Identity",
                      "The up producer must not be eliminated.");
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() != "SwiGLU") continue;
      const auto* edge = graph_utils::GetInputEdge(node, 1);
      ORT_RETURN_IF_NOT(edge != nullptr && edge->GetNode().Index() == producer_index &&
                            edge->GetSrcArgIndex() == 0 && edge->GetDstArgIndex() == 1,
                        "The producer must stay connected to SwiGLU input 1.");
    }
    return Status::OK();
  };
  Model model("ProducedUp", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 14}, {kMSDomain, 1}}, {}, *logger_);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);
  build(builder);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_STATUS_OK(before(graph));
  bool modified = false;
  // Check explicit rewiring before Resolve can rebuild edges; do not run Identity elimination.
  ASSERT_STATUS_OK(MakeCudaTransformer()->ApplyImpl(graph, modified, 0, *logger_));
  ASSERT_TRUE(modified);
  ASSERT_STATUS_OK(after(graph));
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_STATUS_OK(after(graph));
}

TEST_F(GraphTransformationTests, SwiGluFusionSkipsBroadcastMul) {
  BuildOptions opts;
  opts.broadcast_up = true;
  RunFusionTest(opts, /*expect_fused=*/false, *logger_);
}

TEST_F(GraphTransformationTests, SwiGluFusionSkipsSharedQuickGeluOutput) {
  BuildOptions opts;
  opts.extra_gelu_consumer = true;
  RunFusionTest(opts, /*expect_fused=*/false, *logger_);
}

TEST_F(GraphTransformationTests, SwiGluFusionSkipsCpuAssignedNodes) {
  BuildOptions opts;
  opts.ep = kCpuExecutionProvider;
  RunFusionTest(opts, /*expect_fused=*/false, *logger_);
}

TEST_F(GraphTransformationTests, SwiGluFusionCudaFloatParity) {
  RunCudaParityTest<float>(1e-5);
}

TEST_F(GraphTransformationTests, SwiGluFusionCudaFloat16Parity) {
  RunCudaParityTest<MLFloat16>(5e-3);
}

// The next tests compare the fused node bit for bit with QuickGelu followed by Mul.
TEST_F(GraphTransformationTests, SwiGluFusionCudaFloatExactAfterQuickGeluRounding) {
  // alpha * gate crosses -87, where QuickGelu switches to double and narrows the result to float.
  RunCudaQuickGeluMulExactParityTest<float>(-110.0f, -60.0f, 1e20f, 1e30f);
}

TEST_F(GraphTransformationTests, SwiGluFusionCudaFloat16ExactAfterQuickGeluRounding) {
  RunCudaQuickGeluMulExactParityTest<MLFloat16>(-12.0f, 6.0f, -8.0f, 8.0f);
}

TEST_F(GraphTransformationTests, SwiGluFusionCudaFloat16NegativeTailExact) {
  // QuickGelu lands in the fp16 subnormal range or on zero, and a large up scales the rounding error back up.
  RunCudaQuickGeluMulExactParityTest<MLFloat16>(-32.0f, -16.0f, 30000.0f, 65504.0f);
}

TEST_F(GraphTransformationTests, SwiGluFusionCudaBFloat16ExactAfterQuickGeluRounding) {
  RunCudaQuickGeluMulExactParityTest<BFloat16>(-12.0f, 6.0f, -8.0f, 8.0f);
}

TEST_F(GraphTransformationTests, SwiGluFusionCudaBFloat16NegativeTailExact) {
  // QuickGelu's bf16 output falls to subnormals and then to zero near -97, which an unrounded product with a
  // large up would hide.
  RunCudaQuickGeluMulExactParityTest<BFloat16>(-110.0f, -60.0f, 1e37f, 1e38f);
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime

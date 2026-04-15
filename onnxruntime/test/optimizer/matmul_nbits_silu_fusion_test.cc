// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/node_attr_utils.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/matmul_nbits_silu_fusion.h"
#include "core/optimizer/utils.h"

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

void SetWebGpuProvider(Node& node) {
  node.SetExecutionProviderType(kWebGpuExecutionProvider);
}

NodeAttributes MakeMatMulNBitsAttrs(int64_t k, int64_t n, int64_t block_size, int64_t bits, int64_t accuracy_level) {
  NodeAttributes attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("K", k), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("N", n), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("bits", bits), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level), attrs);
  return attrs;
}

Status CheckMatMulNBitsSiluFusedGraph(const Graph& graph) {
  const auto op_to_count = CountOpsInGraph(graph);
  if (OpCount(op_to_count, "com.microsoft.MatMulNBitsSiluMul") != 1 ||
      OpCount(op_to_count, "com.microsoft.MatMulNBits") != 0 ||
      OpCount(op_to_count, "Sigmoid") != 0 ||
      OpCount(op_to_count, "Mul") != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected operator counts after MatMulNBitsSiluFusion.");
  }

  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MatMulNBitsSiluMul") {
      ORT_RETURN_IF_NOT(node.Domain() == kMSDomain, "Fused node must be in com.microsoft domain.");
      ORT_RETURN_IF_NOT(node.GetExecutionProviderType() == kWebGpuExecutionProvider,
                        "Fused node must be assigned to WebGPU EP.");
    }
  }

  return Status::OK();
}

void BuildMatMulNBitsSiluWebGpuPattern(ModelTestBuilder& builder) {
  constexpr int64_t k = 16;
  constexpr int64_t n = 8;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  constexpr int64_t accuracy_level = 4;
  constexpr int64_t blob_size = block_size * bits / 8;

  NodeArg* input = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{1, k},
      std::vector<MLFloat16>{
          MLFloat16(-1.0f), MLFloat16(-0.875f), MLFloat16(-0.75f), MLFloat16(-0.625f),
          MLFloat16(-0.5f), MLFloat16(-0.375f), MLFloat16(-0.25f), MLFloat16(-0.125f),
          MLFloat16(0.125f), MLFloat16(0.25f), MLFloat16(0.375f), MLFloat16(0.5f),
          MLFloat16(0.625f), MLFloat16(0.75f), MLFloat16(0.875f), MLFloat16(1.0f)});
  NodeArg* optional_tensor = builder.MakeOptionalTensor();

  NodeArg* gate_weight = builder.MakeInitializer<uint8_t>({n, 1, blob_size}, uint8_t{0}, uint8_t{15});
  NodeArg* gate_scale = builder.MakeInitializer<MLFloat16>({n, 1}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* gate_bias = builder.MakeInitializer<MLFloat16>({n}, MLFloat16(0.0f), MLFloat16(0.0f));
  NodeArg* up_weight = builder.MakeInitializer<uint8_t>({n, 1, blob_size}, uint8_t{0}, uint8_t{15});
  NodeArg* up_scale = builder.MakeInitializer<MLFloat16>({n, 1}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* up_bias = builder.MakeInitializer<MLFloat16>({n}, MLFloat16(0.0f), MLFloat16(0.0f));

  NodeArg* gate_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
  NodeArg* up_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
  NodeArg* sigmoid_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
  NodeArg* silu_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
  NodeArg* output = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, n});

  NodeAttributes matmul_attrs = MakeMatMulNBitsAttrs(k, n, block_size, bits, accuracy_level);
  Node& gate_matmul = builder.AddNode("MatMulNBits", {input, gate_weight, gate_scale, optional_tensor, optional_tensor, gate_bias}, {gate_out}, kMSDomain, &matmul_attrs);
  Node& up_matmul = builder.AddNode("MatMulNBits", {input, up_weight, up_scale, optional_tensor, optional_tensor, up_bias}, {up_out}, kMSDomain, &matmul_attrs);
  Node& sigmoid = builder.AddNode("Sigmoid", {gate_out}, {sigmoid_out});
  Node& silu_mul = builder.AddNode("Mul", {gate_out, sigmoid_out}, {silu_out});
  Node& final_mul = builder.AddNode("Mul", {silu_out, up_out}, {output});

  SetWebGpuProvider(gate_matmul);
  SetWebGpuProvider(up_matmul);
  SetWebGpuProvider(sigmoid);
  SetWebGpuProvider(silu_mul);
  SetWebGpuProvider(final_mul);
}

}  // namespace

TEST_F(GraphTransformationTests, MatMulNBitsSiluFusionFusesWebGpuPattern) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      BuildMatMulNBitsSiluWebGpuPattern,
      21,
      *logger_,
      std::make_unique<MatMulNBitsSiluFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      TransformerLevel::Level2,
      1,
      nullptr,
      CheckMatMulNBitsSiluFusedGraph));
}

TEST_F(GraphTransformationTests, MatMulNBitsSiluFusionMatchesUnfusedWebGpuResults) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckMatMulNBitsSiluFusedGraph(session.GetGraph()));
  };

  TransformerTester(
      BuildMatMulNBitsSiluWebGpuPattern,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      std::make_unique<MatMulNBitsSiluFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      {},
      {},
      std::move(webgpu_ep));
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime

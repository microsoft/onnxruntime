// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/matmul_nbits_fusion.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#if !defined(DISABLE_CONTRIB_OPS)

TEST_F(GraphTransformationTests, MatMulNBitsBiasFusionPreservesFixedProfiles) {
  for (const std::string profile : {"auto", "fixed", "fixed_iq2_xxs"}) {
    const bool fixed = profile != "auto", iq2 = profile == "fixed_iq2_xxs";
    for (bool fp16_scale : {false, true}) {
      if (iq2 && !fp16_scale) continue;
      for (bool bias_first : {false, true}) {
        SCOPED_TRACE(testing::Message() << "profile=" << profile << " fp16_scale=" << fp16_scale
                                        << " bias_first=" << bias_first);
        const auto build = [&](ModelTestBuilder& builder) {
          constexpr int64_t k = 256, n = 8;
          const int64_t block_size = iq2 ? 256 : 32, bits = iq2 ? 2 : (fixed ? 8 : 4);
          auto* input = !fixed && fp16_scale
                            ? builder.MakeInput<MLFloat16>(std::vector<int64_t>{1, k},
                                                           std::vector<MLFloat16>(k, MLFloat16(0.125f)))
                            : builder.MakeInput<BFloat16>(std::vector<int64_t>{1, k},
                                                          std::vector<BFloat16>(k, BFloat16(0.125f)));
          auto* weight = builder.MakeInitializer<uint8_t>({n, k / block_size, block_size * bits / 8},
                                                          uint8_t{0}, uint8_t{255});
          auto* scales = fp16_scale
                             ? builder.MakeInitializer<MLFloat16>({n, k / block_size},
                                                                  MLFloat16(1.f), MLFloat16(1.f))
                             : builder.MakeInitializer<BFloat16>({n, k / block_size},
                                                                 BFloat16(1.f), BFloat16(1.f));
          auto* empty = builder.MakeOptionalTensor();
          auto* projected = builder.MakeIntermediate();
          std::vector<NodeArg*> inputs = fixed
                                             ? std::vector<NodeArg*>{input, weight, empty, empty, empty, empty, scales}
                                             : std::vector<NodeArg*>{input, weight, scales};
          auto& matmul = builder.AddNode("MatMulNBits", inputs, {projected}, kMSDomain);
          matmul.AddAttribute("K", k);
          matmul.AddAttribute("N", n);
          matmul.AddAttribute("bits", bits);
          matmul.AddAttribute("block_size", block_size);
          matmul.AddAttribute("gemv_kernel", profile);
          matmul.SetExecutionProviderType(kCudaExecutionProvider);
          auto* bias = !fixed && fp16_scale
                           ? builder.MakeInput<MLFloat16>(std::vector<int64_t>{n},
                                                          std::vector<MLFloat16>(n, MLFloat16(1.f)))
                           : builder.MakeInput<BFloat16>(std::vector<int64_t>{n},
                                                         std::vector<BFloat16>(n, BFloat16(1.f)));
          auto* output = builder.MakeOutput();
          builder.AddNode("Add", {bias_first ? bias : projected, bias_first ? projected : bias}, {output})
              .SetExecutionProviderType(kCudaExecutionProvider);
        };
        const auto check = [fixed](const Graph& graph) {
          auto counts = CountOpsInGraph(graph);
          EXPECT_EQ(counts["Add"], fixed ? 1 : 0);
          EXPECT_EQ(counts["com.microsoft.MatMulNBits"], 1);
          for (const auto& node : graph.Nodes()) {
            if (node.OpType() == "MatMulNBits") {
              EXPECT_EQ(node.InputDefs().size(), fixed ? 7u : 6u);
              EXPECT_EQ(node.InputDefs()[5]->Exists(), !fixed);
              if (fixed) EXPECT_TRUE(node.InputDefs()[6]->Exists());
            }
          }
          return Status::OK();
        };
        ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::make_unique<MatMulNBitsFusion>(),
                                              TransformerLevel::Level2, 1, nullptr, check));
#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
        if (fixed && HasCudaEnvironment(800)) {
          TransformerTester(build, [&](InferenceSessionWrapper& session) { ASSERT_STATUS_OK(check(session.GetGraph())); }, TransformerLevel::Level1, TransformerLevel::Level2, 21, 0.0, 0.0, nullptr, {}, {}, DefaultCudaExecutionProvider());
        }
#endif
      }
    }
  }
}

#endif
}  // namespace onnxruntime::test

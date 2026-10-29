// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <bit>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
namespace onnxruntime::test {
#ifdef USE_WEBGPU
namespace {
std::unique_ptr<IExecutionProvider> CreateWebGpuProvider(bool nchw = false) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.storageBufferCacheMode", "disabled"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.enableInt64", "1"));
  if (nchw) {
    ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.preferredLayout", "NCHW"));
  }
  return WebGpuExecutionProviderWithOptions(options);
}
void RunWebGpu(OpTester& test, std::unique_ptr<IExecutionProvider> provider) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  options.graph_optimization_level = TransformerLevel::Default;
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}
}  // namespace

TEST(WebGpuBroadcastTest, EinsumEllipsisWithUnbatchedOperand) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Einsum", 12);
  test.AddAttribute<std::string>("equation", "...ij,jk->...ik");
  test.AddInput<float>("A", {1, 1, 2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("B", {2, 2}, {1, 0, 0, 1});
  test.AddOutput<float>("Y", {1, 1, 2, 2}, {1, 2, 3, 4});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, EinsumBroadcastEllipsis) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Einsum", 12);
  test.AddAttribute<std::string>("equation", "...ij,...jk->...ik");
  test.AddInput<float>("A", {1, 2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("B", {2, 2, 2}, {1, 0, 0, 1, 1, 0, 0, 1});
  test.AddOutput<float>("Y", {2, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, WhereBoolBroadcast) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Where", 16);
  test.AddInput<bool>("condition", {2, 2}, {true, false, false, true});
  test.AddInput<bool>("X", {2, 1}, {true, false});
  test.AddInput<bool>("Y", {1, 2}, {false, true});
  test.AddOutput<bool>("Z", {2, 2}, {true, true, false, false});
  RunWebGpu(test, std::move(ep));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuBroadcastTest, BiasSplitGeluFourChannels) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("BiasSplitGelu", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 4}, {2, 3, 1, 1});
  test.AddInput<float>("bias", {4}, {0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 2}, {1.6826895f, 2.5240343f});
  RunWebGpu(test, std::move(ep));
}
#endif

TEST(WebGpuBroadcastTest, RmsNormBroadcastScale) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 0.f);
  test.AddInput<float>("X", {1, 2}, {3, 4});
  test.AddInput<float>("scale", {}, {2});
  test.AddOutput<float>("Y", {1, 2}, {1.6970563f, 2.2627417f});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, EinsumBroadcastMultipleBatchDimensions) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Einsum", 13);
  test.AddAttribute<std::string>("equation", "...ij,...jk->...ik");
  test.AddInput<float>("A", {2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<float>("B", {1, 3, 2, 2}, {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1});
  test.AddOutput<float>("Y", {2, 3, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, EinsumPrefixBeforeEllipsisAndScalar) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Einsum", 13);
  test.AddAttribute<std::string>("equation", "i...j,->j...i");
  test.AddInput<float>("A", {2, 1, 1, 2}, {1, 2, 3, 4});
  test.AddInput<float>("scale", {}, {2});
  test.AddOutput<float>("Y", {2, 1, 1, 2}, {2, 6, 4, 8});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, EinsumDiagonalAndScalarReduction) {
  for (const std::string equation : {"ii->i", "ii->", "ii"}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("Einsum", 13);
    test.AddAttribute<std::string>("equation", equation);
    test.AddInput<float>("X", {2, 2}, {1, 100, 200, 3});
    if (equation == "ii->i")
      test.AddOutput<float>("Y", {2}, {1, 3});
    else
      test.AddOutput<float>("Y", {}, {4});
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuBroadcastTest, EinsumCachedSingletonReduction) {
  auto owner = CreateWebGpuProvider();
  if (!owner) GTEST_SKIP();
  for (const bool singleton_first : {true, false, true}) {
    OpTester test("Einsum", 13);
    test.AddAttribute<std::string>("equation", "i,i->");
    test.AddInput<float>("A", {singleton_first ? 1 : 3}, singleton_first ? std::vector<float>{2} : std::vector<float>{1, 2, 3});
    test.AddInput<float>("B", {singleton_first ? 3 : 1}, singleton_first ? std::vector<float>{1, 2, 3} : std::vector<float>{2});
    test.AddOutput<float>("Y", {}, {12});
    RunWebGpu(test, CreateWebGpuProvider());
  }
}

TEST(WebGpuBroadcastTest, RmsNormPerRowBroadcastScale) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 0.f);
  test.AddInput<float>("X", {2, 2}, {3, 4, 3, 4});
  test.AddInput<float>("scale", {2, 1}, {2, 3});
  test.AddOutput<float>("Y", {2, 2}, {1.6970563f, 2.2627417f, 2.5455844f, 3.3941126f});
  RunWebGpu(test, std::move(ep));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuBroadcastTest, BiasSplitGeluVectorWidths) {
  for (const int64_t half : {1, 2, 3, 4}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("BiasSplitGelu", 1, kMSDomain);
    test.AddInput<float>("X", {1, 1, 2 * half}, std::vector<float>(static_cast<size_t>(2 * half), 1));
    test.AddInput<float>("bias", {2 * half}, std::vector<float>(static_cast<size_t>(2 * half), 0));
    test.AddOutput<float>("Y", {1, 1, half}, std::vector<float>(static_cast<size_t>(half), 0.84134475f));
    RunWebGpu(test, std::move(ep));
  }
}
#endif

TEST(WebGpuBroadcastTest, RmsNormHalfScalarScale) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 0.f);
  test.AddInput<MLFloat16>("X", {1, 2}, {MLFloat16(3.f), MLFloat16(4.f)});
  test.AddInput<MLFloat16>("scale", {}, {MLFloat16(3.f)});
  std::vector<MLFloat16> expected;
  for (const float value : {3.f, 4.f}) {
    const MLFloat16 normalized(value / std::sqrt(12.5f));
    expected.emplace_back(normalized.ToFloat() * 3.f);
  }
  test.AddOutput<MLFloat16>("Y", {1, 2}, expected);
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, RmsNormEmptyBroadcastScale) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("RMSNormalization", 23);
  test.AddInput<float>("X", {0, 2}, {});
  test.AddInput<float>("scale", {}, {2});
  test.AddOutput<float>("Y", {0, 2}, {});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, RmsNormRejectsInvalidScaleRank) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("RMSNormalization", 23);
  test.AddInput<float>("X", {1, 2}, {3, 4});
  test.AddInput<float>("scale", {1, 1, 2}, {1, 1});
  test.AddOutput<float>("Y", {1, 2}, {0, 0});
  test.Config(OpTester::ExpectResult::kExpectFailure, "Scale/Bias rank cannot exceed Input rank.");
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, RmsNormRejectsInvalidAxis) {
  for (const int64_t axis : {-3, 2}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("RMSNormalization", 23);
    test.AddAttribute<int64_t>("axis", axis);
    test.AddInput<float>("X", {1, 2}, {3, 4});
    test.AddInput<float>("scale", {2}, {1, 1});
    test.AddOutput<float>("Y", {1, 2}, {0, 0});
    test.Config(OpTester::ExpectResult::kExpectFailure, "axis");
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuBroadcastTest, EinsumWithoutReductionPreservesSignedZero) {
  for (const bool diagonal : {false, true}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    const float negative_zero = std::bit_cast<float>(uint32_t{0x80000000});
    OpTester test("Einsum", 13);
    test.AddAttribute<std::string>("equation", diagonal ? "ii->i" : "i->i");
    if (diagonal) {
      test.AddInput<float>("X", {2, 2}, {negative_zero, 5, 6, 1});
    } else {
      test.AddInput<float>("X", {2}, {negative_zero, 1});
    }
    test.AddOutput<float>("Y", {2}, {negative_zero, 1});
    test.SetCustomOutputVerifier([](const std::vector<OrtValue>& fetches, const std::string&) {
      ASSERT_EQ(fetches.size(), 1u);
      const auto& output = fetches[0].Get<Tensor>();
      ASSERT_EQ(output.Shape(), TensorShape({2}));
      EXPECT_EQ(std::bit_cast<uint32_t>(output.Data<float>()[0]), 0x80000000u);
      EXPECT_EQ(output.Data<float>()[1], 1.f);
    });
    RunWebGpu(test, std::move(ep));
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuBroadcastTest, BiasSplitGeluRejectsOddChannels) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("BiasSplitGelu", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3}, {1, 1, 1});
  test.AddInput<float>("bias", {3}, {0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 1}, {0});
  test.Config(OpTester::ExpectResult::kExpectFailure, "even channel");
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuBroadcastTest, BiasSplitGeluEmptyInput) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("BiasSplitGelu", 1, kMSDomain);
  test.AddInput<float>("X", {0, 2, 4}, {});
  test.AddInput<float>("bias", {4}, {0, 0, 0, 0});
  test.AddOutput<float>("Y", {0, 2, 2}, {});
  RunWebGpu(test, std::move(ep));
}
#endif

#endif
}  // namespace onnxruntime::test

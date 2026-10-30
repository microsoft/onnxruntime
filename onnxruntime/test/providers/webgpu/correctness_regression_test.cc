// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#if defined(USE_WEBGPU)
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#endif
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

TEST(WebGpuCorrectnessTest, TopKPaddingFiniteMinimum) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("TopK", 11);
  const float v = -std::numeric_limits<float>::max();
  test.AddInput<float>("X", {3}, {v, v, v});
  test.AddInput<int64_t>("K", {1}, {3});
  test.AddOutput<float>("Y", {3}, {v, v, v});
  test.AddOutput<int64_t>("Indices", {3}, {0, 1, 2});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, TopKLargePaddingFiniteMaximum) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("TopK", 11);
  test.AddAttribute<int64_t>("largest", 0);
  const float v = std::numeric_limits<float>::max();
  test.AddInput<float>("X", {2049}, std::vector<float>(2049, v));
  test.AddInput<int64_t>("K", {1}, {1});
  test.AddOutput<float>("Y", {1}, {v});
  test.AddOutput<int64_t>("Indices", {1}, {0});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, TriluLargePositiveK) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Trilu", 14);
  test.AddInput<float>("X", {2, 2}, {1, 2, 3, 4});
  test.AddInput<int64_t>("K", {}, {2147483647});
  test.AddOutput<float>("Y", {2, 2}, {0, 0, 0, 0});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, RangeInt32DifferenceOverflow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Range", 11);
  test.AddInput<int32_t>("start", {}, {-2147483647 - 1});
  test.AddInput<int32_t>("limit", {}, {2147483647});
  test.AddInput<int32_t>("delta", {}, {2147483647});
  test.AddOutput<int32_t>("Y", {3}, {-2147483647 - 1, -1, 2147483646});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, InstanceNormHalfConstant) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("InstanceNormalization", 13);
  test.AddInput<MLFloat16>("X", {1, 1, 4}, std::vector<MLFloat16>(4, MLFloat16(1000.f)));
  test.AddInput<MLFloat16>("scale", {1}, {MLFloat16(1.f)});
  test.AddInput<MLFloat16>("B", {1}, {MLFloat16(0.f)});
  test.AddOutput<MLFloat16>("Y", {1, 1, 4}, std::vector<MLFloat16>(4, MLFloat16(0.f)));
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, InstanceNormSmallEpsilon) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("InstanceNormalization", 13);
  test.AddAttribute<float>("epsilon", 1e-8f);
  test.AddInput<float>("X", {1, 1, 4}, {1, 1, 1, 1});
  test.AddInput<float>("scale", {1}, {1});
  test.AddInput<float>("B", {1}, {0});
  test.AddOutput<float>("Y", {1, 1, 4}, {0, 0, 0, 0});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, BatchNormMixedTypes) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("BatchNormalization", 15);
  test.AddAttribute<float>("epsilon", 0.f);
  test.AddInput<MLFloat16>("X", {1, 1, 4}, {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)});
  test.AddInput<float>("scale", {1}, {1});
  test.AddInput<float>("B", {1}, {0});
  test.AddInput<float>("mean", {1}, {0});
  test.AddInput<float>("var", {1}, {1});
  test.AddOutput<MLFloat16>("Y", {1, 1, 4}, {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, SoftmaxHalfLongRow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Softmax", 13);
  test.AddInput<MLFloat16>("X", {1, 65536}, std::vector<MLFloat16>(65536, MLFloat16(0.f)));
  test.AddOutput<MLFloat16>("Y", {1, 65536}, std::vector<MLFloat16>(65536, MLFloat16(1.f / 65536)));
  test.SetOutputAbsErr("Y", 1e-7f);
  test.SetOutputRelErr("Y", 0.f);
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, AveragePoolHalfOverflow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("AveragePool", 19);
  test.AddAttribute<std::vector<int64_t>>("kernel_shape", {2});
  test.AddInput<MLFloat16>("X", {1, 1, 2}, {MLFloat16(40000.f), MLFloat16(40000.f)});
  test.AddOutput<MLFloat16>("Y", {1, 1, 1}, {MLFloat16(40000.f)});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ReduceLogSumExpOverflow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("ReduceLogSumExp", 13);
  test.AddInput<float>("X", {2}, {100, 100});
  test.AddOutput<float>("Y", {1}, {100.69314718f});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ReduceSumInt32NaivePrecision) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("ReduceSum", 13);
  std::vector<int32_t> x(1025 * 2);
  for (size_t i = 0; i < 1025; ++i) {
    x[2 * i] = 16777217;
    x[2 * i + 1] = 1;
  }
  test.AddInput<int32_t>("X", {1025, 2}, x);
  test.AddInput<int64_t>("axes", {1}, {1});
  test.AddOutput<int32_t>("Y", {1025, 1}, std::vector<int32_t>(1025, 16777218));
  RunWebGpu(test, std::move(ep));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuCorrectnessTest, MultiHeadAttentionDifferentValueHeadSize) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("MultiHeadAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 1);
  test.AddInput<float>("query", {1, 1, 2}, {0, 0});
  test.AddInput<float>("key", {1, 1, 2}, {0, 0});
  test.AddInput<float>("value", {1, 1, 4}, {1, 2, 3, 4});
  test.AddOutput<float>("output", {1, 1, 4}, {1, 2, 3, 4});
  test.AddOutput<float>("present_key", {1, 1, 1, 2}, {0, 0});
  test.AddOutput<float>("present_value", {1, 1, 1, 4}, {1, 2, 3, 4});
  RunWebGpu(test, std::move(ep));
}
#endif

TEST(WebGpuCorrectnessTest, DequantizeInt32ToHalfRange) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("DequantizeLinear", 19);
  test.AddInput<int32_t>("X", {1}, {131072});
  test.AddInput<MLFloat16>("scale", {}, {MLFloat16(1.f / 16384)});
  test.AddOutput<MLFloat16>("Y", {1}, {MLFloat16(8.f)});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, EinsumEllipsisWithUnbatchedOperand) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Einsum", 12);
  test.AddAttribute<std::string>("equation", "...ij,jk->...ik");
  test.AddInput<float>("A", {1, 1, 2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("B", {2, 2}, {1, 0, 0, 1});
  test.AddOutput<float>("Y", {1, 1, 2, 2}, {1, 2, 3, 4});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, EinsumBroadcastEllipsis) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Einsum", 12);
  test.AddAttribute<std::string>("equation", "...ij,...jk->...ik");
  test.AddInput<float>("A", {1, 2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("B", {2, 2, 2}, {1, 0, 0, 1, 1, 0, 0, 1});
  test.AddOutput<float>("Y", {2, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ResizeNearestRankOne) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Resize", 13);
  test.AddAttribute<std::string>("mode", "nearest");
  test.AddInput<float>("X", {2}, {1, 2});
  test.AddInput<float>("roi", {0}, {});
  test.AddInput<float>("scales", {1}, {2});
  test.AddOutput<float>("Y", {4}, {1, 1, 2, 2});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ResizeCropUnitScale) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Resize", 13);
  test.AddAttribute<std::string>("mode", "nearest");
  test.AddAttribute<std::string>("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddInput<float>("X", {2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("roi", {4}, {0, 1, 1, 0});
  test.AddInput<float>("scales", {2}, {1, 1});
  test.AddOutput<float>("Y", {2, 2}, {2, 1, 4, 3});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, SliceReversePartialRange) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Slice", 13);
  test.AddInput<float>("X", {5}, {0, 1, 2, 3, 4});
  test.AddInput<int64_t>("starts", {1}, {3});
  test.AddInput<int64_t>("ends", {1}, {0});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddInput<int64_t>("steps", {1}, {-1});
  test.AddOutput<float>("Y", {3}, {3, 2, 1});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, WhereBoolBroadcast) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Where", 16);
  test.AddInput<bool>("condition", {2, 2}, {true, false, false, true});
  test.AddInput<bool>("X", {2, 1}, {true, false});
  test.AddInput<bool>("Y", {1, 2}, {false, true});
  test.AddOutput<bool>("Z", {2, 2}, {true, true, false, false});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ConcatEmptyChunk) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Concat", 13);
  test.AddAttribute<int64_t>("axis", 0);
  for (int i = 0; i < 128; ++i) test.AddInput<float>(("empty" + std::to_string(i)).c_str(), {0}, {});
  test.AddInput<float>("X", {1}, {7});
  test.AddOutput<float>("Y", {1}, {7});
  RunWebGpu(test, std::move(ep));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuCorrectnessTest, BiasSplitGeluFourChannels) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("BiasSplitGelu", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 4}, {2, 3, 1, 1});
  test.AddInput<float>("bias", {4}, {0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 2}, {1.6826895f, 2.5240343f});
  RunWebGpu(test, std::move(ep));
}
#endif

TEST(WebGpuCorrectnessTest, ConvPointwiseTrailingPadding) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Conv", 11);
  test.AddAttribute<std::vector<int64_t>>("pads", {0, 0, 1, 1});
  test.AddInput<float>("X", {1, 1, 2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("W", {1, 1, 1, 1}, {1});
  test.AddOutput<float>("Y", {1, 1, 3, 3}, {1, 2, 0, 3, 4, 0, 0, 0, 0});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, InstanceNormLargeOffset) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("InstanceNormalization", 13);
  test.AddInput<float>("X", {1, 1, 4}, {10000, 10001, 10000, 10001});
  test.AddInput<float>("scale", {1}, {1});
  test.AddInput<float>("B", {1}, {0});
  test.AddOutput<float>("Y", {1, 1, 4}, {-.99998f, .99998f, -.99998f, .99998f});
  RunWebGpu(test, std::move(ep));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuCorrectnessTest, SkipLayerNormLargeOffset) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("SkipLayerNormalization", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddInput<float>("X", {1, 1, 4}, {10000, 10001, 10000, 10001});
  test.AddInput<float>("skip", {1, 1, 4}, {0, 0, 0, 0});
  test.AddInput<float>("gamma", {4}, {1, 1, 1, 1});
  test.AddOutput<float>("Y", {1, 1, 4}, {-.99998f, .99998f, -.99998f, .99998f});
  RunWebGpu(test, std::move(ep));
}
#endif

TEST(WebGpuCorrectnessTest, RmsNormBroadcastScale) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 0.f);
  test.AddInput<float>("X", {1, 2}, {3, 4});
  test.AddInput<float>("scale", {}, {2});
  test.AddOutput<float>("Y", {1, 2}, {1.6970563f, 2.2627417f});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ConvAbsentBiasSlot) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Conv", 11);
  test.AddInput<float>("X", {1, 1, 1, 1}, {2});
  test.AddInput<float>("W", {1, 1, 1, 1}, {3});
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("Y", {1, 1, 1, 1}, {6});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ConvTransposeAbsentBiasSlot) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("ConvTranspose", 11);
  test.AddInput<float>("X", {1, 1, 1, 1}, {2});
  test.AddInput<float>("W", {1, 1, 1, 1}, {3});
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("Y", {1, 1, 1, 1}, {6});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, SoftmaxEmptyAxis) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Softmax", 13);
  test.AddInput<float>("X", {1, 0}, {});
  test.AddOutput<float>("Y", {1, 0}, {});
  RunWebGpu(test, std::move(ep));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuCorrectnessTest, AttentionAbsentBias) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Attention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 1);
  test.AddInput<float>("input", {1, 1, 2}, {1, 2});
  test.AddInput<float>("weights", {2, 6}, std::vector<float>(12, 0));
  test.AddOutput<float>("output", {1, 1, 2}, {0, 0});
  RunWebGpu(test, std::move(ep));
}
#endif

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuCorrectnessTest, MultiHeadAttentionFirstPromptCaches) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("MultiHeadAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 1);
  test.AddInput<float>("query", {1, 1, 2}, {0, 0});
  test.AddInput<float>("key", {1, 1, 2}, {2, 3});
  test.AddInput<float>("value", {1, 1, 2}, {4, 5});
  test.AddOutput<float>("output", {1, 1, 2}, {4, 5});
  test.AddOutput<float>("present_key", {1, 1, 1, 2}, {2, 3});
  test.AddOutput<float>("present_value", {1, 1, 1, 2}, {4, 5});
  RunWebGpu(test, std::move(ep));
}
#endif

TEST(WebGpuCorrectnessTest, SoftmaxHalfLongRowNormalPeak) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Softmax", 13);
  const size_t n = 131072;
  std::vector<MLFloat16> x(n, MLFloat16(0.f));
  x.back() = MLFloat16(16.f);
  const float low = std::exp(-16.f) / (1.f + (n - 1) * std::exp(-16.f));
  std::vector<MLFloat16> y(n, MLFloat16(low));
  y.back() = MLFloat16(1.f / (1.f + (n - 1) * std::exp(-16.f)));
  test.AddInput<MLFloat16>("X", {1, static_cast<int64_t>(n)}, x);
  test.AddOutput<MLFloat16>("Y", {1, static_cast<int64_t>(n)}, y);
  test.SetOutputAbsErr("Y", 1e-3f);
  test.SetOutputRelErr("Y", 0.f);
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ConvTrailingPaddingLayoutsAndInitializers) {
  for (const bool nchw : {false, true}) {
    for (const bool initializer : {false, true}) {
      SCOPED_TRACE(testing::Message() << "nchw=" << nchw << " initializer=" << initializer);
      auto ep = CreateWebGpuProvider(nchw);
      if (!ep) GTEST_SKIP();
      OpTester test("Conv", 13);
      test.AddAttribute<std::vector<int64_t>>("pads", {0, 0, 1, 1});
      test.AddInput<float>("X", {1, 2, 2, 2}, {1, 2, 3, 4, 10, 20, 30, 40});
      test.AddInput<float>("W", {2, 2, 1, 1}, {1, 0, 0, 1}, initializer);
      test.AddOutput<float>("Y", {1, 2, 3, 3}, {1, 2, 0, 3, 4, 0, 0, 0, 0, 10, 20, 0, 30, 40, 0, 0, 0, 0});
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuCorrectnessTest, ConvWholeInputKernelWithTrailingPadding) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Conv", 13);
  test.AddAttribute<std::vector<int64_t>>("pads", {0, 0, 1, 1});
  test.AddInput<float>("X", {1, 1, 2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("W", {1, 1, 2, 2}, {1, 1, 1, 1}, true);
  test.AddOutput<float>("Y", {1, 1, 2, 2}, {10, 6, 7, 4});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, SliceReverseStridesAndAxes) {
  for (const int64_t step : {int64_t{-1}, int64_t{-2}, std::numeric_limits<int64_t>::min()}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("Slice", 13);
    test.AddInput<float>("X", {2, 5}, {0, 1, 2, 3, 4, 10, 11, 12, 13, 14});
    test.AddInput<int64_t>("starts", {1}, {3});
    test.AddInput<int64_t>("ends", {1}, {std::numeric_limits<int64_t>::min()});
    test.AddInput<int64_t>("axes", {1}, {-1});
    test.AddInput<int64_t>("steps", {1}, {step});
    const std::vector<float> expected = step == -1   ? std::vector<float>{3, 2, 1, 0, 13, 12, 11, 10}
                                        : step == -2 ? std::vector<float>{3, 1, 13, 11}
                                                     : std::vector<float>{3, 13};
    test.AddOutput<float>("Y", {2, static_cast<int64_t>(expected.size() / 2)}, expected);
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuCorrectnessTest, SliceForwardVectorAndScalarViews) {
  for (const int64_t start : {0, 1, 4}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("Slice", 13);
    test.AddInput<float>("X", {2, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17});
    test.AddInput<int64_t>("starts", {1}, {start});
    test.AddInput<int64_t>("ends", {1}, {start + 4});
    test.AddInput<int64_t>("axes", {1}, {1});
    std::vector<float> expected;
    for (int row = 0; row < 2; ++row)
      for (int j = 0; j < 4; ++j) expected.push_back(static_cast<float>(row * 10 + start + j));
    test.AddOutput<float>("Y", {2, 4}, expected);
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuCorrectnessTest, TopKPaddingInfinitiesAndDirections) {
  for (const bool largest : {false, true}) {
    for (const int64_t size : {3, 2049}) {
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      OpTester test("TopK", 11);
      test.AddAttribute<int64_t>("largest", largest ? 1 : 0);
      const float value = largest ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
      test.AddInput<float>("X", {size}, std::vector<float>(static_cast<size_t>(size), value));
      test.AddInput<int64_t>("K", {1}, {2});
      test.AddOutput<float>("Y", {2}, {value, value});
      test.AddOutput<int64_t>("Indices", {2}, {0, 1});
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuCorrectnessTest, TriluExtremeOffsetsBothDirections) {
  for (const int64_t upper : {0, 1}) {
    for (const int64_t k : {int64_t{INT32_MIN}, int64_t{INT32_MAX}}) {
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      OpTester test("Trilu", 14);
      test.AddAttribute<int64_t>("upper", upper);
      const std::vector<float> input{1, 2, 3, 4};
      test.AddInput<float>("X", {2, 2}, input);
      test.AddInput<int64_t>("K", {}, {k});
      test.AddOutput<float>("Y", {2, 2}, (upper ? k < 0 : k > 0) ? input : std::vector<float>(4, 0));
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuCorrectnessTest, EinsumBroadcastMultipleBatchDimensions) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Einsum", 13);
  test.AddAttribute<std::string>("equation", "...ij,...jk->...ik");
  test.AddInput<float>("A", {2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<float>("B", {1, 3, 2, 2}, {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1});
  test.AddOutput<float>("Y", {2, 3, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, EinsumPrefixBeforeEllipsisAndScalar) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Einsum", 13);
  test.AddAttribute<std::string>("equation", "i...j,->j...i");
  test.AddInput<float>("A", {2, 1, 1, 2}, {1, 2, 3, 4});
  test.AddInput<float>("scale", {}, {2});
  test.AddOutput<float>("Y", {2, 1, 1, 2}, {2, 6, 4, 8});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, EinsumDiagonalAndScalarReduction) {
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

TEST(WebGpuCorrectnessTest, EinsumCachedSingletonReduction) {
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

TEST(WebGpuCorrectnessTest, ResizeNearestHighRank) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Resize", 13);
  test.AddAttribute<std::string>("mode", "nearest");
  test.AddInput<float>("X", {1, 1, 1, 1, 1, 2}, {1, 2});
  test.AddInput<float>("roi", {0}, {});
  test.AddInput<float>("scales", {6}, {1, 1, 1, 1, 1, 2});
  test.AddOutput<float>("Y", {1, 1, 1, 1, 1, 4}, {1, 1, 2, 2});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ReduceLogSumExpNaiveLargeValues) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("ReduceLogSumExp", 13);
  test.AddAttribute<std::vector<int64_t>>("axes", {1});
  test.AddInput<float>("X", {1025, 2}, std::vector<float>(2050, 100));
  test.AddOutput<float>("Y", {1025, 1}, std::vector<float>(1025, 100 + std::log(2.f)));
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, ReduceLogSumExpInfinities) {
  for (const float v : {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("ReduceLogSumExp", 13);
    test.AddInput<float>("X", {3}, {v, v, v});
    test.AddOutput<float>("Y", {1}, {v});
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuCorrectnessTest, ReduceIntegerNaiveSumMeanProd) {
  for (const std::string op : {"ReduceSum", "ReduceMean", "ReduceProd"}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test(op.c_str(), 13);
    if (op != "ReduceSum") test.AddAttribute<std::vector<int64_t>>("axes", {1});
    std::vector<int32_t> input(2050);
    for (size_t i = 0; i < 1025; ++i) {
      input[2 * i] = 16777217;
      input[2 * i + 1] = op == "ReduceMean" ? 16777219 : 1;
    }
    test.AddInput<int32_t>("X", {1025, 2}, input);
    if (op == "ReduceSum") test.AddInput<int64_t>("axes", {1}, {1});
    test.AddOutput<int32_t>("Y", {1025, 1}, std::vector<int32_t>(1025, op == "ReduceProd" ? 16777217 : 16777218));
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuCorrectnessTest, RmsNormPerRowBroadcastScale) {
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
TEST(WebGpuCorrectnessTest, SkipLayerNormCenteredSplitWithResidualOutput) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("SkipLayerNormalization", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-5f);
  std::vector<float> x(1024), expected(1024);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = 10000.f + (i % 2);
    expected[i] = (i % 2 ? 1.f : -1.f) / std::sqrt(1.00004f);
  }
  test.AddInput<float>("X", {1, 1, 1024}, x);
  test.AddInput<float>("skip", {1, 1, 1024}, std::vector<float>(1024, 0));
  test.AddInput<float>("gamma", {1024}, std::vector<float>(1024, 1));
  test.AddOutput<float>("Y", {1, 1, 1024}, expected);
  test.AddOptionalOutputEdge<float>();
  test.AddOptionalOutputEdge<float>();
  test.AddOutput<float>("sum", {1, 1, 1024}, x);
  RunWebGpu(test, std::move(ep));
}
#endif

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuCorrectnessTest, GroupNormCenteredVariance) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("GroupNorm", 1, kMSDomain);
  test.AddAttribute<int64_t>("groups", 1);
  test.AddAttribute<int64_t>("channels_last", 1);
  test.AddAttribute<int64_t>("activation", 0);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddInput<float>("X", {1, 1, 1, 4}, {10000, 10001, 10000, 10001});
  test.AddInput<float>("gamma", {4}, {1, 1, 1, 1});
  test.AddInput<float>("beta", {4}, {0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 1, 4}, {-.99998f, .99998f, -.99998f, .99998f});
  RunWebGpu(test, std::move(ep));
}
#endif

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuCorrectnessTest, MultiHeadAttentionDecodePresentKeyOnly) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("MultiHeadAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 1);
  test.AddInput<float>("query", {1, 1, 2}, {0, 0});
  test.AddInput<float>("key", {1, 1, 2}, {2, 3});
  test.AddInput<float>("value", {1, 1, 2}, {4, 6});
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("past_key", {1, 1, 1, 2}, {1, 1});
  test.AddInput<float>("past_value", {1, 1, 1, 2}, {2, 4});
  test.AddOutput<float>("output", {1, 1, 2}, {3, 5});
  test.AddOutput<float>("present_key", {1, 1, 2, 2}, {1, 1, 2, 3});
  RunWebGpu(test, std::move(ep));
}
#endif

TEST(WebGpuCorrectnessTest, CenteredNormalizationFractionalConstants) {
  for (const int64_t length : {3, 7, 33, 512, 1024}) {
    for (const bool instance : {false, true}) {
#ifdef DISABLE_CONTRIB_OPS
      if (!instance) continue;
#endif
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      SCOPED_TRACE(testing::Message() << "length=" << length << " instance=" << instance);
      OpTester test(instance ? "InstanceNormalization" : "SkipLayerNormalization", instance ? 13 : 1,
                    instance ? kOnnxDomain : kMSDomain);
      test.AddAttribute<float>("epsilon", 1e-5f);
      test.AddInput<float>("X", {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 4080.11767578125f));
      if (instance) {
        test.AddInput<float>("scale", {1}, {1000});
        test.AddInput<float>("bias", {1}, {0});
      } else {
        test.AddInput<float>("skip", {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 0));
        test.AddInput<float>("gamma", {length}, std::vector<float>(static_cast<size_t>(length), 1000));
      }
      test.AddOutput<float>("Y", {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 0));
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuCorrectnessTest, CenteredNormalizationCancellationAndScale) {
  for (const int64_t length : {4, 512}) {
    for (const bool instance : {false, true}) {
#ifdef DISABLE_CONTRIB_OPS
      if (!instance) continue;
#endif
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      SCOPED_TRACE(testing::Message() << "length=" << length << " instance=" << instance);
      std::vector<float> input(static_cast<size_t>(length), 0);
      input[0] = 1e8f;
      input[2] = -1e8f;
      input[3] = 1;
      const double mean = 1.0 / length;
      double variance = 0;
      for (float value : input) variance += (value - mean) * (value - mean);
      variance /= length;
      std::vector<float> expected;
      for (float value : input) expected.push_back(static_cast<float>((value - mean) / std::sqrt(variance + 1e-5) * 1e8));
      OpTester test(instance ? "InstanceNormalization" : "SkipLayerNormalization", instance ? 13 : 1,
                    instance ? kOnnxDomain : kMSDomain);
      test.AddAttribute<float>("epsilon", 1e-5f);
      test.AddInput<float>("X", {1, 1, length}, input);
      if (instance) {
        test.AddInput<float>("scale", {1}, {1e8f});
        test.AddInput<float>("bias", {1}, {0});
      } else {
        test.AddInput<float>("skip", {1, 1, length}, std::vector<float>(static_cast<size_t>(length), 0));
        test.AddInput<float>("gamma", {length}, std::vector<float>(static_cast<size_t>(length), 1e8f));
      }
      test.AddOutput<float>("Y", {1, 1, length}, expected);
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuCorrectnessTest, AveragePoolHalfParallelRange) {
  for (const int64_t length : {128, 65536}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("AveragePool", 19);
    test.AddAttribute<std::vector<int64_t>>("kernel_shape", {length});
    test.AddInput<MLFloat16>("X", {1, 1, length}, std::vector<MLFloat16>(static_cast<size_t>(length), MLFloat16(2.f)));
    test.AddOutput<MLFloat16>("Y", {1, 1, 1}, {MLFloat16(2.f)});
    RunWebGpu(test, std::move(ep));
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuCorrectnessTest, BiasSplitGeluVectorWidths) {
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

TEST(WebGpuCorrectnessTest, IntegerMeanAvoidsOverflowInBothReductionPaths) {
  for (const int64_t rows : {1, 1025}) {
    for (const bool negative : {false, true}) {
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      OpTester test("ReduceMean", 13);
      test.AddAttribute<std::vector<int64_t>>("axes", {1});
      const int32_t value = negative ? INT32_MIN : INT32_MAX;
      test.AddInput<int32_t>("X", {rows, 2}, std::vector<int32_t>(static_cast<size_t>(rows * 2), value));
      test.AddOutput<int32_t>("Y", {rows, 1}, std::vector<int32_t>(static_cast<size_t>(rows), value));
      RunWebGpu(test, std::move(ep));
    }
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("ReduceMean", 13);
    test.AddAttribute<std::vector<int64_t>>("axes", {1});
    test.AddInput<uint32_t>("X", {rows, 2}, std::vector<uint32_t>(static_cast<size_t>(rows * 2), UINT32_MAX));
    test.AddOutput<uint32_t>("Y", {rows, 1}, std::vector<uint32_t>(static_cast<size_t>(rows), UINT32_MAX));
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuCorrectnessTest, EmptySoftmaxStillValidatesAxis) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Softmax", 13);
  test.AddAttribute<int64_t>("axis", 3);
  test.AddInput<float>("X", {1, 0}, {});
  test.AddOutput<float>("Y", {1, 0}, {});
  test.Config(OpTester::ExpectResult::kExpectFailure, "axis");
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuCorrectnessTest, CenteredNormalizationKeepsSmallMeanOffsets) {
  for (const int kind : {0, 1, 2}) {
    SCOPED_TRACE(kind);
#ifdef DISABLE_CONTRIB_OPS
    if (kind != 0) continue;
#endif
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    const float low = 4080.11767578125f;
    const float high = std::nextafter(low, std::numeric_limits<float>::infinity());
    const double half_delta = (static_cast<double>(high) - low) / 2;
    const float expected = static_cast<float>(half_delta / std::sqrt(half_delta * half_delta + 1e-5) * 1000);
    OpTester test(kind == 0 ? "InstanceNormalization" : kind == 1 ? "SkipLayerNormalization"
                                                                  : "GroupNorm",
                  kind == 0 ? 13 : 1, kind == 0 ? kOnnxDomain : kMSDomain);
    test.AddAttribute<float>("epsilon", 1e-5f);
    const std::vector<int64_t> shape = kind == 2 ? std::vector<int64_t>{1, 1, 1, 4} : std::vector<int64_t>{1, 1, 4};
    test.AddInput<float>("X", shape, {low, high, low, high});
    if (kind == 0) {
      test.AddInput<float>("scale", {1}, {1000});
      test.AddInput<float>("bias", {1}, {0});
    } else {
      if (kind == 1)
        test.AddInput<float>("skip", shape, {0, 0, 0, 0});
      else {
        test.AddAttribute<int64_t>("groups", 1);
        test.AddAttribute<int64_t>("channels_last", 1);
        test.AddAttribute<int64_t>("activation", 0);
      }
      test.AddInput<float>("gamma", {4}, {1000, 1000, 1000, 1000});
      if (kind == 2) test.AddInput<float>("beta", {4}, {0, 0, 0, 0});
    }
    test.AddOutput<float>("Y", shape, {-expected, expected, -expected, expected});
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuCorrectnessTest, BatchNormMixedStatisticsLayouts) {
  for (const bool nchw : {false, true}) {
    auto ep = CreateWebGpuProvider(nchw);
    if (!ep) GTEST_SKIP();
    OpTester test("BatchNormalization", 14);
    test.AddAttribute<float>("epsilon", 0.f);
    test.AddInput<MLFloat16>("X", {1, 2, 1, 2}, {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)});
    test.AddInput<MLFloat16>("scale", {2}, {MLFloat16(2.f), MLFloat16(3.f)});
    test.AddInput<MLFloat16>("bias", {2}, {MLFloat16(1.f), MLFloat16(-1.f)});
    test.AddInput<float>("mean", {2}, {0, 1});
    test.AddInput<float>("variance", {2}, {1, 4});
    test.AddOutput<MLFloat16>("Y", {1, 2, 1, 2}, {MLFloat16(3.f), MLFloat16(5.f), MLFloat16(2.f), MLFloat16(3.5f)});
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuCorrectnessTest, DequantizeInt32HalfVectorWidths) {
  for (const int64_t length : {2, 4}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("DequantizeLinear", 19);
    test.AddInput<int32_t>("X", {length}, std::vector<int32_t>(static_cast<size_t>(length), -131072));
    test.AddInput<MLFloat16>("scale", {}, {MLFloat16(1.f / 16384)});
    test.AddOutput<MLFloat16>("Y", {length}, std::vector<MLFloat16>(static_cast<size_t>(length), MLFloat16(-8.f)));
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuCorrectnessTest, FailedRunMustNotBecomeCapturedGraph) {
  for (const std::string op : {"Gemm", "Einsum"}) {
    SCOPED_TRACE(op);
    ConfigOptions config;
    ASSERT_STATUS_OK(config.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
    ASSERT_STATUS_OK(config.AddConfigEntry("ep.webgpuexecutionprovider.enableGraphCapture", "1"));
    auto ep = WebGpuExecutionProviderWithOptions(config);
    if (!ep) GTEST_SKIP();
    auto* provider = ep.get();
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);
    auto* graph = model.mutable_graph();
    graph->set_name("failed_capture");
    for (const char* name : {"A", "B"}) {
      auto* arg = graph->add_input();
      arg->set_name(name);
      auto* type = arg->mutable_type()->mutable_tensor_type();
      type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      type->mutable_shape()->add_dim()->set_dim_param(std::string(name) + "0");
      type->mutable_shape()->add_dim()->set_dim_param(std::string(name) + "1");
    }
    auto* out = graph->add_output();
    out->set_name("Y");
    auto* out_type = out->mutable_type()->mutable_tensor_type();
    out_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    out_type->mutable_shape()->add_dim()->set_dim_param("M");
    out_type->mutable_shape()->add_dim()->set_dim_param("N");
    auto* node = graph->add_node();
    node->set_name("gemm");
    node->set_op_type(op);
    if (op == "Einsum") {
      auto* equation = node->add_attribute();
      equation->set_name("equation");
      equation->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
      equation->set_s("ij,jk->ik");
    }
    node->add_input("A");
    node->add_input("B");
    node->add_output("Y");
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    InferenceSession session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
    const auto bytes = model.SerializeAsString();
    ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());
    const auto allocators = provider->CreatePreferredAllocators();
    const auto allocator_it = std::find_if(allocators.begin(), allocators.end(), [](const auto& candidate) {
      return candidate->Info().device.Type() == OrtDevice::GPU && candidate->Info().alloc_type == OrtDeviceAllocator;
    });
    ASSERT_NE(allocator_it, allocators.end());
    const auto allocator = *allocator_it;
    OrtValue a, b;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{2, 3}, allocator, a);
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{4, 2}, allocator, b);
    std::vector<OrtValue> outputs;
    const std::vector<std::string> output_names{"Y"};
    const auto status = session.Run(RunOptions{}, {{"A", a}, {"B", b}}, output_names, &outputs);
    ASSERT_FALSE(status.IsOK());
    EXPECT_FALSE(provider->IsGraphCaptured(0)) << status.ErrorMessage();
    const auto retry_status = session.Run(RunOptions{}, {{"A", a}, {"B", b}}, output_names, &outputs);
    EXPECT_FALSE(retry_status.IsOK()) << "An identical invalid run must not become successful replay.";
    EXPECT_FALSE(provider->IsGraphCaptured(0));

    Tensor cpu_a(DataTypeImpl::GetType<float>(), TensorShape{2, 3}, CPUAllocator::DefaultInstance());
    Tensor cpu_b(DataTypeImpl::GetType<float>(), TensorShape{3, 2}, CPUAllocator::DefaultInstance());
    std::fill_n(cpu_a.MutableData<float>(), 6, 1.f);
    std::fill_n(cpu_b.MutableData<float>(), 6, 1.f);
    OrtValue valid_b;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{3, 2}, allocator, valid_b);
    ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(cpu_a, *a.GetMutable<Tensor>()));
    ASSERT_STATUS_OK(session.GetDataTransferManager().CopyTensor(cpu_b, *valid_b.GetMutable<Tensor>()));
    ASSERT_STATUS_OK(session.Run(RunOptions{}, {{"A", a}, {"B", valid_b}}, output_names, &outputs));
    EXPECT_TRUE(provider->IsGraphCaptured(0));
    ASSERT_EQ(outputs.size(), 1u);
    const auto& result = outputs[0].Get<Tensor>();
    ASSERT_EQ(result.Shape(), TensorShape({2, 2}));
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(result.Data<float>()[i], 3.f);

#if !defined(ORT_USE_EP_API_ADAPTERS)
    // Plugin RunOptions currently has no termination query in the public EP API.
    ASSERT_STATUS_OK(provider->ReleaseCapturedGraph(0));
    RunOptions cancelled;
    cancelled.terminate = true;
    EXPECT_FALSE(session.Run(cancelled, {{"A", a}, {"B", valid_b}}, output_names, &outputs).IsOK());
    EXPECT_FALSE(provider->IsGraphCaptured(0));
    ASSERT_STATUS_OK(session.Run(RunOptions{}, {{"A", a}, {"B", valid_b}}, output_names, &outputs));
    EXPECT_TRUE(provider->IsGraphCaptured(0));
#endif
  }
}

#endif
}  // namespace onnxruntime::test

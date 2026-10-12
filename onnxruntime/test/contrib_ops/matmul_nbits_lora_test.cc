// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/cpu/quantization/matmul_nbits_lora_helper.h"
#include "core/framework/allocator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/webgpu/graph_capture_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

template <typename T>
void RunLoraTest(int64_t rank, bool gpu = false, bool empty = false,
                 bool mismatched_rank = false, bool vector = false,
                 bool negative_weights = false, float alpha = 1.0f,
                 int64_t weight_prepacked = 0) {
  constexpr int64_t K = 35;
  constexpr int64_t N = 16;
  const int64_t rows = empty ? 0 : 3;
  const int64_t count = vector ? 1 : 2 * rows;
  const std::vector<int64_t> input_shape = vector ? std::vector<int64_t>{K}
                                                  : std::vector<int64_t>{2, rows, K};
  const std::vector<int64_t> output_shape = vector ? std::vector<int64_t>{N}
                                                   : std::vector<int64_t>{2, rows, N};
  WebGpuGraphCaptureTester test("MatMulNBitsLora", 1, kMSDomain);
  test.AddAttribute("K", K);
  test.AddAttribute("N", N);
  test.AddAttribute("bits", int64_t{4});
  test.AddAttribute("block_size", int64_t{32});
  test.AddAttribute("lora_alpha", alpha);
  test.AddAttribute("weight_prepacked", weight_prepacked);
  test.AddInput<T>("X", input_shape, std::vector<T>(count * K, T{1.0f}));
  test.AddInput<uint8_t>("B", {N, 2, 16}, std::vector<uint8_t>(N * 2 * 16, 0x99), true);
  test.AddInput<T>("scales", {N, 2}, std::vector<T>(N * 2, T{0.5f}), true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<T>("bias", {N}, std::vector<T>(N, T{0.75f}), true);
  test.AddInput<int8_t>("Q_A", {K, rank}, std::vector<int8_t>(K * rank, negative_weights ? -1 : 1));
  const int64_t b_rank = mismatched_rank ? rank + 1 : rank;
  test.AddInput<int8_t>("Q_B", {b_rank, N}, std::vector<int8_t>(b_rank * N, 1));
  std::vector<float> scale_a(2 * rank, 0.25f);
  for (int64_t index = rank; index < 2 * rank; ++index) scale_a[index] = 0.5f;
  test.AddInput<float>("S_A", {2, rank}, scale_a);
  test.AddInput<float>("S_B", {(b_rank + 31) / 32, N},
                       std::vector<float>(((b_rank + 31) / 32) * N, 0.5f));
  const std::vector<T> expected(count * N, T{18.25f + (negative_weights ? -1.0f : 1.0f) *
                                                          9.5f * static_cast<float>(rank) * 0.5f * alpha});
  test.AddOutput<T>("Y", output_shape, expected);
  if (gpu) {
#ifdef USE_WEBGPU
    ConfigOptions config;
    ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                           webgpu::options::kEnableGraphCapture_ON));
    ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kValidationMode,
                                           webgpu::options::kValidationMode_full));
    auto ep = WebGpuExecutionProviderWithOptions(config);
    if (!ep) {
      GTEST_SKIP() << "WebGPU execution provider is unavailable.";
    }
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    if (weight_prepacked != 0) {
      test.Config(options)
          .ConfigEp(std::move(ep))
          .Config(OpTester::ExpectResult::kExpectFailure, "weight_prepacked=0")
          .RunWithConfig();
    } else {
      test.RunWithBoundCapture<T>(std::move(ep), options, TensorShape(output_shape), expected);
    }
    return;
#else
    GTEST_SKIP() << "WebGPU execution provider is not compiled.";
#endif
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  const bool invalid_alpha = !std::isfinite(alpha);
  test.Run(mismatched_rank || invalid_alpha || weight_prepacked != 0 ? OpTester::ExpectResult::kExpectFailure
                                                                     : OpTester::ExpectResult::kExpectSuccess,
           weight_prepacked != 0 ? "weight_prepacked=0" : invalid_alpha ? "lora_alpha must be finite"
                                                      : mismatched_rank ? "MatMulNBitsLora requires"
                                                                        : "",
           {}, nullptr, &providers);
}

}  // namespace

TEST(MatMulNBitsLora, CpuZeroRank) { RunLoraTest<float>(0); }
TEST(MatMulNBitsLora, CpuActiveRank) { RunLoraTest<float>(3); }
TEST(MatMulNBitsLora, CpuMultipleScaleBlocks) { RunLoraTest<float>(33); }
TEST(MatMulNBitsLora, CpuEmptyOutput) { RunLoraTest<float>(3, false, true); }
TEST(MatMulNBitsLora, CpuVector) { RunLoraTest<float>(3, false, false, false, true); }
TEST(MatMulNBitsLora, CpuNegativeQuantizedWeights) { RunLoraTest<float>(3, false, false, false, false, true); }
TEST(MatMulNBitsLora, CpuRejectMismatchedRank) { RunLoraTest<float>(3, false, false, true); }
TEST(MatMulNBitsLora, CpuScalarScale) { RunLoraTest<float>(3, false, false, false, false, false, 0.5f); }
TEST(MatMulNBitsLora, CpuNegativeScalarScale) { RunLoraTest<float>(3, false, false, false, false, false, -2.0f); }
TEST(MatMulNBitsLora, CpuZeroRankWithScalarScale) { RunLoraTest<float>(0, false, false, false, false, false, 0.5f); }
TEST(MatMulNBitsLora, CpuZeroRankRejectsNonfiniteScale) {
  RunLoraTest<float>(0, false, false, false, false, false, std::numeric_limits<float>::infinity());
}
TEST(MatMulNBitsLora, WebGpuFloatZeroRank) { RunLoraTest<float>(0, true); }
TEST(MatMulNBitsLora, WebGpuFloatActiveRank) { RunLoraTest<float>(3, true); }
TEST(MatMulNBitsLora, WebGpuHalfZeroRank) { RunLoraTest<MLFloat16>(0, true); }
TEST(MatMulNBitsLora, WebGpuHalfActiveRank) { RunLoraTest<MLFloat16>(3, true); }
TEST(MatMulNBitsLora, WebGpuHalfMultipleScaleBlocks) { RunLoraTest<MLFloat16>(33, true); }
TEST(MatMulNBitsLora, WebGpuHalfEmptyOutput) { RunLoraTest<MLFloat16>(3, true, true); }
TEST(MatMulNBitsLora, WebGpuHalfVector) { RunLoraTest<MLFloat16>(3, true, false, false, true); }
TEST(MatMulNBitsLora, WebGpuHalfNegativeQuantizedWeights) { RunLoraTest<MLFloat16>(3, true, false, false, false, true); }
TEST(MatMulNBitsLora, WebGpuHalfScalarScale) { RunLoraTest<MLFloat16>(3, true, false, false, false, false, 0.5f); }

TEST(MatMulNBitsLora, CpuRejectsPrepackedLayouts) {
  for (int64_t layout : {-1, 1, 2}) {
    for (int64_t rank : {0, 3}) {
      RunLoraTest<float>(rank, false, false, false, false, false, 1.0f, layout);
    }
  }
}

TEST(MatMulNBitsLora, WebGpuRejectsPrepackedLayouts) {
  for (int64_t layout : {-1, 1, 2}) {
    for (int64_t rank : {0, 3}) {
      RunLoraTest<float>(rank, true, false, false, false, false, 1.0f, layout);
      RunLoraTest<MLFloat16>(rank, true, false, false, false, false, 1.0f, layout);
    }
  }
}

TEST(MatMulNBitsLora, ValidatesHalfShapesWithoutGpu) {
  const auto allocator = CPUAllocator::DefaultInstance();
  Tensor input(DataTypeImpl::GetType<MLFloat16>(), TensorShape({1, 3, 35}), allocator);
  Tensor a(DataTypeImpl::GetType<int8_t>(), TensorShape({35, 0}), allocator);
  Tensor b(DataTypeImpl::GetType<int8_t>(), TensorShape({0, 16}), allocator);
  Tensor sa(DataTypeImpl::GetType<float>(), TensorShape({2, 0}), allocator);
  Tensor sb(DataTypeImpl::GetType<float>(), TensorShape({0, 16}), allocator);
  EXPECT_TRUE(contrib::CheckMatMulNBitsLoraInputs(&input, &a, &b, &sa, &sb, 35, 16).IsOK());
  Tensor bad_scale(DataTypeImpl::GetType<float>(), TensorShape({1, 0}), allocator);
  EXPECT_FALSE(contrib::CheckMatMulNBitsLoraInputs(&input, &a, &b, &bad_scale, &sb, 35, 16).IsOK());
}

#ifdef USE_WEBGPU
TEST(MatMulNBitsLora, UnfusedWebGpuEmptyBlockedDequantization) {
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                         webgpu::options::kEnableGraphCapture_ON));
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kValidationMode,
                                         webgpu::options::kValidationMode_full));
  auto ep = WebGpuExecutionProviderWithOptions(config);
  if (!ep) GTEST_SKIP() << "WebGPU execution provider is unavailable.";
  WebGpuGraphCaptureTester test("DequantizeLinear", 21);
  test.AddAttribute("axis", int64_t{0});
  test.AddAttribute("block_size", int64_t{32});
  test.AddInput<int8_t>("Q", {35, 0}, {});
  test.AddInput<float>("S", {2, 0}, {});
  test.AddOutput<float>("Y", {35, 0}, {});
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.RunWithBoundCapture<float>(std::move(ep), options, TensorShape({35, 0}), {});
}
#endif

}  // namespace onnxruntime::test

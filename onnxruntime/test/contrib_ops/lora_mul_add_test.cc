// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/cpu/quantization/lora_mul_add_helper.h"
#include "core/framework/allocator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

template <typename T>
void RunLoraTest(int64_t rank, bool gpu = false, bool empty = false,
                 bool mismatched_rank = false, bool vector = false,
                 bool negative_weights = false) {
  constexpr int64_t K = 35;
  constexpr int64_t N = 16;
  const int64_t rows = empty ? 0 : 3;
  const int64_t count = vector ? 1 : 2 * rows;
  const std::vector<int64_t> input_shape = vector ? std::vector<int64_t>{K}
                                                  : std::vector<int64_t>{2, rows, K};
  const std::vector<int64_t> output_shape = vector ? std::vector<int64_t>{N}
                                                   : std::vector<int64_t>{2, rows, N};
  OpTester test("LoraMulAdd", 1, kMSDomain);
  test.AddInput<T>("base", output_shape, std::vector<T>(count * N, T{0.75f}));
  test.AddInput<T>("X", input_shape, std::vector<T>(count * K, T{1.0f}));
  test.AddInput<int8_t>("Q_A", {K, rank}, std::vector<int8_t>(K * rank, negative_weights ? -1 : 1));
  const int64_t b_rank = mismatched_rank ? rank + 1 : rank;
  test.AddInput<int8_t>("Q_B", {b_rank, N}, std::vector<int8_t>(b_rank * N, 1));
  std::vector<float> scale_a(2 * rank, 0.25f);
  for (int64_t index = rank; index < 2 * rank; ++index) scale_a[index] = 0.5f;
  test.AddInput<float>("S_A", {2, rank}, scale_a);
  test.AddInput<float>("S_B", {(b_rank + 31) / 32, N},
                       std::vector<float>(((b_rank + 31) / 32) * N, 0.5f));
  test.AddOutput<T>("Y", output_shape,
                    std::vector<T>(count * N, T{0.75f + (negative_weights ? -1.0f : 1.0f) *
                                                            9.5f * static_cast<float>(rank) * 0.5f}));
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
    test.SetNumRunCalls(4);
    test.Config(options).ConfigEp(std::move(ep)).RunWithConfig();
    return;
#else
    GTEST_SKIP() << "WebGPU execution provider is not compiled.";
#endif
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(mismatched_rank ? OpTester::ExpectResult::kExpectFailure : OpTester::ExpectResult::kExpectSuccess,
           mismatched_rank ? "LoraMulAdd requires" : "", {}, nullptr, &providers);
}

}  // namespace

TEST(LoraMulAdd, CpuZeroRank) { RunLoraTest<float>(0); }
TEST(LoraMulAdd, CpuActiveRank) { RunLoraTest<float>(3); }
TEST(LoraMulAdd, CpuMultipleScaleBlocks) { RunLoraTest<float>(33); }
TEST(LoraMulAdd, CpuEmptyOutput) { RunLoraTest<float>(3, false, true); }
TEST(LoraMulAdd, CpuVector) { RunLoraTest<float>(3, false, false, false, true); }
TEST(LoraMulAdd, CpuNegativeQuantizedWeights) { RunLoraTest<float>(3, false, false, false, false, true); }
TEST(LoraMulAdd, CpuRejectMismatchedRank) { RunLoraTest<float>(3, false, false, true); }
TEST(LoraMulAdd, WebGpuFloatZeroRank) { RunLoraTest<float>(0, true); }
TEST(LoraMulAdd, WebGpuFloatActiveRank) { RunLoraTest<float>(3, true); }
TEST(LoraMulAdd, WebGpuHalfZeroRank) { RunLoraTest<MLFloat16>(0, true); }
TEST(LoraMulAdd, WebGpuHalfActiveRank) { RunLoraTest<MLFloat16>(3, true); }
TEST(LoraMulAdd, WebGpuHalfMultipleScaleBlocks) { RunLoraTest<MLFloat16>(33, true); }
TEST(LoraMulAdd, WebGpuHalfEmptyOutput) { RunLoraTest<MLFloat16>(3, true, true); }
TEST(LoraMulAdd, WebGpuHalfVector) { RunLoraTest<MLFloat16>(3, true, false, false, true); }
TEST(LoraMulAdd, WebGpuHalfNegativeQuantizedWeights) { RunLoraTest<MLFloat16>(3, true, false, false, false, true); }

TEST(LoraMulAdd, ValidatesHalfShapesWithoutGpu) {
  const auto allocator = CPUAllocator::DefaultInstance();
  Tensor base(DataTypeImpl::GetType<MLFloat16>(), TensorShape({1, 3, 16}), allocator);
  Tensor input(DataTypeImpl::GetType<MLFloat16>(), TensorShape({1, 3, 35}), allocator);
  Tensor a(DataTypeImpl::GetType<int8_t>(), TensorShape({35, 0}), allocator);
  Tensor b(DataTypeImpl::GetType<int8_t>(), TensorShape({0, 16}), allocator);
  Tensor sa(DataTypeImpl::GetType<float>(), TensorShape({2, 0}), allocator);
  Tensor sb(DataTypeImpl::GetType<float>(), TensorShape({0, 16}), allocator);
  EXPECT_TRUE(contrib::CheckLoraMulAddInputs(&base, &input, &a, &b, &sa, &sb).IsOK());
  Tensor bad_scale(DataTypeImpl::GetType<float>(), TensorShape({1, 0}), allocator);
  EXPECT_FALSE(contrib::CheckLoraMulAddInputs(&base, &input, &a, &b, &bad_scale, &sb).IsOK());
}

#ifdef USE_WEBGPU
TEST(LoraMulAdd, UnfusedWebGpuEmptyBlockedDequantization) {
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                         webgpu::options::kEnableGraphCapture_ON));
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kValidationMode,
                                         webgpu::options::kValidationMode_full));
  auto ep = WebGpuExecutionProviderWithOptions(config);
  if (!ep) GTEST_SKIP() << "WebGPU execution provider is unavailable.";
  OpTester test("DequantizeLinear", 21);
  test.AddAttribute("axis", int64_t{0});
  test.AddAttribute("block_size", int64_t{32});
  test.AddInput<int8_t>("Q", {35, 0}, {});
  test.AddInput<float>("S", {2, 0}, {});
  test.AddOutput<float>("Y", {35, 0}, {});
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.SetNumRunCalls(4);
  test.Config(options).ConfigEp(std::move(ep)).RunWithConfig();
}
#endif

}  // namespace onnxruntime::test

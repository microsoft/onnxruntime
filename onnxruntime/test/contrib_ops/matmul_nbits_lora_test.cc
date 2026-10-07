// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/cpu/quantization/matmul_nbits_lora_helper.h"
#include "core/framework/allocator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

void RunLoraTest(int64_t rank, bool bias, bool empty = false, bool mismatched_rank = false,
                 bool vector = false) {
  constexpr int64_t K = 32;
  constexpr int64_t N = 16;
  const int64_t rows = empty ? 0 : 3;
  OpTester test("MatMulNBitsLora", 1, kMSDomain);
  test.AddAttribute("K", K);
  test.AddAttribute("N", N);
  test.AddAttribute("bits", int64_t{4});
  test.AddAttribute("block_size", int64_t{32});
  const std::vector<int64_t> input_shape = vector ? std::vector<int64_t>{K}
                                                  : std::vector<int64_t>{2, rows, K};
  const std::vector<int64_t> output_shape = vector ? std::vector<int64_t>{N}
                                                   : std::vector<int64_t>{2, rows, N};
  test.AddInput<float>("A", input_shape, std::vector<float>(vector ? K : 2 * rows * K, 1.0f));
  test.AddInput<uint8_t>("B", {N, 1, 16}, std::vector<uint8_t>(N * 16, 0x88), true);
  test.AddInput<float>("scales", {N, 1}, std::vector<float>(N, 1.0f), true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddOptionalInputEdge<int32_t>();
  if (bias) {
    test.AddInput<float>("bias", {N}, std::vector<float>(N, 1.0f), true);
  } else {
    test.AddOptionalInputEdge<float>();
  }
  test.AddInput<float>("lora_A", {K, rank}, std::vector<float>(K * rank, 1.0f));
  const int64_t b_rank = mismatched_rank ? rank + 1 : rank;
  test.AddInput<float>("lora_B", {b_rank, N}, std::vector<float>(b_rank * N, 1.0f));
  test.AddOutput<float>("Y", output_shape,
                        std::vector<float>(vector ? N : 2 * rows * N,
                                           static_cast<float>(K * rank + (bias ? 1 : 0))));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(mismatched_rank ? OpTester::ExpectResult::kExpectFailure : OpTester::ExpectResult::kExpectSuccess,
           mismatched_rank ? "MatMulNBitsLora requires" : "", {}, nullptr, &providers);
}

}  // namespace

TEST(MatMulNBitsLora, ZeroRank) { RunLoraTest(0, false); }
TEST(MatMulNBitsLora, ZeroRankWithBias) { RunLoraTest(0, true); }
TEST(MatMulNBitsLora, ActiveRank) { RunLoraTest(2, false); }
TEST(MatMulNBitsLora, ActiveRankWithBias) { RunLoraTest(2, true); }
TEST(MatMulNBitsLora, EmptyOutput) { RunLoraTest(2, false, true); }
TEST(MatMulNBitsLora, RejectMismatchedRank) { RunLoraTest(2, false, false, true); }
TEST(MatMulNBitsLora, VectorZeroRank) { RunLoraTest(0, false, false, false, true); }
TEST(MatMulNBitsLora, VectorActiveRank) { RunLoraTest(2, true, false, false, true); }

TEST(MatMulNBitsLora, ValidateFloat16ShapesWithoutGpu) {
  const auto allocator = CPUAllocator::DefaultInstance();
  Tensor input(DataTypeImpl::GetType<MLFloat16>(), TensorShape({1, 3, 32}), allocator);
  Tensor lora_a(DataTypeImpl::GetType<MLFloat16>(), TensorShape({32, 0}), allocator);
  Tensor lora_b(DataTypeImpl::GetType<MLFloat16>(), TensorShape({0, 16}), allocator);
  EXPECT_TRUE(contrib::CheckMatMulNBitsLoraInputs(&input, &lora_a, &lora_b, 32, 16).IsOK());
  Tensor wrong_type(DataTypeImpl::GetType<float>(), TensorShape({0, 16}), allocator);
  EXPECT_FALSE(contrib::CheckMatMulNBitsLoraInputs(&input, &lora_a, &wrong_type, 32, 16).IsOK());
}

#ifdef USE_WEBGPU
namespace {

void RunHalfWebGpuTest(int64_t rank, bool bias, bool empty = false, bool vector = false) {
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                         webgpu::options::kEnableGraphCapture_ON));
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kValidationMode,
                                         webgpu::options::kValidationMode_full));
  auto ep = WebGpuExecutionProviderWithOptions(config);
  if (!ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  constexpr int64_t K = 32;
  constexpr int64_t N = 16;
  const int64_t rows = empty ? 0 : 3;
  OpTester test("MatMulNBitsLora", 1, kMSDomain);
  test.AddAttribute("K", K);
  test.AddAttribute("N", N);
  test.AddAttribute("bits", int64_t{4});
  test.AddAttribute("block_size", int64_t{32});
  const std::vector<int64_t> input_shape = vector ? std::vector<int64_t>{K}
                                                  : std::vector<int64_t>{2, rows, K};
  const std::vector<int64_t> output_shape = vector ? std::vector<int64_t>{N}
                                                   : std::vector<int64_t>{2, rows, N};
  test.AddInput<MLFloat16>("A", input_shape,
                           std::vector<MLFloat16>(vector ? K : 2 * rows * K, MLFloat16{1.0f}));
  test.AddInput<uint8_t>("B", {N, 1, 16}, std::vector<uint8_t>(N * 16, 0x88), true);
  test.AddInput<MLFloat16>("scales", {N, 1},
                           std::vector<MLFloat16>(N, MLFloat16{1.0f}), true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddOptionalInputEdge<int32_t>();
  if (bias) {
    test.AddInput<MLFloat16>("bias", {N},
                             std::vector<MLFloat16>(N, MLFloat16{1.0f}), true);
  } else {
    test.AddOptionalInputEdge<MLFloat16>();
  }
  test.AddInput<MLFloat16>("lora_A", {K, rank},
                           std::vector<MLFloat16>(K * rank, MLFloat16{1.0f}));
  test.AddInput<MLFloat16>("lora_B", {rank, N},
                           std::vector<MLFloat16>(rank * N, MLFloat16{1.0f}));
  test.AddOutput<MLFloat16>("Y", output_shape,
                            std::vector<MLFloat16>(vector ? N : 2 * rows * N,
                                                   MLFloat16{static_cast<float>(K * rank + (bias ? 1 : 0))}));
  test.SetNumRunCalls(4);
  test.ConfigEp(std::move(ep)).RunWithConfig();
}

}  // namespace

TEST(MatMulNBitsLora, WebGpuFloat16ZeroRankCapture) { RunHalfWebGpuTest(0, false); }
TEST(MatMulNBitsLora, WebGpuFloat16ActiveRankCapture) { RunHalfWebGpuTest(2, true); }
TEST(MatMulNBitsLora, WebGpuFloat16EmptyOutput) { RunHalfWebGpuTest(2, false, true); }
TEST(MatMulNBitsLora, WebGpuFloat16VectorActiveRankCapture) { RunHalfWebGpuTest(2, true, false, true); }
#endif

}  // namespace onnxruntime::test

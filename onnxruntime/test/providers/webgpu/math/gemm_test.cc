// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
namespace {

const onnxruntime::RunOptions run_options = []() {
  onnxruntime::RunOptions options{};
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOpTesterRunOptionsConfigTestTunableOp, "true"));
  return options;
}();

const constexpr auto run_with_tunable_op = &run_options;

}  // namespace

#endif

#if defined(USE_WEBGPU)
TEST(GemmOpTest, WebGpuPackedFp16LongReduction) {
  for (const auto& dimensions : {std::pair<int64_t, int64_t>{900, 4}, {899, 3}, {900, 3}}) {
    const auto [reduction_size, columns] = dimensions;
    const bool transpose_b = reduction_size == 900 && columns == 3;
    SCOPED_TRACE(reduction_size);
    SCOPED_TRACE(columns);
    auto webgpu_ep = DefaultWebGpuExecutionProvider();
    ASSERT_NE(webgpu_ep, nullptr);
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    const MLFloat16 value{0.1f};
    const MLFloat16 expected{0.5f * value.ToFloat() * value.ToFloat() * static_cast<float>(reduction_size) + 0.25f};
    OpTester test("Gemm", 13);
    test.AddAttribute("alpha", 0.5f);
    test.AddAttribute("beta", 0.5f);
    test.AddAttribute("transB", static_cast<int64_t>(transpose_b));
    test.AddInput<MLFloat16>("A", {4, reduction_size}, std::vector<MLFloat16>(4 * reduction_size, value));
    test.AddInput<MLFloat16>("B", transpose_b ? std::vector<int64_t>{columns, reduction_size}
                                           : std::vector<int64_t>{reduction_size, columns},
                           std::vector<MLFloat16>(reduction_size * columns, value));
    test.AddInput<MLFloat16>("C", {columns}, std::vector<MLFloat16>(columns, MLFloat16(0.5f)));
    test.AddOutput<MLFloat16>("Y", {4, columns}, std::vector<MLFloat16>(4 * columns, expected));
    test.SetOutputAbsErr("Y", 0.01f);
    test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
  }
}

TEST(GemmOpTest, WebGpuPackedFp16SplitKCancellation) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(webgpu_ep, nullptr);
  constexpr int64_t reduction_size = 1024;
  constexpr int64_t columns = 16;
  std::vector<MLFloat16> weights(reduction_size * columns);
  for (int64_t reduction_index = 0; reduction_index < reduction_size; ++reduction_index) {
    for (int64_t column = 0; column < columns; ++column) {
      weights[reduction_index * columns + column] = MLFloat16(reduction_index < 512 ? 512.0f : -512.0f);
    }
  }
  for (int64_t column = 0; column < columns; ++column) {
    weights[column] = MLFloat16(513.0f);
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  OpTester test("Gemm", 13);
  test.AddInput<MLFloat16>("A", {8, reduction_size}, std::vector<MLFloat16>(8 * reduction_size, MLFloat16(1.0f)));
  test.AddInput<MLFloat16>("B", {reduction_size, columns}, weights, true);
  test.AddOutput<MLFloat16>("Y", {8, columns}, std::vector<MLFloat16>(8 * columns, MLFloat16(1.0f)));
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// Test int32 with M=128, K=128, N=128, transA=True
TEST(GemmOpTest, GemmTransA_int32_128x128x128) {
  OpTester test("Gemm", 13);

  test.AddAttribute("transA", (int64_t)1);  // transposeA = 1
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  const int64_t M = 128, K = 128, N = 128;

  // Initialize input matrices with int values
  std::vector<int32_t> A_data(K * M);  // A shape is {K, M} because transposeA=1
  std::vector<int32_t> B_data(K * N);
  std::vector<int32_t> C_data(M * N);

  // Fill A matrix with pattern (will be transposed)
  for (int64_t i = 0; i < K * M; ++i) {
    A_data[i] = static_cast<int32_t>((i % 7) + 1);
  }

  // Fill B matrix with pattern
  for (int64_t i = 0; i < K * N; ++i) {
    B_data[i] = static_cast<int32_t>((i % 5) + 1);
  }

  // Fill C matrix (bias) with small values
  for (int64_t i = 0; i < M * N; ++i) {
    C_data[i] = static_cast<int32_t>((i % 3) + 1);
  }

  // Calculate expected output: Y = alpha * A^T * B + beta * C
  std::vector<int32_t> Y_data(M * N, 0);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      int64_t sum = 0;
      for (int64_t k = 0; k < K; ++k) {
        // A is transposed, so A^T[i][k] = A[k][i]
        sum += static_cast<int64_t>(A_data[k * M + i]) * static_cast<int64_t>(B_data[k * N + j]);
      }
      Y_data[i * N + j] = static_cast<int32_t>(sum + C_data[i * N + j]);  // alpha=1.0, beta=1.0
    }
  }

  test.AddInput<int32_t>("A", {K, M}, A_data);  // A shape is {K, M} because transA=True
  test.AddInput<int32_t>("B", {K, N}, B_data);
  test.AddInput<int32_t>("C", {M, N}, C_data);
  test.AddOutput<int32_t>("Y", {M, N}, Y_data);

  test.ConfigExcludeEps({kQnnExecutionProvider, kCpuExecutionProvider, kCoreMLExecutionProvider})
      .Config(run_with_tunable_op)
      .RunWithConfig();
}
#endif  // defined(USE_WEBGPU)

}  // namespace test
}  // namespace onnxruntime

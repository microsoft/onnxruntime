// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
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

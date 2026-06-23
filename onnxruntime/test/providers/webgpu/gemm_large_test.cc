// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/cpu/math/gemm_helper.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

static float GetBiasValue(const std::vector<float>& c_vals, const TensorShape& c_shape,
                          int64_t M, int64_t N, int64_t m, int64_t n) {
  if (c_vals.empty())
    return 0.0f;
  if (c_shape.Size() == 1)
    return c_vals[0];
  if (c_shape.NumDimensions() == 1 && c_shape[0] == N)
    return c_vals[n];
  if (c_shape.NumDimensions() == 2) {
    if (c_shape[0] == M && c_shape[1] == 1)
      return c_vals[m];
    if (c_shape[0] == M && c_shape[1] == N)
      return c_vals[m * N + n];
    if (c_shape[0] == 1 && c_shape[1] == N)
      return c_vals[n];
  }
  ADD_FAILURE() << "Unsupported bias shape in test reference";
  return 0.0f;
}

static void ComputeExpectedResult(int64_t M, int64_t K, int64_t N,
                                  const std::vector<float>& a_vals, const std::vector<float>& b_vals,
                                  const std::vector<float>& c_vals, std::vector<float>& expected_vals,
                                  int64_t a_trans, int64_t b_trans, const TensorShape& c_shape,
                                  float alpha, float beta) {
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float a_val = a_trans ? a_vals[k * M + m] : a_vals[m * K + k];
        float b_val = b_trans ? b_vals[n * K + k] : b_vals[k * N + n];
        sum += a_val * b_val;
      }
      expected_vals[m * N + n] = sum * alpha + GetBiasValue(c_vals, c_shape, M, N, m, n) * beta;
    }
  }
}

template <typename T, int version = 13>
void RunTestTyped(std::initializer_list<int64_t> a_dims, int64_t a_trans, std::initializer_list<int64_t> b_dims,
                  int64_t b_trans, std::initializer_list<int64_t> c_dims, float alpha = 1.0f, float beta = 1.0f) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, MLFloat16>, "unexpected type for T");

  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  TensorShape a_shape(a_dims);
  TensorShape b_shape(b_dims);
  TensorShape c_shape(c_dims);
  GemmHelper helper(a_shape, a_trans != 0, b_shape, b_trans != 0, c_shape);
  ASSERT_STATUS_OK(helper.State());
  const auto M = helper.M();
  const auto K = helper.K();
  const auto N = helper.N();

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));
  std::vector<float> c_vals;
  if (c_dims.size() != 0) {
    c_vals = random.Gaussian<float>(AsSpan(c_dims), 0.0f, 0.25f);
  }

  std::vector<float> expected_vals(M * N);
  ComputeExpectedResult(M, K, N, a_vals, b_vals, c_vals, expected_vals, a_trans, b_trans, c_shape, alpha, beta);

  OpTester test("Gemm", version);
  test.AddAttribute("transA", a_trans);
  test.AddAttribute("transB", b_trans);
  test.AddAttribute("alpha", alpha);
  test.AddAttribute("beta", beta);
  if constexpr (std::is_same_v<T, float>) {
    test.AddInput<T>("A", a_dims, a_vals);
    test.AddInput<T>("B", b_dims, b_vals);
    if (c_dims.size() != 0)
      test.AddInput<T>("C", c_dims, c_vals);
    test.AddOutput<T>("Y", {M, N}, expected_vals);
  } else {
    test.AddInput<T>("A", a_dims, FloatsToMLFloat16s(a_vals));
    test.AddInput<T>("B", b_dims, FloatsToMLFloat16s(b_vals));
    if (c_dims.size() != 0)
      test.AddInput<T>("C", c_dims, FloatsToMLFloat16s(c_vals));
    test.AddOutput<T>("Y", {M, N}, FloatsToMLFloat16s(expected_vals));
    test.SetOutputAbsErr("Y", 0.055f);
    test.SetOutputRelErr("Y", 0.02f);
  }

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

template <int version = 13>
void RunBothTypes(std::initializer_list<int64_t> a_dims, int64_t a_trans,
                  std::initializer_list<int64_t> b_dims, int64_t b_trans,
                  std::initializer_list<int64_t> c_dims, float alpha = 1.0f, float beta = 1.0f) {
  RunTestTyped<float, version>(a_dims, a_trans, b_dims, b_trans, c_dims, alpha, beta);
  RunTestTyped<MLFloat16, version>(a_dims, a_trans, b_dims, b_trans, c_dims, alpha, beta);
}

// Aligned dimensions and baseline large shapes.
TEST(Gemm_Large, DISABLED_Aligned) {
  RunBothTypes({512, 1024}, 0, {1024, 1024}, 0, {512, 1024});
  RunBothTypes({127, 1024}, 0, {1024, 1024}, 0, {1024});
}

// Unaligned dimensions and edge shapes.
TEST(Gemm_Large, DISABLED_Unaligned) {
  RunBothTypes({127, 1023}, 0, {1023, 1023}, 0, {1023});
  RunBothTypes({511, 1024}, 0, {1024, 1023}, 0, {511, 1});
  RunBothTypes({6, 1024}, 0, {1024, 192}, 0, {6, 1});
  RunBothTypes({49, 1024}, 0, {1024, 600}, 0, {49, 600});
}

// Transpose combinations and transposed bias cases.
TEST(Gemm_Large, DISABLED_TransAB) {
  RunBothTypes({1024, 512}, 1, {1024, 1024}, 0, {512, 1});
  RunBothTypes({512, 1024}, 0, {1024, 1024}, 1, {512, 1});
  RunBothTypes({1024, 512}, 1, {1024, 1024}, 1, {512, 1});
  RunBothTypes({1024, 512}, 1, {1024, 1024}, 1, {512, 1}, 1.5f, 1.3f);
  RunBothTypes({1024, 16}, 1, {1024, 192}, 0, {192});
  RunBothTypes({16, 1024}, 0, {192, 1024}, 1, {192});
  RunBothTypes({1024, 16}, 1, {192, 1024}, 1, {192});
}

// Bias broadcast coverage, including no-bias and alpha/beta scaling.
TEST(Gemm_Large, DISABLED_BiasBroadcast) {
  RunBothTypes({16, 1024}, 0, {1024, 191}, 0, {1, 191});
  RunBothTypes({15, 1024}, 0, {1024, 191}, 0, {15, 191});
  RunBothTypes({15, 1024}, 0, {1024, 192}, 0, {15, 1});
  RunTestTyped<float>({16, 1024}, 0, {1024, 192}, 0, {});  // no bias
  RunBothTypes({16, 1024}, 0, {1024, 192}, 0, {16, 1});
  RunBothTypes({16, 1024}, 0, {1024, 192}, 0, {1, 192});
  RunBothTypes({16, 1024}, 0, {1024, 192}, 0, {192});
  RunBothTypes({16, 1024}, 0, {1024, 192}, 0, {16, 192});
  RunBothTypes({16, 1024}, 0, {1024, 600}, 0, {1, 600});

  RunBothTypes({16, 1024}, 0, {1024, 192}, 0, {16, 1}, 1.5f, 1.3f);
  RunBothTypes({16, 1024}, 0, {1024, 192}, 0, {1, 192}, 1.5f, 1.3f);
  RunBothTypes({16, 1024}, 0, {1024, 192}, 0, {192}, 1.5f, 1.3f);
  RunBothTypes({16, 1024}, 0, {1024, 192}, 0, {16, 192}, 1.5f, 1.3f);
}

}  // namespace test
}  // namespace onnxruntime

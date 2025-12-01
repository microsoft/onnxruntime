// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

bool IsValidBroadcast(const TensorShape& bias_shape, int64_t M, int64_t N) {
  // valid shapes are (,) , (1, N) , (M, 1) , (M, N)
  if (bias_shape.NumDimensions() > 2)
    return false;
  // shape is (1,) or (1, 1), or (,)
  if (bias_shape.Size() == 1)
    return true;
  // valid bias_shape (s) are (N,) or (1, N) or (M, 1) or (M, N),
  // In last case no broadcasting needed, so don't fail it
  return ((bias_shape.NumDimensions() == 1 && bias_shape[0] == N) ||
          (bias_shape.NumDimensions() == 2 && bias_shape[0] == M && (bias_shape[1] == 1 || bias_shape[1] == N)) ||
          (bias_shape.NumDimensions() == 2 && bias_shape[0] == 1 && bias_shape[1] == N));
}

Status ComputeGemmOutputShape(const TensorShape& left, int64_t trans_left, const TensorShape& right,
                              int64_t trans_right, const TensorShape& bias, int64_t& M, int64_t& K, int64_t& N) {
  // dimension check
  ORT_ENFORCE(left.NumDimensions() == 2 || left.NumDimensions() == 1);
  ORT_ENFORCE(right.NumDimensions() == 2);

  for (size_t i = 0; i != left.NumDimensions(); ++i) {
    ORT_ENFORCE(left[i] >= 0);
    ORT_ENFORCE(left[i] <= std::numeric_limits<int64_t>::max());
  }

  for (size_t i = 0; i != right.NumDimensions(); ++i) {
    ORT_ENFORCE(right[i] >= 0);
    ORT_ENFORCE(right[i] <= std::numeric_limits<int64_t>::max());
  }

  if (trans_left == 1) {
    M = left.NumDimensions() == 2 ? left[1] : left[0];
    K = left.NumDimensions() == 2 ? left[0] : 1;
  } else {
    M = left.NumDimensions() == 2 ? left[0] : 1;
    K = left.NumDimensions() == 2 ? left[1] : left[0];
  }

  N = trans_right == 1 ? right[0] : right[1];
  int k_dim = trans_right == 1 ? 1 : 0;

  Status status;
  if (right[k_dim] != K) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GEMM: Dimension mismatch, W: ", right.ToString(),
                             " K: " + std::to_string(K), " N:" + std::to_string(N));
    return status;
  }

  if (!IsValidBroadcast(bias, M, N)) {
    status = common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Gemm: Invalid bias shape for broadcast");
    return status;
  }

  // it is possible the input is empty tensor, for example the output of roipool in fast rcnn.
  // it is also possible that K == 0
  ORT_ENFORCE(M >= 0 && K >= 0 && N >= 0);

  return status;
}

float GetScale(const std::vector<float>& c_vals, const TensorShape& c_shape, int64_t M, int64_t N, int64_t m, int64_t n) {
  if (c_vals.empty())
    return 0.0f;
  if (c_shape.Size() == 1)
    return c_vals[0];
  // valid c_shape (s) are (N,) or (1, N) or (M, 1) or (M, N),
  // In last case no broadcasting needed, so don't fail it
  if (c_shape.NumDimensions() == 1 && c_shape[0] == N) {
    return c_vals[n];
  }

  if (c_shape.NumDimensions() == 2 && c_shape[0] == M) {
    if (c_shape[1] == 1) {
      return c_vals[m];
    } else if (c_shape[1] == N) {
      return c_vals[m * N + n];
    }
  }

  if (c_shape.NumDimensions() == 2 && c_shape[0] == 1 && c_shape[1] == N) {
    return c_vals[n];
  }
  return 0.0f;
}

Status GetExpectedResult(const int64_t M, const int64_t K, const int64_t N, const std::vector<float>& a_vals,
                         const std::vector<float>& b_vals, const std::vector<float>& c_vals,
                         std::vector<float>& expected_vals, const TensorShape& a_shape, int64_t a_trans,
                         const TensorShape& b_shape, int64_t b_trans,
                         const TensorShape& c_shape, float alpha, float beta) {
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        if (a_trans == 0 && b_trans == 0) {
          sum += a_vals[m * K + k] * b_vals[k * N + n];
        } else if (a_trans == 0 && b_trans == 1) {
          sum += a_vals[m * K + k] * b_vals[n * K + k];
        } else if (a_trans == 1 && b_trans == 0) {
          sum += a_vals[k * M + m] * b_vals[k * N + n];
        } else {
          sum += a_vals[k * M + m] * b_vals[n * K + k];
        }
      }
      expected_vals[m * N + n] = sum * alpha + GetScale(c_vals, c_shape, M, N, m, n) * beta;
    }
  }

  return Status::OK();
}

template <typename T1, int version>
void RunTestTyped(std::initializer_list<int64_t> a_dims, int64_t a_trans, std::initializer_list<int64_t> b_dims,
                  int64_t b_trans, std::initializer_list<int64_t> c_dims, float alpha = 1.0f, float beta = 1.0f) {
  static_assert(std::is_same_v<T1, float> || std::is_same_v<T1, MLFloat16>, "unexpected type for T1");

  int64_t M = 0;
  int64_t K = 0;
  int64_t N = 0;
  TensorShape a_shape = TensorShape(a_dims);
  TensorShape b_shape = TensorShape(b_dims);
  TensorShape c_shape = TensorShape(c_dims);
  ComputeGemmOutputShape(a_shape, a_trans, b_shape, b_trans, c_shape, M, K, N);

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));
  std::vector<float> c_vals;
  if (c_dims.size() > 0) {
    c_vals = std::vector<float>(random.Gaussian<float>(AsSpan(c_dims), 0.0f, 0.25f));
  }
  std::vector<float> expected_vals(M * N);
  GetExpectedResult(M, K, N, a_vals, b_vals, c_vals, expected_vals, a_shape, a_trans, b_shape, b_trans, c_shape, alpha, beta);

  OpTester test("Gemm", version);
  test.AddAttribute("transA", a_trans);
  test.AddAttribute("transB", b_trans);
  test.AddAttribute("alpha", alpha);
  test.AddAttribute("beta", beta);
  if constexpr (std::is_same_v<T1, float>) {
    test.AddInput<T1>("A", a_dims, a_vals);
    test.AddInput<T1>("B", b_dims, b_vals);
    if (c_dims.size() != 0) {
      test.AddInput<T1>("C", c_dims, c_vals);
    }
    test.AddOutput<T1>("Y", {M, N}, expected_vals);
  } else if constexpr (std::is_same<T1, MLFloat16>::value) {
    test.AddInput<T1>("A", a_dims, FloatsToMLFloat16s(a_vals));
    test.AddInput<T1>("B", b_dims, FloatsToMLFloat16s(b_vals));
    if (c_dims.size() != 0) {
      test.AddInput<T1>("C", c_dims, FloatsToMLFloat16s(c_vals));
    }
    test.AddOutput<T1>("Y", {M, N}, FloatsToMLFloat16s(expected_vals));
    test.SetOutputAbsErr("Y", 0.055f);
    test.SetOutputRelErr("Y", 0.02f);
  }

  test.RunWithConfig();
}

TEST(Gemm_Large, Float32) {
  RunTestTyped<float, 13>({512, 1024}, 0, {1024, 1024}, 0, {512, 1024});
  RunTestTyped<float, 13>({127, 1024}, 0, {1024, 1024}, 0, {1024});
  RunTestTyped<float, 13>({127, 1023}, 0, {1023, 1023}, 0, {1023});
  RunTestTyped<float, 13>({511, 1024}, 0, {1024, 1023}, 0, {511, 1});
  RunTestTyped<float, 13>({1024, 512}, 1, {1024, 1024}, 0, {512, 1});
  RunTestTyped<float, 13>({512, 1024}, 0, {1024, 1024}, 1, {512, 1});
  RunTestTyped<float, 13>({1024, 512}, 1, {1024, 1024}, 1, {512, 1});
  RunTestTyped<float, 13>({1024, 512}, 1, {1024, 1024}, 1, {512, 1}, 1.5f, 1.3f);

  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 191}, 0, {1, 191});
  RunTestTyped<float, 13>({15, 1024}, 0, {1024, 191}, 0, {15, 191});
  RunTestTyped<float, 13>({15, 1024}, 0, {1024, 192}, 0, {15, 1});
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {});
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {16, 1});
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {1, 192});
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {192});
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {16, 192});
  RunTestTyped<float, 13>({6, 1024}, 0, {1024, 192}, 0, {6, 1});
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 600}, 0, {1, 600});
  RunTestTyped<float, 13>({49, 1024}, 0, {1024, 600}, 0, {49, 600});
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {16, 1}, 1.5f, 1.3f);
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {1, 192}, 1.5f, 1.3f);
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {192}, 1.5f, 1.3f);
  RunTestTyped<float, 13>({16, 1024}, 0, {1024, 192}, 0, {16, 192}, 1.5f, 1.3f);
  RunTestTyped<float, 13>({1024, 16}, 1, {1024, 192}, 0, {192});
  RunTestTyped<float, 13>({16, 1024}, 0, {192, 1024}, 1, {192});
  RunTestTyped<float, 13>({1024, 16}, 1, {192, 1024}, 1, {192});
}

TEST(Gemm_Large, Float16) {
  RunTestTyped<MLFloat16, 13>({512, 1024}, 0, {1024, 1024}, 0, {512, 1024});
  RunTestTyped<MLFloat16, 13>({127, 1024}, 0, {1024, 1024}, 0, {1024});
  RunTestTyped<MLFloat16, 13>({127, 1023}, 0, {1023, 1023}, 0, {1023});
  RunTestTyped<MLFloat16, 13>({511, 1024}, 0, {1024, 1023}, 0, {511, 1});
  RunTestTyped<MLFloat16, 13>({1024, 512}, 1, {1024, 1024}, 0, {512, 1});
  RunTestTyped<MLFloat16, 13>({512, 1024}, 0, {1024, 1024}, 1, {512, 1});
  RunTestTyped<MLFloat16, 13>({1024, 512}, 1, {1024, 1024}, 1, {512, 1});
  RunTestTyped<MLFloat16, 13>({1024, 512}, 1, {1024, 1024}, 1, {512, 1}, 1.5f, 1.3f);

  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 191}, 0, {1, 191});
  RunTestTyped<MLFloat16, 13>({15, 1024}, 0, {1024, 191}, 0, {15, 191});
  RunTestTyped<MLFloat16, 13>({15, 1024}, 0, {1024, 192}, 0, {15, 1});
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 192}, 0, {16, 1});
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 192}, 0, {1, 192});
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 192}, 0, {192});
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 192}, 0, {16, 192});
  RunTestTyped<MLFloat16, 13>({6, 1024}, 0, {1024, 192}, 0, {6, 1});
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 600}, 0, {1, 600});
  RunTestTyped<MLFloat16, 13>({49, 1024}, 0, {1024, 600}, 0, {49, 600});
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 192}, 0, {16, 1}, 1.5, 1.3);
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 192}, 0, {1, 192}, 1.5, 1.3);
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 192}, 0, {192}, 1.5, 1.3);
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {1024, 192}, 0, {16, 192}, 1.5, 1.3);
  RunTestTyped<MLFloat16, 13>({1024, 16}, 1, {1024, 192}, 0, {192});
  RunTestTyped<MLFloat16, 13>({16, 1024}, 0, {192, 1024}, 1, {192});
  RunTestTyped<MLFloat16, 13>({1024, 16}, 1, {192, 1024}, 1, {192});
}

}  // namespace test
}  // namespace onnxruntime

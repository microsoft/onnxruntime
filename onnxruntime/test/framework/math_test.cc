/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Modifications Copyright (c) Microsoft.

#include "core/util/math.h"
#include "core/common/float16.h"  // BFloat16 helpers
#include <gtest/gtest.h>
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"
#include "core/util/thread_utils.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

#define VECTOR_HEAD(x) x.size() > 0 ? &x[0] : NULL

// parameter is thread pool size
class MathGemmTest : public testing::TestWithParam<int> {
 protected:
  static OrtThreadPoolParams CreateThreadPoolOptions(int size) {
    OrtThreadPoolParams option;
    option.thread_pool_size = size;
    return option;
  }
  std::unique_ptr<concurrency::ThreadPool> tp{concurrency::CreateThreadPool(&Env::Default(), CreateThreadPoolOptions(GetParam()), concurrency::ThreadPoolType::INTRA_OP)};
};

TEST_P(MathGemmTest, GemmNoTransNoTrans) {
  auto& provider = CPUMathUtil::Instance();
  std::vector<float> X(50);  // 5 * 10
  std::vector<float> W(60);  // 10 * 6
  std::vector<float> Y(30);  // 5 * 6
  math::Set<float, CPUMathUtil>(X.size(), 1, VECTOR_HEAD(X), &provider);
  math::Set<float, CPUMathUtil>(W.size(), 1, VECTOR_HEAD(W), &provider);
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X[i], 1);
  }
  for (size_t i = 0; i < W.size(); ++i) {
    EXPECT_EQ(W[i], 1);
  }

  constexpr float kOne = 1.0;
  constexpr float kPointFive = 0.5;
  constexpr float kZero = 0.0;
  math::Gemm<float>(CblasNoTrans, CblasNoTrans, 5, 6, 10, kOne,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kZero, VECTOR_HEAD(Y),
                    tp.get());
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float>(CblasNoTrans, CblasNoTrans, 5, 6, 10, kOne,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kPointFive,
                    VECTOR_HEAD(Y), tp.get());
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 15) << i;
  }
  // Test Accumulate
  math::Gemm<float>(CblasNoTrans, CblasNoTrans, 5, 6, 10,
                    kPointFive,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kOne, VECTOR_HEAD(Y),
                    tp.get());
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 20) << i;
  }
}

TEST_P(MathGemmTest, GemmNoTransTrans) {
  auto& provider = CPUMathUtil::Instance();

  std::vector<float> X(50);  // 5 * 10
  std::vector<float> W(60);  // 10 * 6
  std::vector<float> Y(30);  // 5 * 6
  math::Set<float, CPUMathUtil>(X.size(), 1, VECTOR_HEAD(X), &provider);
  math::Set<float, CPUMathUtil>(W.size(), 1, VECTOR_HEAD(W), &provider);
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X[i], 1);
  }
  for (size_t i = 0; i < W.size(); ++i) {
    EXPECT_EQ(W[i], 1);
  }

  constexpr float kOne = 1.0;
  constexpr float kPointFive = 0.5;
  constexpr float kZero = 0.0;
  math::Gemm<float>(CblasNoTrans, CblasTrans, 5, 6, 10, kOne,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kZero, VECTOR_HEAD(Y),
                    tp.get());
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float>(CblasNoTrans, CblasTrans, 5, 6, 10, kOne,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kPointFive,
                    VECTOR_HEAD(Y), tp.get());
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 15) << i;
  }
  math::Gemm<float>(CblasNoTrans, CblasTrans, 5, 6, 10, kPointFive,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kOne, VECTOR_HEAD(Y),
                    tp.get());
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 20) << i;
  }
}

INSTANTIATE_TEST_SUITE_P(MathGemmTests, MathGemmTest,
                         testing::Values(1, 0));

class MathBFloat16GemmTest : public testing::TestWithParam<int> {
 protected:
  static OrtThreadPoolParams CreateThreadPoolOptions(int size) {
    OrtThreadPoolParams option;
    option.thread_pool_size = size;
    return option;
  }
  std::unique_ptr<concurrency::ThreadPool> tp{concurrency::CreateThreadPool(&Env::Default(), CreateThreadPoolOptions(GetParam()), concurrency::ThreadPoolType::INTRA_OP)};
};

TEST_P(MathBFloat16GemmTest, GemmNoTransNoTrans) {
  if (!MlasBf16AccelerationSupported()) {
    GTEST_SKIP() << "BF16 acceleration is not supported on this platform.";
  }
  const int M = 5;
  const int N = 6;
  const int K = 10;

  std::vector<float> X_fp32(M * K);
  std::vector<float> W_fp32(K * N);
  std::vector<float> Y_fp32(M * N);
  std::vector<float> Y_ref(M * N);

  // Initialize with random data
  for (size_t i = 0; i < X_fp32.size(); ++i) X_fp32[i] = (float)rand() / (float)RAND_MAX;
  for (size_t i = 0; i < W_fp32.size(); ++i) W_fp32[i] = (float)rand() / (float)RAND_MAX;

  std::vector<BFloat16> X_bf16(M * K);
  std::vector<BFloat16> W_bf16(K * N);
  std::vector<BFloat16> Y_bf16(M * N);

  FloatToBFloat16(X_fp32.data(), X_bf16.data(), X_bf16.size());
  FloatToBFloat16(W_fp32.data(), W_bf16.data(), W_bf16.size());

  const BFloat16 kOne_bf16(1.f);
  const BFloat16 kZero_bf16(0.f);

  math::Gemm<BFloat16>(CblasNoTrans, CblasNoTrans, M, N, K, kOne_bf16,
                       VECTOR_HEAD(X_bf16), VECTOR_HEAD(W_bf16), kZero_bf16, VECTOR_HEAD(Y_bf16),
                       tp.get());

  BFloat16ToFloat(Y_bf16.data(), Y_fp32.data(), Y_fp32.size());

  // Reference computation
  math::Gemm<float>(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                    VECTOR_HEAD(X_fp32), VECTOR_HEAD(W_fp32), 0.0f, VECTOR_HEAD(Y_ref),
                    tp.get());

  for (size_t i = 0; i < Y_fp32.size(); ++i) {
    EXPECT_NEAR(Y_fp32[i], Y_ref[i], 0.05);
  }
}

TEST_P(MathBFloat16GemmTest, GemmTransTrans) {
  if (!MlasBf16AccelerationSupported()) {
    GTEST_SKIP() << "BF16 acceleration is not supported on this platform.";
  }
  const int M = 5;
  const int N = 6;
  const int K = 10;

  std::vector<float> X_fp32(K * M);
  std::vector<float> W_fp32(N * K);
  std::vector<float> Y_fp32(M * N);
  std::vector<float> Y_ref(M * N);

  // Initialize with random data
  for (size_t i = 0; i < X_fp32.size(); ++i) X_fp32[i] = (float)rand() / (float)RAND_MAX;
  for (size_t i = 0; i < W_fp32.size(); ++i) W_fp32[i] = (float)rand() / (float)RAND_MAX;

  std::vector<BFloat16> X_bf16(K * M);
  std::vector<BFloat16> W_bf16(N * K);
  std::vector<BFloat16> Y_bf16(M * N);

  FloatToBFloat16(X_fp32.data(), X_bf16.data(), X_bf16.size());
  FloatToBFloat16(W_fp32.data(), W_bf16.data(), W_bf16.size());

  const BFloat16 kOne_bf16(1.f);
  const BFloat16 kZero_bf16(0.f);

  math::Gemm<BFloat16>(CblasTrans, CblasTrans, M, N, K, kOne_bf16,
                       VECTOR_HEAD(X_bf16), VECTOR_HEAD(W_bf16), kZero_bf16, VECTOR_HEAD(Y_bf16),
                       tp.get());

  BFloat16ToFloat(Y_bf16.data(), Y_fp32.data(), Y_fp32.size());

  // Reference computation
  math::Gemm<float>(CblasTrans, CblasTrans, M, N, K, 1.0f,
                    VECTOR_HEAD(X_fp32), VECTOR_HEAD(W_fp32), 0.0f, VECTOR_HEAD(Y_ref),
                    tp.get());

  for (size_t i = 0; i < Y_fp32.size(); ++i) {
    EXPECT_NEAR(Y_fp32[i], Y_ref[i], 0.05);
  }
}

TEST_P(MathBFloat16GemmTest, MatMul) {
  if (!MlasBf16AccelerationSupported()) {
    GTEST_SKIP() << "BF16 acceleration is not supported on this platform.";
  }
  const int M = 5;
  const int N = 6;
  const int K = 10;

  std::vector<float> X_fp32(M * K);
  std::vector<float> W_fp32(K * N);
  std::vector<float> Y_fp32(M * N);
  std::vector<float> Y_ref(M * N);

  // Initialize with random data
  for (size_t i = 0; i < X_fp32.size(); ++i) X_fp32[i] = (float)rand() / (float)RAND_MAX;
  for (size_t i = 0; i < W_fp32.size(); ++i) W_fp32[i] = (float)rand() / (float)RAND_MAX;

  std::vector<BFloat16> X_bf16(M * K);
  std::vector<BFloat16> W_bf16(K * N);
  std::vector<BFloat16> Y_bf16(M * N);

  FloatToBFloat16(X_fp32.data(), X_bf16.data(), X_bf16.size());
  FloatToBFloat16(W_fp32.data(), W_bf16.data(), W_bf16.size());

  math::MatMul<BFloat16>(M, N, K, VECTOR_HEAD(X_bf16), VECTOR_HEAD(W_bf16), VECTOR_HEAD(Y_bf16), tp.get());

  BFloat16ToFloat(Y_bf16.data(), Y_fp32.data(), Y_fp32.size());

  // Reference computation
  math::MatMul<float>(M, N, K, VECTOR_HEAD(X_fp32), VECTOR_HEAD(W_fp32), VECTOR_HEAD(Y_ref), tp.get());

  for (size_t i = 0; i < Y_fp32.size(); ++i) {
    EXPECT_NEAR(Y_fp32[i], Y_ref[i], 0.05);
  }
}

INSTANTIATE_TEST_SUITE_P(MathBFloat16GemmTests, MathBFloat16GemmTest,
                         testing::Values(1, 0));

TEST(MathTest, GemvNoTrans) {
  auto& provider = CPUMathUtil::Instance();
  std::vector<float> A(50);  // 5 * 10
  std::vector<float> X(10);
  std::vector<float> Y(5);
  math::Set<float, CPUMathUtil>(A.size(), 1, VECTOR_HEAD(A), &provider);
  math::Set<float, CPUMathUtil>(X.size(), 1, VECTOR_HEAD(X), &provider);
  EXPECT_EQ(Y.size(), 5u);
  for (size_t i = 0; i < A.size(); ++i) {
    EXPECT_EQ(A[i], 1);
  }
  for (size_t i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X[i], 1);
  }

  constexpr float kOne = 1.0;
  constexpr float kPointFive = 0.5;
  constexpr float kZero = 0.0;
  math::Gemv<float, CPUMathUtil>(CblasNoTrans, 5, 10, kOne, VECTOR_HEAD(A), VECTOR_HEAD(X),
                                 kZero, VECTOR_HEAD(Y), &provider);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 10) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUMathUtil>(CblasNoTrans, 5, 10, kOne, VECTOR_HEAD(A), VECTOR_HEAD(X),
                                 kPointFive, VECTOR_HEAD(Y), &provider);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 15) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUMathUtil>(CblasNoTrans, 5, 10, kPointFive, VECTOR_HEAD(A),
                                 VECTOR_HEAD(X), kOne, VECTOR_HEAD(Y),
                                 &provider);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 20) << i;
  }
}

TEST(MathTest, GemvTrans) {
  auto& provider = CPUMathUtil::Instance();
  std::vector<float> A(60);  // 6 * 10
  std::vector<float> X(6);
  std::vector<float> Y(10);
  math::Set<float, CPUMathUtil>(A.size(), 1, VECTOR_HEAD(A), &provider);
  math::Set<float, CPUMathUtil>(X.size(), 1, VECTOR_HEAD(X), &provider);
  EXPECT_EQ(Y.size(), 10u);
  for (size_t i = 0; i < A.size(); ++i) {
    EXPECT_EQ(A[i], 1);
  }
  for (size_t i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X[i], 1);
  }

  constexpr float kOne = 1.0;
  constexpr float kPointFive = 0.5;
  constexpr float kZero = 0.0;
  math::Gemv<float, CPUMathUtil>(CblasTrans, 6, 10, kOne, VECTOR_HEAD(A), VECTOR_HEAD(X),
                                 kZero, VECTOR_HEAD(Y), &provider);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 6) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUMathUtil>(CblasTrans, 6, 10, kOne, VECTOR_HEAD(A), VECTOR_HEAD(X),
                                 kPointFive, VECTOR_HEAD(Y), &provider);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 9) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUMathUtil>(CblasTrans, 6, 10, kPointFive, VECTOR_HEAD(A),
                                 VECTOR_HEAD(X), kOne, VECTOR_HEAD(Y),
                                 &provider);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 12) << i;
  }
}

TEST(MathTest, HalfFloatConversion) {
  constexpr float original_values[] = {-4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f};
  for (const auto original_value : original_values) {
    const auto half_value = math::floatToHalf(original_value);
    const auto round_trip_value = math::halfToFloat(half_value);
    EXPECT_EQ(round_trip_value, original_value);
  }
}

TEST(MathTest, HalfDoubleConversion) {
  constexpr double original_values[] = {-4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f};
  for (const auto original_value : original_values) {
    const auto half_value = math::doubleToHalf(original_value);
    const auto round_trip_value = static_cast<double>(math::halfToFloat(half_value));
    EXPECT_EQ(round_trip_value, original_value);
  }
}

}  // namespace onnxruntime

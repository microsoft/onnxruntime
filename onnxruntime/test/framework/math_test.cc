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
#include <gtest/gtest.h>
#include "core/util/math_cpuonly.h"
namespace onnxruntime {

#define VECTOR_HEAD(x) x.size() > 0 ? &x[0] : NULL

TEST(MathTest, GemmNoTransNoTrans) {
  auto& provider = CPUMathUtil::Instance();
  std::vector<float> X(50);  // 5 * 10
  std::vector<float> W(60);  // 10 * 6
  std::vector<float> Y(30);  // 5 * 6
  math::Set<float, CPUMathUtil>(X.size(), 1, VECTOR_HEAD(X), &provider);
  math::Set<float, CPUMathUtil>(W.size(), 1, VECTOR_HEAD(W), &provider);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X[i], 1);
  }
  for (int i = 0; i < W.size(); ++i) {
    EXPECT_EQ(W[i], 1);
  }

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasNoTrans, 5, 6, 10, kOne,
                                 VECTOR_HEAD(X), VECTOR_HEAD(W), kZero, VECTOR_HEAD(Y),
                                 &provider);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasNoTrans, 5, 6, 10, kOne,
                                 VECTOR_HEAD(X), VECTOR_HEAD(W), kPointFive,
                                 VECTOR_HEAD(Y), &provider);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 15) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasNoTrans, 5, 6, 10,
                                 kPointFive,
                                 VECTOR_HEAD(X), VECTOR_HEAD(W), kOne, VECTOR_HEAD(Y),
                                 &provider);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 20) << i;
  }
}

TEST(MathTest, GemmNoTransTrans) {
  auto& provider = CPUMathUtil::Instance();
  std::vector<float> X(50);  // 5 * 10
  std::vector<float> W(60);  // 10 * 6
  std::vector<float> Y(30);  // 5 * 6
  math::Set<float, CPUMathUtil>(X.size(), 1, VECTOR_HEAD(X), &provider);
  math::Set<float, CPUMathUtil>(W.size(), 1, VECTOR_HEAD(W), &provider);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X[i], 1);
  }
  for (int i = 0; i < W.size(); ++i) {
    EXPECT_EQ(W[i], 1);
  }

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasTrans, 5, 6, 10, kOne,
                                 VECTOR_HEAD(X), VECTOR_HEAD(W), kZero, VECTOR_HEAD(Y),
                                 &provider);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasTrans, 5, 6, 10, kOne,
                                 VECTOR_HEAD(X), VECTOR_HEAD(W), kPointFive,
                                 VECTOR_HEAD(Y), &provider);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 15) << i;
  }
  math::Gemm<float, CPUMathUtil>(CblasNoTrans, CblasTrans, 5, 6, 10, kPointFive,
                                 VECTOR_HEAD(X), VECTOR_HEAD(W), kOne, VECTOR_HEAD(Y),
                                 &provider);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 20) << i;
  }
}

TEST(MathTest, GemvNoTrans) {
  auto& provider = CPUMathUtil::Instance();
  std::vector<float> A(50);  // 5 * 10
  std::vector<float> X(10);
  std::vector<float> Y(5);
  math::Set<float, CPUMathUtil>(A.size(), 1, VECTOR_HEAD(A), &provider);
  math::Set<float, CPUMathUtil>(X.size(), 1, VECTOR_HEAD(X), &provider);
  EXPECT_EQ(Y.size(), 5);
  for (int i = 0; i < A.size(); ++i) {
    EXPECT_EQ(A[i], 1);
  }
  for (int i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X[i], 1);
  }

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemv<float, CPUMathUtil>(CblasNoTrans, 5, 10, kOne, VECTOR_HEAD(A), VECTOR_HEAD(X),
                                 kZero, VECTOR_HEAD(Y), &provider);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 10) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUMathUtil>(CblasNoTrans, 5, 10, kOne, VECTOR_HEAD(A), VECTOR_HEAD(X),
                                 kPointFive, VECTOR_HEAD(Y), &provider);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 15) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUMathUtil>(CblasNoTrans, 5, 10, kPointFive, VECTOR_HEAD(A),
                                 VECTOR_HEAD(X), kOne, VECTOR_HEAD(Y),
                                 &provider);
  for (int i = 0; i < Y.size(); ++i) {
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
  EXPECT_EQ(Y.size(), 10);
  for (int i = 0; i < A.size(); ++i) {
    EXPECT_EQ(A[i], 1);
  }
  for (int i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X[i], 1);
  }

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemv<float, CPUMathUtil>(CblasTrans, 6, 10, kOne, VECTOR_HEAD(A), VECTOR_HEAD(X),
                                 kZero, VECTOR_HEAD(Y), &provider);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 6) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUMathUtil>(CblasTrans, 6, 10, kOne, VECTOR_HEAD(A), VECTOR_HEAD(X),
                                 kPointFive, VECTOR_HEAD(Y), &provider);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 9) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUMathUtil>(CblasTrans, 6, 10, kPointFive, VECTOR_HEAD(A),
                                 VECTOR_HEAD(X), kOne, VECTOR_HEAD(Y),
                                 &provider);
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 12) << i;
  }
}

TEST(MathTest, WeightsDiagonalTransformation) {
  std::vector<float> X{
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0};
  std::vector<float> Y;
  std::vector<float> expected{
      1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 16.0, 17.0, 18.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0, 20.0, 21.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.0, 23.0, 24.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 26.0, 27.0};

  math::WeightsDiagonalTransformation(X.data(), 1, 3, 3, 3, Y.data());
  EXPECT_EQ(Y.size(), expected.size());
  for (int i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], expected[i]);
  }
}

}  // namespace onnxruntime

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
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <gtest/gtest.h>
#include "core/common/inlined_containers.h"
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"
#include "core/util/thread_utils.h"

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
                    tp.get(), nullptr);
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float>(CblasNoTrans, CblasNoTrans, 5, 6, 10, kOne,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kPointFive,
                    VECTOR_HEAD(Y), tp.get(), nullptr);
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 15) << i;
  }
  // Test Accumulate
  math::Gemm<float>(CblasNoTrans, CblasNoTrans, 5, 6, 10,
                    kPointFive,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kOne, VECTOR_HEAD(Y),
                    tp.get(), nullptr);
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
                    tp.get(), nullptr);
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float>(CblasNoTrans, CblasTrans, 5, 6, 10, kOne,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kPointFive,
                    VECTOR_HEAD(Y), tp.get(), nullptr);
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 15) << i;
  }
  math::Gemm<float>(CblasNoTrans, CblasTrans, 5, 6, 10, kPointFive,
                    VECTOR_HEAD(X), VECTOR_HEAD(W), kOne, VECTOR_HEAD(Y),
                    tp.get(), nullptr);
  EXPECT_EQ(Y.size(), 30u);
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y[i], 20) << i;
  }
}

INSTANTIATE_TEST_SUITE_P(MathGemmTests, MathGemmTest,
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

TEST(MathTest, Col2im2dLayouts) {
  struct TestCase {
    int64_t height;
    int64_t width;
    std::array<int64_t, 2> kernel;
    std::array<int64_t, 2> stride;
    std::array<int64_t, 2> dilation;
    std::array<int64_t, 4> pads;
  };
  const TestCase cases[] = {
      {2, 2, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {6, 10, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {34, 130, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {6, 128, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {6, 134, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      // An extra output row or column must still be set to zero.
      {7, 10, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {6, 11, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {7, 11, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {6, 10, {2, 2}, {2, 2}, {1, 1}, {1, 0, 0, 0}},
      {6, 10, {2, 2}, {2, 2}, {1, 1}, {0, 1, 0, 0}},
      {6, 10, {2, 2}, {2, 2}, {1, 1}, {0, 0, 1, 0}},
      {6, 10, {2, 2}, {2, 2}, {1, 1}, {0, 0, 0, 1}},
      {6, 10, {2, 2}, {2, 2}, {2, 1}, {0, 0, 0, 0}},
      {6, 10, {2, 2}, {2, 2}, {1, 2}, {0, 0, 0, 0}},
      {6, 10, {2, 2}, {1, 2}, {1, 1}, {0, 0, 0, 0}},
      {6, 10, {2, 2}, {2, 1}, {1, 1}, {0, 0, 0, 0}},
      {6, 10, {3, 2}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {6, 10, {2, 3}, {2, 2}, {1, 1}, {0, 0, 0, 0}},
      {6, 10, {3, 3}, {1, 1}, {1, 1}, {1, 1, 1, 1}},
      {6, 10, {2, 2}, {1, 1}, {2, 2}, {0, 0, 0, 0}},
      {4, 4, {4, 4}, {2, 2}, {1, 1}, {1, 1, 1, 1}},
      {2, 2, {3, 3}, {3, 3}, {1, 1}, {2, 2, 2, 2}},
  };
  auto& provider = CPUMathUtil::Instance();
  for (const auto& c : cases) {
    SCOPED_TRACE(testing::Message() << c.height << "x" << c.width
                                    << " kernel=" << testing::PrintToString(c.kernel)
                                    << " stride=" << testing::PrintToString(c.stride)
                                    << " dilation=" << testing::PrintToString(c.dilation)
                                    << " pads=" << testing::PrintToString(c.pads));
    const int64_t channels = 3;
    const int64_t image_shape[] = {c.height, c.width};
    const int64_t column_channels = channels * c.kernel[0] * c.kernel[1];
    const int64_t column_shape[] = {
        (c.height + c.pads[0] + c.pads[2] - (c.dilation[0] * (c.kernel[0] - 1) + 1)) / c.stride[0] + 1,
        (c.width + c.pads[1] + c.pads[3] - (c.dilation[1] * (c.kernel[1] - 1) + 1)) / c.stride[1] + 1};
    const size_t column_size = static_cast<size_t>(column_channels * column_shape[0] * column_shape[1]);
    const int64_t image_size = channels * c.height * c.width;
    InlinedVector<float> column;
    column.reserve(column_size);
    for (size_t i = 0; i < column_size; ++i) {
      column.push_back(static_cast<float>(static_cast<int>(i % 257) - 128) * 0.25f);
    }
    InlinedVector<float> expected(static_cast<size_t>(image_size));
    // Leave a guard before and after an output that is not aligned to 16 bytes.
    InlinedVector<float> actual(static_cast<size_t>(image_size) + 2, -12345.0f);
    math::Col2imNd<float, CPUMathUtil, StorageOrder::NCHW>(
        column.data(), image_shape, column_shape, column_channels, image_size,
        c.kernel.data(), c.stride.data(), c.dilation.data(), c.pads.data(), 2,
        expected.data(), &provider);
    math::Col2im<float, CPUMathUtil, StorageOrder::NCHW>(
        column.data(), channels, c.height, c.width, c.kernel[0], c.kernel[1],
        c.dilation[0], c.dilation[1], c.pads[0], c.pads[1], c.pads[2], c.pads[3],
        c.stride[0], c.stride[1], actual.data() + 1, &provider);
    EXPECT_EQ(0, std::memcmp(expected.data(), actual.data() + 1, expected.size() * sizeof(float)));
    EXPECT_EQ(actual.front(), -12345.0f);
    EXPECT_EQ(actual.back(), -12345.0f);
  }
}

TEST(MathTest, Col2im2x2SpecialValues) {
  const float column[] = {-0.0f, 0.0f, std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::denorm_min(),
                          -std::numeric_limits<float>::denorm_min(), 1.0f};
  float actual[8];
  auto& provider = CPUMathUtil::Instance();
  math::Col2im<float, CPUMathUtil, StorageOrder::NCHW>(
      column, 1, 2, 4, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2, actual, &provider);
  const int source_indices[] = {0, 2, 1, 3, 4, 6, 5, 7};
  for (size_t i = 0; i < std::size(actual); ++i) {
    const float expected = 0.0f + column[source_indices[i]];
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual[i]));
    } else {
      EXPECT_EQ(0, std::memcmp(&expected, &actual[i], sizeof(float))) << i;
    }
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

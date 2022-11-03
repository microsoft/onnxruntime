// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <numeric>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/common/gsl.h"
using namespace std;
namespace onnxruntime {
namespace test {

class ArrayFeatureExtractorTest : public ::testing::Test {
 protected:
  OpTester test_{"ArrayFeatureExtractor", 1, onnxruntime::kMLDomain};
};

TEST_F(ArrayFeatureExtractorTest, Basic) {
  constexpr int N = 3;
  const std::vector<float> X = {0.8f, -1.5f, 2.0f, 3.8f, -4.0f, 5.0f,
                                6.8f, -7.5f, 8.0f, 9.8f, -9.0f, 4.0f,
                                4.8f, -4.5f, 4.0f, 4.8f, -4.0f, 4.0f};
  constexpr int kCols = 6;
  const vector<int64_t> x_dims = {N, kCols};
  test_.AddInput<float>("X", x_dims, X);

  const std::vector<int64_t> Y = {1L, 2L, 4L};
  const vector<int64_t> y_dims = {1, 3};
  test_.AddInput<int64_t>("Y", y_dims, Y);

  // prepare expected output
  vector<float> expected_output;
  for (int i = 0; i < N; ++i) {
    auto offset = i * kCols;
    for (size_t j = 0; j < Y.size(); ++j) {
      expected_output.push_back(X[offset + Y[j]]);
    }
  }
  const vector<int64_t> expected_dims{N, gsl::narrow_cast<int64_t>(Y.size())};
  test_.AddOutput<float>("Z", expected_dims, expected_output);

  test_.Run();
}

TEST_F(ArrayFeatureExtractorTest, HigherDimensionalX) {
  const std::vector<int64_t> x_dims{2, 3, 4, 5};
  const int64_t x_size = std::accumulate(
      x_dims.begin(), x_dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  const std::vector<int32_t> X = [x_size]() {
    std::vector<int32_t> v(x_size);
    std::iota(v.begin(), v.end(), 0);
    return v;
  }();
  test_.AddInput("X", x_dims, X);

  const std::vector<int64_t> Y{0, 1, 1, 2};
  const int64_t y_size = gsl::narrow_cast<int64_t>(Y.size());
  const std::vector<int64_t> y_dims{1, y_size};
  test_.AddInput("Y", y_dims, Y);

  // prepare expected output
  const std::vector<int64_t> z_dims = [&x_dims, y_size]() {
    std::vector<int64_t> v{x_dims};
    v[v.size() - 1] = y_size;
    return v;
  }();
  const std::vector<int32_t> Z = [&x_dims, x_size, y_size, &X, &Y]() {
    const int64_t x_last_dim_size = x_dims.back();  // stride
    const int64_t x_leading_dims_size = x_size / x_last_dim_size;
    std::vector<int32_t> v(x_leading_dims_size * y_size);
    int32_t* v_output = v.data();
    for (int64_t x_idx = 0; x_idx < x_size; x_idx += x_last_dim_size) {
      for (int64_t y_idx = 0; y_idx < y_size; ++y_idx) {
        *(v_output++) = X[x_idx + Y[y_idx]];
      }
    }
    return v;
  }();
  test_.AddOutput<int32_t>("Z", z_dims, Z);

  test_.Run();
}

TEST_F(ArrayFeatureExtractorTest, OneDimensionalX) {
  test_.AddInput<int32_t>("X", {1}, {42});
  test_.AddInput<int64_t>("Y", {1, 3}, {0, 0, 0});
  test_.AddOutput("Z", {1, 3}, {42, 42, 42});
  test_.Run();
}

TEST_F(ArrayFeatureExtractorTest, InvalidInputEmptyX) {
  test_.AddInput<int32_t>("X", {0}, {});
  test_.AddInput<int64_t>("Y", {1}, {1});
  test_.AddOutput<int32_t>("Z", {0}, {});
  test_.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST_F(ArrayFeatureExtractorTest, InvalidInputEmptyY) {
  test_.AddInput<int32_t>("X", {1}, {1});
  test_.AddInput<int64_t>("Y", {0}, {});
  test_.AddOutput<int32_t>("Z", {0}, {});
  test_.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST_F(ArrayFeatureExtractorTest, InvalidInputOutOfBoundsY) {
  test_.AddInput<int32_t>("X", {2, 2}, {1, 2, 3, 4});
  test_.AddInput<int64_t>("Y", {1}, {10});
  test_.AddOutput<int32_t>("Z", {0}, {});
  test_.Run(OpTester::ExpectResult::kExpectFailure, "index is out of range");
}

}  // namespace test
}  // namespace onnxruntime

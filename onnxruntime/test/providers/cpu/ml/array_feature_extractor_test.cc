// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "gsl/gsl"
using namespace std;
namespace onnxruntime {
namespace test {

TEST(MLOpTest, ArrayFeatureExtractorTest) {
  OpTester test("ArrayFeatureExtractor", 1, onnxruntime::kMLDomain);
  const int N = 3;
  const std::vector<float> X = {0.8f, -1.5f, 2.0f, 3.8f, -4.0f, 5.0f,
                                6.8f, -7.5f, 8.0f, 9.8f, -9.0f, 4.0f,
                                4.8f, -4.5f, 4.0f, 4.8f, -4.0f, 4.0f};
  const int kCols = 6;
  const vector<int64_t> x_dims = {N, kCols};
  test.AddInput<float>("X", x_dims, X);

  const std::vector<int64_t> Y = {1L, 2L, 4L};
  const vector<int64_t> y_dims = {1, 3};
  test.AddInput<int64_t>("Y", y_dims, Y);

  // prepare expected output
  vector<float> expected_output;
  for (int i = 0; i < N; ++i) {
    auto offset = i * kCols;
    for (size_t j = 0; j < Y.size(); ++j) {
      expected_output.push_back(X[offset + Y[j]]);
    }
  }
  const vector<int64_t> expected_dims{N, gsl::narrow_cast<int64_t>(Y.size())};
  test.AddOutput<float>("Z", expected_dims, expected_output);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

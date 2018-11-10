// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

template <typename T>
void TestScalar() {
  OpTester test("Scaler", 1, onnxruntime::kMLDomain);
  vector<float> scale{3.f, -4.f, 3.0f};
  vector<float> offset{4.8f, -0.5f, 77.0f};
  test.AddAttribute("scale", scale);
  test.AddAttribute("offset", offset);
  vector<T> input{1, -2, 3, 4, 5, -6};
  vector<int64_t> dims{2, 3};

  // prepare expected output
  vector<float> expected_output;
  for (size_t i = 0; i < input.size(); ++i) {
    expected_output.push_back((static_cast<float>(input[i]) - offset[i % dims[1]]) * scale[i % dims[1]]);
  }

  test.AddInput<T>("X", dims, input);
  test.AddOutput<float>("Y", dims, expected_output);
  test.Run();
}

TEST(MLOpTest, ScalerOp) {
  TestScalar<float>();
  TestScalar<double>();
  TestScalar<int64_t>();
  TestScalar<int32_t>();
}

TEST(MLOpTest, ScalerOpScaleOffsetSize1) {
  OpTester test("Scaler", 1, onnxruntime::kMLDomain);
  vector<float> scale{3.f};
  vector<float> offset{4.8f};
  test.AddAttribute("scale", scale);
  test.AddAttribute("offset", offset);
  vector<float> input{0.8f, -0.5f, 0.0f, 0.8f, 1.0f, 1.0f};
  vector<int64_t> dims{2, 3};

  // prepare expected output
  vector<float> expected_output;
  for (size_t i = 0; i < input.size(); ++i) {
    expected_output.push_back((input[i] - offset[0]) * scale[0]);
  }

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, expected_output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(FeatureVectorizer, BasicFunctionality) {
  OpTester test("FeatureVectorizer", 1, onnxruntime::kMLDomain);

  test.AddAttribute("inputdimensions", std::vector<int64_t>{3, 2, 1, 4});

  std::vector<int64_t> input0_dims = {1, 3};
  test.AddInput<int32_t>("X0", input0_dims, {1, 2, 3});

  std::vector<int64_t> input1_dims = {1, 2};
  test.AddInput<int32_t>("X1", input1_dims, {4, 5});

  std::vector<int64_t> input2_dims = {1};
  test.AddInput<int32_t>("X2", input2_dims, {6});

  std::vector<int64_t> input3_dims = {1, 4};
  test.AddInput<int32_t>("X3", input3_dims, {7, 8, 9, 10});

  test.AddOutput<float>("Y", std::vector<int64_t>{1, 10},
                        {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});
  test.Run();
}

TEST(FeatureVectorizer, HandleInputDimensionMismatch) {
  OpTester test("FeatureVectorizer", 1, onnxruntime::kMLDomain);

  test.AddAttribute("inputdimensions", std::vector<int64_t>{2, 3});

  std::vector<int64_t> input0_dims = {1, 3};  // long - ignore extra
  test.AddInput<int32_t>("X0", input0_dims, {1, 2, 3});

  std::vector<int64_t> input1_dims = {1, 2};  // short - pad with 0.f
  test.AddInput<int32_t>("X1", input1_dims, {1, 2});

  test.AddOutput<float>("Y", std::vector<int64_t>{1, 5}, {1.f, 2.f, 1.f, 2.f, 0.f});

  test.Run();
}

// test with batch size of 2.
TEST(FeatureVectorizer, Batch) {
  OpTester test("FeatureVectorizer", 1, onnxruntime::kMLDomain);

  test.AddAttribute("inputdimensions", std::vector<int64_t>{2, 2});

  std::vector<int64_t> input0_dims = {2, 2};
  test.AddInput<double>("X0", input0_dims, {1., 2., 3., 4.});

  std::vector<int64_t> input1_dims = {2, 2};
  test.AddInput<double>("X1", input1_dims, {10., 11., 12., 13.});

  test.AddOutput<float>("Y", std::vector<int64_t>{2, 4},
                        {1.f, 2.f, 10.f, 11.f,
                         3.f, 4.f, 12.f, 13.f});

  test.Run();
}

// test with batch size of 2.
TEST(FeatureVectorizer, BatchWith3DInput) {
  OpTester test("FeatureVectorizer", 1, onnxruntime::kMLDomain);

  test.AddAttribute("inputdimensions", std::vector<int64_t>{2, 4});

  std::vector<int64_t> input0_dims = {2, 2};
  test.AddInput<double>("X0", input0_dims, {1., 2., 3., 4.});

  std::vector<int64_t> input1_dims = {2, 2, 2};
  test.AddInput<double>("X1", input1_dims,
                        {10., 11., 12., 13.,
                         14., 15., 16., 17.});

  test.AddOutput<float>("Y", std::vector<int64_t>{2, 6},
                        {1.f, 2.f, 10.f, 11.f, 12.f, 13.f,
                         3.f, 4.f, 14.f, 15.f, 16.f, 17.f});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

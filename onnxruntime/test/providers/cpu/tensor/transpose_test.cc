// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Some of the tests can't run on TensorrtExecutionProvider because of errors.
// Those tests will fallback to other EPs.

template <class T>
void TransposeTest(std::vector<int64_t>& input_shape,
                   std::vector<T>& input_vals,
                   std::vector<int64_t>* p_perm,
                   std::vector<int64_t> expected_shape,
                   std::initializer_list<T>& expected_vals,
                   bool is_tensorrt_supported = true) {
  OpTester test("Transpose");
  if (nullptr != p_perm)
    test.AddAttribute("perm", *p_perm);
  test.AddInput<T>("X", input_shape, input_vals);
  test.AddOutput<T>("Y", expected_shape, expected_vals);
  // Disable TensorRT on unsupported tests
  std::unordered_set<std::string> excluded_providers;
  if (!is_tensorrt_supported) {
    excluded_providers.insert(kTensorrtExecutionProvider);
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

// Test 2 dimensional transpose, with no permutation attribute specified
TEST(TransposeOpTest, TwoDimNoAttr) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<float> input_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f};

  std::vector<int64_t> expected_shape({3, 2});
  auto expected_vals = {
      1.0f, 4.0f,
      2.0f, 5.0f,
      3.0f, 6.0f};

  TransposeTest(input_shape, input_vals, nullptr, expected_shape, expected_vals, false);  //TensorRT: SegFault error
}

TEST(TransposeOpTest, TwoDimNoAttrStr) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<std::string> input_vals = {
      "1", "2", "3",
      "4", "5", "6"};

  std::vector<int64_t> expected_shape({3, 2});
  std::initializer_list<std::string> expected_vals = {
      "1", "4",
      "2", "5",
      "3", "6"};

  TransposeTest(input_shape, input_vals, nullptr, expected_shape, expected_vals);
}

// Test 2 dimensional transpose, with permutation attribute specified
TEST(TransposeOpTest, TwoDim) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<float> input_vals = {1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  auto expected_vals = {1.0f, 4.0f,
                        2.0f, 5.0f,
                        3.0f, 6.0f};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, TwoDim_double) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<double> input_vals = {1.0, 2.0, 3.0,
                                    4.0, 5.0, 6.0};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::initializer_list<double> expected_vals = {1.0, 4.0,
                                                 2.0, 5.0,
                                                 3.0, 6.0};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, TwoDim_int32) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<int32_t> input_vals = {1, 2, 3,
                                     4, 5, 6};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::initializer_list<int32_t> expected_vals = {1, 4,
                                                  2, 5,
                                                  3, 6};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, TwoDim_int16) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<int16_t> input_vals = {
      1, 2, 3,
      4, 5, 6};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::initializer_list<int16_t> expected_vals = {
      1, 4,
      2, 5,
      3, 6};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, TwoDim_mlfloat16) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<MLFloat16> input_vals;
  for (uint16_t i = 0; i < 6; ++i)
    input_vals.push_back(MLFloat16(i));

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::initializer_list<MLFloat16> expected_vals = {MLFloat16(1), MLFloat16(4),
                                                    MLFloat16(2), MLFloat16(5),
                                                    MLFloat16(3), MLFloat16(6)};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, TwoDim_int8) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<int8_t> input_vals = {1, 2, 3,
                                    4, 5, 6};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::initializer_list<int8_t> expected_vals = {1, 4,
                                                 2, 5,
                                                 3, 6};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, TwoDimStr) {
  std::vector<int64_t> input_shape({2, 3});
  std::vector<std::string> input_vals = {
      "1", "2", "3",
      "4", "5", "6"};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  std::initializer_list<std::string> expected_vals = {
      "1", "4",
      "2", "5",
      "3", "6"};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

// Test 3 dimensional transpose, with permutation attribute specified
TEST(TransposeOpTest, ThreeDim) {
  std::vector<int64_t> input_shape({4, 2, 3});
  std::vector<float> input_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,

      1.1f, 2.1f, 3.1f,
      4.1f, 5.1f, 6.1f,

      1.2f, 2.2f, 3.2f,
      4.2f, 5.2f, 6.2f,

      1.3f, 2.3f, 3.3f,
      4.3f, 5.3f, 6.3f};

  std::vector<int64_t> perm = {0, 2, 1};
  std::vector<int64_t> expected_shape({4, 3, 2});
  auto expected_vals = {
      1.0f, 4.0f,
      2.0f, 5.0f,
      3.0f, 6.0f,

      1.1f, 4.1f,
      2.1f, 5.1f,
      3.1f, 6.1f,

      1.2f, 4.2f,
      2.2f, 5.2f,
      3.2f, 6.2f,

      1.3f, 4.3f,
      2.3f, 5.3f,
      3.3f, 6.3f};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, false);  //TensorRT: illegal error
}

TEST(TransposeOpTest, ThreeDimStr) {
  std::vector<int64_t> input_shape({4, 2, 3});
  std::vector<std::string> input_vals = {
      "1", "2", "3",
      "4", "5", "6",

      "1", "2", "3",
      "4", "5", "6",

      "1", "2", "3",
      "4", "5", "6",

      "1", "2", "3",
      "4", "5", "6"};

  std::vector<int64_t> perm = {0, 2, 1};
  std::vector<int64_t> expected_shape({4, 3, 2});
  std::initializer_list<std::string> expected_vals = {
      "1", "4",
      "2", "5",
      "3", "6",

      "1", "4",
      "2", "5",
      "3", "6",

      "1", "4",
      "2", "5",
      "3", "6",

      "1", "4",
      "2", "5",
      "3", "6"};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

TEST(TransposeOpTest, NCHW2NHWC) {
  std::vector<int64_t> input_shape({1, 3, 2, 2});
  std::vector<std::string> input_vals = {
      "1", "2", "3", "4",
      "5", "6", "7", "8",
      "9", "10", "11", "12"};

  std::vector<int64_t> perm = {0, 2, 3, 1};
  std::vector<int64_t> expected_shape({1, 2, 2, 3});
  std::initializer_list<std::string> expected_vals = {
      "1", "5", "9",
      "2", "6", "10",
      "3", "7", "11",
      "4", "8", "12"};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, false);
}

TEST(TransposeOpTest, NHWC2NCHW) {
  std::vector<int64_t> input_shape({1, 2, 2, 3});
  std::vector<std::string> input_vals = {
      "1", "2", "3",
      "4", "5", "6",
      "7", "8", "9",
      "10", "11", "12"};

  std::vector<int64_t> perm = {0, 3, 1, 2};
  std::vector<int64_t> expected_shape({1, 3, 2, 2});
  std::initializer_list<std::string> expected_vals = {
      "1", "4", "7", "10",
      "2", "5", "8", "11",
      "3", "6", "9", "12"};

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals, false);
}

}  // namespace test
}  // namespace onnxruntime

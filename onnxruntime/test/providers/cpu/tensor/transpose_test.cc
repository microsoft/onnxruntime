// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <class T>
void TransposeTest(std::vector<int64_t>& input_shape,
                   std::vector<T>& input_vals,
                   std::vector<int64_t>* p_perm,
                   std::vector<int64_t> expected_shape,
                   std::initializer_list<T>& expected_vals) {
  OpTester test("Transpose");
  if (nullptr != p_perm)
    test.AddAttribute("perm", *p_perm);
  test.AddInput<T>("X", input_shape, input_vals);
  test.AddOutput<T>("Y", expected_shape, expected_vals);
  test.Run();
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

  TransposeTest(input_shape, input_vals, nullptr, expected_shape, expected_vals);
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
  std::vector<float> input_vals = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f};

  std::vector<int64_t> perm = {1, 0};
  std::vector<int64_t> expected_shape({3, 2});
  auto expected_vals = {
      1.0f, 4.0f,
      2.0f, 5.0f,
      3.0f, 6.0f};

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
      1.0f,
      4.0f,
      2.0f,
      5.0f,
      3.0f,
      6.0f,

      1.1f,
      4.1f,
      2.1f,
      5.1f,
      3.1f,
      6.1f,

      1.2f,
      4.2f,
      2.2f,
      5.2f,
      3.2f,
      6.2f,

      1.3f,
      4.3f,
      2.3f,
      5.3f,
      3.3f,
      6.3f,

  };

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
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
      "1",
      "4",
      "2",
      "5",
      "3",
      "6",

      "1",
      "4",
      "2",
      "5",
      "3",
      "6",

      "1",
      "4",
      "2",
      "5",
      "3",
      "6",

      "1",
      "4",
      "2",
      "5",
      "3",
      "6"

  };

  TransposeTest(input_shape, input_vals, &perm, expected_shape, expected_vals);
}

}  // namespace test
}  // namespace onnxruntime

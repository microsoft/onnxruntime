// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on the tests because axis=0 is not supported

template <typename T>
void RunUniqueTest(const std::vector<int64_t>& X_dims,
                   const std::vector<T>& X,
                   const int64_t* axis,
                   bool sorted,
                   const std::vector<int64_t>& Y_dims,
                   const std::vector<T>& Y,
                   const std::vector<int64_t>& indices_dims,
                   const std::vector<int64_t>& indices,
                   const std::vector<int64_t>& inverse_indices_dims,
                   const std::vector<int64_t>& inverse_indices,
                   const std::vector<int64_t>& counts_dims,
                   const std::vector<int64_t>& counts) {
  OpTester test("Unique", 11);

  if (axis) {
    test.AddAttribute("axis", *axis);
  }

  test.AddAttribute("sorted", static_cast<int64_t>(sorted));

  test.AddInput<T>("X", X_dims, X);
  test.AddOutput<T>("Y", Y_dims, Y);
  test.AddOutput<int64_t>("indices", indices_dims, indices);
  test.AddOutput<int64_t>("inverse_indices", inverse_indices_dims, inverse_indices);
  test.AddOutput<int64_t>("counts", counts_dims, counts);

  test.Run();
}

TEST(Unique, Flatten_Nan) {
  const std::vector<int64_t> X_dims{2, 3};
  const std::vector<float> X{1.f, std::numeric_limits<float>::quiet_NaN(), 1.f, 2.f, 2.f, std::numeric_limits<float>::quiet_NaN()};
  const int64_t* axis = nullptr;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{4};
  const std::vector<float> Y{1.f, std::numeric_limits<float>::quiet_NaN(), 2.f, std::numeric_limits<float>::quiet_NaN()};

  const std::vector<int64_t> indices_dims{4};
  const std::vector<int64_t> indices{0, 1, 3, 5};
  const std::vector<int64_t> inverse_indices_dims{6};
  const std::vector<int64_t> inverse_indices{0, 1, 0, 2, 2, 3};
  const std::vector<int64_t> counts_dims{4};
  const std::vector<int64_t> counts{2, 1, 2, 1};

  RunUniqueTest<float>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                       inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Flatten_Unsorted) {
  const std::vector<int64_t> X_dims{2, 3};
  const std::vector<float> X{1.f, 4.f, 1.f, 2.f, 2.f, 0.f};
  const int64_t* axis = nullptr;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{4};
  const std::vector<float> Y{1.f, 4.f, 2.f, 0.f};

  const std::vector<int64_t> indices_dims{4};
  const std::vector<int64_t> indices{0, 1, 3, 5};
  const std::vector<int64_t> inverse_indices_dims{6};
  const std::vector<int64_t> inverse_indices{0, 1, 0, 2, 2, 3};
  const std::vector<int64_t> counts_dims{4};
  const std::vector<int64_t> counts{2, 1, 2, 1};

  RunUniqueTest<float>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                       inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Flatten_Unsorted_Double) {
  const std::vector<int64_t> X_dims{2, 2};
  const std::vector<double> X{3.14, -1.3, 3.14, -1.3};
  const int64_t* axis = nullptr;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{2};
  const std::vector<double> Y{3.14, -1.3};

  const std::vector<int64_t> indices_dims{2};
  const std::vector<int64_t> indices{0, 1};
  const std::vector<int64_t> inverse_indices_dims{4};
  const std::vector<int64_t> inverse_indices{0, 1, 0, 1};
  const std::vector<int64_t> counts_dims{2};
  const std::vector<int64_t> counts{2, 2};

  RunUniqueTest<double>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                        inverse_indices_dims, inverse_indices, counts_dims, counts);
}

// TEMPORARY. The ONNX test expected data for Y for unique_not_sorted_without_axis doesn't match the comments in that
// test and is in sorted order. This unit test validates we have the correct behavior, pending fixing the onnx test
// data.
TEST(Unique, Flatten_Unsorted_MatchOnnxTest) {
  const std::vector<int64_t> X_dims{6};
  const std::vector<float> X{2.f, 1.f, 1.f, 3.f, 4.f, 3.f};
  const int64_t* axis = nullptr;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{4};
  const std::vector<float> Y{2.f, 1.f, 3.f, 4.f};

  const std::vector<int64_t> indices_dims{4};
  const std::vector<int64_t> indices{0, 1, 3, 4};
  const std::vector<int64_t> inverse_indices_dims{6};
  const std::vector<int64_t> inverse_indices{0, 1, 1, 2, 3, 2};
  const std::vector<int64_t> counts_dims{4};
  const std::vector<int64_t> counts{1, 2, 2, 1};

  RunUniqueTest<float>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                       inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Flatten_Sorted) {
  const std::vector<int64_t> X_dims{2, 3};
  const std::vector<float> X{1.f, 4.f, 1.f, 2.f, 2.f, 0.f};
  const int64_t* axis = nullptr;
  bool sorted = true;
  const std::vector<int64_t> Y_dims{4};
  const std::vector<float> Y{0.f, 1.f, 2.f, 4.f};

  const std::vector<int64_t> indices_dims{4};
  const std::vector<int64_t> indices{5, 0, 3, 1};
  const std::vector<int64_t> inverse_indices_dims{6};
  const std::vector<int64_t> inverse_indices{1, 3, 1, 2, 2, 0};
  const std::vector<int64_t> counts_dims{4};
  const std::vector<int64_t> counts{1, 2, 2, 1};

  RunUniqueTest<float>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                       inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Flatten_Sorted_String) {
  const std::vector<int64_t> X_dims{2, 3};
  const std::vector<std::string> X{"1.f", "4.f", "1.f", "2.f", "2.f", "0.f"};
  const int64_t* axis = nullptr;
  bool sorted = true;
  const std::vector<int64_t> Y_dims{4};
  const std::vector<std::string> Y{"0.f", "1.f", "2.f", "4.f"};

  const std::vector<int64_t> indices_dims{4};
  const std::vector<int64_t> indices{5, 0, 3, 1};
  const std::vector<int64_t> inverse_indices_dims{6};
  const std::vector<int64_t> inverse_indices{1, 3, 1, 2, 2, 0};
  const std::vector<int64_t> counts_dims{4};
  const std::vector<int64_t> counts{1, 2, 2, 1};

  RunUniqueTest<std::string>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                             inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, NoOptionalOutput) {
  const std::vector<int64_t> X_dims{2, 4};
  const std::vector<int8_t> X{1, 4, -1, 2, 2, 0, -1, 4};
  bool sorted = true;
  const std::vector<int64_t> Y_dims{5};
  const std::vector<int8_t> Y{-1, 0, 1, 2, 4};

  OpTester test("Unique", 11);

  test.AddAttribute("sorted", static_cast<int64_t>(sorted));

  test.AddInput("X", X_dims, X);
  test.AddOutput("Y", Y_dims, Y);

  test.Run();
}

TEST(Unique, Axis0_Unsorted) {
  const std::vector<int64_t> X_dims{4, 2};
  const std::vector<float> X{0.f, 1.f,
                             1.f, 1.f,
                             0.f, 1.f,
                             1.f, 0.f};

  constexpr int64_t axis = 0;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{3, 2};
  const std::vector<float> Y{0.f, 1.f,
                             1.f, 1.f,
                             1.f, 0.f};

  const std::vector<int64_t> indices_dims{3};
  const std::vector<int64_t> indices{0, 1, 3};
  const std::vector<int64_t> inverse_indices_dims{4};
  const std::vector<int64_t> inverse_indices{0, 1, 0, 2};
  const std::vector<int64_t> counts_dims{3};
  const std::vector<int64_t> counts{2, 1, 1};

  RunUniqueTest<float>(X_dims, X, &axis, sorted, Y_dims, Y, indices_dims, indices,
                       inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Axis0_Sorted) {
  const std::vector<int64_t> X_dims{4, 2};
  const std::vector<float> X{0.f, 1.f,
                             1.f, 1.f,
                             0.f, 1.f,
                             1.f, 0.f};

  constexpr int64_t axis = 0;
  bool sorted = true;
  const std::vector<int64_t> Y_dims{3, 2};
  const std::vector<float> Y{0.f, 1.f,
                             1.f, 0.f,
                             1.f, 1.f};

  const std::vector<int64_t> indices_dims{3};
  const std::vector<int64_t> indices{0, 3, 1};
  const std::vector<int64_t> inverse_indices_dims{4};
  const std::vector<int64_t> inverse_indices{0, 2, 0, 1};
  const std::vector<int64_t> counts_dims{3};
  const std::vector<int64_t> counts{2, 1, 1};

  RunUniqueTest<float>(X_dims, X, &axis, sorted, Y_dims, Y, indices_dims, indices,
                       inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Axis0_Unsorted_String) {
  const std::vector<int64_t> X_dims{4, 2};
  const std::vector<std::string> X{"0.f", "1.f",
                                   "1.f", "1.f",
                                   "0.f", "1.f",
                                   "1.f", "0.f"};

  constexpr int64_t axis = 0;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{3, 2};
  const std::vector<std::string> Y{"0.f", "1.f",
                                   "1.f", "1.f",
                                   "1.f", "0.f"};

  const std::vector<int64_t> indices_dims{3};
  const std::vector<int64_t> indices{0, 1, 3};
  const std::vector<int64_t> inverse_indices_dims{4};
  const std::vector<int64_t> inverse_indices{0, 1, 0, 2};
  const std::vector<int64_t> counts_dims{3};
  const std::vector<int64_t> counts{2, 1, 1};

  RunUniqueTest<std::string>(X_dims, X, &axis, sorted, Y_dims, Y, indices_dims, indices,
                             inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Axis1_Unsorted) {
  const std::vector<int64_t> X_dims{2, 4, 2};
  const std::vector<int8_t> X{1, 1,
                              0, 1,
                              2, 1,
                              0, 1,

                              1, 1,
                              0, 1,
                              2, 1,
                              0, 1};

  constexpr int64_t axis = 1;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{2, 3, 2};
  const std::vector<int8_t> Y{1, 1,
                              0, 1,
                              2, 1,

                              1, 1,
                              0, 1,
                              2, 1};

  const std::vector<int64_t> indices_dims{3};
  const std::vector<int64_t> indices{0, 1, 2};
  const std::vector<int64_t> inverse_indices_dims{4};
  const std::vector<int64_t> inverse_indices{0, 1, 2, 1};
  const std::vector<int64_t> counts_dims{3};
  const std::vector<int64_t> counts{1, 2, 1};

  RunUniqueTest<int8_t>(X_dims, X, &axis, sorted, Y_dims, Y, indices_dims, indices,
                        inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Axis1_Sorted) {
  const std::vector<int64_t> X_dims{2, 4, 2};
  const std::vector<int64_t> X{1, 1,
                               0, 1,
                               2, 1,
                               0, 1,

                               1, 1,
                               0, 1,
                               2, 1,
                               0, 1};

  constexpr int64_t axis = 1;
  bool sorted = true;
  const std::vector<int64_t> Y_dims{2, 3, 2};
  const std::vector<int64_t> Y{0, 1,
                               1, 1,
                               2, 1,

                               0, 1,
                               1, 1,
                               2, 1};

  const std::vector<int64_t> indices_dims{3};
  const std::vector<int64_t> indices{1, 0, 2};
  const std::vector<int64_t> inverse_indices_dims{4};
  const std::vector<int64_t> inverse_indices{1, 0, 2, 0};
  const std::vector<int64_t> counts_dims{3};
  const std::vector<int64_t> counts{2, 1, 1};

  RunUniqueTest<int64_t>(X_dims, X, &axis, sorted, Y_dims, Y, indices_dims, indices,
                         inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Axis2_Unsorted) {
  const std::vector<int64_t> X_dims{2, 2, 4};
  const std::vector<int64_t> X{1, 1, 0, 1,
                               2, 1, 0, 1,

                               1, 1, 0, 1,
                               2, 1, 0, 1};

  constexpr int64_t axis = 2;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{2, 2, 3};
  const std::vector<int64_t> Y{1, 1, 0,
                               2, 1, 0,

                               1, 1, 0,
                               2, 1, 0};

  const std::vector<int64_t> indices_dims{3};
  const std::vector<int64_t> indices{0, 1, 2};
  const std::vector<int64_t> inverse_indices_dims{4};
  const std::vector<int64_t> inverse_indices{0, 1, 2, 1};
  const std::vector<int64_t> counts_dims{3};
  const std::vector<int64_t> counts{1, 2, 1};

  RunUniqueTest<int64_t>(X_dims, X, &axis, sorted, Y_dims, Y, indices_dims, indices,
                         inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, Axis2_Sorted) {
  const std::vector<int64_t> X_dims{2, 2, 4};
  const std::vector<int64_t> X{1, 1, 0, 1,
                               2, 1, 0, 1,

                               1, 1, 0, 1,
                               2, 1, 0, 1};

  constexpr int64_t axis = 2;
  bool sorted = true;
  const std::vector<int64_t> Y_dims{2, 2, 3};
  const std::vector<int64_t> Y{0, 1, 1,
                               0, 1, 2,

                               0, 1, 1,
                               0, 1, 2};

  const std::vector<int64_t> indices_dims{3};
  const std::vector<int64_t> indices{2, 1, 0};
  const std::vector<int64_t> inverse_indices_dims{4};
  const std::vector<int64_t> inverse_indices{2, 1, 0, 1};
  const std::vector<int64_t> counts_dims{3};
  const std::vector<int64_t> counts{1, 2, 1};

  RunUniqueTest<int64_t>(X_dims, X, &axis, sorted, Y_dims, Y, indices_dims, indices,
                         inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(Unique, InvalidAxis) {
  constexpr int64_t axis = 12;
  const std::vector<int64_t> X_dims{2, 3};
  const std::vector<float> X{1.f, 4.f, 1.f, 2.f, 2.f, 0.f};
  const std::vector<int64_t> Y_dims{};
  const std::vector<float> Y{0.f};

  OpTester test("Unique", 11);

  test.AddAttribute("axis", axis);

  test.AddInput("X", X_dims, X);
  test.AddOutput("Y", Y_dims, Y);

  test.Run(OpTester::ExpectResult::kExpectFailure, "[ShapeInferenceError] Invalid value for attribute axis");
}

// check empty input is gracefully handled
TEST(Unique, EmptyInput) {
  const std::vector<int64_t> X_dims{0};
  const std::vector<float> X{};
  const std::vector<int64_t> Y_dims{0};
  const std::vector<float> Y{};
  const std::vector<int64_t> indices_dims{0};
  const std::vector<int64_t> indices{};
  const std::vector<int64_t> inverse_indices_dims{0};
  const std::vector<int64_t> inverse_indices{};
  const std::vector<int64_t> counts_dims{0};
  const std::vector<int64_t> counts{};

  OpTester test("Unique", 11);

  test.AddInput("X", X_dims, X);
  test.AddOutput("Y", Y_dims, Y);
  test.AddOutput<int64_t>("indices", indices_dims, indices);
  test.AddOutput<int64_t>("inverse_indices", inverse_indices_dims, inverse_indices);
  test.AddOutput<int64_t>("counts", counts_dims, counts);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

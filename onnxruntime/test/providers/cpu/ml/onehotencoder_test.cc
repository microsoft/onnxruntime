// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

template <typename T>
void TestIntCategory(std::vector<T>& input) {
  std::vector<int64_t> categories{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expected_output;
  for (size_t i = 0; i < input.size(); ++i)
    for (size_t j = 0; j < categories.size(); ++j)
      if (static_cast<int64_t>(input[i]) != categories[j])
        expected_output.push_back(0.0);
      else
        expected_output.push_back(1.0);

  // Test Matrix [Batch * Labels]
  OpTester test_matrix("OneHotEncoder", 1, onnxruntime::kMLDomain);
  test_matrix.AddAttribute("cats_int64s", categories);
  test_matrix.AddInput<T>("X", {1, 7}, input);
  test_matrix.AddOutput<float>("Y", {1, 7, 8}, expected_output);

  test_matrix.AddAttribute("zeros", int64_t{1});
  test_matrix.Run();

  test_matrix.AddAttribute("zeros", int64_t{0});
  test_matrix.Run(OpTester::ExpectResult::kExpectFailure);

  // Test Vector [Labels]
  OpTester test_vector("OneHotEncoder", 1, onnxruntime::kMLDomain);
  test_vector.AddAttribute("cats_int64s", categories);
  test_vector.AddInput<T>("X", {7}, input);
  test_vector.AddOutput<float>("Y", {7, 8}, expected_output);

  test_vector.AddAttribute("zeros", int64_t{1});
  test_vector.Run();

  test_vector.AddAttribute("zeros", int64_t{0});
  test_vector.Run(OpTester::ExpectResult::kExpectFailure);

  // Test MultiDimensional [:, :, Labels]
  OpTester test_multiD("OneHotEncoder", 1, onnxruntime::kMLDomain);
  test_multiD.AddAttribute("cats_int64s", categories);
  test_multiD.AddInput<T>("X", {1, 1, 7}, input);
  test_multiD.AddOutput<float>("Y", {1, 1, 7, 8}, expected_output);

  test_multiD.AddAttribute("zeros", int64_t{1});
  test_multiD.Run();

  test_multiD.AddAttribute("zeros", int64_t{0});
  test_multiD.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(OneHotEncoderOpTest, IntegerWithInt64) {
  vector<int64_t> input{8, 1, 0, 0, 3, 7, 4};
  TestIntCategory<int64_t>(input);
}

/*
// TODO: Support int32_t type kernel for the op and uncomment the test
TEST(OneHotEncoderOpTest, IntegerWithInt32) {
  vector<int32_t> input{ 8, 1, 0, 0, 3, 7, 4 };
  TestIntCategory<int32_t>(input);
}
*/

TEST(OneHotEncoderOpTest, IntegerWithDouble) {
  vector<double> input{8.1f, 1.2f, 0.0f, 0.7f, 3.4f, 7.9f, 4.4f};
  TestIntCategory<double>(input);
}

TEST(OneHotEncoderOpTest, String) {
  std::vector<std::string> categories{"Apple", "Orange", "Watermelon", "Blueberry", "Coconut", "Mango", "Tangerine"};
  vector<std::string> input{"Watermelon", "Orange", "Tangerine", "Apple", "Kit"};
  vector<float> expected_output;

  for (size_t i = 0; i < input.size(); ++i)
    for (size_t j = 0; j < categories.size(); ++j)
      if (input[i] != categories[j])
        expected_output.push_back(0.0);
      else
        expected_output.push_back(1.0);

  // Test Matrix [Batch, Labels]
  OpTester test_matrix("OneHotEncoder", 1, onnxruntime::kMLDomain);
  test_matrix.AddAttribute("cats_strings", categories);
  test_matrix.AddInput<string>("X", {1, 5}, input);
  test_matrix.AddOutput<float>("Y", {1, 5, 7}, expected_output);

  test_matrix.AddAttribute("zeros", int64_t{1});
  test_matrix.Run();

  test_matrix.AddAttribute("zeros", int64_t{0});
  test_matrix.Run(OpTester::ExpectResult::kExpectFailure);

  // Test Vector [Labels]
  OpTester test_vector("OneHotEncoder", 1, onnxruntime::kMLDomain);
  test_vector.AddAttribute("cats_strings", categories);
  test_vector.AddInput<string>("X", {5}, input);
  test_vector.AddOutput<float>("Y", {5, 7}, expected_output);

  test_vector.AddAttribute("zeros", int64_t{1});
  test_vector.Run();

  test_vector.AddAttribute("zeros", int64_t{0});
  test_vector.Run(OpTester::ExpectResult::kExpectFailure);

  // Test MultiDimensional [:, Labels, :]
  OpTester test_multiD("OneHotEncoder", 1, onnxruntime::kMLDomain);
  test_multiD.AddAttribute("cats_strings", categories);
  test_multiD.AddInput<string>("X", {1, 5, 1}, input);
  test_multiD.AddOutput<float>("Y", {1, 5, 1, 7}, expected_output);

  test_multiD.AddAttribute("zeros", int64_t{1});
  test_multiD.Run();

  test_multiD.AddAttribute("zeros", int64_t{0});
  test_multiD.Run(OpTester::ExpectResult::kExpectFailure);
}

}  // namespace test
}  // namespace onnxruntime

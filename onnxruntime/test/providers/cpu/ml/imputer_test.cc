// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MLOpTest, ImputerOpFloat) {
  OpTester test("Imputer", 1, onnxruntime::kMLDomain);
  const int N = 5;
  std::vector<float> impute = {10.0f};
  float replace = 1.f;
  test.AddAttribute("imputed_value_floats", impute);
  test.AddAttribute("replaced_value_float", replace);
  std::vector<float> X = {0.8f, -0.5f, 0.0f, 1.f, 1.0f};

  // setup expected output
  std::vector<float> expected_output;
  for (auto& elem : X) {
    if (elem == replace) {
      expected_output.push_back(impute[0]);
    } else {
      expected_output.push_back(elem);
    }
  }
  test.AddInput<float>("X", {N}, X);
  test.AddOutput<float>("Y", {N}, expected_output);
  test.Run();
}

TEST(MLOpTest, ImputerOpInts) {
  OpTester test("Imputer", 1, onnxruntime::kMLDomain);
  std::vector<int64_t> impute = {10, 20, 30, 40, 50};
  int64_t replace = 2;
  test.AddAttribute("imputed_value_int64s", impute);
  test.AddAttribute("replaced_value_int64", replace);
  std::vector<int64_t> X = {2, 0, 2, 1, 1};

  // setup expected output
  std::vector<int64_t> expected_output;
  int impute_idx = 0;
  for (auto& elem : X) {
    if (elem == replace) {
      expected_output.push_back(impute[impute_idx]);
    } else {
      expected_output.push_back(elem);
    }
    ++impute_idx;
  }
  std::vector<int64_t> dims{1, 5};
  test.AddInput<int64_t>("X", dims, X);
  test.AddOutput<int64_t>("Y", dims, expected_output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

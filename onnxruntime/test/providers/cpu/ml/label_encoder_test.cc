// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename TInput, typename TOutput>
static void RunTest(const std::vector<int64_t>& dims, const std::vector<TInput>& input, const std::vector<TOutput>& output) {
  OpTester test("LabelEncoder", 1, onnxruntime::kMLDomain);

  static const std::vector<std::string> labels = {"Beer", "Wine", "Tequila"};

  test.AddAttribute("classes_strings", labels);

  test.AddAttribute("default_string", "Water");
  test.AddAttribute<int64_t>("default_int64", 99);

  test.AddInput<TInput>("X", dims, input);
  test.AddOutput<TOutput>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, StringToInt) {
  std::vector<int64_t> dims{2, 2, 2};

  std::vector<std::string> input{"Beer", "Burger", "Tequila", "Burrito", "Wine", "Cheese", "Tequila", "Floor"};
  std::vector<int64_t> output{0, 99, 2, 99, 1, 99, 2, 99};

  RunTest(dims, input, output);
}

TEST(LabelEncoder, IntToString) {
  std::vector<int64_t> dims{2, 3};

  std::vector<int64_t> input{0, 10, 2, 3, 1, -1};
  std::vector<std::string> output{"Beer", "Water", "Tequila", "Water", "Wine", "Water"};

  RunTest(dims, input, output);
}

}  // namespace test
}  // namespace onnxruntime

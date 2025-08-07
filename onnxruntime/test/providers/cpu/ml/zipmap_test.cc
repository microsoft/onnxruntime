// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {
template <typename T>
void TestHelper(const std::vector<T>& classes,
                const std::string& type,
                const vector<int64_t>& input_dims,
                OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {
  OpTester test("ZipMap", 1, onnxruntime::kMLDomain);

  std::vector<float> input{1.f, 0.f, 3.f, 44.f, 23.f, 11.3f};

  if (type == "string") {
    test.AddAttribute("classlabels_strings", classes);
  } else if (type == "int64_t") {
    test.AddAttribute("classlabels_int64s", classes);
  } else {
    ORT_THROW("Invalid type: ", type);
  }

  int64_t batch_size = (input_dims.size() > 1) ? input_dims[0] : 1;

  // prepare expected output
  std::vector<std::map<T, float>> expected_output;
  if (expect_result == OpTester::ExpectResult::kExpectSuccess) {
    for (int64_t i = 0; i < batch_size; ++i) {
      std::map<T, float> var_map;
      for (size_t j = 0; j < classes.size(); ++j) {
        var_map.emplace(classes[j], input[i * 3 + j]);
      }
      expected_output.push_back(var_map);
    }
  }

  test.AddInput<float>("X", input_dims, input);
  test.AddOutput<T, float>("Z", expected_output);
  test.Run(expect_result);
}

// Positive test cases
TEST(MLOpTest, ZipMapOpStringFloat) {
  TestHelper<string>({"class1", "class2", "class3"}, "string", {2, 3});
}

TEST(MLOpTest, ZipMapOpInt64Float) {
  TestHelper<int64_t>({10, 20, 30}, "int64_t", {2, 3});
}

TEST(MLOpTest, ZipMapOpInt64Float1D) {
  TestHelper<int64_t>({10, 20, 30, 40, 50, 60}, "int64_t", {6});
}

// Negative test cases
TEST(MLOpTest, ZipMapOpStringFloatStrideMoreThanNumLabels) {
  TestHelper<string>({"class1", "class2", "class3"}, "string", {1, 6}, OpTester::ExpectResult::kExpectFailure);
}

TEST(MLOpTest, ZipMapOpStringFloatStrideLessThanNumLabels) {
  TestHelper<string>({"class1", "class2", "class3"}, "string", {3, 2}, OpTester::ExpectResult::kExpectFailure);
}

TEST(MLOpTest, ZipMapOpInt64FloatStrideMoreThanNumLabels) {
  TestHelper<int64_t>({10, 20, 30}, "int64_t", {1, 6}, OpTester::ExpectResult::kExpectFailure);
}

TEST(MLOpTest, ZipMapOpInt64FloatStrideLessThanNumLabels) {
  TestHelper<int64_t>({10, 20, 30}, "int64_t", {3, 2}, OpTester::ExpectResult::kExpectFailure);
}
}  // namespace test
}  // namespace onnxruntime

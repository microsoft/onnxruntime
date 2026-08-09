// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// cast from map<int64_t,TFrom> to Tensor<TCastTo>
template <typename TFrom, typename TCastTo>
static void RunTest(const std::map<int64_t, TFrom>& input,
                    const std::vector<TCastTo>& output,
                    const std::string& cast_to,
                    int64_t max_map = -1,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {
  OpTester test("CastMap", 1, onnxruntime::kMLDomain);

  test.AddAttribute("cast_to", cast_to);

  if (max_map <= 0) {
    test.AddAttribute("map_form", "DENSE");
  } else {
    test.AddAttribute("map_form", "SPARSE");
    test.AddAttribute("max_map", max_map);
  }

  test.AddInput<int64_t, TFrom>("X", input);

  std::vector<int64_t> dims{1, gsl::narrow_cast<int64_t>(output.size())};
  test.AddOutput("Y", dims, output);

  test.Run(expect_result);
}

/*
Cast to Tensor<float>.
*/

// test dense input, converting from string to float
// also validate the output is ordered based on the index in the map, and not the order the entries were added.
TEST(CastMap, DenseStringToFloat) {
  std::map<int64_t, std::string> map = {{1, "1.0"},
                                        {2, "2"},
                                        {3, "-3.0f"},
                                        {0, "-1"}};  // this should be first in the output

  std::vector<float> output{-1.0f, 1.0f, 2.0f, -3.0f};

  RunTest(map, output, "TO_FLOAT");
}  // namespace test

// Test sparse input, converting from float to float
TEST(CastMap, SparseFloatToFloat) {
  // test with gaps at start and end
  std::map<int64_t, float> map{{1, 1.0f}, {2, 2.0f}};

  std::vector<float> output{0.0f, 1.0f, 2.0f, 0.0f};

  RunTest(map, output, "TO_FLOAT", 4);
}

/*
Cast to Tensor<int64_t>
*/
TEST(CastMap, SparseStringToInt64) {
  // gaps in middle
  std::map<int64_t, std::string> map{{0, "-1.0"}, {3, "3"}};
  std::vector<int64_t> output{-1, 0, 0, 3};

  RunTest(map, output, "TO_INT64", 4);
}

TEST(CastMap, DenseFloatToInt64) {
  // float to int64 is just a static_cast, so no rounding
  std::map<int64_t, float> map{{0, -1.f}, {1, 1.9f}, {2, -2.4f}, {3, 3.0f}};
  std::vector<int64_t> output{-1, 1, -2, 3};

  RunTest(map, output, "TO_INT64");
}

/*
Cast to Tensor<string>
*/
TEST(CastMap, StringToString) {
  std::map<int64_t, std::string> map{{0, "-1.0f"}, {1, "3"}};

  std::vector<std::string> output{"-1.0f", "3"};

  RunTest(map, output, "TO_STRING");
}

TEST(CastMap, FloatToString) {
  std::map<int64_t, float> map{{0, -1.0f}, {1, 3.0f}};

  // std::stof converts to these values.
  std::vector<std::string> output{"-1.000000", "3.000000"};

  RunTest(map, output, "TO_STRING");
}

/*
Miscellaneous tests
*/

// cast from map<int64_t,TFrom> to Tensor<TCastTo>
void RunBadAttributeTest(const std::string& cast_to,
                         const std::string& map_form,
                         int64_t max_map = -1,
                         OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {
  OpTester test("CastMap", 1, onnxruntime::kMLDomain);

  test.AddAttribute("cast_to", cast_to);
  test.AddAttribute("map_form", map_form);
  test.AddAttribute("max_map", max_map);

  std::map<int64_t, float> input{{0, 1.0f}};
  std::vector<float> output{1.0f};

  test.AddInput<int64_t, float>("X", input);

  std::vector<int64_t> dims{1, gsl::narrow_cast<int64_t>(output.size())};
  test.AddOutput<float>("Y", dims, output);

  test.Run(expect_result);
}

// test invalid attributes are detected
TEST(CastMap, InvalidAttributes) {
  // bad cast_to
  RunBadAttributeTest("UNKNOWN", "DENSE", -1, OpTester::ExpectResult::kExpectFailure);

  // bad map_form
  RunBadAttributeTest("TO_FLOAT", "UNKNOWN", -1, OpTester::ExpectResult::kExpectFailure);

  // bad max_map
  RunBadAttributeTest("TO_FLOAT", "SPARSE", -2, OpTester::ExpectResult::kExpectFailure);
}

TEST(CastMap, InvalidIndexInMap) {
  // negative index values in map aren't allowed
  std::map<int64_t, std::string> map{{-3, "-3"}, {0, "0"}};
  std::vector<int64_t> output{0, 0};

  RunTest(map, output, "TO_INT64", 5, OpTester::ExpectResult::kExpectFailure);
}

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(CompressTest, Compress0) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(0));

  test.AddInput<float>("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<bool>("condition", {3}, {0, 1, 1});
  test.AddOutput<float>("output", {2, 2}, {3.0f, 4.0f, 5.0f, 6.0f});
  test.Run();
}

TEST(CompressTest, Compress1) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<bool>("condition", {2}, {0, 1});
  test.AddOutput<float>("output", {3, 1}, {2.0f, 4.0f, 6.0f});
  test.Run();
}

TEST(CompressTest, Compress_3dims) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<bool>("condition", {2}, {0, 1});
  test.AddOutput<float>("output", {2, 1, 3}, {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f});
  test.Run();
}

TEST(CompressTest, Compress_condition_all_false) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<bool>("condition", {2}, {0, 0});
  test.AddOutput<float>("output", {2, 0, 3}, {});
  test.Run();
}

TEST(CompressTest, Compress_3dims_has_extra_condition) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  // has condition length = 3 > input_dim[axis] = 2
  test.AddInput<bool>("condition", {3}, {0, 1, 1});
  test.AddOutput<float>("output", {2, 1, 3}, {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(CompressTest, Compress_3dims_has_extra_input) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {2, 3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,

                                            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f});
  // has condition length = 2 < input_dim[axis] = 3
  test.AddInput<bool>("condition", {2}, {0, 1});
  test.AddOutput<float>("output", {2, 1, 3}, {4.0f, 5.0f, 6.0f, 13.0f, 14.0f, 15.0f});
  test.Run();
}

TEST(CompressTest, Compress_default_axis) {
  OpTester test("Compress", 9);

  test.AddInput<float>("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<bool>("condition", {5}, {0, 1, 0, 0, 1});
  test.AddOutput<float>("output", {2}, {2.0f, 5.0f});
  test.Run();
}

// Test that we accumulate to a buffer that does not overflow
TEST(CompressTest, Compress_default_axis_issue_9247_cumulative_sum_overflow) {
  OpTester test("Compress", 9);

  // Generate input longer than 127
  constexpr size_t elements = 150;
  std::vector<float> input;
  for (size_t i = 0; i < elements; ++i) {
    input.push_back(static_cast<float>(i));
  }

  // Let's select all of the elements
  std::unique_ptr<bool[]> all_true = std::make_unique<bool[]>(elements);
  std::fill_n(all_true.get(), elements, true);
  std::vector<int64_t> output_shape{static_cast<int64_t>(elements)};

  test.AddInput<float>("input", {2, 75}, input);
  test.AddInput<bool>("condition", output_shape, all_true.get(), elements);
  // Should get all of the input
  test.AddOutput<float>("output", output_shape, input);
  test.Run();
}

TEST(CompressTest, Compress0_string) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(0));

  test.AddInput<std::string>("input", {3, 2}, {"1", "2", "3", "4", "5", "6"});
  test.AddInput<bool>("condition", {3}, {0, 1, 1});
  test.AddOutput<std::string>("output", {2, 2}, {"3", "4", "5", "6"});
  test.Run();
}

TEST(CompressTest, Compress_default_axis_string) {
  OpTester test("Compress", 9);

  test.AddInput<std::string>("input", {3, 2}, {"1", "2", "3", "4", "5", "6"});
  test.AddInput<bool>("condition", {5}, {0, 1, 0, 0, 1});
  test.AddOutput<std::string>("output", {2}, {"2", "5"});
  test.Run();
}

TEST(CompressTest, Compress_3dims_neg_axis) {
  OpTester test("Compress", 11);

  test.AddAttribute("axis", int64_t(-2));

  test.AddInput<float>("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<bool>("condition", {2}, {0, 1});
  test.AddOutput<float>("output", {2, 1, 3}, {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

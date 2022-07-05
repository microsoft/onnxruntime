// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <limits>

namespace onnxruntime {
namespace test {

constexpr float FLOAT_INF = std::numeric_limits<float>::infinity();
constexpr float FLOAT_NINF = -std::numeric_limits<float>::infinity();
constexpr float FLOAT_NAN = std::numeric_limits<float>::quiet_NaN();

constexpr double DOUBLE_INF = std::numeric_limits<double>::infinity();
constexpr double DOUBLE_NINF = -std::numeric_limits<double>::infinity();
constexpr double DOUBLE_NAN = std::numeric_limits<double>::quiet_NaN();

TEST(IsInfTest, test_isinf_float) {
  // Defaults for detect_negative = 1
  // detect_positive = 1
  OpTester test("IsInf", 10);

  std::vector<int64_t> input_dim{6};
  std::vector<float> input = {-1.2f, FLOAT_NAN, FLOAT_INF, 2.8f, FLOAT_NINF, FLOAT_INF};
  test.AddInput<float>("X", input_dim, input);

  std::vector<int64_t> output_dim(input_dim);
  test.AddOutput<bool>("Y", output_dim, {false, false, true, false, true, true});
  test.Run();
}

TEST(IsInfTest, test_isinf_double) {
  // Defaults for detect_negative = 1
  // detect_positive = 1
  OpTester test("IsInf", 10);

  std::vector<int64_t> input_dim{6};
  std::vector<double> input = {-1.2, DOUBLE_NAN, DOUBLE_INF, 2.8, DOUBLE_NINF, DOUBLE_INF};
  test.AddInput<double>("X", input_dim, input);

  std::vector<int64_t> output_dim(input_dim);
  test.AddOutput<bool>("Y", output_dim, {false, false, true, false, true, true});
  test.Run();
}

TEST(IsInfTest, test_isinf_positive_float) {
  OpTester test("IsInf", 10);
  test.AddAttribute<int64_t>("detect_negative", 0);

  std::vector<int64_t> input_dim{6};
  std::vector<float> input = {-1.7f, FLOAT_NAN, FLOAT_INF, 3.6f, FLOAT_NINF, FLOAT_INF};
  test.AddInput<float>("X", input_dim, input);

  std::vector<int64_t> output_dim(input_dim);
  test.AddOutput<bool>("Y", output_dim, {false, false, true, false, false, true});
  test.Run();
}

TEST(IsInfTest, test_isinf_positive_double) {
  OpTester test("IsInf", 10);
  test.AddAttribute<int64_t>("detect_negative", 0);

  std::vector<int64_t> input_dim{6};
  std::vector<double> input = {-1.7, DOUBLE_NAN, DOUBLE_INF, 3.6, DOUBLE_NINF, DOUBLE_INF};
  test.AddInput<double>("X", input_dim, input);

  std::vector<int64_t> output_dim(input_dim);
  test.AddOutput<bool>("Y", output_dim, {false, false, true, false, false, true});
  test.Run();
}

TEST(IsInfTest, test_isinf_negative_float) {
  OpTester test("IsInf", 10);
  test.AddAttribute<int64_t>("detect_positive", 0);

  std::vector<int64_t> input_dim{6};
  std::vector<float> input = {-1.7f, FLOAT_NAN, FLOAT_INF, 3.6f, FLOAT_NINF, FLOAT_INF};
  test.AddInput<float>("X", input_dim, input);

  std::vector<int64_t> output_dim(input_dim);
  test.AddOutput<bool>("Y", output_dim, {false, false, false, false, true, false});
  test.Run();
}

TEST(IsInfTest, test_isinf_negative_double) {
  OpTester test("IsInf", 10);
  test.AddAttribute<int64_t>("detect_positive", 0);

  std::vector<int64_t> input_dim{6};
  std::vector<double> input = {-1.7, DOUBLE_NAN, DOUBLE_INF, 3.6, DOUBLE_NINF, DOUBLE_INF};
  test.AddInput<double>("X", input_dim, input);

  std::vector<int64_t> output_dim(input_dim);
  test.AddOutput<bool>("Y", output_dim, {false, false, false, false, true, false});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

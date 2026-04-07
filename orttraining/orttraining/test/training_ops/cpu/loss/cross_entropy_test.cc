// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Regression tests for OOB reads when label values are outside [0, C).

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLarge) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, 5, 2};  // 5 is out of range [0, 5)

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_NegativeLabel) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-100));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, -1, 2};  // -1 is out of range (and != ignore_index)

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLargeWithWeights) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, 100, 2};  // 100 is out of range
  std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddInput<float>("weight", {5}, weight_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_LabelTooLarge) {
  OpTester test("SoftmaxCrossEntropyLossGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> dY_data = {1.0f};
  std::vector<float> log_prob_data(3 * 5, -1.6094f);
  std::vector<int64_t> index_data = {0, 5, 2};  // 5 is out of range [0, 5)

  test.AddInput<float>("dY", {}, dY_data);
  test.AddInput<float>("log_prob", {3, 5}, log_prob_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOutput<float>("dX", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_LabelTooLargeWithWeights) {
  OpTester test("SoftmaxCrossEntropyLossGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> dY_data = {1.0f};
  std::vector<float> log_prob_data(3 * 5, -1.6094f);
  std::vector<int64_t> index_data = {0, 5, 2};  // 5 is out of range [0, 5)
  std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  test.AddInput<float>("dY", {}, dY_data);
  test.AddInput<float>("log_prob", {3, 5}, log_prob_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddInput<float>("weight", {5}, weight_data);
  test.AddOutput<float>("dX", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

}  // namespace test
}  // namespace onnxruntime

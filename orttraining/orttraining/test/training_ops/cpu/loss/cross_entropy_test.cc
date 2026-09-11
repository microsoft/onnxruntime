// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

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

// Covers the weighted, non-MEAN forward loop (the second per-sample loop in Compute).
TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLargeWithWeightsSumReduction) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("sum"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, 7, 2};  // 7 is out of range [0, 5)
  std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddInput<float>("weight", {5}, weight_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

// int32 label type — kernel is registered for both int32_t and int64_t.
TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLargeInt32) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int32_t> index_data = {0, 5, 2};  // 5 is out of range [0, 5)

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int32_t>("index", {3}, index_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

// Higher-dimensional inputs: logit [N, C, D1, D2], label [N, D1, D2].
TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLargeHighDim) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  // [N=2, C=4, D1=2, D2=3] -> label shape [2, 2, 3] -> 12 label entries.
  std::vector<float> X_data(2 * 4 * 2 * 3, 1.0f);
  std::vector<int64_t> index_data = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 99, 3};  // 99 is out of range
  std::vector<float> log_prob_init(2 * 4 * 2 * 3, 0.0f);

  test.AddInput<float>("X", {2, 4, 2, 3}, X_data);
  test.AddInput<int64_t>("index", {2, 2, 3}, index_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {2, 4, 2, 3}, log_prob_init);

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

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_LabelTooLargeInt32) {
  OpTester test("SoftmaxCrossEntropyLossGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> dY_data = {1.0f};
  std::vector<float> log_prob_data(3 * 5, -1.6094f);
  std::vector<int32_t> index_data = {0, 5, 2};

  test.AddInput<float>("dY", {}, dY_data);
  test.AddInput<float>("log_prob", {3, 5}, log_prob_data);
  test.AddInput<int32_t>("index", {3}, index_data);
  test.AddOutput<float>("dX", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

// SoftmaxCrossEntropyLossInternal shares the same Compute as SoftmaxCrossEntropyLoss but
// is registered separately under kMSDomain v1 with an optional runtime ignore_index input.
TEST(CrossEntropyTest, SoftmaxCrossEntropyLossInternal_LabelTooLarge) {
  OpTester test("SoftmaxCrossEntropyLossInternal", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, 5, 2};
  int64_t ignore_index_val = -1;

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  // weight is optional; pass empty input to skip and still provide ignore_index input below.
  test.AddOptionalInputEdge<float>();
  test.AddInput<int64_t>("ignore_index", {}, &ignore_index_val, 1);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossInternalGrad_LabelTooLarge) {
  OpTester test("SoftmaxCrossEntropyLossInternalGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));

  std::vector<float> dY_data = {1.0f};
  std::vector<float> log_prob_data(3 * 5, -1.6094f);
  std::vector<int64_t> index_data = {0, 5, 2};
  int64_t ignore_index_val = -1;

  test.AddInput<float>("dY", {}, dY_data);
  test.AddInput<float>("log_prob", {3, 5}, log_prob_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOptionalInputEdge<float>();  // weight
  test.AddInput<int64_t>("ignore_index", {}, &ignore_index_val, 1);
  test.AddOutput<float>("dX", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

}  // namespace test
}  // namespace onnxruntime

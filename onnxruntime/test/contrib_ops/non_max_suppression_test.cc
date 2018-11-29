// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(NonMaxSuppressionOpTest, WithIOUThreshold) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {6, 4},
      {0.0f, 0.0f, 1.0f, 1.0f,
       0.0f, 0.1f, 1.0f, 1.1f,
       0.0f, -0.1f, 1.0f, 0.9f,
       0.0f, 10.0f, 1.0f, 11.0f,
       0.0f, 10.1f, 1.0f, 11.1f,
       0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddAttribute<int64_t>("max_output_size", 3LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {3}, {3L, 0L, 5L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, WithScoreThreshold) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddAttribute<int64_t>("max_output_size", 3LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.4f);
  test.AddOutput<int32_t>("selected_indices", {2}, {3L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, WithScoreThresholdZeroScores) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {6}, {0.1f, 0.0f, 0.0f, 0.3f, 0.2f, -5.0f});
  test.AddAttribute<int64_t>("max_output_size", 6LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", -3.0f);
  test.AddOutput<int32_t>("selected_indices", {2}, {3L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, FlippedCoordinates) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {6, 4},
                       {1.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, 0.9f, 1.0f, -0.1f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        1.0f, 10.1f, 0.0f, 11.1f,
                        1.0f, 101.0f, 0.0f, 100.0f});
  test.AddInput<float>("scores", {6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddAttribute<int64_t>("max_output_size", 3LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {3}, {3L, 0L, 5L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, SelectTwo) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddAttribute<int64_t>("max_output_size", 2LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {2}, {3L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, SelectThirty) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddAttribute<int64_t>("max_output_size", 30LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {3}, {3L, 0L, 5L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, SelectSingleBox) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {1, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f});
  test.AddInput<float>("scores", {1}, {0.9f});
  test.AddAttribute<int64_t>("max_output_size", 3LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {1}, {0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, SelectFromIdenticalBoxes) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {10, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f,

                        0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 1.0f, 1.0f});
  test.AddInput<float>("scores", {10}, {0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f});
  test.AddAttribute<int64_t>("max_output_size", 3LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {1}, {0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, InconsistentBoxAndScoreShapes) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {5}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f});
  test.AddAttribute<int64_t>("max_output_size", 30LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {1}, {0L});
  test.Run(OpTester::ExpectResult::kExpectFailure, "scores and boxes should have same num_boxes.");
}

TEST(NonMaxSuppressionOpTest, InvalidIOUThreshold) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {1, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f});
  test.AddInput<float>("scores", {1}, {0.9f});
  test.AddAttribute<int64_t>("max_output_size", 3LL);
  test.AddAttribute<float>("iou_threshold", 1.2f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {1}, {0L});
  test.Run(OpTester::ExpectResult::kExpectFailure, "iou_threshold must be in range [0, 1]");
}

TEST(NonMaxSuppressionOpTest, EmptyInput) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {0, 4},
                       {});
  test.AddInput<float>("scores", {0}, {});
  test.AddAttribute<int64_t>("max_output_size", 30LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {0}, {});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, ZeroMaxOutputSize) {
  OpTester test("NonMaxSuppression", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("boxes", {6, 4},
                       {1.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, 0.9f, 1.0f, -0.1f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        1.0f, 10.1f, 0.0f, 11.1f,
                        1.0f, 101.0f, 0.0f, 100.0f});
  test.AddInput<float>("scores", {6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddAttribute<int64_t>("max_output_size", 0LL);
  test.AddAttribute<float>("iou_threshold", 0.5f);
  test.AddAttribute<float>("score_threshold", 0.0f);
  test.AddOutput<int32_t>("selected_indices", {0}, {});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

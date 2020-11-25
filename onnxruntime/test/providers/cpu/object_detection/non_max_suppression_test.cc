// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(NonMaxSuppressionOpTest, WithIOUThreshold) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {3, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L,
                           0L, 0L, 5L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, CenterPointBoxFormat) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.5f, 0.5f, 1.0f, 1.0f,
                        0.5f, 0.6f, 1.0f, 1.0f,
                        0.5f, 0.4f, 1.0f, 1.0f,
                        0.5f, 10.5f, 1.0f, 1.0f,
                        0.5f, 10.6f, 1.0f, 1.0f,
                        0.5f, 100.5f, 1.0f, 1.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {3, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L,
                           0L, 0L, 5L});
  test.AddAttribute<int64_t>("center_point_box", 1LL);
  test.Run();
}

TEST(NonMaxSuppressionOpTest, TwoClasses) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 2, 6},
                       {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f,
                        0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {6L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {6, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L,
                           0L, 0L, 5L,
                           0L, 1L, 3L,
                           0L, 1L, 0L,
                           0L, 1L, 5L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, TwoBatches_OneClass) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {2, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f,

                        0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {2, 1, 6},
                       {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f,
                        0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {2L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {4, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L,
                           1L, 0L, 3L,
                           1L, 0L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, TwoBatches_TwoClasses) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {2, 5, 4},
                       {0.0f, 0.0f, 0.3f, 0.3f,
                        0.0f, 0.0f, 0.4f, 0.4f,
                        0.0f, 0.0f, 0.5f, 0.5f,
                        0.5f, 0.5f, 0.9f, 0.9f,
                        0.5f, 0.5f, 1.0f, 1.0f,

                        0.0f, 0.0f, 0.3f, 0.3f,
                        0.0f, 0.0f, 0.4f, 0.4f,
                        0.5f, 0.5f, 0.95f, 0.95f,
                        0.5f, 0.5f, 0.96f, 0.96f,
                        0.5f, 0.5f, 1.0f, 1.0f});
  test.AddInput<float>("scores", {2, 2, 5},
                       {0.1f, 0.2f, 0.6f, 0.3f, 0.9f,
                        0.1f, 0.2f, 0.6f, 0.3f, 0.9f,

                        0.1f, 0.2f, 0.6f, 0.3f, 0.9f,
                        0.1f, 0.2f, 0.6f, 0.3f, 0.9f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {2L});
  test.AddInput<float>("iou_threshold", {}, {0.8f});
  test.AddOutput<int64_t>("selected_indices", {8, 3},
                          {0L, 0L, 4L,
                           0L, 0L, 2L,
                           0L, 1L, 4L,
                           0L, 1L, 2L,

                           1L, 0L, 4L,
                           1L, 0L, 1L,
                           1L, 1L, 4L,
                           1L, 1L, 1L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, WithScoreThreshold) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.4f});
  test.AddOutput<int64_t>("selected_indices", {2, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, WithoutScoreThreshold) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddOutput<int64_t>("selected_indices", {3, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L,
                           0L, 0L, 5L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, WithScoreThresholdZeroScores) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.1f, 0.0f, 0.0f, 0.3f, 0.2f, -5.0f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {6L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {-3.0f});
  test.AddOutput<int64_t>("selected_indices", {2, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, FlippedCoordinates) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {1.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, 0.9f, 1.0f, -0.1f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        1.0f, 10.1f, 0.0f, 11.1f,
                        1.0f, 101.0f, 0.0f, 100.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {3, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L,
                           0L, 0L, 5L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, SelectTwo) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {2L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {2, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, SelectThirty) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {30L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {3, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L,
                           0L, 0L, 5L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, SelectSingleBox) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 1, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f});
  test.AddInput<float>("scores", {1, 1, 1}, {0.9f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {1, 3}, {0L, 0L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, SelectFromIdenticalBoxes) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 10, 4},
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
  test.AddInput<float>("scores", {1, 1, 10}, {0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {1, 3}, {0L, 0L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, InconsistentBoxAndScoreShapes) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 5}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {30L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {0, 3}, {});
  test.Run(OpTester::ExpectResult::kExpectFailure, "boxes and scores should have same spatial_dimension.");
}

TEST(NonMaxSuppressionOpTest, InvalidIOUThreshold) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
  test.AddInput<float>("scores", {1, 1, 1}, {0.9f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {1.2f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {0, 3}, {});
  test.Run(OpTester::ExpectResult::kExpectFailure, "iou_threshold must be in range [0, 1]");
}

TEST(NonMaxSuppressionOpTest, EmptyInput) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 0, 4}, {});
  test.AddInput<float>("scores", {1, 1, 0}, {});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {30L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {0, 3}, {});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, ZeroMaxOutputPerClass) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {0L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.4f});
  test.AddOutput<int64_t>("selected_indices", {0, 3}, {});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, BigIntMaxOutputBoxesPerClass) {
  OpTester test("NonMaxSuppression", 10, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {9223372036854775807L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.4f});
  test.AddOutput<int64_t>("selected_indices", {2, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L});
  test.Run();
}

TEST(NonMaxSuppressionOpTest, WithIOUThresholdOpset11) {
  OpTester test("NonMaxSuppression", 11, kOnnxDomain);
  test.AddInput<float>("boxes", {1, 6, 4},
                       {0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.1f, 1.0f, 1.1f,
                        0.0f, -0.1f, 1.0f, 0.9f,
                        0.0f, 10.0f, 1.0f, 11.0f,
                        0.0f, 10.1f, 1.0f, 11.1f,
                        0.0f, 100.0f, 1.0f, 101.0f});
  test.AddInput<float>("scores", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  test.AddInput<int64_t>("max_output_boxes_per_class", {}, {3L});
  test.AddInput<float>("iou_threshold", {}, {0.5f});
  test.AddInput<float>("score_threshold", {}, {0.0f});
  test.AddOutput<int64_t>("selected_indices", {3, 3},
                          {0L, 0L, 3L,
                           0L, 0L, 0L,
                           0L, 0L, 5L});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

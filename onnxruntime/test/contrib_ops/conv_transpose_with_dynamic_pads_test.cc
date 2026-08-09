// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {
TEST(ContribOpTest, ConvTransposeWithDynamicPads) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("output_padding", std::vector<int64_t>{1, 1});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>{0.16857791f, -0.15161794f, 0.08540368f, 0.1820628f, -0.21746576f, 0.08245695f, 0.1431433f, -0.43156421f, 0.30591947f});
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>{-0.06230065f, 0.37932432f, -0.25388849f, 0.33878803f, 0.43709868f, -0.22477469f, 0.04118127f, -0.44696793f, 0.06373066f});
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{1, 1, 1, 1});
  test.AddOutput<float>("Y", {1, 1, 6, 6}, std::vector<float>{0.07368518f, -0.08925839f, -0.06627201f, 0.06301362f, 0.03732984f, -0.01919658f, -0.00628807f, -0.02817563f, -0.01472169f, 0.04392925f, -0.00689478f, -0.01549204f, 0.07957941f, -0.11459791f, -0.09505399f, 0.07681622f, 0.03604182f, -0.01853423f, -0.0270785f, -0.00680824f, -0.06650258f, 0.08004665f, 0.07918708f, -0.0724144f, 0.06256775f, -0.17838378f, -0.18863615f, 0.20064656f, 0.133717f, -0.06876295f, -0.06398046f, -0.00864975f, 0.19289537f, -0.01490572f, -0.13673618f, 0.01949645f});
  test.Run();
}

// Test that a rank-0 W input is gracefully rejected rather than causing undefined behavior.
// These tests exercise shape inference which uses fail_shape_inference (throws InferenceError).
// In no-exception builds, fail_shape_inference calls abort(), so these tests must be skipped.
#ifndef ORT_NO_EXCEPTIONS
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidWeightRank0) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {}, std::vector<float>{1.0f});  // scalar (rank 0)
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{1, 1, 1, 1});
  test.AddOutput<float>("Y", {}, std::vector<float>{0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "Weight tensor must have at least 3 dimensions",
           {kTensorrtExecutionProvider});
}

// Test that a rank-1 W input is gracefully rejected.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidWeightRank1) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {9}, std::vector<float>(9, 1.0f));  // rank 1
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{1, 1, 1, 1});
  test.AddOutput<float>("Y", {}, std::vector<float>{0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "Weight tensor must have at least 3 dimensions",
           {kTensorrtExecutionProvider});
}

// Test that a rank-2 input is rejected (requires at least 3 dims for ConvTranspose).
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidInputRank2) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1}, std::vector<float>{1.0f});  // rank 2
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{1, 1, 1, 1});
  test.AddOutput<float>("Y", {}, std::vector<float>{0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "Input tensor must have at least 3 dimensions",
           {kTensorrtExecutionProvider});
}
#endif  // !ORT_NO_EXCEPTIONS

// Test that incorrectly sized dynamic pads are rejected.
// This runs through kernel validation (not shape inference) so it works in no-exception builds.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidPadsSize) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {3}, std::vector<int64_t>{0, 0, 0});  // Wrong size: should be 4
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "does not match expected size",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// Test that negative pad values are rejected.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_NegativePads) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{-1, 0, 0, 0});  // Negative pad
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "non-negative",
           {kTensorrtExecutionProvider, kQnnExecutionProvider, kDmlExecutionProvider});
}

// DML-specific tests for invalid dynamic pads.
// DML validates operator parameters internally before ORT kernel code runs. When inputs are
// invalid, DML's COM/HRESULT boundary strips the descriptive message and re-throws with just
// E_INVALIDARG (0x80070057), surfacing as "The parameter is incorrect." on Windows.
// We still want to verify DML rejects these inputs rather than crashing, so we test separately
// with the DML-specific error text.
#ifdef USE_DML
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidPadsSize_Dml) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {3}, std::vector<int64_t>{0, 0, 0});  // Wrong size: should be 4
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.ConfigEp(DefaultDmlExecutionProvider())
      .Config(OpTester::ExpectResult::kExpectFailure, "The parameter is incorrect")
      .RunWithConfig();
}

TEST(ContribOpTest, ConvTransposeWithDynamicPads_NegativePads_Dml) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{-1, 0, 0, 0});  // Negative pad
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.ConfigEp(DefaultDmlExecutionProvider())
      .Config(OpTester::ExpectResult::kExpectFailure, "The parameter is incorrect")
      .RunWithConfig();
}
#endif  // USE_DML

}  // namespace test
}  // namespace onnxruntime

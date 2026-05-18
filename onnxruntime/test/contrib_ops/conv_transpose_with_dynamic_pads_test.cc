// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Basic functional test - 2D convolution transpose with dynamic pads
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

// Basic functional test with zero pads
TEST(ContribOpTest, ConvTransposeWithDynamicPads_ZeroPads) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f});
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 4, 4},
                        std::vector<float>{1.0f, 2.0f, 0.0f, 0.0f,
                                           3.0f, 5.0f, 2.0f, 0.0f,
                                           0.0f, 3.0f, 5.0f, 2.0f,
                                           0.0f, 0.0f, 3.0f, 4.0f});
  test.Run();
}

// Test with group > 1
TEST(ContribOpTest, ConvTransposeWithDynamicPads_Groups) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("group", static_cast<int64_t>(2));

  // X: {N=1, C=2, H=2, W=2}
  test.AddInput<float>("X", {1, 2, 2, 2}, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f});
  // W: {C_in=2, C_out/group=1, kH=3, kW=3} - each group has its own filter
  test.AddInput<float>("W", {2, 1, 3, 3}, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{1, 1, 1, 1});
  // Output: {N=1, C_out=2, H=2, W=2}
  test.AddOutput<float>("Y", {1, 2, 2, 2}, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f});
  test.Run();
}

#ifndef ORT_NO_EXCEPTIONS
// Tests below trigger fail_shape_inference() which throws.
// In no-exception builds, this would abort, so these tests are skipped.

// Security: Rank-0 W tensor should fail with proper error, not crash.
// This is an MS domain contrib op, so we control the shape inference code.
// Shape inference rejects the model at load time with fail_shape_inference.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidRank0W) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {}, std::vector<float>{1.0f});  // Rank-0 W
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "Filter W must have at least 2 dimensions");
}

// Security: Rank-1 X tensor should fail with proper error, not crash.
// Shape inference rejects the model at load time.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidRank1X) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("strides", std::vector<int64_t>{1});

  test.AddInput<float>("X", {3}, std::vector<float>(3, 1.0f));  // Rank-1 X
  test.AddInput<float>("W", {1, 1, 3}, std::vector<float>(3, 1.0f));
  test.AddInput<int64_t>("Pads", {2}, std::vector<int64_t>{0, 0});
  test.AddOutput<float>("Y", {1, 1, 5}, std::vector<float>(5, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "Input X must have at least 2 dimensions");
}

// Security: Rank-0 X tensor should fail with proper error, not crash.
// Shape inference rejects the model at load time.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidRank0X) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {}, std::vector<float>{1.0f});  // Rank-0 X
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "Input X must have at least 2 dimensions");
}

// Security: Pads tensor with wrong number of elements should fail.
// When Pads is available as initializer, shape inference catches this with fail_shape_inference.
// When Pads is only available at runtime, kernel validation catches it.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidPadsSize) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {3}, std::vector<int64_t>{0, 0, 0});  // Wrong size: should be 4
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "Pads has incorrect number of values");
}

// Security: 2D Pads tensor should fail.
// Shape inference catches this (fail_shape_inference) when pads initializer is available.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_InvalidPadsRank) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {2, 2}, std::vector<int64_t>{0, 0, 0, 0});  // 2D instead of 1D
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "'pads' input must be a 1D");
}

#endif  // !ORT_NO_EXCEPTIONS

// Security: Negative pad values should fail.
// This is caught by kernel validation (ORT_MAKE_STATUS), not fail_shape_inference,
// so it works in both exception and no-exception builds.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_NegativePads) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{-1, 0, 0, 0});  // Negative pad
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "Pad values must be non-negative");
}

// Security: X and W dimension mismatch should fail.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_DimMismatch) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));      // 4D
  test.AddInput<float>("W", {1, 1, 3, 3, 3}, std::vector<float>(27, 1.0f));  // 5D - mismatch
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 1, 5, 5}, std::vector<float>(25, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "X num_dims does not match W num_dims");
}

// 1D convolution transpose with dynamic pads
TEST(ContribOpTest, ConvTransposeWithDynamicPads_1D) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3});
  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("dilations", std::vector<int64_t>{1});
  test.AddAttribute("output_padding", std::vector<int64_t>{1});

  // X: {N=1, C=1, L=3}
  test.AddInput<float>("X", {1, 1, 3}, std::vector<float>{1.0f, 2.0f, 3.0f});
  // W: {C_in=1, C_out/group=1, kL=3}
  test.AddInput<float>("W", {1, 1, 3}, std::vector<float>{1.0f, 1.0f, 1.0f});
  test.AddInput<int64_t>("Pads", {2}, std::vector<int64_t>{1, 1});
  // Output: stride*(L-1) + output_padding + kernel - pad_begin - pad_end = 2*(3-1) + 1 + 3 - 1 - 1 = 6
  test.AddOutput<float>("Y", {1, 1, 6}, std::vector<float>{1.0f, 2.0f, 4.0f, 3.0f, 5.0f, 3.0f});

  test.Run();
}

// Batch size > 1
TEST(ContribOpTest, ConvTransposeWithDynamicPads_BatchSize2) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  // X: {N=2, C=1, H=2, W=2} - two identical images
  test.AddInput<float>("X", {2, 1, 2, 2}, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  // W: {C_in=1, C_out/group=1, kH=2, kW=2} - identity-like filter
  test.AddInput<float>("W", {1, 1, 2, 2}, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  // Output: {N=2, C=1, H=3, W=3} - each batch element processed independently
  test.AddOutput<float>("Y", {2, 1, 3, 3}, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.Run();
}

// Multiple output channels (C_out/group > 1)
TEST(ContribOpTest, ConvTransposeWithDynamicPads_MultipleOutputChannels) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 1});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  // X: {N=1, C_in=1, H=1, W=1}
  test.AddInput<float>("X", {1, 1, 1, 1}, std::vector<float>{2.0f});
  // W: {C_in=1, C_out/group=2, kH=1, kW=1} - maps 1 channel to 2 channels
  test.AddInput<float>("W", {1, 2, 1, 1}, std::vector<float>{1.0f, 3.0f});
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  // Output: {N=1, C_out=2, H=1, W=1}
  test.AddOutput<float>("Y", {1, 2, 1, 1}, std::vector<float>{2.0f, 6.0f});

  test.Run();
}

// With bias (optional 4th input)
TEST(ContribOpTest, ConvTransposeWithDynamicPads_WithBias) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 1});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  // X: {N=1, C_in=1, H=2, W=2}
  test.AddInput<float>("X", {1, 1, 2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  // W: {C_in=1, C_out/group=1, kH=1, kW=1}
  test.AddInput<float>("W", {1, 1, 1, 1}, std::vector<float>{1.0f});
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  // B: {C_out=1}
  test.AddInput<float>("B", {1}, std::vector<float>{10.0f});
  // Output: X * W + B = X + 10
  test.AddOutput<float>("Y", {1, 1, 2, 2}, std::vector<float>{11.0f, 12.0f, 13.0f, 14.0f});

  test.Run();
}

// Dilations > 1
TEST(ContribOpTest, ConvTransposeWithDynamicPads_Dilations) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{2, 2});

  // X: {N=1, C=1, H=2, W=2}
  test.AddInput<float>("X", {1, 1, 2, 2}, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
  // W: {C_in=1, C_out/group=1, kH=2, kW=2}
  test.AddInput<float>("W", {1, 1, 2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  // Output shape: (2-1)*1 + 0 + (2-1)*2 + 1 = 4 in each dim
  // With dilation=2, effective kernel size is 3 (positions at 0 and 2)
  test.AddOutput<float>("Y", {1, 1, 4, 4}, std::vector<float>{1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.Run();
}

// Asymmetric pads (different begin/end values)
TEST(ContribOpTest, ConvTransposeWithDynamicPads_AsymmetricPads) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  // X: {N=1, C=1, H=1, W=1}
  test.AddInput<float>("X", {1, 1, 1, 1}, std::vector<float>{1.0f});
  // W: {C_in=1, C_out/group=1, kH=3, kW=3}
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  // Pads: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end] = [1, 0, 0, 1]
  // Unpadded output would be 3x3 (kernel applied to single pixel)
  // After removing 1 row from top and 1 col from right: 2x2
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{1, 0, 0, 1});
  // Output: (1-1)*1 + 0 + 3 - 1 - 0 = 2 (height), (1-1)*1 + 0 + 3 - 0 - 1 = 2 (width)
  test.AddOutput<float>("Y", {1, 1, 2, 2}, std::vector<float>{4.0f, 5.0f, 7.0f, 8.0f});
  test.Run();
}

// Security: Input channels not divisible by group should fail.
TEST(ContribOpTest, ConvTransposeWithDynamicPads_ChannelsNotDivisibleByGroup) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("group", static_cast<int64_t>(2));

  // X has 3 input channels, not divisible by group=2
  test.AddInput<float>("X", {1, 3, 3, 3}, std::vector<float>(27, 1.0f));
  test.AddInput<float>("W", {3, 1, 3, 3}, std::vector<float>(27, 1.0f));
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{0, 0, 0, 0});
  test.AddOutput<float>("Y", {1, 2, 5, 5}, std::vector<float>(50, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "Input channels is not divisible by group");
}

}  // namespace test
}  // namespace onnxruntime

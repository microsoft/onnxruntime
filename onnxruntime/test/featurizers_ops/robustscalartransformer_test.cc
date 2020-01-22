// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/RobustScalarFeaturizer.h"

namespace onnxruntime {
namespace test {

TEST(FeaturizersTests, RobustScalarTransformer_default_with_centering) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 160, 64, 0, 0, 128, 64});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float>("?2", {5}, {-1.0f,-0.5f, 0.0f, 0.5f, 1.0f});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_no_centering) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 64});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float>("?2", {5}, {0.25f, 0.75f, 1.25f, 1.75f, 2.25f});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_no_centering_zero_scale) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {3}, {10, 10, 10});

  // Expected output.
  test.AddOutput<float>("?2", {3}, {10.f, 10.f, 10.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_with_centering_no_scaling) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 160, 64, 0, 0, 128, 63});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float>("?2", {5}, {-4.f, -2.f, 0.f, 2.f, 4.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_with_centering_custom_scaling) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 160, 64, 0, 0, 0, 65});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float>("?2", {5}, {-0.5f, -0.25f, 0.f, 0.25f, 0.5f});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_no_centering_custom_scaling) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float>("?2", {5}, {0.125f, 0.375f, 0.625f, 0.875f, 1.125f});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


}
}

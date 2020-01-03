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
  test.AddInput<uint8_t>("State", {8}, {0, 0, 160, 64, 0, 0, 128, 64});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float>("?2", {5}, {-1.0,-0.5, 0.0, 0.5, 1.0});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_no_centering) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {8}, {0, 0, 0, 0, 0, 0, 128, 64});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float>("?2", {5}, {0.25, 0.75, 1.25, 1.75, 2.25});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_no_centering_zero_scale) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {8}, {0, 0, 0, 0, 0, 0, 0, 0});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {3}, {10, 10, 10});

  // Expected output.
  test.AddOutput<float>("?2", {3}, {10, 10, 10});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_with_centering_no_scaling) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {8}, {0, 0, 160, 64, 0, 0, 128, 63});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float_t>("?2", {5}, {-4, -2, 0, 2, 4});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_with_centering_custom_scaling) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {8}, {0, 0, 160, 64, 0, 0, 0, 65});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float_t>("?2", {5}, {-0.5, -0.25, 0, 0.25, 0.5});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, RobustScalarTransformer_default_no_centering_custom_scaling) {
  OpTester test("RobustScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {8}, {0, 0, 0, 0, 0, 0, 0, 65});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {5}, {1, 3, 5, 7, 9});

  // Expected output.
  test.AddOutput<float_t>("?2", {5}, {0.125, 0.375, 0.625, 0.875, 1.125});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


}
}

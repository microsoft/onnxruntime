// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/MinMaxScalarFeaturizer.h"

namespace onnxruntime {
namespace test {

TEST(FeaturizersTests, MinMaxScalarTransformer_int8) {
  OpTester test("MinMaxScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {6}, {1, 0, 0, 0, 1, 9});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {1}, {15});

  // Expected output.
  test.AddOutput<double>("?2", {1}, {1.75});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(FeaturizersTests, MinMaxScalarTransformer_float_t) {
  OpTester test("MinMaxScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 128, 191, 0, 0, 128, 63});

  // We are adding a scalar Tensor in this instance
  test.AddInput<float>("?1", {1}, {2.f});

  // Expected output.
  test.AddOutput<double>("?2", {1}, {1.5});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, MinMaxScalarTransformer_only_one_input) {
  OpTester test("MinMaxScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  test.AddInput<uint8_t>("State", {6}, {1, 0, 0, 0, 255, 255});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("?1", {1}, {2});

  // Expected output.
  test.AddOutput<double>("?2", {1}, {0});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


}
}
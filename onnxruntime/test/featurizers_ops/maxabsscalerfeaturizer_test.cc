// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/MaxAbsScalarFeaturizer.h"

namespace dft = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace test {

TEST(FeaturizersTests, MaxAbsScaler_int8_values) {

  OpTester test("MaxAbsScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  
  // State from when the transformer was trained. Corresponds to Version 1 and a 
  // scale of 0
  test.AddInput<uint8_t>("State", {8}, {1, 0, 0, 0, 0, 0, 128, 64});

  // We are adding a scalar Tensor in this instance
  test.AddInput<int8_t>("X", {5}, {-4,3,0,2,-1});

  // Expected output.
  test.AddOutput<float>("ScaledValues", {5}, {-1.f,.75f,0.f,.5f,-.25f});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, MaxAbsScaler_double_values) {
  OpTester test("MaxAbsScalarTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // State from when the transformer was trained. Corresponds to Version 1 and a
  // scale of 0
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 64});

  // We are adding a scalar Tensor in this instance
  test.AddInput<double>("X", {5}, {-4, 3, 0, 2, -1});

  // Expected output.
  test.AddOutput<double>("ScaledValues", {5}, {-1, .75, 0, .5, -.25});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime

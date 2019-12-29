// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/CatImputerFeaturizer.h"

namespace dft = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace test {

TEST(CategoryImputer, Float_values) {

  OpTester test("CatImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  
  // State from when the transformer was trained. Corresponds to a 
  // most frequent value during training of 1.5 and version of 1
  test.AddInput<uint8_t>("State", {8}, {1, 0, 0, 0, 0, 0, 192, 63});

  // We are adding a scalar Tensor in this instance
  test.AddInput<float_t>("Input", {5}, {1, std::nanf("1"), std::nanf("1"), 2, std::nanf("1")});

  // Expected output.
  test.AddOutput<float_t>("Output", {5}, {1, 1.5, 1.5, 2, 1.5});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(CategoryImputer, Double_values) {
  OpTester test("CatImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // State from when the transformer was trained. Corresponds to a
  // most frequent value during training of 1.5 and version of 1
  test.AddInput<uint8_t>("State", {12}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 63});

  // We are adding a scalar Tensor in this instance
  test.AddInput<double_t>("Input", {5}, {1, std::nan("1"), std::nan("1"), 2, std::nan("1")});

  // Expected output.
  test.AddOutput<double_t>("Output", {5}, {1, 1.5, 1.5, 2, 1.5});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(CategoryImputer, String_values) {
  OpTester test("CatImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // State from when the transformer was trained. Corresponds to a
  // most frequent value during training of "one" and version of 1
  test.AddInput<uint8_t>("State", {11}, {1, 0, 0, 0, 3, 0, 0, 0, 111, 110, 101});

  // We are adding a scalar Tensor in this instance
  test.AddInput<std::string>("Input", {5}, {"ONE", "", "FIVE", "", "NINE"});

  // Expected output.
  test.AddOutput<std::string>("Output", {5}, {"ONE", "one", "FIVE", "one", "NINE"});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime

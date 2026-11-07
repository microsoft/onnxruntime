// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime::test {
namespace {

enum class BiasInput {
  Omitted,
  EmptySlot,
  Present,
};

template <typename T>
void AddBiasInput(OpTester& test, BiasInput bias, bool initializer) {
  if (bias == BiasInput::EmptySlot) {
    test.AddOptionalInputEdge<T>();
  } else if (bias == BiasInput::Present) {
    test.AddInput<T>("B", {1}, {T(5.f)}, initializer);
  }
}

template <typename T>
void RunConvolution(OpTester& test, const std::vector<int64_t>& output_shape, float expected, bool initializer) {
  test.AddOutput<T>("Y", output_shape,
                    std::vector<T>(static_cast<size_t>(TensorShape(output_shape).Size()), T(expected)));
  test.SetCustomOutputVerifier([output_shape, expected](const std::vector<OrtValue>& fetches,
                                                        const std::string& provider_type) {
    // Record providers that produced outputs, so unsupported kernels remain distinguishable.
    ::testing::Test::RecordProperty(provider_type, "executed");
    ASSERT_EQ(fetches.size(), 1u);
    const auto& output = fetches[0].Get<Tensor>();
    ASSERT_EQ(output.Shape(), TensorShape(output_shape));
    for (const auto& value : output.DataAsSpan<T>()) {
      EXPECT_EQ(value, T(expected));
    }
  });
  std::unordered_set<std::string> excluded_providers;
  if (!initializer) {
    // Match the common Conv tests: TensorRT requires constant weights, and QNN has a dynamic-bias limitation.
    excluded_providers.insert(kTensorrtExecutionProvider);
    excluded_providers.insert(kQnnExecutionProvider);
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

void TestPointwiseConvolution(const char* op) {
  for (const size_t rank : {4u, 5u}) {
    for (const bool initializer : {false, true}) {
      for (const auto bias : {BiasInput::Omitted, BiasInput::EmptySlot, BiasInput::Present}) {
        SCOPED_TRACE(testing::Message() << "rank=" << rank << " initializer=" << initializer
                                        << " bias=" << static_cast<int>(bias));
        OpTester test(op, 11);
        const std::vector<int64_t> shape(rank, 1);
        test.AddInput<float>("X", shape, {2});
        test.AddInput<float>("W", shape, {3}, initializer);
        AddBiasInput<float>(test, bias, initializer);
        RunConvolution<float>(test, shape, bias == BiasInput::Present ? 11.f : 6.f, initializer);
      }
    }
  }
}

template <typename T>
void TestSpatialConvolution() {
  for (const bool initializer : {false, true}) {
    for (const auto bias : {BiasInput::Omitted, BiasInput::EmptySlot, BiasInput::Present}) {
      SCOPED_TRACE(testing::Message() << "initializer=" << initializer << " bias=" << static_cast<int>(bias));
      OpTester test("Conv", 11);
      test.AddInput<T>("X", {1, 1, 3, 3}, std::vector<T>(9, T(2.f)));
      test.AddInput<T>("W", {1, 1, 2, 2}, std::vector<T>(4, T(3.f)), initializer);
      AddBiasInput<T>(test, bias, initializer);
      RunConvolution<T>(test, {1, 1, 2, 2}, bias == BiasInput::Present ? 29.f : 24.f, initializer);
    }
  }
}

}  // namespace

TEST(ConvOptionalInputTest, Conv2DAnd3D) {
  TestPointwiseConvolution("Conv");
}

TEST(ConvOptionalInputTest, ConvTranspose2DAnd3D) {
  TestPointwiseConvolution("ConvTranspose");
}

TEST(ConvOptionalInputTest, ConvSpatial) {
  TestSpatialConvolution<float>();
}

TEST(ConvOptionalInputTest, ConvSpatialHalf) {
  TestSpatialConvolution<MLFloat16>();
}

TEST(ConvOptionalInputTest, MissingRequiredWeightsWithBias) {
  for (const char* op : {"Conv", "ConvTranspose"}) {
    SCOPED_TRACE(op);
    OpTester test(op, 11);
    test.AddInput<float>("X", {1, 1, 1, 1}, {2});
    test.AddOptionalInputEdge<float>();
    test.AddInput<float>("B", {1}, {5});
    test.AddOutput<float>("Y", {1, 1, 1, 1}, {0});
    // W is required by both schemas, even when the optional bias is supplied.
    test.Run(OpTester::ExpectResult::kExpectFailure, "input 1 is marked single but has an empty string");
  }
}

}  // namespace onnxruntime::test

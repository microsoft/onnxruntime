// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <random>
#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

void BERTLayerNormCore(const int opset_version) {
  // LayerNormalization was wrongly registered 
  // in ai.onnx domain since version 1. To maintain BC,
  // we keep the wrong operator as is and officially
  // propose a new one in opset 16 to hide the old one
  // for future opset.  This function should test
  // the old (since opset 1) LayerNormalization
  // and the new (since opset 16) LayerNormalization
  // because they have the same definition.
  OpTester tester("LayerNormalization", opset_version);
  tester.AddAttribute<int64_t>("axis", -1);
  tester.AddAttribute<float>("epsilon", 1e-12f);

  // create rand inputs
  RandomValueGenerator random{};

  std::vector<int64_t> X_dims{4, 128};
  std::vector<float> X_data = random.Uniform<float>(X_dims, 0.0f, 1.0f);
  tester.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> scale_dims{128};
  std::vector<float> scale_data = random.Uniform<float>(scale_dims, 0.0f, 1.0f);
  tester.AddInput<float>("Scale", scale_dims, scale_data);

  std::vector<int64_t> B_dims{128};
  std::vector<float> B_data = random.Uniform<float>(B_dims, 0.0f, 1.0f);
  tester.AddInput<float>("B", B_dims, B_data);

  tester.AddReferenceOutputs("testdata/layernorm.onnx");

  tester.Run();
}

TEST(LayerNormTest, BERTLayerNorm) {
  BERTLayerNormCore(1);
  BERTLayerNormCore(16);
}

void BERTLayerNormNoBiasCoreCore(const int opset_version) {
  OpTester tester("LayerNormalization", opset_version);
  tester.AddAttribute<int64_t>("axis", -1);
  tester.AddAttribute<float>("epsilon", 1e-12f);

  // create rand inputs
  RandomValueGenerator random{};

  std::vector<int64_t> X_dims{4, 128};
  std::vector<float> X_data = random.Uniform<float>(X_dims, 0.0f, 1.0f);
  tester.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> scale_dims{128};
  std::vector<float> scale_data = random.Uniform<float>(scale_dims, 0.0f, 1.0f);
  tester.AddInput<float>("Scale", scale_dims, scale_data);

  tester.AddOptionalInputEdge<float>();

  tester.AddReferenceOutputs("testdata/layernorm_no_bias.onnx");

  tester.Run();
}

TEST(LayerNormTest, BERTLayerNorm_NoBias) {
  BERTLayerNormNoBiasCoreCore(1);
  BERTLayerNormNoBiasCoreCore(16);
}

void LayerNormCore(const int opset_version) {
  OpTester test("LayerNormalization", opset_version);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<float>("x", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("gamma", {3}, {1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("output", dims, {-1.2247f, 0.0f, 1.2247f, -1.2247f, 0.0f, 1.2247f});
  test.Run();
}

TEST(LayerNormTest, LayerNorm) {
  LayerNormCore(1);
  LayerNormCore(16);
}

void LayerNormScaleCore(const int opset_version) {
  OpTester test("LayerNormalization", opset_version);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("x", dims, {-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddOutput<float>("output", dims, {0.6953f, 5.1824f, -0.6953f, -5.1824f, 0.6953f, 5.1824f, -0.6953f, -5.1824f});
  test.Run();
}

TEST(LayerNormTest, LayerNorm_Scale) {
  LayerNormScaleCore(1);
  LayerNormScaleCore(16);
}

void LayerNormScaleBiasCore(const int opset_version) {
  OpTester test("LayerNormalization", opset_version);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<float>("x", dims, {1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f});
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddInput<float>("bias", {2}, {0.6435f, -0.3964f});
  test.AddOutput<float>("output", dims, {-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f});
  test.Run();
}

TEST(LayerNormTest, LayerNorm_Scale_Bias) {
  LayerNormScaleBiasCore(1);
  LayerNormScaleBiasCore(16);
}

}  // namespace test
}  // namespace onnxruntime

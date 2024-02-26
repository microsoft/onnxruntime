// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <random>
#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

TEST(LayerNormTest, BERTLayerNorm) {
  OpTester tester("LayerNormalization", 17 /*opset_version*/);
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

TEST(LayerNormTest, BERTLayerNorm_NoBias) {
  OpTester tester("LayerNormalization", 17 /*opset_version*/);
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

  tester.AddReferenceOutputs("testdata/layernorm_no_bias.onnx");

  tester.Run();
}

TEST(LayerNormTest, LayerNorm) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<float>("x", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("gamma", {3}, {1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("output", dims, {-1.2247f, 0.0f, 1.2247f, -1.2247f, 0.0f, 1.2247f});
  test.Run();
}

TEST(LayerNormTest, LayerNorm_BFloat16Input) {
// prevents test from running on non-BF16-supporting hardware
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<BFloat16>("x", dims, MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  test.AddInput<BFloat16>("gamma", {3}, MakeBFloat16({1.0f, 1.0f, 1.0f}));
  test.AddOutput<BFloat16>("output", dims, MakeBFloat16({-1.2247f, 0.0f, 1.2247f, -1.2247f, 0.0f, 1.2247f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_Scale) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("x", dims, {-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddOutput<float>("output", dims, {0.6953f, 5.1824f, -0.6953f, -5.1824f, 0.6953f, 5.1824f, -0.6953f, -5.1824f});
  test.Run();
}

TEST(LayerNormTest, LayerNorm_Scale_Float16Input) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddOutput<float>("output", dims, {0.6953f, 5.1824f, -0.6953f, -5.1824f, 0.6953f, 5.1824f, -0.6953f, -5.1824f});
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_Scale_Float16ScaleOutput) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("x", dims, {-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({0.6953f, 5.1824f, -0.6953f, -5.1824f, 0.6953f, 5.1824f, -0.6953f, -5.1824f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_Scale_Float16InputScaleOutput) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({0.6953f, 5.1824f, -0.6953f, -5.1824f, 0.6953f, 5.1824f, -0.6953f, -5.1824f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_Scale_Bias) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<float>("x", dims, {1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f});
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddInput<float>("bias", {2}, {0.6435f, -0.3964f});
  test.AddOutput<float>("output", dims, {-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f});
  test.Run();
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Float16Input) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f}));
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddInput<float>("bias", {2}, {0.6435f, -0.3964f});
  test.AddOutput<float>("output", dims, {-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f});
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kQnnExecutionProvider,
            kOpenVINOExecutionProvider, kNnapiExecutionProvider, kCoreMLExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Float16ScaleBiasOutput) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<float>("x", dims, {1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f});
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}));
  test.AddInput<MLFloat16>("bias", {2}, ToFloat16({0.6435f, -0.3964f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Float16InputScaleBiasOutput) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f}));
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}));
  test.AddInput<MLFloat16>("bias", {2}, ToFloat16({0.6435f, -0.3964f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

// LayerNormalization became an ONNX operator in opset 17. It uses the same implementation so this is a sanity check.
TEST(LayerNormTest, LayerNorm17_float) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<float>("x", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("gamma", {3}, {1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("output", dims, {-1.2247f, 0.0f, 1.2247f, -1.2247f, 0.0f, 1.2247f});
  test.Run();
}

TEST(LayerNormTest, LayerNorm17_double) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<double>("x", dims, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  test.AddInput<double>("gamma", {3}, {1.0, 1.0, 1.0});
  test.AddOutput<double>("output", dims, {-1.2247, 0.0, 1.2247, -1.2247, 0.0, 1.2247});
  // DNNL does not support double
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDnnlExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_InvalidScaleBias) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  // as axis is 1, the scale and bias should have size 6
  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<float>("x", dims, {1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f});
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddInput<float>("bias", {2}, {0.6435f, -0.3964f});
  test.AddAttribute<int64_t>("axis", 1);
  test.AddOutput<float>("output", dims, {-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f});
  // CPU and CUDA EPs have check for unexpected scale or bias sizes. Exclude other EPs with a LayerNormalization
  // implementation for which we don't control the check or error message.
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Size of X.shape()[axis:] == 6. Size of scale and bias (if provided) must match this",
           {kDnnlExecutionProvider, kDmlExecutionProvider, kTensorrtExecutionProvider});
}

#if defined(USE_DNNL)
TEST(LayerNormTest, LayerNorm17_Scale_Bias_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<BFloat16>("x", dims, MakeBFloat16({1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f}));
  test.AddInput<BFloat16>("gamma", {2}, MakeBFloat16({-0.6953f, 5.1824f}));
  test.AddInput<BFloat16>("bias", {2}, MakeBFloat16({0.6435f, -0.3964f}));
  test.AddOutput<BFloat16>("output", dims, MakeBFloat16({-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f}));
  test.Run();
}
#endif  //  USE_DNNL
}  // namespace test
}  // namespace onnxruntime

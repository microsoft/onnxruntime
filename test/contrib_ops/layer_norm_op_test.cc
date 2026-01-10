// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <random>
#include "core/framework/tensor.h"
#include "core/providers/cpu/nn/layer_norm_helper.h"
#include "core/session/inference_session.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

// Some feature (like broadcast support) are implemented in CPU and CUDA provider only. A helper to run tests.
void RunTestOnCpuAndCuda(OpTester& test, const std::string& expected_failure_msg = "") {
  auto expected_result = expected_failure_msg.empty()
                             ? OpTester::ExpectResult::kExpectSuccess
                             : OpTester::ExpectResult::kExpectFailure;

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_provider;
  cpu_execution_provider.push_back(DefaultCpuExecutionProvider());
  test.Run(expected_result, expected_failure_msg, {}, nullptr, &cpu_execution_provider);

  constexpr int min_cuda_architecture = 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);

  if (enable_cuda) {
    std::vector<std::unique_ptr<IExecutionProvider>> gpu_execution_provider;
    if (enable_cuda) {
      gpu_execution_provider.push_back(DefaultCudaExecutionProvider());
    }

    if (gpu_execution_provider.size() > 0) {
      test.Run(expected_result, expected_failure_msg, {}, nullptr, &gpu_execution_provider);
    }
  }
}

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
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider, kWebGpuExecutionProvider});
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
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider, kWebGpuExecutionProvider});
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

TEST(LayerNormTest, LayerNorm_Scale_Float16InputScaleOutput_Initializers) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}), true);
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({0.6953f, 5.1824f, -0.6953f, -5.1824f, 0.6953f, 5.1824f, -0.6953f, -5.1824f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_Scale_Bias) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<float>("x", dims, {1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f});
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddInput<float>("bias", {2}, {0.6435f, -0.3964f});
  test.AddOutput<float>("output", dims, {-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f});
  test.SetOutputTolerance(0.0001f);
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
  test.SetOutputTolerance(0.0001f);

  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kQnnExecutionProvider,
            kOpenVINOExecutionProvider, kNnapiExecutionProvider, kCoreMLExecutionProvider, kWebGpuExecutionProvider});
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
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider, kWebGpuExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_NoBroadcast) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("x", dims, {-1.0f, 2.0f, 3.0f, -4.0f, -10.264f, 8.6453f, 43.1561f, -0.641239f});
  test.AddInput<float>("gamma", {2, 2, 2}, {-0.1f, 1.7f, -0.6953f, 5.1824f, -0.1f, 1.7f, -0.6953f, 5.1824f});
  test.AddInput<float>("bias", {2, 2, 2}, {-2.0f, 0.3f, 0.0f, 0.0f, -2.0f, 0.3f, 0.0f, 0.0f});
  test.AddOutput<float>("output", dims, {-1.9f, 2.0f, -0.6953f, -5.1824f, -1.9f, 2.0f, -0.6953f, -5.1824f});

  test.SetOutputTolerance(0.0001f);

  RunTestOnCpuAndCuda(test);
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_NoBroadcast_Fp16) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-1.0f, 2.0f, 3.0f, -4.0f, -10.264f, 8.6453f, 43.1561f, -0.641239f}));
  test.AddInput<MLFloat16>("gamma", {2, 2, 2}, ToFloat16({-0.1f, 1.7f, -0.6953f, 5.1824f, -0.1f, 1.7f, -0.6953f, 5.1824f}));
  test.AddInput<MLFloat16>("bias", {2, 2, 2}, ToFloat16({-2.0f, 0.3f, 0.0f, 0.0f, -2.0f, 0.3f, 0.0f, 0.0f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({-1.9f, 2.0f, -0.6953f, -5.1824f, -1.9f, 2.0f, -0.6953f, -5.1824f}));

  RunTestOnCpuAndCuda(test);
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Broadcast_Dim0) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{4, 2, 2};
  test.AddInput<float>("x", dims, {-1.0f, 2.0f, -10.264f, 8.6453f, 3.0f, -4.0f, 43.1561f, -0.641239f, -5.0f, 6.0f, -8.2164f, 0.11412f, 7.0f, 8.0f, 41.3156f, 3.0458f});
  test.AddInput<float>("gamma", {1, 2, 2}, {-0.1f, 1.7f, -0.6953f, 5.1824f});
  test.AddInput<float>("bias", {1, 2, 2}, {-2.0f, 0.3f, 0.0f, 0.0f});
  test.AddOutput<float>("output", dims, {-1.9f, 2.0f, 0.6953f, 5.1824f, -2.1f, -1.4f, -0.6953f, -5.1824f, -1.9f, 2.0f, 0.6953f, 5.1824f, -1.9f, 2.0f, -0.6953f, -5.1824f});
  test.SetOutputTolerance(0.0001f);

  RunTestOnCpuAndCuda(test);
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Broadcast_Dim0_Fp16) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{4, 2, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-1.0f, 2.0f, -10.264f, 8.6453f, 3.0f, -4.0f, 43.1561f, -0.641239f, -5.0f, 6.0f, -8.2164f, 0.11412f, 7.0f, 8.0f, 41.3156f, 3.0458f}));
  test.AddInput<MLFloat16>("gamma", {1, 2, 2}, ToFloat16({-0.1f, 1.7f, -0.6953f, 5.1824f}));
  test.AddInput<MLFloat16>("bias", {1, 2, 2}, ToFloat16({-2.0f, 0.3f, 0.0f, 0.0f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({-1.9f, 2.0f, 0.6953f, 5.1824f, -2.1f, -1.4f, -0.6953f, -5.1824f, -1.9f, 2.0f, 0.6953f, 5.1824f, -1.9f, 2.0f, -0.6953f, -5.1824f}));

  RunTestOnCpuAndCuda(test);
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Broadcast_Dim1) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 4, 2};
  test.AddInput<float>("x", dims, {-1.0f, 2.0f, 3.0f, -4.0f, -5.0f, 6.0f, 7.0f, 8.0f, -10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  test.AddInput<float>("gamma", {2, 1, 2}, {-0.1f, 1.7f, -0.6953f, 5.1824f});
  test.AddInput<float>("bias", {2, 1, 2}, {-2.0f, 0.3f, 0.0f, 0.0f});
  test.AddOutput<float>("output", dims, {-1.9f, 2.0f, -2.1f, -1.4f, -1.9f, 2.0f, -1.9f, 2.0f, 0.6953f, 5.1824f, -0.6953f, -5.1824f, 0.6953f, 5.1824f, -0.6953f, -5.1824f});
  test.SetOutputTolerance(0.0001f);

  RunTestOnCpuAndCuda(test);
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Broadcast_Dim1_Fp16) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 4, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-1.0f, 2.0f, 3.0f, -4.0f, -5.0f, 6.0f, 7.0f, 8.0f, -10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  test.AddInput<MLFloat16>("gamma", {2, 1, 2}, ToFloat16({-0.1f, 1.7f, -0.6953f, 5.1824f}));
  test.AddInput<MLFloat16>("bias", {2, 1, 2}, ToFloat16({-2.0f, 0.3f, 0.0f, 0.0f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({-1.9f, 2.0f, -2.1f, -1.4f, -1.9f, 2.0f, -1.9f, 2.0f, 0.6953f, 5.1824f, -0.6953f, -5.1824f, 0.6953f, 5.1824f, -0.6953f, -5.1824f}));

  RunTestOnCpuAndCuda(test);
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Broadcast_Fp16) {
  auto run_test = [](bool is_initializer) {
    OpTester test("LayerNormalization");
    test.AddAttribute<float>("epsilon", 1e-05f);

    std::vector<int64_t> dims{1, 3, 2};
    test.AddInput<MLFloat16>("x", dims, ToFloat16({1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f}));
    test.AddInput<MLFloat16>("gamma", {1, 1, 2}, ToFloat16({-0.6953f, 5.1824f}), is_initializer);
    test.AddInput<MLFloat16>("bias", {1, 1, 2}, ToFloat16({0.6435f, -0.3964f}), is_initializer);
    test.AddOutput<MLFloat16>("output", dims, ToFloat16({-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f}));

    RunTestOnCpuAndCuda(test);
  };

  run_test(false);
  run_test(true);
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Float16InputScaleBiasOutput) {
  auto run_test = [](bool is_initializer) {
    OpTester test("LayerNormalization");
    test.AddAttribute<float>("epsilon", 1e-05f);

    std::vector<int64_t> dims{1, 3, 2};
    test.AddInput<MLFloat16>("x", dims, ToFloat16({1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f}));
    test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}), is_initializer);
    test.AddInput<MLFloat16>("bias", {2}, ToFloat16({0.6435f, -0.3964f}), is_initializer);
    test.AddOutput<MLFloat16>("output", dims, ToFloat16({-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f}));
    // TRT, DNNL, OpenVINO and NNAPI don't support this combination of datatypes
    test.Run(OpTester::ExpectResult::kExpectSuccess, "",
             {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
              kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider, kWebGpuExecutionProvider});
  };
  run_test(false);
  run_test(true);
}

template <typename T>
class LayerNormTest : public ::testing::Test {
};

using LayerNormTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(LayerNormTest, LayerNormTestTypes);

TEST(LayerNormTest, LayerNorm_Scale_Bias_Float16InputScaleBiasOutput_Initializers) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f}));
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}), true);
  test.AddInput<MLFloat16>("bias", {2}, ToFloat16({0.6435f, -0.3964f}), true);
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({-0.0516f, -5.5776f, -0.0518f, -5.5788f, -0.0518f, -5.5788f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

// LayerNormalization became an ONNX operator in opset 17. It uses the same implementation so this is a sanity check.
TYPED_TEST(LayerNormTest, LayerNorm17_opset) {
  auto run_test = [](bool is_initializer) {
    OpTester test("LayerNormalization", 17);
    test.AddAttribute<float>("epsilon", 1e-05f);

    std::vector<int64_t> dims{1, 2, 3};
    test.AddInput<TypeParam>("x", dims, GetTypedArray<TypeParam>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    test.AddInput<TypeParam>("gamma", {3}, GetTypedArray<TypeParam>({1.0f, 1.0f, 1.0f}), is_initializer);
    test.AddOutput<TypeParam>("output", dims, GetTypedArray<TypeParam>({-1.2247f, 0.0f, 1.2247f, -1.2247f, 0.0f, 1.2247f}));
    if (std::is_same<TypeParam, MLFloat16>::value) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCoreMLExecutionProvider(true));
      // coreml EP requires weight and bias to be initializers
      test.Run(OpTester::ExpectResult::kExpectSuccess, "",
               {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
                kNnapiExecutionProvider, kQnnExecutionProvider},
               nullptr, &execution_providers);
    } else {
      test.Run();
    }
  };
  // Execution provider entry invalid.
  // when other EPs support layer-norm fp16, this test should be updated to include them.
  if (std::is_same<TypeParam, MLFloat16>::value) {
#if !defined(USE_COREML)
    return;
#endif
  }

  run_test(false);
  run_test(true);
}

TEST(LayerNormTest, LayerNorm17_double) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<double>("x", dims, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  test.AddInput<double>("gamma", {3}, {1.0, 1.0, 1.0});
  test.AddOutput<double>("output", dims, {-1.2247, 0.0, 1.2247, -1.2247, 0.0, 1.2247});

  test.SetOutputTolerance(0.0001f);

  // DNNL does not support double
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDnnlExecutionProvider});
}

TEST(LayerNormTest, LayerNorm_NormSize1_Valid) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> dims{1, 3, 1};
  test.AddInput<float>("x", dims, {1.2416f, 0.946123f, 13.1685f});
  test.AddInput<float>("gamma", {1}, {-0.6953f});
  test.AddInput<float>("bias", {1}, {0.6435f});
  test.AddAttribute<int64_t>("axis", 2);
  test.AddOutput<float>("output", dims, {0.6435f, 0.6435f, 0.6435f});
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_ValidScaleBias_Broadcast) {
  OpTester test("LayerNormalization");
  test.AddAttribute<float>("epsilon", 1e-05f);

  // With axis = 1, scale and bias of shape {2} are NumPy-broadcastable
  // to X.shape[axis:] = {3, 2}, so this configuration is now valid.
  std::vector<int64_t> dims{1, 3, 2};
  test.AddInput<float>("x", dims, {1.2416f, 0.946123f, 13.1685f, 0.36423f, 21.145f, 0.03941f});
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddInput<float>("bias", {2}, {0.6435f, -0.3964f});
  test.AddAttribute<int64_t>("axis", 1);
  test.AddOutput<float>("output", dims,
                        {1.063606f, -3.716114f,
                         0.042961f, -4.087264f,
                         -0.639629f, -4.294445f});

  // This configuration used to be rejected, but with generic NumPy-style
  // broadcasting support it is now valid and should run successfully.
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_Scale_Scalar_NoBias_Axis2) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x(2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {2, 2, 2}, x);

  test.AddInput<float>("Scale", {}, {1.5f}, true);
  test.AddOutput<float>("Y", {2, 2, 2},
                        {
                            -1.5f,
                            1.5f,
                            -1.5f,
                            1.5f,
                            -1.5f,
                            1.5f,
                            -1.5f,
                            1.5f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Scalar_Axis2) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x(2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {2, 2, 2}, x);

  test.AddInput<float>("Scale", {}, {1.5f}, true);

  test.AddInput<float>("B", {}, {0.1f}, true);
  test.AddOutput<float>("Y", {2, 2, 2},
                        {
                            -1.4f,
                            1.6f,
                            -1.4f,
                            1.6f,
                            -1.4f,
                            1.6f,
                            -1.4f,
                            1.6f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_Axis2) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x(2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {2, 2, 2}, x);
  test.AddInput<float>("Scale", {2}, {1.0f, 2.0f}, true);
  test.AddInput<float>("B", {2}, {0.0f, 0.5f}, true);
  test.AddOutput<float>("Y", {2, 2, 2},
                        {
                            -1.0f,
                            2.5f,
                            -1.0f,
                            2.5f,
                            -1.0f,
                            2.5f,
                            -1.0f,
                            2.5f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_4D_OuterInnerBroadcast_Axis3) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 3);
  std::vector<float> x(1 * 2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {1, 2, 2, 2}, x);
  test.AddInput<float>("Scale", {1, 2, 1, 2},
                       {
                           1.0f,
                           1.1f,
                           1.2f,
                           1.3f,
                       },
                       true);
  test.AddInput<float>("B", {1, 2, 1, 2},
                       {
                           0.0f,
                           0.5f,
                           1.0f,
                           1.5f,
                       },
                       true);
  test.AddOutput<float>("Y", {1, 2, 2, 2},
                        {
                            -1.0f,
                            1.6f,
                            -1.0f,
                            1.6f,

                            -0.2f,
                            2.8f,
                            -0.2f,
                            2.8f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_Scale_Scalar_NoBias) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x(2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {2, 2, 2}, x);
  test.AddInput<float>("Scale", {}, {1.5f}, true);
  test.AddOutput<float>("Y", {2, 2, 2},
                        {
                            -1.5f,
                            1.5f,
                            -1.5f,
                            1.5f,
                            -1.5f,
                            1.5f,
                            -1.5f,
                            1.5f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}
TEST(LayerNormTest, LayerNorm_Scale_Bias_Scalar) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x(2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {2, 2, 2}, x);
  test.AddInput<float>("Scale", {}, {1.5f}, true);
  test.AddInput<float>("B", {}, {0.1f}, true);

  test.AddOutput<float>("Y", {2, 2, 2},
                        {
                            -1.4f,
                            1.6f,
                            -1.4f,
                            1.6f,
                            -1.4f,
                            1.6f,
                            -1.4f,
                            1.6f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_PerLastDim) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {2, 2, 2}, x);

  test.AddInput<float>("Scale", {2}, {1.0f, 2.0f}, true);

  test.AddInput<float>("B", {2}, {0.0f, 0.5f}, true);

  test.AddOutput<float>("Y", {2, 2, 2},
                        {
                            -1.0f,
                            2.5f,
                            -1.0f,
                            2.5f,
                            -1.0f,
                            2.5f,
                            -1.0f,
                            2.5f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_Scale_Bias_4D_OuterInnerBroadcast) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 3);

  std::vector<float> x(1 * 2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {1, 2, 2, 2}, x);

  test.AddInput<float>("Scale", {1, 2, 1, 2},
                       {
                           1.0f,
                           1.1f,
                           1.2f,
                           1.3f,
                       },
                       true);

  test.AddInput<float>("B", {1, 2, 1, 2},
                       {
                           0.0f,
                           0.5f,
                           1.0f,
                           1.5f,
                       },
                       true);
  test.AddOutput<float>("Y", {1, 2, 2, 2},
                        {
                            -1.0f,
                            1.6f,
                            -1.0f,
                            1.6f,

                            -0.2f,
                            2.8f,
                            -0.2f,
                            2.8f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}
TEST(LayerNormTest, LayerNorm_NormSize1_NoBias) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<int64_t>("axis", 2);
  test.AddAttribute<float>("epsilon", 1e-5f);

  std::vector<float> x = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f};
  test.AddInput<float>("X", {2, 3, 1}, x);
  test.AddInput<float>("Scale", {1}, {1.0f});
  std::vector<float> expected = {
      0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f};

  test.AddOutput<float>("Y", {2, 3, 1}, expected);
  test.AddOutput<float>("Mean", {2, 3, 1},
                        {1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f});

  float inv_std = 1.0f / sqrtf(1e-5f);
  test.AddOutput<float>("InvStdDev", {2, 3, 1},
                        {inv_std, inv_std, inv_std,
                         inv_std, inv_std, inv_std});

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}
TEST(LayerNormTest, LayerNorm_NormSize1_WithBiasScale) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<int64_t>("axis", 2);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddInput<float>("X", {1, 2, 1}, {10.0f, 20.0f});
  test.AddInput<float>("Scale", {1}, {2.0f});
  test.AddInput<float>("Bias", {1}, {5.0f});
  test.AddOutput<float>("Y", {1, 2, 1}, {5.0f, 5.0f});
  test.AddOutput<float>("Mean", {1, 2, 1}, {10.0f, 20.0f});
  float inv_std = 1.0f / sqrtf(1e-5f);
  test.AddOutput<float>("InvStdDev", {1, 2, 1}, {inv_std, inv_std});
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(LayerNormTest, LayerNorm_Scale_Broadcast_Inner_Mixed) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 1);
  std::vector<int64_t> dims{1, 2, 4};
  std::vector<float> x = {
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f};
  test.AddInput<float>("X", dims, x);
  std::vector<float> scale = {1.0f, 0.5f, 1.0f, 0.5f};
  test.AddInput<float>("Scale", {1, 4}, scale);
  std::vector<float> expected_y = {
      -1.527524f, -0.545544f, -0.654653f, -0.109109f,
      0.218218f, 0.327327f, 1.091088f, 0.763762f};
  test.AddOutput<float>("Y", dims, expected_y);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
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

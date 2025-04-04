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
#include "test/framework/test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

TEST(RMSNormTest, RMSNorm) {
  OpTester test("RMSNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<float>("x", dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("gamma", {3}, {1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("output", dims, {0.4629f, 0.9258f, 1.3887f, 0.7895f, 0.9869f, 1.1843f});
  test.Run();
}

TEST(RMSNormTest, RMSNorm_BFloat16Input) {
// prevents test from running on non-BF16-supporting hardware
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  OpTester test("RMSNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{1, 2, 3};
  test.AddInput<BFloat16>("x", dims, MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  test.AddInput<BFloat16>("gamma", {3}, MakeBFloat16({1.0f, 1.0f, 1.0f}));
  test.AddOutput<BFloat16>("output", dims, MakeBFloat16({0.4629f, 0.9258f, 1.3887f, 0.7895f, 0.9869f, 1.1843f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

TEST(RMSNormTest, RMSNorm_Scale) {
  OpTester test("RMSNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("x", dims, {-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddOutput<float>("output", dims, {0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f});
  test.Run();
}

TEST(RMSNormTest, RMSNorm_Scale_Float16Input) {
  OpTester test("RMSNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  test.AddInput<float>("gamma", {2}, {-0.6953f, 5.1824f});
  test.AddOutput<float>("output", dims, {0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f});
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider, kWebGpuExecutionProvider});
}

TEST(RMSNormTest, RMSNorm_Scale_Float16ScaleOutput) {
  OpTester test("RMSNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("x", dims, {-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider, kWebGpuExecutionProvider});
}

TEST(RMSNormTest, RMSNorm_Scale_Float16InputScaleOutput) {
  OpTester test("RMSNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}));
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

TEST(RMSNormTest, RMSNorm_Scale_Float16InputScaleOutput_Initializers) {
  OpTester test("RMSNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-05f);

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<MLFloat16>("x", dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  test.AddInput<MLFloat16>("gamma", {2}, ToFloat16({-0.6953f, 5.1824f}), true);
  test.AddOutput<MLFloat16>("output", dims, ToFloat16({0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime

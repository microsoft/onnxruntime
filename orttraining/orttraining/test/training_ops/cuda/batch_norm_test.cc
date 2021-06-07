// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

TEST(BatchNormInternalTest, ForwardTrainingTest) {
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{2, 2, 2, 2};
  std::vector<int64_t> channel_dims{2};
  test.AddInput<float>("X", input_output_dims, {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f, -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f});
  test.AddInput<float>("scale", channel_dims, {1.0f, 1.0f});
  test.AddInput<float>("B", channel_dims, {0.0f, 0.0f});
  test.AddInput<float>("mean", channel_dims, {1.0f, 2.0f});
  test.AddInput<float>("var", channel_dims, {1.0f, 2.0f});

  test.AddOutput<float>("Y", input_output_dims, {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f, -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f});

  test.AddOutput<float>("running_mean", channel_dims, {-0.1754f, 0.303106f});
  test.AddOutput<float>("running_var", channel_dims, {0.696052f, 1.41316f});
  test.AddOutput<float>("saved_mean", channel_dims, {-0.306f, 0.114562f});
  test.AddOutput<float>("saved_inv_std", channel_dims, {1.2288f, 0.861317f});

  // exclude CUDA Execution Provider due to flakiness
  // exclude TRT and OpenVINO for same reasons as seen in TestBatchNorm()
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(BatchNormInternalTest, SimpleShape5Training) {
  OpTester test("BatchNormalization", 9, kOnnxDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{1, 1, 3};
  std::vector<int64_t> channel_dims{1};
  test.AddInput<float>("X", input_output_dims, {-1.0f, 0.0f, 1.0f});
  test.AddInput<float>("scale", channel_dims, {1.0f});
  test.AddInput<float>("B", channel_dims, {-1.0f});
  test.AddInput<float>("mean", channel_dims, {0.0f});
  test.AddInput<float>("var", channel_dims, {1.0f});

  test.AddOutput<float>("Y", input_output_dims, {-2.2247f, -1.0f, 0.2247f});

  test.AddOutput<float>("running_mean", channel_dims, {0.0f});
  test.AddOutput<float>("running_var", channel_dims, {1.0f});
  // mean and variance of X across channel dimension
  // With Opset9 we output saved_inv_std instead of saved_var to match CUDA EP
  test.AddOutput<float>("saved_mean", channel_dims, {0.0f});
  test.AddOutput<float>("saved_inv_std", channel_dims, {1.2247f});

  // exclude CUDA Execution Provider due to flakiness
  // exclude TRT and OpenVINO for same reasons as seen in TestBatchNorm()
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(BatchNormInternalTest, SimpleShape5Training16) {
  OpTester test("BatchNormalization", 9, kOnnxDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{1, 1, 1, 1, 3};
  std::vector<int64_t> channel_dims{1};

  std::vector<float> X = {-1.0f, 0.0f, 1.0f};
  std::vector<float> scale = {1.0f};
  std::vector<float> B = {-1.0f};
  std::vector<float> mean = {3.0f};
  std::vector<float> var = {4.0f};

  std::vector<MLFloat16> X_half(3);
  std::vector<MLFloat16> scale_half(1);
  std::vector<MLFloat16> B_half(1);
  std::vector<MLFloat16> mean_half(1);
  std::vector<MLFloat16> var_half(1);

  ConvertFloatToMLFloat16(X.data(), X_half.data(), 3);
  ConvertFloatToMLFloat16(scale.data(), scale_half.data(), 1);
  ConvertFloatToMLFloat16(B.data(), B_half.data(), 1);
  ConvertFloatToMLFloat16(mean.data(), mean_half.data(), 1);
  ConvertFloatToMLFloat16(var.data(), var_half.data(), 1);

  test.AddInput<MLFloat16>("X", input_output_dims, X_half);
  test.AddInput<MLFloat16>("scale", channel_dims, scale_half);
  test.AddInput<MLFloat16>("B", channel_dims, B_half);
  test.AddInput<MLFloat16>("mean", channel_dims, mean_half);
  test.AddInput<MLFloat16>("var", channel_dims, var_half);

  std::vector<float> Y = {-2.2247f, -1.0f, 0.2247f};
  std::vector<float> running_mean = {2.7f};
  std::vector<float> running_var = {3.7f};
  // std::vector<float> Y = {0.3933f, 3.699, 7.0078f};
  // std::vector<float> running_mean = {0.0f};
  // std::vector<float> running_var = {0.0f};
  std::vector<float> saved_mean = {0.0f};
  std::vector<float> saved_inv_std = {1.2247f};
  // std::vector<float> saved_inv_std = {2.0f};

  std::vector<MLFloat16> Y_half(3);
  std::vector<MLFloat16> running_mean_half(1);
  std::vector<MLFloat16> running_var_half(1);
  std::vector<MLFloat16> saved_mean_half(1);
  std::vector<MLFloat16> saved_inv_std_half(1);

  ConvertFloatToMLFloat16(Y.data(), Y_half.data(), 3);
  ConvertFloatToMLFloat16(running_mean.data(), running_mean_half.data(), 1);
  ConvertFloatToMLFloat16(running_var.data(), running_var_half.data(), 1);
  ConvertFloatToMLFloat16(saved_mean.data(), saved_mean_half.data(), 1);
  ConvertFloatToMLFloat16(saved_inv_std.data(), saved_inv_std_half.data(), 1);

  test.AddOutput<MLFloat16>("Y", input_output_dims, Y_half);
  test.AddOutput<MLFloat16>("running_mean", channel_dims, running_mean_half);
  test.AddOutput<MLFloat16>("running_var", channel_dims, running_var_half);
  test.AddOutput<MLFloat16>("saved_mean", channel_dims, saved_mean_half);
  test.AddOutput<MLFloat16>("saved_inv_std", channel_dims, saved_inv_std_half);

  // test.AddOutput<MLFloat16>("Y", input_output_dims, {-2.2247f, -1.0f, 0.2247f});

  // test.AddOutput<MLFloat16>("running_mean", channel_dims, {2.7f});
  // test.AddOutput<MLFloat16>("running_var", channel_dims, {3.7f});
  // mean and variance of X across channel dimension
  // With Opset9 we output saved_inv_std instead of saved_var to match CUDA EP
  // test.AddOutput<MLFloat16>("saved_mean", channel_dims, {0.0f});
  // test.AddOutput<MLFloat16>("saved_inv_std", channel_dims, {1.2247f});

  // exclude CUDA Execution Provider due to flakiness
  // exclude TRT and OpenVINO for same reasons as seen in TestBatchNorm()
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(BatchNormInternalTest, SimpleShape6Training) {
  OpTester test("BatchNormInternal", 1, kMSDomain);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{1, 1, 1, 1, 1, 2};
  std::vector<int64_t> channel_dims{1};
  test.AddInput<float>("X", input_output_dims, {-0.999f, 1.0f});
  test.AddInput<float>("scale", channel_dims, {1.0f});
  test.AddInput<float>("B", channel_dims, {-1.0f});
  test.AddInput<float>("mean", channel_dims, {0.0f});
  test.AddInput<float>("var", channel_dims, {1.0f});

  test.AddOutput<float>("Y", input_output_dims, {-2.0f, 0.0f});

  test.AddOutput<float>("running_mean", channel_dims, {0.0f});
  test.AddOutput<float>("running_var", channel_dims, {1.0f});
  // mean and variance of X across channel dimension
  // With Opset9 we output saved_inv_std instead of saved_var to match CUDA EP
  test.AddOutput<float>("saved_mean", channel_dims, {0.0f});
  test.AddOutput<float>("saved_inv_std", channel_dims, {1.0f});

  // exclude CUDA Execution Provider due to flakiness
  // exclude TRT and OpenVINO for same reasons as seen in TestBatchNorm()
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

}  // namespace test
} // namespace contrib
}  // namespace onnxruntime

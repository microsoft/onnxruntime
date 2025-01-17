// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

struct ScaleInputOutput {
  ScaleInputOutput() {
    input_half.resize(input_float.size());
    output_up_half.resize(output_up_float.size());
    output_down_half.resize(output_down_float.size());
    scale_half.resize(scale_float.size());

    input_bf16.resize(input_float.size());
    output_up_bf16.resize(output_up_float.size());
    output_down_bf16.resize(output_down_float.size());
    scale_bf16.resize(scale_float.size());

    ConvertFloatToMLFloat16(input_float.data(), input_half.data(), int(input_float.size()));
    ConvertFloatToMLFloat16(output_up_float.data(), output_up_half.data(), int(output_up_float.size()));
    ConvertFloatToMLFloat16(output_down_float.data(), output_down_half.data(), int(output_down_float.size()));
    ConvertFloatToMLFloat16(scale_float.data(), scale_half.data(), int(scale_float.size()));

    input_bf16 = FloatsToBFloat16s(input_float);
    output_up_bf16 = FloatsToBFloat16s(output_up_float);
    output_down_bf16 = FloatsToBFloat16s(output_down_float);
    scale_bf16 = FloatsToBFloat16s(scale_float);
  }

  // Fp32 Inputs/Output
  std::vector<float> scale_float = {2.0f};
  std::vector<int64_t> scale_int64 = {2LL};
  std::vector<int32_t> scale_int32 = {2};
  std::vector<double> scale_double = {2.0};
  std::vector<float> input_float = {1.0f, 2.0f, 3.0f};
  std::vector<double> input_double = {1.0, 2.0, 3.0};
  std::vector<float> output_up_float = {2.0f, 4.0f, 6.0f};
  std::vector<float> output_down_float = {0.5f, 1.0f, 1.5f};
  std::vector<double> output_up_double = {2.0, 4.0, 6.0};
  std::vector<double> output_down_double = {0.5, 1.0, 1.5};

  // Fp16 Inputs/Outputs
  std::vector<MLFloat16> input_half;
  std::vector<MLFloat16> output_up_half;
  std::vector<MLFloat16> output_down_half;
  std::vector<MLFloat16> scale_half;

  // BFloat16 Inputs/Output
  std::vector<BFloat16> input_bf16;
  std::vector<BFloat16> output_up_bf16;
  std::vector<BFloat16> output_down_bf16;
  std::vector<BFloat16> scale_bf16;
};

TEST(CudaKernelTest, ScaleFloatFloatScaleUp) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {3}, data.input_float);
  test.AddInput<float>("scale", {1}, data.scale_float);
  test.AddOutput<float>("output", {3}, data.output_up_float);
  test.Run();
}

TEST(CudaKernelTest, ScaleDoubleInt32ScaleUp) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddInput<double>("input", {3}, data.input_double);
  test.AddInput<int32_t>("scale", {1}, data.scale_int32);
  test.AddOutput<double>("output", {3}, data.output_up_double);
  test.Run();
}

TEST(CudaKernelTest, ScaleHalfHalfScaleUp) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("scale_down", int64_t(0));
  test.AddInput<MLFloat16>("input", {3}, data.input_half);
  test.AddInput<MLFloat16>("scale", {1}, data.scale_half);
  test.AddOutput<MLFloat16>("output", {3}, data.output_up_half);
  test.Run();
}

TEST(CudaKernelTest, ScaleHalfInt64ScaleUp) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("scale_down", int64_t(0));
  test.AddInput<MLFloat16>("input", {3}, data.input_half);
  test.AddInput<int64_t>("scale", {1}, data.scale_int64);
  test.AddOutput<MLFloat16>("output", {3}, data.output_up_half);
  test.Run();
}

TEST(CudaKernelTest, ScaleFloatDoubleScaleDown) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("scale_down", int64_t(1));
  test.AddInput<float>("input", {3}, data.input_float);
  test.AddInput<double>("scale", {1}, data.scale_double);
  test.AddOutput<float>("output", {3}, data.output_down_float);
  test.Run();
}

TEST(CudaKernelTest, ScaleDoubleInt64ScaleDown) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("scale_down", int64_t(1));
  test.AddInput<double>("input", {3}, data.input_double);
  test.AddInput<int64_t>("scale", {1}, data.scale_int64);
  test.AddOutput<double>("output", {3}, data.output_down_double);
  test.Run();
}

TEST(CudaKernelTest, ScaleHalfHalfScaleDown) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("scale_down", int64_t(1));
  test.AddInput<MLFloat16>("input", {3}, data.input_half);
  test.AddInput<MLFloat16>("scale", {1}, data.scale_half);
  test.AddOutput<MLFloat16>("output", {3}, data.output_down_half);
  test.Run();
}

TEST(CudaKernelTest, ScaleHalfInt64ScaleDown) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("scale_down", int64_t(1));
  test.AddInput<MLFloat16>("input", {3}, data.input_half);
  test.AddInput<int64_t>("scale", {1}, data.scale_int64);
  test.AddOutput<MLFloat16>("output", {3}, data.output_down_half);
  test.Run();
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(CudaKernelTest, ScaleBFloat16BFloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware does not support BFP16";
    return;
  }
#endif
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddInput<BFloat16>("input", {3}, data.input_bf16);
  test.AddInput<BFloat16>("scale", {1}, data.scale_bf16);
  test.AddOutput<BFloat16>("output", {3}, data.output_up_bf16);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(CudaKernelTest, ScaleFloatBFloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware does not support BFP16";
    return;
  }
#endif
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {3}, data.input_float);
  test.AddInput<BFloat16>("scale", {1}, data.scale_bf16);
  test.AddOutput<float>("output", {3}, data.output_up_float);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif
}  // namespace test
}  // namespace onnxruntime

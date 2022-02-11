// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

struct MixedPrecisionScaleInputOutput {
  MixedPrecisionScaleInputOutput() {
    input1_half.resize(input1.size());
    output1_half.resize(output1.size());
    ConvertFloatToMLFloat16(input1.data(), input1_half.data(), int(input1.size()));
    ConvertFloatToMLFloat16(output1.data(), output1_half.data(), int(output1.size()));

    input2_half.resize(input2.size());
    output2_half.resize(output2.size());
    ConvertFloatToMLFloat16(input2.data(), input2_half.data(), int(input2.size()));
    ConvertFloatToMLFloat16(output2.data(), output2_half.data(), int(output2.size()));

    input1_bf16.resize(input1.size());
    output1_bf16.resize(output1.size());
    input1_bf16 = FloatsToBFloat16s(input1);
    output1_bf16 = FloatsToBFloat16s(output1);

    input2_bf16.resize(input2.size());
    output2_bf16.resize(output2.size());
    input2_bf16 = FloatsToBFloat16s(input2);
    output2_bf16 = FloatsToBFloat16s(output2);
  }

  // Fp32 Inputs/Output
  std::vector<float> scale = {0.1f};
  std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
  std::vector<float> input2 = {4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> output1 = {0.1f, 0.2f, 0.3f};
  std::vector<float> output2 = {0.4f, 0.5f, 0.6f, 0.7f};

  // Fp16 Inputs/Outputs
  std::vector<MLFloat16> input1_half;
  std::vector<MLFloat16> input2_half;
  std::vector<MLFloat16> output1_half;
  std::vector<MLFloat16> output2_half;

  // BF16 Inputs/Output
  std::vector<BFloat16> input1_bf16;
  std::vector<BFloat16> input2_bf16;
  std::vector<BFloat16> output1_bf16;
  std::vector<BFloat16> output2_bf16;
};

TEST(CudaKernelTest, MixedPrecisionScaleF2F) {
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<float>("input1", {3}, data.input1);
  test.AddOutput<float>("output1", {3}, data.output1);
  test.Run();
}

TEST(CudaKernelTest, MixedPrecisionScaleF2F_MultiInputs) {
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<float>("input1", {3}, data.input1);
  test.AddInput<float>("input2", {4}, data.input2);
  test.AddOutput<float>("output1", {3}, data.output1);
  test.AddOutput<float>("output2", {4}, data.output2);
  test.Run();
}

TEST(CudaKernelTest, MixedPrecisionScaleF2H) {
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<float>("input1", {3}, data.input1);
  test.AddOutput<MLFloat16>("output1", {3}, data.output1_half);
  test.Run();
}

TEST(CudaKernelTest, MixedPrecisionScaleF2H_MultiInput_FuseOutput) {
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  test.AddAttribute("fuse_outputs", int64_t(1));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<float>("input1", {3}, data.input1);
  test.AddInput<float>("input2", {4}, data.input2);

  std::vector<MLFloat16> output;
  output.insert(output.end(), data.output1_half.begin(), data.output1_half.end());
  output.insert(output.end(), data.output2_half.begin(), data.output2_half.end());
  test.AddOutput<MLFloat16>("output", {7}, output);
  test.Run();
}

TEST(CudaKernelTest, MixedPrecisionScaleH2F) {
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<MLFloat16>("input1", {3}, data.input1_half);
  test.AddOutput<float>("output1", {3}, data.output1);
  test.Run();
}

TEST(CudaKernelTest, MixedPrecisionScaleH2F_MultiInputs) {
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<MLFloat16>("input1", {3}, data.input1_half);
  test.AddInput<MLFloat16>("input2", {4}, data.input2_half);
  test.AddOutput<float>("output1", {3}, data.output1);
  test.AddOutput<float>("output2", {4}, data.output2);
  test.Run();
}

TEST(CudaKernelTest, MixedPrecisionScaleH2F_MultiInput_FuseOutput) {
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  test.AddAttribute("fuse_outputs", int64_t(1));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<MLFloat16>("input1", {3}, data.input1_half);
  test.AddInput<MLFloat16>("input2", {4}, data.input2_half);

  std::vector<float> output;
  output.insert(output.end(), data.output1.begin(), data.output1.end());
  output.insert(output.end(), data.output2.begin(), data.output2.end());
  test.AddOutput<float>("output", {7}, output);
  test.Run();
}

TEST(CudaKernelTest, MixedPrecisionScaleH2H) {
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<MLFloat16>("input1", {3}, data.input1_half);
  test.AddOutput<MLFloat16>("output1", {3}, data.output1_half);
  test.Run();
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(CudaKernelTest, MixedPrecisionScale_bfloat16_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<BFloat16>("input1", {3}, data.input1_bf16);
  test.AddOutput<BFloat16>("output1", {3}, data.output1_bf16);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif 
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(CudaKernelTest, MixedPrecisionScale_float_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<float>("input1", {3}, data.input1);
  test.AddOutput<BFloat16>("output1", {3}, data.output1_bf16);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif 
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(CudaKernelTest, MixedPrecisionScale_bfloat16_float) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<BFloat16>("input1", {3}, data.input1_bf16);
  test.AddOutput<float>("output1", {3}, data.output1);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif 
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(CudaKernelTest, MixedPrecisionScale_half_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<MLFloat16>("input1", {3}, data.input1_half);
  test.AddOutput<BFloat16>("output1", {3}, data.output1_bf16);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif 
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(CudaKernelTest, MixedPrecisionScale_bfloat16_half) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  MixedPrecisionScaleInputOutput data;
  OpTester test("MixedPrecisionScale", 1, onnxruntime::kMSDomain);
  test.AddAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  test.AddInput<float>("scale", {1}, data.scale);
  test.AddInput<BFloat16>("input1", {3}, data.input1_bf16);
  test.AddOutput<MLFloat16>("output1", {3}, data.output1_half);

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
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>

#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

TEST(MathOpTest, AffineDefaultAttributes) {
  OpTester test("Affine");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Affine) {
  OpTester test("Affine");
  std::vector<int64_t> dims{2, 2};
  test.AddAttribute("alpha", 2.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {1.0f, 3.0f, 5.0f, 7.0f});
  test.Run();
}

TEST(MathOpTest, Scale) {
  OpTester test("Scale");
  std::vector<int64_t> dims{2, 2};
  test.AddAttribute("scale", 2.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 2.0f, 4.0f, 6.0f});
  test.Run();
}

TEST(MathOpTest, Scale_Default) {
  OpTester test("Scale");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

std::vector<float> Add_Simple(const std::vector<float>& input_a_data, const std::vector<float>& input_b_data) {
  EXPECT_TRUE(input_a_data.size() % input_b_data.size() == 0 || input_b_data.size() % input_a_data.size() == 0);
  const std::vector<float>& input_large_size = input_a_data.size() >= input_b_data.size() ? input_a_data : input_b_data;
  const std::vector<float>& input_small_size = input_a_data.size() < input_b_data.size() ? input_a_data : input_b_data;

  std::vector<float> output(input_large_size.size());
  for (size_t iter = 0; iter < input_large_size.size() / input_small_size.size(); iter++) {
    std::transform(input_large_size.begin() + iter * input_small_size.size(),
                   input_large_size.begin() + (iter + 1) * input_small_size.size(),
                   input_small_size.begin(),
                   output.begin() + iter * input_small_size.size(),
                   [](float a, float b) {
                     return a + b;
                   });
  }
  return output;
}

const std::vector<float> ComputeGeluWithErf(const std::vector<float>& input_data) {
  std::vector<float> output(input_data.size());

  std::transform(input_data.begin(),
                 input_data.end(),
                 output.begin(),
                 [](float x) {
                   float y = erf(x * static_cast<float>(M_SQRT1_2));
                   return x * 0.5f * (y + 1.0f);
                 });

  return output;
}

static void RunBiasGeluTestFloat(
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& bias_dims) {
  RandomValueGenerator random{2333};
  std::vector<float> input_data = random.Uniform<float>(input_dims, -1.0f, 1.0f);
  std::vector<float> bias_data = random.Uniform<float>(bias_dims, -1.0f, 1.0f);
  std::vector<float> output_data = ComputeGeluWithErf(Add_Simple(input_data, bias_data));

  OpTester tester("BiasGelu", 1, onnxruntime::kMSDomain);
  tester.AddInput<float>("A", input_dims, input_data);
  tester.AddInput<float>("B", bias_dims, bias_data);
  tester.AddOutput<float>("C", input_dims, output_data);
  tester.Run();
}

TEST(BiasGeluTest, Float) {
  RunBiasGeluTestFloat({2, 4}, {4});
  RunBiasGeluTestFloat({3, 7}, {7});
  RunBiasGeluTestFloat({2, 4, 512}, {512});
  RunBiasGeluTestFloat({2, 3, 333}, {333});
  RunBiasGeluTestFloat({2, 2048}, {2048});
  RunBiasGeluTestFloat({2, 2333}, {2333});
}

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)
static void RunBiasGeluTestHalf(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& bias_dims) {
  RandomValueGenerator random{2333};
  std::vector<float> input_data = random.Uniform<float>(input_dims, -1.0f, 1.0f);
  std::vector<float> bias_data = random.Uniform<float>(bias_dims, -1.0f, 1.0f);
  std::vector<float> output_data = ComputeGeluWithErf(Add_Simple(input_data, bias_data));
  std::vector<MLFloat16> input_data_half(input_data.size());
  std::vector<MLFloat16> bias_data_half(bias_data.size());
  std::vector<MLFloat16> output_data_half(output_data.size());
  ConvertFloatToMLFloat16(input_data.data(), input_data_half.data(), input_data.size());
  ConvertFloatToMLFloat16(bias_data.data(), bias_data_half.data(), bias_data.size());
  ConvertFloatToMLFloat16(output_data.data(), output_data_half.data(), output_data.size());

  OpTester tester("BiasGelu", 1, onnxruntime::kMSDomain);
  tester.AddInput<MLFloat16>("A", input_dims, input_data_half);
  tester.AddInput<MLFloat16>("B", bias_dims, bias_data_half);
  tester.AddOutput<MLFloat16>("C", input_dims, output_data_half);
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "",
             {kTensorrtExecutionProvider});  // TensorRT: fp16 is not supported
}

TEST(BiasGeluTest, MLFloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  RunBiasGeluTestHalf({2, 4}, {4});
  RunBiasGeluTestHalf({3, 7}, {7});
  RunBiasGeluTestHalf({2, 4, 512}, {512});
  RunBiasGeluTestHalf({2, 3, 333}, {333});
  RunBiasGeluTestHalf({2, 2048}, {2048});
  RunBiasGeluTestHalf({2, 2333}, {2333});
}
#endif

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DNNL)
static void RunBiasGeluTestBFloat16(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& bias_dims) {
  RandomValueGenerator random{2333};
  std::vector<float> input_data = random.Uniform<float>(input_dims, 0.5f, 1.5f);
  std::vector<float> bias_data = random.Uniform<float>(bias_dims, 0.5f, 1.5f);
  std::vector<float> output_data = ComputeGeluWithErf(Add_Simple(input_data, bias_data));
  std::vector<BFloat16> input_data_bf16 = FloatsToBFloat16s(input_data);
  std::vector<BFloat16> bias_data_bf16 = FloatsToBFloat16s(bias_data);
  std::vector<BFloat16> output_data_bf16 = FloatsToBFloat16s(output_data);

  OpTester tester("BiasGelu", 1, onnxruntime::kMSDomain);
  tester.AddInput<BFloat16>("A", input_dims, input_data_bf16);
  tester.AddInput<BFloat16>("B", bias_dims, bias_data_bf16);
  tester.AddOutput<BFloat16>("C", input_dims, output_data_bf16);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#elif USE_DML
  execution_providers.push_back(DefaultDmlExecutionProvider());
#endif
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(BiasGeluTest, BFloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  RunBiasGeluTestBFloat16({2, 4}, {4});
  RunBiasGeluTestBFloat16({3, 7}, {7});
  RunBiasGeluTestBFloat16({2, 4, 512}, {512});
  RunBiasGeluTestBFloat16({2, 3, 333}, {333});
  RunBiasGeluTestBFloat16({2, 2048}, {2048});
  RunBiasGeluTestBFloat16({2, 2333}, {2333});
}
#endif

TEST(MathOpTest, ComplexMul) {
  if (DefaultCudaExecutionProvider() == nullptr) return;

  std::vector<float> input_a_data = {
      -0.5f, 0.6f};

  std::vector<float> input_b_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> output_data = {
      -0.10f, 0.73f,
      -0.60f, -0.50f,
      -0.37f, 0.20f,
      0.21f, 0.48f};

  OpTester tester("ComplexMul", 1, onnxruntime::kMSDomain);
  tester.AddInput<float>("A", {1, 2}, input_a_data);
  tester.AddInput<float>("B", {4, 2}, input_b_data);
  tester.AddOutput<float>("C", {4, 2}, output_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, ComplexMulConj) {
  if (DefaultCudaExecutionProvider() == nullptr) return;

  std::vector<float> input_a_data = {
      -0.5f, 0.6f};

  std::vector<float> input_b_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> output_data = {
      -0.70f, 0.23f,
      0.60f, 0.50f,
      -0.13f, 0.40f,
      -0.51f, -0.12f};

  OpTester tester("ComplexMulConj", 1, onnxruntime::kMSDomain);
  tester.AddInput<float>("A", {1, 2}, input_a_data);
  tester.AddInput<float>("B", {4, 2}, input_b_data);
  tester.AddOutput<float>("C", {4, 2}, output_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, ComplexMul_fp16) {
  if (DefaultCudaExecutionProvider() == nullptr) return;

  std::vector<MLFloat16> input_a_data = {
      MLFloat16(-0.5f), MLFloat16(0.6f)};

  std::vector<MLFloat16> input_b_data = {
      MLFloat16(0.8f), MLFloat16(-0.5f), MLFloat16(0.0f), MLFloat16(1.f),
      MLFloat16(0.5f), MLFloat16(0.2f), MLFloat16(0.3f), MLFloat16(-0.6f)};

  std::vector<MLFloat16> output_data = {
      MLFloat16(-0.10f), MLFloat16(0.73f),
      MLFloat16(-0.60f), MLFloat16(-0.50f),
      MLFloat16(-0.37f), MLFloat16(0.20f),
      MLFloat16(0.21f), MLFloat16(0.48f)};

  OpTester tester("ComplexMul", 1, onnxruntime::kMSDomain);
  tester.AddInput<MLFloat16>("A", {1, 2}, input_a_data);
  tester.AddInput<MLFloat16>("B", {4, 2}, input_b_data);
  tester.AddOutput<MLFloat16>("C", {4, 2}, output_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, ComplexMulConj_fp16) {
  if (DefaultCudaExecutionProvider() == nullptr) return;

  std::vector<MLFloat16> input_a_data = {
      MLFloat16(-0.5f), MLFloat16(0.6f)};

  std::vector<MLFloat16> input_b_data = {
      MLFloat16(0.8f), MLFloat16(-0.5f), MLFloat16(0.0f), MLFloat16(1.f),
      MLFloat16(0.5f), MLFloat16(0.2f), MLFloat16(0.3f), MLFloat16(-0.6f)};

  std::vector<MLFloat16> output_data = {
      MLFloat16(-0.70f), MLFloat16(0.23f),
      MLFloat16(0.60f), MLFloat16(0.50f),
      MLFloat16(-0.13f), MLFloat16(0.40f),
      MLFloat16(-0.51f), MLFloat16(-0.12f)};

  OpTester tester("ComplexMulConj", 1, onnxruntime::kMSDomain);
  tester.AddInput<MLFloat16>("A", {1, 2}, input_a_data);
  tester.AddInput<MLFloat16>("B", {4, 2}, input_b_data);
  tester.AddOutput<MLFloat16>("C", {4, 2}, output_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime

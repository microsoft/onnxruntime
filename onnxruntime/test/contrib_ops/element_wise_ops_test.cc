// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>

#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "test/common/cuda_op_test_utils.h"
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

static void RunBiasGeluTest(
    const std::vector<float>& input_a_data,
    const std::vector<float>& input_b_data,
    const std::vector<int64_t>& input_a_dims,
    const std::vector<int64_t>& input_b_dims) {
  std::vector<float> output_data = ComputeGeluWithErf(Add_Simple(input_a_data, input_b_data));

  OpTester tester("BiasGelu", 1, onnxruntime::kMSDomain);

  const std::vector<int64_t>& output_dims = input_a_dims.size() >= input_b_dims.size() ? input_a_dims : input_b_dims;
  tester.AddInput<float>("A", input_a_dims, input_a_data);
  tester.AddInput<float>("B", input_b_dims, input_b_data);
  tester.AddOutput<float>("C", output_dims, output_data);

  tester.Run();
}

TEST(BiasGeluTest, Two_One_Dim) {
  std::vector<float> input_a_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> input_b_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  RunBiasGeluTest(input_a_data, input_b_data, {2, 4}, {4});
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(BiasGeluTest, Two_One_Dim_fp16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester tester("BiasGelu", 1, onnxruntime::kMSDomain);

  std::vector<float> A = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> B = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> Y = ComputeGeluWithErf(Add_Simple(A, B));

  std::vector<MLFloat16> f_A(8);
  std::vector<MLFloat16> f_B(4);
  std::vector<MLFloat16> f_Y(8);
  ConvertFloatToMLFloat16(A.data(), f_A.data(), 8);
  ConvertFloatToMLFloat16(B.data(), f_B.data(), 4);
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 8);

  tester.AddInput<MLFloat16>("A", {2, 4}, f_A);
  tester.AddInput<MLFloat16>("B", {4}, f_B);
  tester.AddOutput<MLFloat16>("Y", {2, 4}, f_Y);
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: fp16 is not supported
}
#endif

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(BiasGeluTest, Two_One_Dim_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  OpTester tester("BiasGelu", 1, onnxruntime::kMSDomain);

  std::vector<float> A = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> B = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> Y = ComputeGeluWithErf(Add_Simple(A, B));

  std::vector<BFloat16> f_A = FloatsToBFloat16s(A);
  std::vector<BFloat16> f_B = FloatsToBFloat16s(B);
  std::vector<BFloat16> f_Y = FloatsToBFloat16s(Y);

  tester.AddInput<BFloat16>("A", {2, 4}, f_A);
  tester.AddInput<BFloat16>("B", {4}, f_B);
  tester.AddOutput<BFloat16>("Y", {2, 4}, f_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif 
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
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
        MLFloat16(math::floatToHalf(-0.5f)), MLFloat16(math::floatToHalf(0.6f))};

  std::vector<MLFloat16> input_b_data = {
        MLFloat16(math::floatToHalf(0.8f)), MLFloat16(math::floatToHalf(-0.5f)), MLFloat16(math::floatToHalf(0.0f)), MLFloat16(math::floatToHalf(1.f)),
        MLFloat16(math::floatToHalf(0.5f)), MLFloat16(math::floatToHalf(0.2f)), MLFloat16(math::floatToHalf(0.3f)), MLFloat16(math::floatToHalf(-0.6f))};

  std::vector<MLFloat16> output_data = {
        MLFloat16(math::floatToHalf(-0.10f)), MLFloat16(math::floatToHalf(0.73f)),
        MLFloat16(math::floatToHalf(-0.60f)), MLFloat16(math::floatToHalf(-0.50f)),
        MLFloat16(math::floatToHalf(-0.37f)), MLFloat16(math::floatToHalf(0.20f)),
        MLFloat16(math::floatToHalf(0.21f)), MLFloat16(math::floatToHalf(0.48f))};

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
        MLFloat16(math::floatToHalf(-0.5f)), MLFloat16(math::floatToHalf(0.6f))};

  std::vector<MLFloat16> input_b_data = {
        MLFloat16(math::floatToHalf(0.8f)), MLFloat16(math::floatToHalf(-0.5f)), MLFloat16(math::floatToHalf(0.0f)), MLFloat16(math::floatToHalf(1.f)),
        MLFloat16(math::floatToHalf(0.5f)), MLFloat16(math::floatToHalf(0.2f)), MLFloat16(math::floatToHalf(0.3f)), MLFloat16(math::floatToHalf(-0.6f))};

  std::vector<MLFloat16> output_data = {
        MLFloat16(math::floatToHalf(-0.70f)), MLFloat16(math::floatToHalf(0.23f)),
        MLFloat16(math::floatToHalf(0.60f)), MLFloat16(math::floatToHalf(0.50f)),
        MLFloat16(math::floatToHalf(-0.13f)), MLFloat16(math::floatToHalf(0.40f)),
        MLFloat16(math::floatToHalf(-0.51f)), MLFloat16(math::floatToHalf(-0.12f))};

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

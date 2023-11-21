// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000

TEST(GemmFloat8OpTest, BFloat16) {
  OpTester test("GemmFloat8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddAttribute("activation", "NONE");
  test.AddAttribute("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16));
  test.AddInput<BFloat16>("A", {2, 4}, MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<BFloat16>("B", {4, 3}, MakeBFloat16({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}));
  test.AddInput<BFloat16>("C", {2, 3}, MakeBFloat16({1.f, 1.f, 1.f, 1.f, 1.f, 1.f}));
  test.AddOutput<BFloat16>("Y", {2, 3}, MakeBFloat16({11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(GemmFloat8OpTest, Float) {
  OpTester test("GemmFloat8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddAttribute("activation", "NONE");
  test.AddAttribute("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  test.AddInput<float>("A", {2, 4}, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<float>("B", {4, 3}, std::vector<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}));
  test.AddInput<float>("C", {2, 3}, std::vector<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f}));
  test.AddOutput<float>("Y", {2, 3}, std::vector<float>({11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

std::vector<MLFloat16> _Cvt(const std::vector<float>& tensor) {
  std::vector<MLFloat16> fp16_data(tensor.size());
  ConvertFloatToMLFloat16(tensor.data(), fp16_data.data(), static_cast<int>(tensor.size()));
  return fp16_data;
}

TEST(GemmFloat8OpTest, Float16) {
  OpTester test("GemmFloat8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddAttribute("activation", "NONE");
  test.AddAttribute("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
  test.AddInput<MLFloat16>("A", {2, 4}, _Cvt(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f})));
  test.AddInput<MLFloat16>("B", {4, 3}, _Cvt(std::vector<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
  test.AddInput<MLFloat16>("C", {2, 3}, _Cvt(std::vector<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
  test.AddOutput<MLFloat16>("Y", {2, 3}, _Cvt(std::vector<float>({11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f})));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#if (!defined(DISABLE_FLOAT8_TYPES)) && (CUDA_VERSION >= 12000)

template <typename T>
std::vector<T> _TypedCvt(const std::vector<float>& tensor);

template <>
std::vector<float> _TypedCvt(const std::vector<float>& tensor) {
  return tensor;
}

template <>
std::vector<Float8E4M3FN> _TypedCvt(const std::vector<float>& tensor) {
  std::vector<Float8E4M3FN> out(tensor.size());
  for (size_t i = 0; i < tensor.size(); ++i) {
    out[i] = Float8E4M3FN(tensor[i]);
  }
  return out;
}

template <typename ab_type, typename out_type>
void TestGemmFloat8WithFloat8(int64_t dtype) {
  int min_cuda_architecture = 11080;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support Matrix Multiplication for FLOAT8";
    return;
  }
  OpTester test("GemmFloat8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddAttribute("activation", "NONE");
  test.AddAttribute("dtype", dtype);
  test.AddInput<ab_type>("A", {2, 4}, _TypeCvt<ap_type>(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f})));
  test.AddInput<ab_type>("B", {3, 4}, _TypeCvt<ap_type>(std::vector<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
  test.AddInput<out_type>("C", {2, 3}, _TypeCvt<out_type>(std::vector<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
  test.AddOutput<MLFloat16>("Y", {2, 3}, _TypeCvt<out_type>(std::vector<float>({11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f})));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(GemmFloat8OpTest, Float8E4M3FNToFloat) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, float>(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
}

TEST(GemmFloat8OpTest, Float8E4M3FNToFloat8E4M3FN) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, Float8E4M3FN>(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN));
}

#endif

#endif

}  // namespace test
}  // namespace onnxruntime

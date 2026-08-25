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

TEST(GemmFloat8OpTest, FloatWithFloat16CFails) {
  OpTester test("GemmFloat8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("transA", int64_t{0});
  test.AddAttribute("transB", int64_t{0});
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddAttribute("activation", "NONE");
  test.AddAttribute("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  test.AddInput<float>("A", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  test.AddInput<MLFloat16>("C", {2, 3}, _Cvt({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddOutput<float>("Y", {2, 3}, {11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Non-FP8 output requires input C and output Y to have the same type.",
           {}, nullptr, &execution_providers);
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
std::vector<MLFloat16> _TypedCvt(const std::vector<float>& tensor) {
  return _Cvt(tensor);
}

template <>
std::vector<Float8E4M3FN> _TypedCvt(const std::vector<float>& tensor) {
  std::vector<Float8E4M3FN> out(tensor.size());
  for (size_t i = 0; i < tensor.size(); ++i) {
    out[i] = Float8E4M3FN(tensor[i]);
  }
  return out;
}

template <typename ab_type, typename c_type, typename out_type>
void TestGemmFloat8WithFloat8(int64_t dtype, bool has_scales = false,
                              bool expect_failure = false, bool has_c = true) {
  constexpr int min_cuda_architecture = 890;
  constexpr int64_t M = 16;
  constexpr int64_t N = 16;
  constexpr int64_t K = 16;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    GTEST_SKIP() << "Hardware does NOT support Matrix Multiplication for FLOAT8";
  }

  std::vector<float> a_data(M * K);
  std::vector<float> b_data(N * K, 0.0f);
  std::vector<float> c_data(M * N, 1.0f);
  std::vector<float> y_data(M * N);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t k = 0; k < K; ++k) {
      a_data[static_cast<size_t>(m * K + k)] = static_cast<float>((m + k) % 5 - 2);
    }
  }
  for (int64_t n = 0; n < N; ++n) {
    b_data[static_cast<size_t>(n * K + n)] = 1.0f;
  }
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      y_data[static_cast<size_t>(m * N + n)] =
          a_data[static_cast<size_t>(m * K + n)] + (has_c ? 1.0f : 0.0f);
    }
  }

  OpTester test("GemmFloat8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddAttribute("activation", "NONE");
  test.AddAttribute("dtype", dtype);
  test.AddInput<ab_type>("A", {M, K}, _TypedCvt<ab_type>(a_data));
  test.AddInput<ab_type>("B", {N, K}, _TypedCvt<ab_type>(b_data));
  if (has_c) {
    test.AddInput<c_type>("C", {M, N}, _TypedCvt<c_type>(c_data));
  } else {
    test.AddOptionalInputEdge<c_type>();
  }
  if (has_scales) {
    test.AddInput<float>("scaleA", {1}, {1.0f});
    test.AddInput<float>("scaleB", {1}, {1.0f});
    test.AddInput<float>("scaleY", {1}, {1.0f});
  }
  test.AddOutput<out_type>("Y", {M, N}, _TypedCvt<out_type>(y_data));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(expect_failure ? OpTester::ExpectResult::kExpectFailure
                          : OpTester::ExpectResult::kExpectSuccess,
           expect_failure ? "FP8 output requires input C to be FLOAT16 or BFLOAT16." : "",
           {}, nullptr, &execution_providers);
}

TEST(GemmFloat8OpTest, Float8E4M3FNToFloat) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, float, float>(
      static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
}

TEST(GemmFloat8OpTest, Float8E4M3FNToFloat8E4M3FN) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, MLFloat16, Float8E4M3FN>(
      static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN));
}

TEST(GemmFloat8OpTest, ScaledFloat8E4M3FNWithFloat16CToFloat8E4M3FN) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, MLFloat16, Float8E4M3FN>(
      static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN), true);
}

TEST(GemmFloat8OpTest, Float8E4M3FNWithFloatCToFloat8E4M3FNFails) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, float, Float8E4M3FN>(
      static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN), false, true);
}

TEST(GemmFloat8OpTest, ScaledFloat8E4M3FNToFloat8E4M3FNWithoutC) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, MLFloat16, Float8E4M3FN>(
      static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN),
      true, false, false);
}

#endif

#endif

}  // namespace test
}  // namespace onnxruntime

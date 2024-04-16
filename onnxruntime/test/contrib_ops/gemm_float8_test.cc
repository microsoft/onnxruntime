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
#endif

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename T>
std::vector<T> _TypeCvt(const std::vector<float>& tensor);

template <>
std::vector<float> _TypeCvt(const std::vector<float>& tensor) {
  return tensor;
}

template <>
std::vector<Float8E4M3FN> _TypeCvt(const std::vector<float>& tensor) {
  std::vector<Float8E4M3FN> out(tensor.size());
  for (size_t i = 0; i < tensor.size(); ++i) {
    out[i] = Float8E4M3FN(tensor[i]);
  }
  return out;
}

template <>
std::vector<MLFloat16> _TypeCvt(const std::vector<float>& tensor) {
  std::vector<MLFloat16> out(tensor.size());
  for (size_t i = 0; i < tensor.size(); ++i) {
    out[i] = MLFloat16(tensor[i]);
  }
  return out;
}

void ReferenceGemm(  // note row majored
    bool trans_a, bool trans_b,
    int m, int n, int k,
    float alpha, const float* a, const float* b,
    float beta, const float* c, float* d) {
  // strides
  int sax = trans_a ? 1 : k;
  int say = trans_a ? m : 1;
  int sbx = trans_b ? 1 : n;
  int sby = trans_b ? k : 1;
  int scx = n;
  int scy = 1;

#define A(x, y) a[sax * (x) + say * (y)]
#define B(x, y) b[sbx * (x) + sby * (y)]
#define C(x, y) c[scx * (x) + scy * (y)]
#define D(x, y) d[scx * (x) + scy * (y)]
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) {
        acc += A(i, p) * B(p, j);
      }
      D(i, j) = alpha * acc + (beta != 0.0f ? beta * C(i, j) : 0.0f);
    }
  }
#undef A
#undef B
#undef C
#undef D
}

std::vector<float> GetMatrix(int x, int y) {
  std::vector<float> candidate_values{-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f};

  std::random_device r;
  std::mt19937 rng{r()};
  std::uniform_int_distribution<int> uniform_dist(0, static_cast<int>(candidate_values.size()));
  std::vector<float> ret(x * y);
  for (int i = 0; i < x * y; i++) {
    ret[i] = candidate_values[uniform_dist(rng)];
  }
  return ret;
}

template <typename a_type, typename b_type, typename out_type>
void TestGemmFloat8WithFloat8(int64_t dtype, bool trans_a = false, bool trans_b = true, float alpha = 1.0f, float beta = 1.0f) {
#if defined(USE_CUDA)
  int min_cuda_architecture = 11080;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    GTEST_SKIP_("Hardware NOT support Matrix Multiplication for FLOAT8");
  }
#endif
#if defined(USE_ROCM) && !defined(USE_COMPOSABLE_KERNEL)
  GTEST_SKIP_("GemmFloat8 on AMD hardware requires to build with USE_COMPOSABLE_KERNEL");
#endif
  int m = 16;
  int n = 32;
  int k = 16;

  auto a = GetMatrix(trans_a ? k : m, trans_a ? m : k);
  auto b = GetMatrix(trans_b ? n : k, trans_b ? k : n);
  auto c = GetMatrix(1, n);
  std::vector<float> c_broadcast;
  for (int i = 0; i < m; i++) {  // broadcast for ReferenceGemm
    c_broadcast.insert(c_broadcast.end(), c.begin(), c.begin() + n);
  }

  std::vector<float> d(m * n, 0);
  ReferenceGemm(trans_a, trans_b, m, n, k, alpha, a.data(), b.data(), beta, c_broadcast.data(), d.data());

  OpTester test("GemmFloat8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("transA", (int64_t)trans_a);
  test.AddAttribute("transB", (int64_t)trans_b);
  test.AddAttribute("alpha", alpha);
  test.AddAttribute("beta", beta);
  test.AddAttribute("activation", "NONE");
  test.AddAttribute("dtype", dtype);
  test.AddInput<a_type>("A", {trans_a ? k : m, trans_a ? m : k}, _TypeCvt<a_type>(a));
  test.AddInput<b_type>("B", {trans_b ? n : k, trans_b ? k : n}, _TypeCvt<b_type>(b));
  test.AddInput<out_type>("C", {1, n}, _TypeCvt<out_type>(c));
  test.AddOutput<MLFloat16>("Y", {m, n}, _TypeCvt<out_type>(d));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_CUDA)
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif defined(USE_ROCM)
  execution_providers.push_back(DefaultRocmExecutionProvider(/*test_tunable_op=*/true));
#endif
  test.SetOutputTolerance(0.2f, 0.05f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#if defined(USE_CUDA) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000
TEST(GemmFloat8OpTest, Float8E4M3FNToFloat) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, float>(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
}

TEST(GemmFloat8OpTest, Float8E4M3FNToFloat8E4M3FN) {
  TestGemmFloat8WithFloat8<Float8E4M3FN, Float8E4M3FN>(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN));
}
#endif

#if defined(USE_ROCM)
TEST(GemmFloat8OpTest, F8F16F16_NN_OnlyBias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<Float8E4M3FN, MLFloat16, MLFloat16>(output_dtype, false, false, 0.0f, 1.0f);
}

TEST(GemmFloat8OpTest, F8F16F16_NN_NoBias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<Float8E4M3FN, MLFloat16, MLFloat16>(output_dtype, false, false, 2.0f, 0.0f);
}

TEST(GemmFloat8OpTest, F8F16F16_NN_Bias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<Float8E4M3FN, MLFloat16, MLFloat16>(output_dtype, false, false, 4.0f, 1.0f);
}

TEST(GemmFloat8OpTest, F16F8F16_NN_OnlyBias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<MLFloat16, Float8E4M3FN, MLFloat16>(output_dtype, false, false, 0.0f, 1.0f);
}

TEST(GemmFloat8OpTest, F16F8F16_NN_NoBias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<MLFloat16, Float8E4M3FN, MLFloat16>(output_dtype, false, false, 2.0f, 0.0f);
}

TEST(GemmFloat8OpTest, F16F8F16_NN_Bias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<MLFloat16, Float8E4M3FN, MLFloat16>(output_dtype, false, false, 2.0f, 1.0f);
}

TEST(GemmFloat8OpTest, F16F8F16_NT_OnlyBias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<MLFloat16, Float8E4M3FN, MLFloat16>(output_dtype, false, true, 0.0f, 1.0f);
}

TEST(GemmFloat8OpTest, F16F8F16_NT_NoBias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<MLFloat16, Float8E4M3FN, MLFloat16>(output_dtype, false, true, 2.0f, 0.0f);
}

TEST(GemmFloat8OpTest, F16F8F16_NT_Bias) {
  auto output_dtype = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  TestGemmFloat8WithFloat8<MLFloat16, Float8E4M3FN, MLFloat16>(output_dtype, false, true, 2.0f, 1.0f);
}
#endif

#endif

}  // namespace test
}  // namespace onnxruntime

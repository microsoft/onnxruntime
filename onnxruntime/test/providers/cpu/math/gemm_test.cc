// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void TestGemmNoTrans() {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer = false) {
    OpTester test("Gemm");

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    test.AddInput<T>("A", {2, 4},
                     {1.0f, 2.0f, 3.0f, 4.0f,
                      -1.0f, -2.0f, -3.0f, -4.0f});
    test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f), b_is_initializer);
    test.AddInput<T>("C", {2, 3}, std::vector<T>(6, 1.0f), c_is_initializer);
    test.AddOutput<T>("Y", {2, 3},
                      {11.0f, 11.0f, 11.0f,
                       -9.0f, -9.0f, -9.0f});
    test.Run();
  };

  run_test(false, false);
  // NNAPI EP requires weight to be an initializer
  run_test(true, false);
  // CoreML EP requires weight and bias both to be initializers
  run_test(true, true);
}

TEST(GemmOpTest, GemmNoTrans_float) {
  TestGemmNoTrans<float>();
}

TEST(GemmOpTest, GemmNoTrans_double) {
  TestGemmNoTrans<double>();
}

// Only CUDA kernel has float 16 support
#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(GemmOpTest, GemmNoTrans_f16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f,
                       -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> B(12, 1.0f);
  std::vector<float> C(6, 1.0f);
  std::vector<float> Y{11.0f, 11.0f, 11.0f,
                       -9.0f, -9.0f, -9.0f};

  std::vector<MLFloat16> f_A(8);
  std::vector<MLFloat16> f_B(12);
  std::vector<MLFloat16> f_C(6);
  std::vector<MLFloat16> f_Y(6);
  ConvertFloatToMLFloat16(A.data(), f_A.data(), 8);
  ConvertFloatToMLFloat16(B.data(), f_B.data(), 12);
  ConvertFloatToMLFloat16(C.data(), f_C.data(), 6);
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 6);

  test.AddInput<MLFloat16>("A", {2, 4}, f_A);
  test.AddInput<MLFloat16>("B", {4, 3}, f_B);
  test.AddInput<MLFloat16>("C", {2, 3}, f_C);
  test.AddOutput<MLFloat16>("Y", {2, 3}, f_Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: fp16 is not supported
}
#endif

template <typename T>
void TestGemmBroadcast() {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer) {
    OpTester test("Gemm");

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    test.AddInput<T>("A", {2, 4},
                     {1.0f, 2.0f, 3.0f, 4.0f,
                      -1.0f, -2.0f, -3.0f, -4.0f});
    test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f), b_is_initializer);
    test.AddInput<T>("C", {3}, std::vector<T>{1.0f, 2.0f, 3.0f}, c_is_initializer);
    test.AddOutput<T>("Y", {2, 3},
                      {11.0f, 12.0f, 13.0f,
                       -9.0f, -8.0f, -7.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO : Temporarily disabled due to accuracy issues
#else
    test.Run();
#endif
  };

  run_test(false, false);
  // NNAPI EP requires weight to be an initializer
  run_test(true, false);
  // CoreML EP requires weight and bias both to be initializers
  run_test(true, true);
}

TEST(GemmOpTest, GemmBroadcast) {
  TestGemmBroadcast<float>();
  TestGemmBroadcast<double>();
}

template <typename T>
static void TestGemmTrans() {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)1);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<T>("A", {4, 2},
                   {1.0f, -1.0f,
                    2.0f, -2.0f,
                    3.0f, -3.0f,
                    4.0f, -4.0f});
  test.AddInput<T>("B", {3, 4}, std::vector<T>(12, 1.0f));
  test.AddInput<T>("C", {3}, std::vector<T>(3, 1.0f));
  test.AddOutput<T>("Y", {2, 3},
                    {11.0f, 11.0f, 11.0f,
                     -9.0f, -9.0f, -9.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.Run();
#endif
}

TEST(GemmOpTest, GemmTrans) {
  TestGemmTrans<float>();
  TestGemmTrans<double>();
}

// NNAPI EP's GEMM only works as A*B', add case only B is transposed
// Also test NNAPI EP's handling of non-1D bias (C of Gemm)
template <typename T>
static void TestGemmTransB() {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer = false) {
    OpTester test("Gemm");

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)1);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    test.AddInput<T>("A", {2, 4},
                     {1.0f, 2.0f, 3.0f, 4.0f,
                      -1.0f, -2.0f, -3.0f, -4.0f});
    test.AddInput<T>("B", {3, 4}, std::vector<T>(12, 1.0f), b_is_initializer);
    test.AddInput<T>("C", {1, 3}, std::vector<T>(3, 1.0f), c_is_initializer);
    test.AddOutput<T>("Y", {2, 3},
                      {11.0f, 11.0f, 11.0f,
                       -9.0f, -9.0f, -9.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
    test.Run();
#endif
  };
  run_test(false, false);
  // CoreML EP requires weight and bias both to be initializers
  run_test(true, true);
}

TEST(GemmOpTest, GemmTransB) {
  TestGemmTransB<float>();
  TestGemmTransB<double>();
}

// NNAPI EP's GEMM only works as A*B', add case only B is transposed
// Also test NNAPI EP's handling of non-1D bias (C of Gemm) which is broadcastable but not valid for NNAPI
template <typename T>
static void TestGemmTransB_1() {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer = false) {
    OpTester test("Gemm");

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)1);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    test.AddInput<T>("A", {2, 4},
                     {1.0f, 2.0f, 3.0f, 4.0f,
                      -1.0f, -2.0f, -3.0f, -4.0f});
    test.AddInput<T>("B", {3, 4}, std::vector<T>(12, 1.0f), b_is_initializer);
    test.AddInput<T>("C", {2, 1}, std::vector<T>(2, 1.0f), c_is_initializer);
    test.AddOutput<T>("Y", {2, 3},
                      {11.0f, 11.0f, 11.0f,
                       -9.0f, -9.0f, -9.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
    test.Run();
#endif
  };
  run_test(false, false);
  // CoreML EP requires weight and bias both to be initializers
  run_test(true, true);
}

TEST(GemmOpTest, GemmTransB_1) {
  TestGemmTransB_1<float>();
  TestGemmTransB_1<double>();
}

template <typename T>
void TestGemmAlphaBeta() {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 2.0f);

  test.AddInput<T>("A", {2, 4},
                   {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f));
  test.AddInput<T>("C", {3}, std::vector<T>(3, 1.0f));
  test.AddOutput<T>("Y", {2, 3},
                    {7.0f, 7.0f, 7.0f,
                     -3.0f, -3.0f, -3.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Seg fault in parser
#endif
}

TEST(GemmOpTest, GemmAlphaBeta) {
  TestGemmAlphaBeta<float>();
  TestGemmAlphaBeta<double>();
}

template <typename T>
void TestGemmNaN() {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  test.AddInput<T>("A", {2, 4},
                   {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f));
  test.AddInput<T>("C", {2, 3}, std::vector<T>(6, 1.0f));
  test.AddOutput<T>("Y", {2, 3},
                    {10.0f, 10.0f, 10.0f,
                     -10.0f, -10.0f, -10.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Seg fault in parser
}

TEST(GemmOpTest, GemmNaN) {
  TestGemmNaN<float>();
  TestGemmNaN<double>();
}

template <typename T>
void TestGemmScalarBroadcast() {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<T>("A", {2, 4},
                   {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f));
  test.AddInput<T>("C", {1}, std::vector<T>{1.0f});
  test.AddOutput<T>("Y", {2, 3},
                    {11.0f, 11.0f, 11.0f,
                     -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(GemmOpTest, GemmScalarBroadcast) {
  TestGemmScalarBroadcast<float>();
  TestGemmScalarBroadcast<double>();
}

template <typename T>
void TestGemm2DBroadcast_1() {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<T>("A", {2, 4},
                   {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f));
  test.AddInput<T>("C", {2, 1}, std::vector<T>{1.0, 2.0f});
  test.AddOutput<T>("Y", {2, 3},
                    {11.0f, 11.0f, 11.0f,
                     -8.0f, -8.0f, -8.0f});
  test.Run();
}

TEST(GemmOpTest, Gemm2DBroadcast_1) {
  TestGemm2DBroadcast_1<float>();
  TestGemm2DBroadcast_1<double>();
}

template <typename T>
void TestGemm2DBroadcast_2() {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  // Same as GemmBroadcast, but adding the unnecessary second dimension.
  test.AddInput<T>("A", {2, 4},
                   {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f));
  test.AddInput<T>("C", {1, 3}, std::vector<T>{1.0f, 2.0f, 3.0f});
  test.AddOutput<T>("Y", {2, 3},
                    {11.0f, 12.0f, 13.0f,
                     -9.0f, -8.0f, -7.0f});
  test.Run();
}

TEST(GemmOpTest, Gemm2DBroadcast_2) {
  TestGemm2DBroadcast_2<float>();
  TestGemm2DBroadcast_2<double>();
}

template <typename T>
void TestGemmFalseBroadcast() {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<T>("A", {2, 4},
                   {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f));
  test.AddInput<T>("C", {2, 3}, std::vector<T>{1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f});
  test.AddOutput<T>("Y", {2, 3},
                    {11.0f, 11.0f, 11.0f,
                     -8.0f, -8.0f, -8.0f});
  test.Run();
}

TEST(GemmOpTest, GemmFalseBroadcast) {
  TestGemmFalseBroadcast<float>();
  TestGemmFalseBroadcast<double>();
}

template <typename T>
void TestGemmEmptyTensor() {
  OpTester test("Gemm");

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<T>("A", {0, 4},
                   {});
  test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f));
  test.AddInput<T>("C", {3}, std::vector<T>(3, 1.0f));
  test.AddOutput<T>("Y", {0, 3},
                    {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});  //TensorRT: doesn't support dynamic shape yet
}

TEST(GemmOpTest, GemmEmptyTensor) {
  TestGemmEmptyTensor<float>();
  TestGemmEmptyTensor<double>();
}

template <typename T>
static void TestGemmNoBiasOpset11() {
  OpTester test("Gemm", 11);

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<T>("A", {2, 4},
                   {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<T>("B", {4, 3}, std::vector<T>(12, 1.0f));
  test.AddOutput<T>("Y", {2, 3},
                    {10.0f, 10.0f, 10.0f,
                     -10.0f, -10.0f, -10.0f});
  // tensorRT don't seem to support missing bias
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(GemmOpTest, GemmNoBiasOpset11) {
  TestGemmNoBiasOpset11<float>();
  TestGemmNoBiasOpset11<double>();
}

template <typename T>
static void TestGemmWithAlphaOpset11() {
  OpTester test("Gemm", 11);

  test.AddAttribute("alpha", 2.0f);

  test.AddInput<T>("A", {2, 2},
                   {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<T>("B", {2, 2}, std::vector<T>(4, 1.0f));
  test.AddOutput<T>("Y", {2, 2},
                    {6.0f, 6.0f, 14.0f, 14.0f});
  // tensorRT don't seem to support missing bias
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(GemmOpTest, GemmWithAlphaOpset11) {
  TestGemmWithAlphaOpset11<float>();
  TestGemmWithAlphaOpset11<double>();
}

}  // namespace test
}  // namespace onnxruntime

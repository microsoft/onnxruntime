// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

void TestGemmNoTrans(bool b_is_initializer) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f), b_is_initializer);
  test.AddInput<float>("C", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(GemmOpTest, GemmNoTrans) {
  TestGemmNoTrans(false);
}

// NNAPI EP requires weight to be an initializer
TEST(GemmOpTest, GemmNoTransBIsInitializer) {
  TestGemmNoTrans(true);
}

// Only CUDA kernel has float 16 support
#ifdef USE_CUDA
TEST(GemmOpTest, GemmNoTrans_f16) {
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
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

TEST(GemmOpTest, GemmBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>{1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 12.0f, 13.0f,
                         -9.0f, -8.0f, -7.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO : Temporarily disabled due to accuracy issues
#else
  test.Run();
#endif
}

TEST(GemmOpTest, GemmTrans) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)1);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {4, 2},
                       {1.0f, -1.0f,
                        2.0f, -2.0f,
                        3.0f, -3.0f,
                        4.0f, -4.0f});
  test.AddInput<float>("B", {3, 4}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.Run();
#endif
}

// NNAPI EP's GEMM only works as A*B', add case only B is transposed
TEST(GemmOpTest, GemmTransB) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {3, 4}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.Run();
#endif
}

TEST(GemmOpTest, GemmAlphaBeta) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 2.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {7.0f, 7.0f, 7.0f,
                         -3.0f, -3.0f, -3.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Seg fault in parser
#endif
}

TEST(GemmOpTest, GemmNaN) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Seg fault in parser
}

TEST(GemmOpTest, GemmScalarBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {1}, std::vector<float>{1.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(GemmOpTest, Gemm2DBroadcast_1) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 1}, std::vector<float>{1.0f, 2.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -8.0f, -8.0f, -8.0f});
  test.Run();
}

TEST(GemmOpTest, Gemm2DBroadcast_2) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  // Same as GemmBroadcast, but adding the unnecessary second dimension.
  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {1, 3}, std::vector<float>{1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 12.0f, 13.0f,
                         -9.0f, -8.0f, -7.0f});
  test.Run();
}

TEST(GemmOpTest, GemmFalseBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 3}, std::vector<float>{1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -8.0f, -8.0f, -8.0f});
  test.Run();
}

TEST(GemmOpTest, GemmEmptyTensor) {
  OpTester test("Gemm");

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {0, 4},
                       {});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {0, 3},
                        {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});  //TensorRT: doesn't support dynamic shape yet
}

TEST(GemmOpTest, GemmNoBiasOpset11) {
  OpTester test("Gemm", 11);

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f});
  // NGraph and tensorRT don't seem to support missing bias
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});
}

TEST(GemmOpTest, GemmWithAlphaOpset11) {
  OpTester test("Gemm", 11);

  test.AddAttribute("alpha", 2.0f);

  test.AddInput<float>("A", {2, 2},
                       {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<float>("B", {2, 2}, std::vector<float>(4, 1.0f));
  test.AddOutput<float>("Y", {2, 2},
                        {6.0f, 6.0f, 14.0f, 14.0f});
  // NGraph and tensorRT don't seem to support missing bias
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime

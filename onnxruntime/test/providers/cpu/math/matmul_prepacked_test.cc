// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/mlas/inc/mlas.h"
#include "core/optimizer/matmul_prepacking.h"

namespace onnxruntime {
namespace test {

static void AddGemmParams(const MLAS_GEMM_PARAMETERS& params, OpTester& test) {
  for (const auto& kv : GemmParamsToNodeAttributes(params)) {
    test.AddAttribute(kv.first, kv.second);
  }
}

TEST(MathOpTest, PackForGemm1) {
  OpTester test("PackForGemm", 1, kOnnxRuntimeDomain);

  MLAS_GEMM_PARAMETERS gemm_params;
  gemm_params.K = 1;
  gemm_params.N = 2;
  gemm_params.PackedStrideN = 2;
  gemm_params.PackedStrideK = 1;
  gemm_params.PackedSize = 16;

  AddGemmParams(gemm_params, test);
  test.AddInput("B", {1, 2}, std::vector<float>{23.0f, 42.0f});
  std::vector<float> PackedB(16, 0.0f);
  PackedB[0] = 23.0f;
  PackedB[1] = 42.0f;
  test.AddOutput("PackedB", {16}, PackedB);
  test.Run();
}

TEST(MathOpTest, PackForGemm2) {
  OpTester test("PackForGemm", 1, kOnnxRuntimeDomain);

  MLAS_GEMM_PARAMETERS gemm_params;
  gemm_params.PackedSize = 16;
  gemm_params.K = 1;
  gemm_params.N = 3;
  gemm_params.PackedStrideN = 16;
  gemm_params.PackedStrideK = 16;

  AddGemmParams(gemm_params, test);
  test.AddInput("B", {2, 1, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  std::vector<float> PackedB(2*16, 0.0f);
  PackedB[0] = 1;
  PackedB[1] = 2;
  PackedB[2] = 3;
  PackedB[16] = 4;
  PackedB[16+1] = 5;
  PackedB[16+2] = 6;
  test.AddOutput("PackedB", {2, 16}, PackedB);
  test.Run();
}

TEST(MathOpTest, MatMulPacked1) {
  OpTester test("MatMulPrepacked", 1, kOnnxRuntimeDomain);
  MLAS_GEMM_PARAMETERS gemm_params;
  gemm_params.K = 1;
  gemm_params.N = 1;
  gemm_params.PackedSize = 16;
  gemm_params.PackedStrideN = 16;
  gemm_params.PackedStrideK = 1;
  AddGemmParams(gemm_params, test);
  test.AddInput("A", {1, 1}, std::vector<float>{1.0f});
  test.AddInput("PackedB", {16}, std::vector<float>(16));
  test.AddOutput("Y", {1, 1}, std::vector<float>{0.0f});
  test.Run();
}

TEST(MathOpTest, MatMulPacked2) {
  OpTester test("MatMulPrepacked", 1, kOnnxRuntimeDomain);
  MLAS_GEMM_PARAMETERS gemm_params;
  gemm_params.K = 2;
  gemm_params.N = 2;
  gemm_params.PackedSize = 2*16;
  gemm_params.PackedStrideN = 16;
  gemm_params.PackedStrideK = 16;
  AddGemmParams(gemm_params, test);
  std::vector<float> A{1, 4, 9, 16};
  test.AddInput("A", {2, 2}, A);

  std::vector<float> B(2 * 16, 0.0f);
  B[0] = 1; B[1] = 2;
  B[16] = 3; B[17] = 4;
  test.AddInput("PackedB", {32}, B);

  std::vector<float> Y{1*1 + 4*3, 1*2+4*4, 9*1+16*3, 9*2+16*4};
  test.AddOutput("Y", {2, 2}, Y);

  test.Run();
}

}
}


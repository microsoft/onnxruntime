// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/trt_op_test_utils.h"
#include "core/framework/data_types.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

// Tests are split up "theme-wise" (i.e.) each kind of operation Einsum can be used for
// Within each theme we test "explicit" and "implicit" versions of the Einsum equation (wherever possible)
// Some operations are not possible with implicit notation (reordering, reduction, etc.)

// Theme: Deep copy / No-op

// Explicit
TEST(Einsum, ExplicitEinsumAsIdentity_1D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i->i");
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

// Implicit
TEST(Einsum, ImplicitEinsumAsIdentity_1D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i");
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

// Theme: Transpose/Permutation

// Explicit
TEST(Einsum, ExplicitEinsumAsTransposeOp_2D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ji->ij");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2, 2}, {1.f, 3.f, 2.f, 4.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsTransposeOp_2D_input_With_Broadcasting) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...i->i...");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2, 2}, {1.f, 3.f, 2.f, 4.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsBatchedTransposeOp_3D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ji->...ij");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2, 2, 2}, {1.f, 3.f, 2.f, 4.f, 1.f, 3.f, 2.f, 4.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

// Implicit
TEST(Einsum, ImplicitEinsumAsTransposeOp_2D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ji");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2, 2}, {1.f, 3.f, 2.f, 4.f});
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsBatchedTransposeOp_3D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ji");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2, 2, 2}, {1.f, 3.f, 2.f, 4.f, 1.f, 3.f, 2.f, 4.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

// Theme: Axis/Axes reduction

// Explicit
TEST(Einsum, ExplicitEinsumAsReduceOp_2D_input_0) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij->i");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2}, {3.f, 7.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsReduceOp_2D_input_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij->j");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2}, {4.f, 6.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsBatchedReduceOp_3D_input_0) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ji->...j");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2, 2}, {3.f, 7.f, 3.f, 7.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsBatchedReduceOp_3D_input_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ji->...");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2}, {10.f, 10.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

// Implicit
// Cannot do implicit reduction

// Theme: Outer Product

// Explicit
TEST(Einsum, ExplicitEinsumAsOuterProductOp_2D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i,j->ij");
  test.AddInput<float>("x", {2}, {1.f, 2.f});
  test.AddInput<float>("y", {2}, {3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {3.f, 4.f, 6.f, 8.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsOuterProductWithTransposeOp_2D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i,j->ji");
  test.AddInput<float>("x", {2}, {1.f, 2.f});
  test.AddInput<float>("y", {2}, {3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {3.f, 6.f, 4.f, 8.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsOuterProductWithTransposeOp_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i,j,k->jik");
  test.AddInput<float>("x", {2}, {1.f, 2.f});
  test.AddInput<float>("y", {2}, {3.f, 4.f});
  test.AddInput<float>("z", {2}, {5.f, 6.f});
  test.AddOutput<float>("o", {2, 2, 2}, {15.f, 18.f, 30.f, 36.f, 20.f, 24.f, 40.f, 48.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

// Implicit
TEST(Einsum, ImplicitEinsumAsOuterProductOp_2D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i,j,k");
  test.AddInput<float>("x", {2}, {1.f, 2.f});
  test.AddInput<float>("y", {2}, {3.f, 4.f});
  test.AddInput<float>("z", {2}, {5.f, 6.f});
  test.AddOutput<float>("o", {2, 2, 2}, {15.f, 18.f, 20.f, 24.f, 30.f, 36.f, 40.f, 48.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ImplicitEinsumAsOuterProductOp_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i,j,k");
  test.AddInput<float>("x", {2}, {1.f, 2.f});
  test.AddInput<float>("y", {2}, {3.f, 4.f});
  test.AddInput<float>("z", {2}, {5.f, 6.f});
  test.AddOutput<float>("o", {2, 2, 2}, {15.f, 18.f, 20.f, 24.f, 30.f, 36.f, 40.f, 48.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}
// Theme: MatMul

// Explicit
TEST(Einsum, ExplicitEinsumAsMatmul) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk->ik");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {7.f, 10.f, 15.f, 22.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsMatmulNhcw) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "aibj,ajbk->aibk");
  test.AddInput<float>("x", {1, 3, 1, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddInput<float>("y", {1, 2, 1, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddOutput<float>("o", {1, 3, 1, 3}, {9.f, 12.f, 15.f, 19.f, 26.f, 33.f, 29.f, 40.f, 51.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsMatmulNhcwTransposeA) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ajbi,ajbk->aibk");
  test.AddInput<float>("x", {1, 2, 1, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddInput<float>("y", {1, 2, 1, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddOutput<float>("o", {1, 3, 1, 3}, {17.f, 22.f, 27.f, 22.f, 29.f, 36.f, 27.f, 36.f, 45.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsMatmulNhcwTransposeB) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "aibj,akbj->aibk");
  test.AddInput<float>("x", {1, 3, 1, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddInput<float>("y", {1, 3, 1, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddOutput<float>("o", {1, 3, 1, 3}, {5.f, 11.f, 17.f, 11.f, 25.f, 39.f, 17.f, 39.f, 61.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsMatmulWithUpperCasedLabel) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  // 'K' != 'k' (and dim values differ too) and Einsum should handle be able to handle that
  test.AddAttribute<std::string>("equation", "iK,Kk->ik");
  test.AddInput<float>("x", {2, 1}, {1.f, 2.f});
  test.AddInput<float>("y", {1, 2}, {1.f, 2.f});
  test.AddOutput<float>("o", {2, 2}, {1.f, 2.f, 2.f, 4.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsMatmulWithUpperCasedOutputLabel) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  // Einsum should handle be able to handle upper case on both LHS and RHS
  test.AddAttribute<std::string>("equation", "Ki,ik->Kk");
  test.AddInput<float>("x", {2, 1}, {1.f, 2.f});
  test.AddInput<float>("y", {1, 2}, {1.f, 2.f});
  test.AddOutput<float>("o", {2, 2}, {1.f, 2.f, 2.f, 4.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsMatmul_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk,kl->li");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("z", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {37.f, 81.f, 54.f, 118.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsBatchedMatmul) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "bij,bjk->bik");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2, 2}, {7.f, 10.f, 15.f, 22.f, 7.f, 10.f, 15.f, 22.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsBatchedMatmulWithBroadcasting_0) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ij,...jk->...ik");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2, 2}, {7.f, 10.f, 15.f, 22.f, 7.f, 10.f, 15.f, 22.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsBatchedMatmulWithBroadcasting_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ij,bjk->...ik");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2, 2}, {14.f, 20.f, 30.f, 44.f, 14.f, 20.f, 30.f, 44.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsMatmul_OutputTransposed) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk->ki");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {7.f, 15.f, 10.f, 22.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsMatmul_2) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2068): The parameter is incorrect.";
  }

  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk->ik");
  test.AddInput<float>("x", {2, 1}, {2.f, 3.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {8.f, 12.f, 12.f, 18.f});
  test.Run();
}

// Implicit
TEST(Einsum, ImplicitEinsumAsMatmul) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {7.f, 10.f, 15.f, 22.f});
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsMatmul_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk,kl");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("z", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {37.f, 54.f, 81.f, 118.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}
TEST(Einsum, ImplicitEinsumAsBatchedMatmul) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "bij,bjk");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {14.f, 20.f, 30.f, 44.f});
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsBatchedMatmulWithBroadcasting_0) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ij,...jk");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2, 2}, {7.f, 10.f, 15.f, 22.f, 7.f, 10.f, 15.f, 22.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ImplicitEinsumAsMatmul_2) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2068): The parameter is incorrect.";
  }

  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk");
  test.AddInput<float>("x", {2, 1}, {2.f, 3.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {8.f, 12.f, 12.f, 18.f});
  test.Run();
}

TEST(Einsum, DiagonalWithMatmul) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iij, jk");
  test.AddInput<float>("x", {2, 2, 3}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f});
  test.AddInput<float>("y", {3, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
  test.AddOutput<float>("o", {3}, {60.f, 72.f, 84.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

// Theme: Diagonal parsing

// Explicit
TEST(Einsum, ExplicitEinsumAsDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ii->i");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2}, {1.f, 4.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsDiagonalOp_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iii->i");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2}, {1.f, 4.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithAxisReduced) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->j");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2}, {3.f, 7.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithAxisPreserved) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ij");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {1.f, 3.f, 2.f, 4.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithTranspose) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ji");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

// ROCm doesn't support double
#ifndef USE_ROCM
TEST(Einsum, ExplicitEinsumAsDiagonalOpWithTranspose_double) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ji");
  test.AddInput<double>("x", {2, 2, 2}, {1., 2., 3., 4., 1., 2., 3., 4.});
  test.AddOutput<double>("o", {2, 2}, {1., 2., 3., 4.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}
#endif

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithTranspose_int32) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ji");
  test.AddInput<int32_t>("x", {2, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4});
  test.AddOutput<int32_t>("o", {2, 2}, {1, 2, 3, 4});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithTranspose_int64) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ji");
  test.AddInput<int64_t>("x", {2, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4});
  test.AddOutput<int64_t>("o", {2, 2}, {1, 2, 3, 4});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}
TEST(Einsum, ExplicitEinsumAsBatchedDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ii->...i");
  test.AddInput<float>("x", {3, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {3, 2}, {1.f, 4.f, 1.f, 4.f, 1.f, 4.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsBatchedDiagonalOp_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...iij->...j");
  test.AddInput<float>("x", {2, 2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {4.f, 6.f, 4.f, 6.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

// Implicit (Implicit diagonal ops will sum up diagonal values)
TEST(Einsum, ImplicitEinsumAsDiagonalOp) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 5, which exceeds threshold";
  }

  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ii");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {}, {5.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ImplicitEinsumAsDiagonalOp_1) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: error: The difference between expected[i] and output[i] is 15, which exceeds threshold";
  }

  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iii");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {}, {5.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ImplicitEinsumAsDiagonalOpWithAxisReduced) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2}, {3.f, 7.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ImplicitEinsumAsBatchedDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ii");
  test.AddInput<float>("x", {2, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 1}, {5.f, 5.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ImplicitEinsumAsBatchedDiagonalOp_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...iij");
  test.AddInput<float>("x", {2, 2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {4.f, 6.f, 4.f, 6.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

// Theme: Scalar inputs and outputs

// Explicit
TEST(Einsum, ExplicitEinsumAsElementwiseMulOpWithOneScalar) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",...i->...i");
  test.AddInput<float>("x", {}, {10.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {10.f, 20.f, 30.f, 40.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsElementwiseMulOpWithTwoScalars_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",...i,->...i");
  test.AddInput<float>("x", {}, {10.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("z", {}, {10.f});
  test.AddOutput<float>("o", {2, 2}, {100.f, 200.f, 300.f, 400.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}
TEST(Einsum, ExplicitEinsumAsElementwiseMulOpWithAllScalars) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",->");
  test.AddInput<float>("x", {}, {10.f});
  test.AddInput<float>("y", {}, {2.f});
  test.AddOutput<float>("o", {}, {20.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumReduceAxesInInputToScalarOutput) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij->");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {}, {10.f});
  test.Run();
}

// Implicit
TEST(Einsum, ImplicitEinsumAsElementwiseMulOpWithOneScalar) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",...i");
  test.AddInput<float>("x", {}, {10.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {10.f, 20.f, 30.f, 40.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ImplicitEinsumAsElementwiseMulOpWithThreeScalars_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",...i,,");
  test.AddInput<float>("a", {}, {10.f});
  test.AddInput<float>("b", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("c", {}, {10.f});
  test.AddInput<float>("d", {}, {10.f});
  test.AddOutput<float>("o", {2, 2}, {1000.f, 2000.f, 3000.f, 4000.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}
TEST(Einsum, ImplicitEinsumAsElementwiseMulOpWithAllScalars) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",");
  test.AddInput<float>("x", {}, {10.f});
  test.AddInput<float>("y", {}, {2.f});
  test.AddOutput<float>("o", {}, {20.f});
  test.Run();
}

// Tensor Contraction

// Explicit
TEST(Einsum, ExplicitEinsumAsTensorContraction) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "abcd,ea->bcde");
  test.AddInput<float>("x", {2, 2, 2, 2}, {1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 1.f, 2.f});
  test.AddOutput<float>("o", {2, 2, 2, 2}, {3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsTensorContractionReshapeFinal) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "sbcd,es,eh->bce");
  test.AddInput<float>("x", {2, 2, 2, 2}, {1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, -6.f, 2.f});
  test.AddInput<float>("z", {2, 2}, {3.f, 4.f, 5.f, 6.f});
  test.AddOutput<float>("o", {2, 2, 2}, {63.f, -132.f, 63.f, -132.f, 63.f, -132.f, 63.f, -132.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsTensorContractionReshapeLeft) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "bsnh,btnh->bnts");
  test.AddInput<float>("x", {2, 1, 2, 2}, {1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f});
  test.AddInput<float>("y", {2, 2, 2, 1}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  test.AddOutput<float>("o", {2, 2, 2, 1}, {3.f, 9.f, 6.f, 12.f, 15.f, 21.f, 18.f, 24.f});
  test.Run();
}

// Implicit
TEST(Einsum, ImplicitEinsumAsTensorContraction) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "abcd,ea");
  test.AddInput<float>("x", {2, 2, 2, 2}, {1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 1.f, 2.f});
  test.AddOutput<float>("o", {2, 2, 2, 2}, {3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f});
  test.Run();
}

// Theme: Half support

TEST(Einsum, ExplicitEinsumAsIdentity_1D_input_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i->i");
  std::vector<float> input_x_f = {0.9f, 2.5f, 2.3f, 1.5f, -4.5f};
  std::vector<float> output_f = {0.9f, 2.5f, 2.3f, 1.5f, -4.5f};
  std::vector<MLFloat16> input_x(5);
  std::vector<MLFloat16> output(5);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 5);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 5);
  test.AddInput<MLFloat16>("x", {5}, input_x);
  test.AddOutput<MLFloat16>("y", {5}, output);
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsTransposeOp_2D_input_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ji->ij");
  std::vector<float> input_x_f = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> output_f = {1.f, 3.f, 2.f, 4.f};
  std::vector<MLFloat16> input_x(4);
  std::vector<MLFloat16> output(4);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 4);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 4);
  test.AddInput<MLFloat16>("x", {2, 2}, input_x);
  test.AddOutput<MLFloat16>("y", {2, 2}, output);
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsReduceOp_2D_input_0_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij->i");
  std::vector<float> input_x_f = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> output_f = {3.f, 7.f};
  std::vector<MLFloat16> input_x(4);
  std::vector<MLFloat16> output(2);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 4);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 2);
  test.AddInput<MLFloat16>("x", {2, 2}, input_x);
  test.AddOutput<MLFloat16>("y", {2}, output);
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsOuterProductOp_2D_input_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i,j->ij");
  std::vector<float> input_x_f = {1.f, 2.f};
  std::vector<float> input_y_f = {3.f, 4.f};
  std::vector<float> output_f = {3.f, 4.f, 6.f, 8.f};
  std::vector<MLFloat16> input_x(2);
  std::vector<MLFloat16> input_y(2);
  std::vector<MLFloat16> output(4);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 2);
  ConvertFloatToMLFloat16(input_y_f.data(), input_y.data(), 2);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 4);
  test.AddInput<MLFloat16>("x", {2}, input_x);
  test.AddInput<MLFloat16>("y", {2}, input_y);
  test.AddOutput<MLFloat16>("o", {2, 2}, output);
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsMatmul_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk->ik");
  std::vector<float> input_x_f = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> input_y_f = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> output_f = {7.f, 10.f, 15.f, 22.f};
  std::vector<MLFloat16> input_x(4);
  std::vector<MLFloat16> input_y(4);
  std::vector<MLFloat16> output(4);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 4);
  ConvertFloatToMLFloat16(input_y_f.data(), input_y.data(), 4);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 4);
  test.AddInput<MLFloat16>("x", {2, 2}, input_x);
  test.AddInput<MLFloat16>("y", {2, 2}, input_y);
  test.AddOutput<MLFloat16>("o", {2, 2}, output);
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsBatchedMatmul_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "bij,bjk->bik");
  std::vector<float> input_x_f = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  std::vector<float> input_y_f = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  std::vector<float> output_f = {7.f, 10.f, 15.f, 22.f, 7.f, 10.f, 15.f, 22.f};
  std::vector<MLFloat16> input_x(8);
  std::vector<MLFloat16> input_y(8);
  std::vector<MLFloat16> output(8);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 8);
  ConvertFloatToMLFloat16(input_y_f.data(), input_y.data(), 8);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 8);
  test.AddInput<MLFloat16>("x", {2, 2, 2}, input_x);
  test.AddInput<MLFloat16>("y", {2, 2, 2}, input_y);
  test.AddOutput<MLFloat16>("o", {2, 2, 2}, output);
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsDiagonalOp_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ii->i");
  std::vector<float> input_x_f = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> output_f = {1.f, 4.f};
  std::vector<MLFloat16> input_x(4);
  std::vector<MLFloat16> output(2);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 4);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 2);
  test.AddInput<MLFloat16>("x", {2, 2}, input_x);
  test.AddOutput<MLFloat16>("o", {2}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsElementwiseMulOpWithOneScalar_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",...i->...i");
  std::vector<float> input_x_f = {10.f};
  std::vector<float> input_y_f = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> output_f = {10.f, 20.f, 30.f, 40.f};
  std::vector<MLFloat16> input_x(1);
  std::vector<MLFloat16> input_y(4);
  std::vector<MLFloat16> output(4);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 1);
  ConvertFloatToMLFloat16(input_y_f.data(), input_y.data(), 4);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 4);
  test.AddInput<MLFloat16>("x", {}, input_x);
  test.AddInput<MLFloat16>("y", {2, 2}, input_y);
  test.AddOutput<MLFloat16>("o", {2, 2}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(Einsum, ExplicitEinsumAsTensorContraction_Half) {
  if (!HasCudaEnvironment(600)) {
    return;
  }
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "abcd,ea->bcde");
  std::vector<float> input_x_f = {1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f};
  std::vector<float> input_y_f = {1.f, 2.f, 1.f, 2.f};
  std::vector<float> output_f = {3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f};
  std::vector<MLFloat16> input_x(16);
  std::vector<MLFloat16> input_y(4);
  std::vector<MLFloat16> output(16);
  ConvertFloatToMLFloat16(input_x_f.data(), input_x.data(), 16);
  ConvertFloatToMLFloat16(input_y_f.data(), input_y.data(), 4);
  ConvertFloatToMLFloat16(output_f.data(), output.data(), 16);
  test.AddInput<MLFloat16>("x", {2, 2, 2, 2}, input_x);
  test.AddInput<MLFloat16>("y", {2, 2}, input_y);
  test.AddOutput<MLFloat16>("o", {2, 2, 2, 2}, output);
  test.Run();
}

// Theme: Tests involving MatMul(s) interleaved with Transpose(s)
// for two and three inputs (most common use-case of Einsum operator)

struct EinsumTestCase {
  std::string_view equation;
  gsl::span<const int64_t> shape;
  gsl::span<const float> expected;
};
static constexpr std::string_view equation0 = "abc,cd->abc";
static constexpr std::array<int64_t, 3> shape0{2, 2, 2};
static constexpr std::array<float, 8> expected0{0.f, 5.f, 2.f, 15.f, 4.f, 25.f, 6.f, 35.f};
static constexpr std::string_view equation1 = "abc,cd->abd";
static constexpr std::array<int64_t, 3> shape1{2, 2, 2};
static constexpr std::array<float, 8> expected1{2.f, 3.f, 6.f, 11.f, 10.f, 19.f, 14.f, 27.f};
static constexpr std::string_view equation2 = "abc,cd->acd";
static constexpr std::array<int64_t, 3> shape2{2, 2, 2};
static constexpr std::array<float, 8> expected2{0.f, 2.f, 8.f, 12.f, 0.f, 10.f, 24.f, 36.f};
static constexpr std::string_view equation3 = "abc,dc->abd";
static constexpr std::array<int64_t, 3> shape3{2, 2, 2};
static constexpr std::array<float, 8> expected3{1.f, 3.f, 3.f, 13.f, 5.f, 23.f, 7.f, 33.f};
static constexpr std::string_view equation4 = "abc,dc->abc";
static constexpr std::array<int64_t, 3> shape4{2, 2, 2};
static constexpr std::array<float, 8> expected4{0.f, 4.f, 4.f, 12.f, 8.f, 20.f, 12.f, 28.f};
static constexpr std::string_view equation5 = "abc,dc->acd";
static constexpr std::array<int64_t, 3> shape5{2, 2, 2};
static constexpr std::array<float, 8> expected5{0.f, 4.f, 4.f, 12.f, 0.f, 20.f, 12.f, 36.f};
static constexpr std::string_view equation6 = "acb,cd->acd";
static constexpr std::array<int64_t, 3> shape6{2, 2, 2};
static constexpr std::array<float, 8> expected6{0.f, 1.f, 10.f, 15.f, 0.f, 9.f, 26.f, 39.f};
static constexpr std::string_view equation7 = "acb,cd->abc";
static constexpr std::array<int64_t, 3> shape7{2, 2, 2};
static constexpr std::array<float, 8> expected7{0.f, 10.f, 1.f, 15.f, 4.f, 30.f, 5.f, 35.f};
static constexpr std::string_view equation8 = "acb,cd->abd";
static constexpr std::array<int64_t, 3> shape8{2, 2, 2};
static constexpr std::array<float, 8> expected8{4.f, 6.f, 6.f, 10.f, 12.f, 22.f, 14.f, 26.f};
static constexpr std::string_view equation9 = "acb,dc->acd";
static constexpr std::array<int64_t, 3> shape9{2, 2, 2};
static constexpr std::array<float, 8> expected9{0.f, 2.f, 5.f, 15.f, 0.f, 18.f, 13.f, 39.f};
static constexpr std::string_view equation10 = "acb,dc->abd";
static constexpr std::array<int64_t, 3> shape10{2, 2, 2};
static constexpr std::array<float, 8> expected10{2.f, 6.f, 3.f, 11.f, 6.f, 26.f, 7.f, 31.f};
static constexpr std::string_view equation11 = "acb,dc->abc";
static constexpr std::array<int64_t, 3> shape11{2, 2, 2};
static constexpr std::array<float, 8> expected11{0.f, 8.f, 2.f, 12.f, 8.f, 24.f, 10.f, 28.f};
static constexpr std::string_view equation12 = "bac,cd->bac";
static constexpr std::array<int64_t, 3> shape12{2, 2, 2};
static constexpr std::array<float, 8> expected12{0.f, 5.f, 2.f, 15.f, 4.f, 25.f, 6.f, 35.f};
static constexpr std::string_view equation13 = "bac,cd->bad";
static constexpr std::array<int64_t, 3> shape13{2, 2, 2};
static constexpr std::array<float, 8> expected13{2.f, 3.f, 6.f, 11.f, 10.f, 19.f, 14.f, 27.f};
static constexpr std::string_view equation14 = "bac,cd->bcd";
static constexpr std::array<int64_t, 3> shape14{2, 2, 2};
static constexpr std::array<float, 8> expected14{0.f, 2.f, 8.f, 12.f, 0.f, 10.f, 24.f, 36.f};
static constexpr std::string_view equation15 = "bac,dc->bad";
static constexpr std::array<int64_t, 3> shape15{2, 2, 2};
static constexpr std::array<float, 8> expected15{1.f, 3.f, 3.f, 13.f, 5.f, 23.f, 7.f, 33.f};
static constexpr std::string_view equation16 = "bac,dc->bac";
static constexpr std::array<int64_t, 3> shape16{2, 2, 2};
static constexpr std::array<float, 8> expected16{0.f, 4.f, 4.f, 12.f, 8.f, 20.f, 12.f, 28.f};
static constexpr std::string_view equation17 = "bac,dc->bcd";
static constexpr std::array<int64_t, 3> shape17{2, 2, 2};
static constexpr std::array<float, 8> expected17{0.f, 4.f, 4.f, 12.f, 0.f, 20.f, 12.f, 36.f};
static constexpr std::string_view equation18 = "bca,cd->bcd";
static constexpr std::array<int64_t, 3> shape18{2, 2, 2};
static constexpr std::array<float, 8> expected18{0.f, 1.f, 10.f, 15.f, 0.f, 9.f, 26.f, 39.f};
static constexpr std::string_view equation19 = "bca,cd->bac";
static constexpr std::array<int64_t, 3> shape19{2, 2, 2};
static constexpr std::array<float, 8> expected19{0.f, 10.f, 1.f, 15.f, 4.f, 30.f, 5.f, 35.f};
static constexpr std::string_view equation20 = "bca,cd->bad";
static constexpr std::array<int64_t, 3> shape20{2, 2, 2};
static constexpr std::array<float, 8> expected20{4.f, 6.f, 6.f, 10.f, 12.f, 22.f, 14.f, 26.f};
static constexpr std::string_view equation21 = "bca,dc->bcd";
static constexpr std::array<int64_t, 3> shape21{2, 2, 2};
static constexpr std::array<float, 8> expected21{0.f, 2.f, 5.f, 15.f, 0.f, 18.f, 13.f, 39.f};
static constexpr std::string_view equation22 = "bca,dc->bad";
static constexpr std::array<int64_t, 3> shape22{2, 2, 2};
static constexpr std::array<float, 8> expected22{2.f, 6.f, 3.f, 11.f, 6.f, 26.f, 7.f, 31.f};
static constexpr std::string_view equation23 = "bca,dc->bac";
static constexpr std::array<int64_t, 3> shape23{2, 2, 2};
static constexpr std::array<float, 8> expected23{0.f, 8.f, 2.f, 12.f, 8.f, 24.f, 10.f, 28.f};
static constexpr std::string_view equation24 = "cab,cd->cad";
static constexpr std::array<int64_t, 3> shape24{2, 2, 2};
static constexpr std::array<float, 8> expected24{0.f, 1.f, 0.f, 5.f, 18.f, 27.f, 26.f, 39.f};
static constexpr std::string_view equation25 = "cab,cd->cbd";
static constexpr std::array<int64_t, 3> shape25{2, 2, 2};
static constexpr std::array<float, 8> expected25{0.f, 2.f, 0.f, 4.f, 20.f, 30.f, 24.f, 36.f};
static constexpr std::string_view equation26 = "cab,dc->cad";
static constexpr std::array<int64_t, 3> shape26{2, 2, 2};
static constexpr std::array<float, 8> expected26{0.f, 2.f, 0.f, 10.f, 9.f, 27.f, 13.f, 39.f};
static constexpr std::string_view equation27 = "cab,dc->cbd";
static constexpr std::array<int64_t, 3> shape27{2, 2, 2};
static constexpr std::array<float, 8> expected27{0.f, 4.f, 0.f, 8.f, 10.f, 30.f, 12.f, 36.f};
static constexpr std::string_view equation28 = "cba,cd->cbd";
static constexpr std::array<int64_t, 3> shape28{2, 2, 2};
static constexpr std::array<float, 8> expected28{0.f, 1.f, 0.f, 5.f, 18.f, 27.f, 26.f, 39.f};
static constexpr std::string_view equation29 = "cba,cd->cad";
static constexpr std::array<int64_t, 3> shape29{2, 2, 2};
static constexpr std::array<float, 8> expected29{0.f, 2.f, 0.f, 4.f, 20.f, 30.f, 24.f, 36.f};
static constexpr std::string_view equation30 = "cba,dc->cbd";
static constexpr std::array<int64_t, 3> shape30{2, 2, 2};
static constexpr std::array<float, 8> expected30{0.f, 2.f, 0.f, 10.f, 9.f, 27.f, 13.f, 39.f};
static constexpr std::string_view equation31 = "cba,dc->cad";
static constexpr std::array<int64_t, 3> shape31{2, 2, 2};
static constexpr std::array<float, 8> expected31{0.f, 4.f, 0.f, 8.f, 10.f, 30.f, 12.f, 36.f};
static constexpr std::array<EinsumTestCase, 32> case0 = {{
    {equation0, shape0, expected0},
    {equation1, shape1, expected1},
    {equation2, shape2, expected2},
    {equation3, shape3, expected3},
    {equation4, shape4, expected4},
    {equation5, shape5, expected5},
    {equation6, shape6, expected6},
    {equation7, shape7, expected7},
    {equation8, shape8, expected8},
    {equation9, shape9, expected9},
    {equation10, shape10, expected10},
    {equation11, shape11, expected11},
    {equation12, shape12, expected12},
    {equation13, shape13, expected13},
    {equation14, shape14, expected14},
    {equation15, shape15, expected15},
    {equation16, shape16, expected16},
    {equation17, shape17, expected17},
    {equation18, shape18, expected18},
    {equation19, shape19, expected19},
    {equation20, shape20, expected20},
    {equation21, shape21, expected21},
    {equation22, shape22, expected22},
    {equation23, shape23, expected23},
    {equation24, shape24, expected24},
    {equation25, shape25, expected25},
    {equation26, shape26, expected26},
    {equation27, shape27, expected27},
    {equation28, shape28, expected28},
    {equation29, shape29, expected29},
    {equation30, shape30, expected30},
    {equation31, shape31, expected31},
}};

static constexpr std::string_view equation32 = "abc,cd,def->abd";
static constexpr std::array<int64_t, 3> shape32{2, 2, 2};
static constexpr std::array<float, 8> expected32{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f};
static constexpr std::string_view equation33 = "abc,cd,def->abe";
static constexpr std::array<int64_t, 3> shape33{2, 2, 2};
static constexpr std::array<float, 8> expected33{29.f, 49.f, 105.f, 173.f, 181.f, 297.f, 257.f, 421.f};
static constexpr std::string_view equation34 = "abc,cd,def->acd";
static constexpr std::array<int64_t, 3> shape34{2, 2, 2};
static constexpr std::array<float, 8> expected34{0.f, 44.f, 48.f, 264.f, 0.f, 220.f, 144.f, 792.f};
static constexpr std::string_view equation35 = "abc,cd,def->ace";
static constexpr std::array<int64_t, 3> shape35{2, 2, 2};
static constexpr std::array<float, 8> expected35{18.f, 26.f, 116.f, 196.f, 90.f, 130.f, 348.f, 588.f};
static constexpr std::string_view equation36 = "abc,cd,dfe->abd";
static constexpr std::array<int64_t, 3> shape36{2, 2, 2};
static constexpr std::array<float, 8> expected36{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f};
static constexpr std::string_view equation37 = "abc,cd,dfe->abf";
static constexpr std::array<int64_t, 3> shape37{2, 2, 2};
static constexpr std::array<float, 8> expected37{29.f, 49.f, 105.f, 173.f, 181.f, 297.f, 257.f, 421.f};
static constexpr std::string_view equation38 = "abc,cd,dfe->acd";
static constexpr std::array<int64_t, 3> shape38{2, 2, 2};
static constexpr std::array<float, 8> expected38{0.f, 44.f, 48.f, 264.f, 0.f, 220.f, 144.f, 792.f};
static constexpr std::string_view equation39 = "abc,cd,dfe->acf";
static constexpr std::array<int64_t, 3> shape39{2, 2, 2};
static constexpr std::array<float, 8> expected39{18.f, 26.f, 116.f, 196.f, 90.f, 130.f, 348.f, 588.f};
static constexpr std::string_view equation40 = "abc,cd,edf->abe";
static constexpr std::array<int64_t, 3> shape40{2, 2, 2};
static constexpr std::array<float, 8> expected40{17.f, 57.f, 61.f, 197.f, 105.f, 337.f, 149.f, 477.f};
static constexpr std::string_view equation41 = "abc,cd,edf->abd";
static constexpr std::array<int64_t, 3> shape41{2, 2, 2};
static constexpr std::array<float, 8> expected41{20.f, 54.f, 60.f, 198.f, 100.f, 342.f, 140.f, 486.f};
static constexpr std::string_view equation42 = "abc,cd,edf->ace";
static constexpr std::array<int64_t, 3> shape42{2, 2, 2};
static constexpr std::array<float, 8> expected42{10.f, 26.f, 68.f, 228.f, 50.f, 130.f, 204.f, 684.f};
static constexpr std::string_view equation43 = "abc,cd,edf->acd";
static constexpr std::array<int64_t, 3> shape43{2, 2, 2};
static constexpr std::array<float, 8> expected43{0.f, 36.f, 80.f, 216.f, 0.f, 180.f, 240.f, 648.f};
static constexpr std::string_view equation44 = "abc,cd,efd->abe";
static constexpr std::array<int64_t, 3> shape44{2, 2, 2};
static constexpr std::array<float, 8> expected44{16.f, 56.f, 56.f, 192.f, 96.f, 328.f, 136.f, 464.f};
static constexpr std::string_view equation45 = "abc,cd,efd->abf";
static constexpr std::array<int64_t, 3> shape45{2, 2, 2};
static constexpr std::array<float, 8> expected45{26.f, 46.f, 90.f, 158.f, 154.f, 270.f, 218.f, 382.f};
static constexpr std::string_view equation46 = "abc,cd,efd->ace";
static constexpr std::array<int64_t, 3> shape46{2, 2, 2};
static constexpr std::array<float, 8> expected46{8.f, 24.f, 64.f, 224.f, 40.f, 120.f, 192.f, 672.f};
static constexpr std::string_view equation47 = "abc,cd,efd->acf";
static constexpr std::array<int64_t, 3> shape47{2, 2, 2};
static constexpr std::array<float, 8> expected47{12.f, 20.f, 104.f, 184.f, 60.f, 100.f, 312.f, 552.f};
static constexpr std::string_view equation48 = "abc,cd,fde->abf";
static constexpr std::array<int64_t, 3> shape48{2, 2, 2};
static constexpr std::array<float, 8> expected48{17.f, 57.f, 61.f, 197.f, 105.f, 337.f, 149.f, 477.f};
static constexpr std::string_view equation49 = "abc,cd,fde->abd";
static constexpr std::array<int64_t, 3> shape49{2, 2, 2};
static constexpr std::array<float, 8> expected49{20.f, 54.f, 60.f, 198.f, 100.f, 342.f, 140.f, 486.f};
static constexpr std::string_view equation50 = "abc,cd,fde->acf";
static constexpr std::array<int64_t, 3> shape50{2, 2, 2};
static constexpr std::array<float, 8> expected50{10.f, 26.f, 68.f, 228.f, 50.f, 130.f, 204.f, 684.f};
static constexpr std::string_view equation51 = "abc,cd,fde->acd";
static constexpr std::array<int64_t, 3> shape51{2, 2, 2};
static constexpr std::array<float, 8> expected51{0.f, 36.f, 80.f, 216.f, 0.f, 180.f, 240.f, 648.f};
static constexpr std::string_view equation52 = "abc,cd,fed->abf";
static constexpr std::array<int64_t, 3> shape52{2, 2, 2};
static constexpr std::array<float, 8> expected52{16.f, 56.f, 56.f, 192.f, 96.f, 328.f, 136.f, 464.f};
static constexpr std::string_view equation53 = "abc,cd,fed->abe";
static constexpr std::array<int64_t, 3> shape53{2, 2, 2};
static constexpr std::array<float, 8> expected53{26.f, 46.f, 90.f, 158.f, 154.f, 270.f, 218.f, 382.f};
static constexpr std::string_view equation54 = "abc,cd,fed->acf";
static constexpr std::array<int64_t, 3> shape54{2, 2, 2};
static constexpr std::array<float, 8> expected54{8.f, 24.f, 64.f, 224.f, 40.f, 120.f, 192.f, 672.f};
static constexpr std::string_view equation55 = "abc,cd,fed->ace";
static constexpr std::array<int64_t, 3> shape55{2, 2, 2};
static constexpr std::array<float, 8> expected55{12.f, 20.f, 104.f, 184.f, 60.f, 100.f, 312.f, 552.f};
static constexpr std::string_view equation56 = "abc,dc,def->abd";
static constexpr std::array<int64_t, 3> shape56{2, 2, 2};
static constexpr std::array<float, 8> expected56{6.f, 66.f, 18.f, 286.f, 30.f, 506.f, 42.f, 726.f};
static constexpr std::string_view equation57 = "abc,dc,def->abe";
static constexpr std::array<int64_t, 3> shape57{2, 2, 2};
static constexpr std::array<float, 8> expected57{28.f, 44.f, 120.f, 184.f, 212.f, 324.f, 304.f, 464.f};
static constexpr std::string_view equation58 = "abc,dc,def->acd";
static constexpr std::array<int64_t, 3> shape58{2, 2, 2};
static constexpr std::array<float, 8> expected58{0.f, 88.f, 24.f, 264.f, 0.f, 440.f, 72.f, 792.f};
static constexpr std::string_view equation59 = "abc,dc,def->ace";
static constexpr std::array<int64_t, 3> shape59{2, 2, 2};
static constexpr std::array<float, 8> expected59{36.f, 52.f, 112.f, 176.f, 180.f, 260.f, 336.f, 528.f};
static constexpr std::string_view equation60 = "abc,dc,dfe->abd";
static constexpr std::array<int64_t, 3> shape60{2, 2, 2};
static constexpr std::array<float, 8> expected60{6.f, 66.f, 18.f, 286.f, 30.f, 506.f, 42.f, 726.f};
static constexpr std::string_view equation61 = "abc,dc,dfe->abf";
static constexpr std::array<int64_t, 3> shape61{2, 2, 2};
static constexpr std::array<float, 8> expected61{28.f, 44.f, 120.f, 184.f, 212.f, 324.f, 304.f, 464.f};
static constexpr std::string_view equation62 = "abc,dc,dfe->acd";
static constexpr std::array<int64_t, 3> shape62{2, 2, 2};
static constexpr std::array<float, 8> expected62{0.f, 88.f, 24.f, 264.f, 0.f, 440.f, 72.f, 792.f};
static constexpr std::string_view equation63 = "abc,dc,dfe->acf";
static constexpr std::array<int64_t, 3> shape63{2, 2, 2};
static constexpr std::array<float, 8> expected63{36.f, 52.f, 112.f, 176.f, 180.f, 260.f, 336.f, 528.f};
static constexpr std::string_view equation64 = "abc,dc,edf->abe";
static constexpr std::array<int64_t, 3> shape64{2, 2, 2};
static constexpr std::array<float, 8> expected64{16.f, 48.f, 68.f, 196.f, 120.f, 344.f, 172.f, 492.f};
static constexpr std::string_view equation65 = "abc,dc,edf->abd";
static constexpr std::array<int64_t, 3> shape65{2, 2, 2};
static constexpr std::array<float, 8> expected65{10.f, 54.f, 30.f, 234.f, 50.f, 414.f, 70.f, 594.f};
static constexpr std::string_view equation66 = "abc,dc,edf->ace";
static constexpr std::array<int64_t, 3> shape66{2, 2, 2};
static constexpr std::array<float, 8> expected66{20.f, 52.f, 64.f, 192.f, 100.f, 260.f, 192.f, 576.f};
static constexpr std::string_view equation67 = "abc,dc,edf->acd";
static constexpr std::array<int64_t, 3> shape67{2, 2, 2};
static constexpr std::array<float, 8> expected67{0.f, 72.f, 40.f, 216.f, 0.f, 360.f, 120.f, 648.f};
static constexpr std::string_view equation68 = "abc,dc,efd->abe";
static constexpr std::array<int64_t, 3> shape68{2, 2, 2};
static constexpr std::array<float, 8> expected68{14.f, 46.f, 58.f, 186.f, 102.f, 326.f, 146.f, 466.f};
static constexpr std::string_view equation69 = "abc,dc,efd->abf";
static constexpr std::array<int64_t, 3> shape69{2, 2, 2};
static constexpr std::array<float, 8> expected69{22.f, 38.f, 90.f, 154.f, 158.f, 270.f, 226.f, 386.f};
static constexpr std::string_view equation70 = "abc,dc,efd->ace";
static constexpr std::array<int64_t, 3> shape70{2, 2, 2};
static constexpr std::array<float, 8> expected70{16.f, 48.f, 56.f, 184.f, 80.f, 240.f, 168.f, 552.f};
static constexpr std::string_view equation71 = "abc,dc,efd->acf";
static constexpr std::array<int64_t, 3> shape71{2, 2, 2};
static constexpr std::array<float, 8> expected71{24.f, 40.f, 88.f, 152.f, 120.f, 200.f, 264.f, 456.f};
static constexpr std::string_view equation72 = "abc,dc,fde->abf";
static constexpr std::array<int64_t, 3> shape72{2, 2, 2};
static constexpr std::array<float, 8> expected72{16.f, 48.f, 68.f, 196.f, 120.f, 344.f, 172.f, 492.f};
static constexpr std::string_view equation73 = "abc,dc,fde->abd";
static constexpr std::array<int64_t, 3> shape73{2, 2, 2};
static constexpr std::array<float, 8> expected73{10.f, 54.f, 30.f, 234.f, 50.f, 414.f, 70.f, 594.f};
static constexpr std::string_view equation74 = "abc,dc,fde->acf";
static constexpr std::array<int64_t, 3> shape74{2, 2, 2};
static constexpr std::array<float, 8> expected74{20.f, 52.f, 64.f, 192.f, 100.f, 260.f, 192.f, 576.f};
static constexpr std::string_view equation75 = "abc,dc,fde->acd";
static constexpr std::array<int64_t, 3> shape75{2, 2, 2};
static constexpr std::array<float, 8> expected75{0.f, 72.f, 40.f, 216.f, 0.f, 360.f, 120.f, 648.f};
static constexpr std::string_view equation76 = "abc,dc,fed->abf";
static constexpr std::array<int64_t, 3> shape76{2, 2, 2};
static constexpr std::array<float, 8> expected76{14.f, 46.f, 58.f, 186.f, 102.f, 326.f, 146.f, 466.f};
static constexpr std::string_view equation77 = "abc,dc,fed->abe";
static constexpr std::array<int64_t, 3> shape77{2, 2, 2};
static constexpr std::array<float, 8> expected77{22.f, 38.f, 90.f, 154.f, 158.f, 270.f, 226.f, 386.f};
static constexpr std::string_view equation78 = "abc,dc,fed->acf";
static constexpr std::array<int64_t, 3> shape78{2, 2, 2};
static constexpr std::array<float, 8> expected78{16.f, 48.f, 56.f, 184.f, 80.f, 240.f, 168.f, 552.f};
static constexpr std::string_view equation79 = "abc,dc,fed->ace";
static constexpr std::array<int64_t, 3> shape79{2, 2, 2};
static constexpr std::array<float, 8> expected79{24.f, 40.f, 88.f, 152.f, 120.f, 200.f, 264.f, 456.f};
static constexpr std::string_view equation80 = "acb,cd,def->acd";
static constexpr std::array<int64_t, 3> shape80{2, 2, 2};
static constexpr std::array<float, 8> expected80{0.f, 22.f, 60.f, 330.f, 0.f, 198.f, 156.f, 858.f};
static constexpr std::string_view equation81 = "acb,cd,def->ace";
static constexpr std::array<int64_t, 3> shape81{2, 2, 2};
static constexpr std::array<float, 8> expected81{9.f, 13.f, 145.f, 245.f, 81.f, 117.f, 377.f, 637.f};
static constexpr std::string_view equation82 = "acb,cd,def->abd";
static constexpr std::array<int64_t, 3> shape82{2, 2, 2};
static constexpr std::array<float, 8> expected82{24.f, 132.f, 36.f, 220.f, 72.f, 484.f, 84.f, 572.f};
static constexpr std::string_view equation83 = "acb,cd,def->abe";
static constexpr std::array<int64_t, 3> shape83{2, 2, 2};
static constexpr std::array<float, 8> expected83{58.f, 98.f, 96.f, 160.f, 210.f, 346.f, 248.f, 408.f};
static constexpr std::string_view equation84 = "acb,cd,dfe->acd";
static constexpr std::array<int64_t, 3> shape84{2, 2, 2};
static constexpr std::array<float, 8> expected84{0.f, 22.f, 60.f, 330.f, 0.f, 198.f, 156.f, 858.f};
static constexpr std::string_view equation85 = "acb,cd,dfe->acf";
static constexpr std::array<int64_t, 3> shape85{2, 2, 2};
static constexpr std::array<float, 8> expected85{9.f, 13.f, 145.f, 245.f, 81.f, 117.f, 377.f, 637.f};
static constexpr std::string_view equation86 = "acb,cd,dfe->abd";
static constexpr std::array<int64_t, 3> shape86{2, 2, 2};
static constexpr std::array<float, 8> expected86{24.f, 132.f, 36.f, 220.f, 72.f, 484.f, 84.f, 572.f};
static constexpr std::string_view equation87 = "acb,cd,dfe->abf";
static constexpr std::array<int64_t, 3> shape87{2, 2, 2};
static constexpr std::array<float, 8> expected87{58.f, 98.f, 96.f, 160.f, 210.f, 346.f, 248.f, 408.f};
static constexpr std::string_view equation88 = "acb,cd,edf->ace";
static constexpr std::array<int64_t, 3> shape88{2, 2, 2};
static constexpr std::array<float, 8> expected88{5.f, 13.f, 85.f, 285.f, 45.f, 117.f, 221.f, 741.f};
static constexpr std::string_view equation89 = "acb,cd,edf->acd";
static constexpr std::array<int64_t, 3> shape89{2, 2, 2};
static constexpr std::array<float, 8> expected89{0.f, 18.f, 100.f, 270.f, 0.f, 162.f, 260.f, 702.f};
static constexpr std::string_view equation90 = "acb,cd,edf->abe";
static constexpr std::array<int64_t, 3> shape90{2, 2, 2};
static constexpr std::array<float, 8> expected90{34.f, 114.f, 56.f, 184.f, 122.f, 394.f, 144.f, 464.f};
static constexpr std::string_view equation91 = "acb,cd,edf->abd";
static constexpr std::array<int64_t, 3> shape91{2, 2, 2};
static constexpr std::array<float, 8> expected91{40.f, 108.f, 60.f, 180.f, 120.f, 396.f, 140.f, 468.f};
static constexpr std::string_view equation92 = "acb,cd,efd->ace";
static constexpr std::array<int64_t, 3> shape92{2, 2, 2};
static constexpr std::array<float, 8> expected92{4.f, 12.f, 80.f, 280.f, 36.f, 108.f, 208.f, 728.f};
static constexpr std::string_view equation93 = "acb,cd,efd->acf";
static constexpr std::array<int64_t, 3> shape93{2, 2, 2};
static constexpr std::array<float, 8> expected93{6.f, 10.f, 130.f, 230.f, 54.f, 90.f, 338.f, 598.f};
static constexpr std::string_view equation94 = "acb,cd,efd->abe";
static constexpr std::array<int64_t, 3> shape94{2, 2, 2};
static constexpr std::array<float, 8> expected94{32.f, 112.f, 52.f, 180.f, 112.f, 384.f, 132.f, 452.f};
static constexpr std::string_view equation95 = "acb,cd,efd->abf";
static constexpr std::array<int64_t, 3> shape95{2, 2, 2};
static constexpr std::array<float, 8> expected95{52.f, 92.f, 84.f, 148.f, 180.f, 316.f, 212.f, 372.f};
static constexpr std::string_view equation96 = "acb,cd,fde->acf";
static constexpr std::array<int64_t, 3> shape96{2, 2, 2};
static constexpr std::array<float, 8> expected96{5.f, 13.f, 85.f, 285.f, 45.f, 117.f, 221.f, 741.f};
static constexpr std::string_view equation97 = "acb,cd,fde->acd";
static constexpr std::array<int64_t, 3> shape97{2, 2, 2};
static constexpr std::array<float, 8> expected97{0.f, 18.f, 100.f, 270.f, 0.f, 162.f, 260.f, 702.f};
static constexpr std::string_view equation98 = "acb,cd,fde->abf";
static constexpr std::array<int64_t, 3> shape98{2, 2, 2};
static constexpr std::array<float, 8> expected98{34.f, 114.f, 56.f, 184.f, 122.f, 394.f, 144.f, 464.f};
static constexpr std::string_view equation99 = "acb,cd,fde->abd";
static constexpr std::array<int64_t, 3> shape99{2, 2, 2};
static constexpr std::array<float, 8> expected99{40.f, 108.f, 60.f, 180.f, 120.f, 396.f, 140.f, 468.f};
static constexpr std::string_view equation100 = "acb,cd,fed->acf";
static constexpr std::array<int64_t, 3> shape100{2, 2, 2};
static constexpr std::array<float, 8> expected100{4.f, 12.f, 80.f, 280.f, 36.f, 108.f, 208.f, 728.f};
static constexpr std::string_view equation101 = "acb,cd,fed->ace";
static constexpr std::array<int64_t, 3> shape101{2, 2, 2};
static constexpr std::array<float, 8> expected101{6.f, 10.f, 130.f, 230.f, 54.f, 90.f, 338.f, 598.f};
static constexpr std::string_view equation102 = "acb,cd,fed->abf";
static constexpr std::array<int64_t, 3> shape102{2, 2, 2};
static constexpr std::array<float, 8> expected102{32.f, 112.f, 52.f, 180.f, 112.f, 384.f, 132.f, 452.f};
static constexpr std::string_view equation103 = "acb,cd,fed->abe";
static constexpr std::array<int64_t, 3> shape103{2, 2, 2};
static constexpr std::array<float, 8> expected103{52.f, 92.f, 84.f, 148.f, 180.f, 316.f, 212.f, 372.f};
static constexpr std::string_view equation104 = "acb,dc,def->acd";
static constexpr std::array<int64_t, 3> shape104{2, 2, 2};
static constexpr std::array<float, 8> expected104{0.f, 44.f, 30.f, 330.f, 0.f, 396.f, 78.f, 858.f};
static constexpr std::string_view equation105 = "acb,dc,def->ace";
static constexpr std::array<int64_t, 3> shape105{2, 2, 2};
static constexpr std::array<float, 8> expected105{18.f, 26.f, 140.f, 220.f, 162.f, 234.f, 364.f, 572.f};

static constexpr std::string_view equation106 = "acb,dc,def->abd";
static constexpr std::array<int64_t, 3> shape106{2, 2, 2};
static constexpr std::array<float, 8> expected106{12.f, 132.f, 18.f, 242.f, 36.f, 572.f, 42.f, 682.f};
static constexpr std::string_view equation107 = "acb,dc,def->abe";
static constexpr std::array<int64_t, 3> shape107{2, 2, 2};
static constexpr std::array<float, 8> expected107{56.f, 88.f, 102.f, 158.f, 240.f, 368.f, 286.f, 438.f};
static constexpr std::string_view equation108 = "acb,dc,dfe->acd";
static constexpr std::array<int64_t, 3> shape108{2, 2, 2};
static constexpr std::array<float, 8> expected108{0.f, 44.f, 30.f, 330.f, 0.f, 396.f, 78.f, 858.f};
static constexpr std::string_view equation109 = "acb,dc,dfe->acf";
static constexpr std::array<int64_t, 3> shape109{2, 2, 2};
static constexpr std::array<float, 8> expected109{18.f, 26.f, 140.f, 220.f, 162.f, 234.f, 364.f, 572.f};
static constexpr std::string_view equation110 = "acb,dc,dfe->abd";
static constexpr std::array<int64_t, 3> shape110{2, 2, 2};
static constexpr std::array<float, 8> expected110{12.f, 132.f, 18.f, 242.f, 36.f, 572.f, 42.f, 682.f};
static constexpr std::string_view equation111 = "acb,dc,dfe->abf";
static constexpr std::array<int64_t, 3> shape111{2, 2, 2};
static constexpr std::array<float, 8> expected111{56.f, 88.f, 102.f, 158.f, 240.f, 368.f, 286.f, 438.f};
static constexpr std::string_view equation112 = "acb,dc,edf->ace";
static constexpr std::array<int64_t, 3> shape112{2, 2, 2};
static constexpr std::array<float, 8> expected112{10.f, 26.f, 80.f, 240.f, 90.f, 234.f, 208.f, 624.f};
static constexpr std::string_view equation113 = "acb,dc,edf->acd";
static constexpr std::array<int64_t, 3> shape113{2, 2, 2};
static constexpr std::array<float, 8> expected113{0.f, 36.f, 50.f, 270.f, 0.f, 324.f, 130.f, 702.f};
static constexpr std::string_view equation114 = "acb,dc,edf->abe";
static constexpr std::array<int64_t, 3> shape114{2, 2, 2};
static constexpr std::array<float, 8> expected114{32.f, 96.f, 58.f, 170.f, 136.f, 392.f, 162.f, 466.f};
static constexpr std::string_view equation115 = "acb,dc,edf->abd";
static constexpr std::array<int64_t, 3> shape115{2, 2, 2};
static constexpr std::array<float, 8> expected115{20.f, 108.f, 30.f, 198.f, 60.f, 468.f, 70.f, 558.f};
static constexpr std::string_view equation116 = "acb,dc,efd->ace";
static constexpr std::array<int64_t, 3> shape116{2, 2, 2};
static constexpr std::array<float, 8> expected116{8.f, 24.f, 70.f, 230.f, 72.f, 216.f, 182.f, 598.f};
static constexpr std::string_view equation117 = "acb,dc,efd->acf";
static constexpr std::array<int64_t, 3> shape117{2, 2, 2};
static constexpr std::array<float, 8> expected117{12.f, 20.f, 110.f, 190.f, 108.f, 180.f, 286.f, 494.f};
static constexpr std::string_view equation118 = "acb,dc,efd->abe";
static constexpr std::array<int64_t, 3> shape118{2, 2, 2};
static constexpr std::array<float, 8> expected118{28.f, 92.f, 50.f, 162.f, 116.f, 372.f, 138.f, 442.f};
static constexpr std::string_view equation119 = "acb,dc,efd->abf";
static constexpr std::array<int64_t, 3> shape119{2, 2, 2};
static constexpr std::array<float, 8> expected119{44.f, 76.f, 78.f, 134.f, 180.f, 308.f, 214.f, 366.f};
static constexpr std::string_view equation120 = "acb,dc,fde->acf";
static constexpr std::array<int64_t, 3> shape120{2, 2, 2};
static constexpr std::array<float, 8> expected120{10.f, 26.f, 80.f, 240.f, 90.f, 234.f, 208.f, 624.f};
static constexpr std::string_view equation121 = "acb,dc,fde->acd";
static constexpr std::array<int64_t, 3> shape121{2, 2, 2};
static constexpr std::array<float, 8> expected121{0.f, 36.f, 50.f, 270.f, 0.f, 324.f, 130.f, 702.f};
static constexpr std::string_view equation122 = "acb,dc,fde->abf";
static constexpr std::array<int64_t, 3> shape122{2, 2, 2};
static constexpr std::array<float, 8> expected122{32.f, 96.f, 58.f, 170.f, 136.f, 392.f, 162.f, 466.f};
static constexpr std::string_view equation123 = "acb,dc,fde->abd";
static constexpr std::array<int64_t, 3> shape123{2, 2, 2};
static constexpr std::array<float, 8> expected123{20.f, 108.f, 30.f, 198.f, 60.f, 468.f, 70.f, 558.f};
static constexpr std::string_view equation124 = "acb,dc,fed->acf";
static constexpr std::array<int64_t, 3> shape124{2, 2, 2};
static constexpr std::array<float, 8> expected124{8.f, 24.f, 70.f, 230.f, 72.f, 216.f, 182.f, 598.f};
static constexpr std::string_view equation125 = "acb,dc,fed->ace";
static constexpr std::array<int64_t, 3> shape125{2, 2, 2};
static constexpr std::array<float, 8> expected125{12.f, 20.f, 110.f, 190.f, 108.f, 180.f, 286.f, 494.f};
static constexpr std::string_view equation126 = "acb,dc,fed->abf";
static constexpr std::array<int64_t, 3> shape126{2, 2, 2};
static constexpr std::array<float, 8> expected126{28.f, 92.f, 50.f, 162.f, 116.f, 372.f, 138.f, 442.f};
static constexpr std::string_view equation127 = "acb,dc,fed->abe";
static constexpr std::array<int64_t, 3> shape127{2, 2, 2};
static constexpr std::array<float, 8> expected127{44.f, 76.f, 78.f, 134.f, 180.f, 308.f, 214.f, 366.f};
static constexpr std::string_view equation128 = "bac,cd,def->bad";
static constexpr std::array<int64_t, 3> shape128{2, 2, 2};
static constexpr std::array<float, 8> expected128{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f};
static constexpr std::string_view equation129 = "bac,cd,def->bae";
static constexpr std::array<int64_t, 3> shape129{2, 2, 2};
static constexpr std::array<float, 8> expected129{29.f, 49.f, 105.f, 173.f, 181.f, 297.f, 257.f, 421.f};
static constexpr std::string_view equation130 = "bac,cd,def->bcd";
static constexpr std::array<int64_t, 3> shape130{2, 2, 2};
static constexpr std::array<float, 8> expected130{0.f, 44.f, 48.f, 264.f, 0.f, 220.f, 144.f, 792.f};
static constexpr std::string_view equation131 = "bac,cd,def->bce";
static constexpr std::array<int64_t, 3> shape131{2, 2, 2};
static constexpr std::array<float, 8> expected131{18.f, 26.f, 116.f, 196.f, 90.f, 130.f, 348.f, 588.f};
static constexpr std::string_view equation132 = "bac,cd,dfe->bad";
static constexpr std::array<int64_t, 3> shape132{2, 2, 2};
static constexpr std::array<float, 8> expected132{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f};
static constexpr std::string_view equation133 = "bac,cd,dfe->baf";
static constexpr std::array<int64_t, 3> shape133{2, 2, 2};
static constexpr std::array<float, 8> expected133{29.f, 49.f, 105.f, 173.f, 181.f, 297.f, 257.f, 421.f};
static constexpr std::string_view equation134 = "bac,cd,dfe->bcd";
static constexpr std::array<int64_t, 3> shape134{2, 2, 2};
static constexpr std::array<float, 8> expected134{0.f, 44.f, 48.f, 264.f, 0.f, 220.f, 144.f, 792.f};
static constexpr std::string_view equation135 = "bac,cd,dfe->bcf";
static constexpr std::array<int64_t, 3> shape135{2, 2, 2};
static constexpr std::array<float, 8> expected135{18.f, 26.f, 116.f, 196.f, 90.f, 130.f, 348.f, 588.f};
static constexpr std::string_view equation136 = "bac,cd,edf->bae";
static constexpr std::array<int64_t, 3> shape136{2, 2, 2};
static constexpr std::array<float, 8> expected136{17.f, 57.f, 61.f, 197.f, 105.f, 337.f, 149.f, 477.f};
static constexpr std::string_view equation137 = "bac,cd,edf->bad";
static constexpr std::array<int64_t, 3> shape137{2, 2, 2};
static constexpr std::array<float, 8> expected137{20.f, 54.f, 60.f, 198.f, 100.f, 342.f, 140.f, 486.f};
static constexpr std::string_view equation138 = "bac,cd,edf->bce";
static constexpr std::array<int64_t, 3> shape138{2, 2, 2};
static constexpr std::array<float, 8> expected138{10.f, 26.f, 68.f, 228.f, 50.f, 130.f, 204.f, 684.f};
static constexpr std::string_view equation139 = "bac,cd,edf->bcd";
static constexpr std::array<int64_t, 3> shape139{2, 2, 2};
static constexpr std::array<float, 8> expected139{0.f, 36.f, 80.f, 216.f, 0.f, 180.f, 240.f, 648.f};
static constexpr std::string_view equation140 = "bac,cd,efd->bae";
static constexpr std::array<int64_t, 3> shape140{2, 2, 2};
static constexpr std::array<float, 8> expected140{16.f, 56.f, 56.f, 192.f, 96.f, 328.f, 136.f, 464.f};
static constexpr std::string_view equation141 = "bac,cd,efd->baf";
static constexpr std::array<int64_t, 3> shape141{2, 2, 2};
static constexpr std::array<float, 8> expected141{26.f, 46.f, 90.f, 158.f, 154.f, 270.f, 218.f, 382.f};
static constexpr std::string_view equation142 = "bac,cd,efd->bce";
static constexpr std::array<int64_t, 3> shape142{2, 2, 2};
static constexpr std::array<float, 8> expected142{8.f, 24.f, 64.f, 224.f, 40.f, 120.f, 192.f, 672.f};
static constexpr std::string_view equation143 = "bac,cd,efd->bcf";
static constexpr std::array<int64_t, 3> shape143{2, 2, 2};
static constexpr std::array<float, 8> expected143{12.f, 20.f, 104.f, 184.f, 60.f, 100.f, 312.f, 552.f};
static constexpr std::string_view equation144 = "bac,cd,fde->baf";
static constexpr std::array<int64_t, 3> shape144{2, 2, 2};
static constexpr std::array<float, 8> expected144{17.f, 57.f, 61.f, 197.f, 105.f, 337.f, 149.f, 477.f};
static constexpr std::string_view equation145 = "bac,cd,fde->bad";
static constexpr std::array<int64_t, 3> shape145{2, 2, 2};
static constexpr std::array<float, 8> expected145{20.f, 54.f, 60.f, 198.f, 100.f, 342.f, 140.f, 486.f};
static constexpr std::string_view equation146 = "bac,cd,fde->bcf";
static constexpr std::array<int64_t, 3> shape146{2, 2, 2};
static constexpr std::array<float, 8> expected146{10.f, 26.f, 68.f, 228.f, 50.f, 130.f, 204.f, 684.f};
static constexpr std::string_view equation147 = "bac,cd,fde->bcd";
static constexpr std::array<int64_t, 3> shape147{2, 2, 2};
static constexpr std::array<float, 8> expected147{0.f, 36.f, 80.f, 216.f, 0.f, 180.f, 240.f, 648.f};
static constexpr std::string_view equation148 = "bac,cd,fed->baf";
static constexpr std::array<int64_t, 3> shape148{2, 2, 2};
static constexpr std::array<float, 8> expected148{16.f, 56.f, 56.f, 192.f, 96.f, 328.f, 136.f, 464.f};
static constexpr std::string_view equation149 = "bac,cd,fed->bae";
static constexpr std::array<int64_t, 3> shape149{2, 2, 2};
static constexpr std::array<float, 8> expected149{26.f, 46.f, 90.f, 158.f, 154.f, 270.f, 218.f, 382.f};
static constexpr std::string_view equation150 = "bac,cd,fed->bcf";
static constexpr std::array<int64_t, 3> shape150{2, 2, 2};
static constexpr std::array<float, 8> expected150{8.f, 24.f, 64.f, 224.f, 40.f, 120.f, 192.f, 672.f};
static constexpr std::string_view equation151 = "bac,cd,fed->bce";
static constexpr std::array<int64_t, 3> shape151{2, 2, 2};
static constexpr std::array<float, 8> expected151{12.f, 20.f, 104.f, 184.f, 60.f, 100.f, 312.f, 552.f};
static constexpr std::string_view equation152 = "bac,dc,def->bad";
static constexpr std::array<int64_t, 3> shape152{2, 2, 2};
static constexpr std::array<float, 8> expected152{6.f, 66.f, 18.f, 286.f, 30.f, 506.f, 42.f, 726.f};
static constexpr std::string_view equation153 = "bac,dc,def->bae";
static constexpr std::array<int64_t, 3> shape153{2, 2, 2};
static constexpr std::array<float, 8> expected153{28.f, 44.f, 120.f, 184.f, 212.f, 324.f, 304.f, 464.f};
static constexpr std::string_view equation154 = "bac,dc,def->bcd";
static constexpr std::array<int64_t, 3> shape154{2, 2, 2};
static constexpr std::array<float, 8> expected154{0.f, 88.f, 24.f, 264.f, 0.f, 440.f, 72.f, 792.f};
static constexpr std::string_view equation155 = "bac,dc,def->bce";
static constexpr std::array<int64_t, 3> shape155{2, 2, 2};
static constexpr std::array<float, 8> expected155{36.f, 52.f, 112.f, 176.f, 180.f, 260.f, 336.f, 528.f};
static constexpr std::string_view equation156 = "bac,dc,dfe->bad";
static constexpr std::array<int64_t, 3> shape156{2, 2, 2};
static constexpr std::array<float, 8> expected156{6.f, 66.f, 18.f, 286.f, 30.f, 506.f, 42.f, 726.f};
static constexpr std::string_view equation157 = "bac,dc,dfe->baf";
static constexpr std::array<int64_t, 3> shape157{2, 2, 2};
static constexpr std::array<float, 8> expected157{28.f, 44.f, 120.f, 184.f, 212.f, 324.f, 304.f, 464.f};
static constexpr std::string_view equation158 = "bac,dc,dfe->bcd";
static constexpr std::array<int64_t, 3> shape158{2, 2, 2};
static constexpr std::array<float, 8> expected158{0.f, 88.f, 24.f, 264.f, 0.f, 440.f, 72.f, 792.f};
static constexpr std::string_view equation159 = "bac,dc,dfe->bcf";
static constexpr std::array<int64_t, 3> shape159{2, 2, 2};
static constexpr std::array<float, 8> expected159{36.f, 52.f, 112.f, 176.f, 180.f, 260.f, 336.f, 528.f};
static constexpr std::string_view equation160 = "bac,dc,edf->bae";
static constexpr std::array<int64_t, 3> shape160{2, 2, 2};
static constexpr std::array<float, 8> expected160{16.f, 48.f, 68.f, 196.f, 120.f, 344.f, 172.f, 492.f};
static constexpr std::string_view equation161 = "bac,dc,edf->bad";
static constexpr std::array<int64_t, 3> shape161{2, 2, 2};
static constexpr std::array<float, 8> expected161{10.f, 54.f, 30.f, 234.f, 50.f, 414.f, 70.f, 594.f};
static constexpr std::string_view equation162 = "bac,dc,edf->bce";
static constexpr std::array<int64_t, 3> shape162{2, 2, 2};
static constexpr std::array<float, 8> expected162{20.f, 52.f, 64.f, 192.f, 100.f, 260.f, 192.f, 576.f};
static constexpr std::string_view equation163 = "bac,dc,edf->bcd";
static constexpr std::array<int64_t, 3> shape163{2, 2, 2};
static constexpr std::array<float, 8> expected163{0.f, 72.f, 40.f, 216.f, 0.f, 360.f, 120.f, 648.f};
static constexpr std::string_view equation164 = "bac,dc,efd->bae";
static constexpr std::array<int64_t, 3> shape164{2, 2, 2};
static constexpr std::array<float, 8> expected164{14.f, 46.f, 58.f, 186.f, 102.f, 326.f, 146.f, 466.f};
static constexpr std::string_view equation165 = "bac,dc,efd->baf";
static constexpr std::array<int64_t, 3> shape165{2, 2, 2};
static constexpr std::array<float, 8> expected165{22.f, 38.f, 90.f, 154.f, 158.f, 270.f, 226.f, 386.f};
static constexpr std::string_view equation166 = "bac,dc,efd->bce";
static constexpr std::array<int64_t, 3> shape166{2, 2, 2};
static constexpr std::array<float, 8> expected166{16.f, 48.f, 56.f, 184.f, 80.f, 240.f, 168.f, 552.f};
static constexpr std::string_view equation167 = "bac,dc,efd->bcf";
static constexpr std::array<int64_t, 3> shape167{2, 2, 2};
static constexpr std::array<float, 8> expected167{24.f, 40.f, 88.f, 152.f, 120.f, 200.f, 264.f, 456.f};
static constexpr std::string_view equation168 = "bac,dc,fde->baf";
static constexpr std::array<int64_t, 3> shape168{2, 2, 2};
static constexpr std::array<float, 8> expected168{16.f, 48.f, 68.f, 196.f, 120.f, 344.f, 172.f, 492.f};
static constexpr std::string_view equation169 = "bac,dc,fde->bad";
static constexpr std::array<int64_t, 3> shape169{2, 2, 2};
static constexpr std::array<float, 8> expected169{10.f, 54.f, 30.f, 234.f, 50.f, 414.f, 70.f, 594.f};
static constexpr std::string_view equation170 = "bac,dc,fde->bcf";
static constexpr std::array<int64_t, 3> shape170{2, 2, 2};
static constexpr std::array<float, 8> expected170{20.f, 52.f, 64.f, 192.f, 100.f, 260.f, 192.f, 576.f};
static constexpr std::string_view equation171 = "bac,dc,fde->bcd";
static constexpr std::array<int64_t, 3> shape171{2, 2, 2};
static constexpr std::array<float, 8> expected171{0.f, 72.f, 40.f, 216.f, 0.f, 360.f, 120.f, 648.f};
static constexpr std::string_view equation172 = "bac,dc,fed->baf";
static constexpr std::array<int64_t, 3> shape172{2, 2, 2};
static constexpr std::array<float, 8> expected172{14.f, 46.f, 58.f, 186.f, 102.f, 326.f, 146.f, 466.f};
static constexpr std::string_view equation173 = "bac,dc,fed->bae";
static constexpr std::array<int64_t, 3> shape173{2, 2, 2};
static constexpr std::array<float, 8> expected173{22.f, 38.f, 90.f, 154.f, 158.f, 270.f, 226.f, 386.f};
static constexpr std::string_view equation174 = "bac,dc,fed->bcf";
static constexpr std::array<int64_t, 3> shape174{2, 2, 2};
static constexpr std::array<float, 8> expected174{16.f, 48.f, 56.f, 184.f, 80.f, 240.f, 168.f, 552.f};
static constexpr std::string_view equation175 = "bac,dc,fed->bce";
static constexpr std::array<int64_t, 3> shape175{2, 2, 2};
static constexpr std::array<float, 8> expected175{24.f, 40.f, 88.f, 152.f, 120.f, 200.f, 264.f, 456.f};
static constexpr std::string_view equation176 = "bca,cd,def->bcd";
static constexpr std::array<int64_t, 3> shape176{2, 2, 2};
static constexpr std::array<float, 8> expected176{0.f, 22.f, 60.f, 330.f, 0.f, 198.f, 156.f, 858.f};
static constexpr std::string_view equation177 = "bca,cd,def->bce";
static constexpr std::array<int64_t, 3> shape177{2, 2, 2};
static constexpr std::array<float, 8> expected177{9.f, 13.f, 145.f, 245.f, 81.f, 117.f, 377.f, 637.f};
static constexpr std::string_view equation178 = "bca,cd,def->bad";
static constexpr std::array<int64_t, 3> shape178{2, 2, 2};
static constexpr std::array<float, 8> expected178{24.f, 132.f, 36.f, 220.f, 72.f, 484.f, 84.f, 572.f};
static constexpr std::string_view equation179 = "bca,cd,def->bae";
static constexpr std::array<int64_t, 3> shape179{2, 2, 2};
static constexpr std::array<float, 8> expected179{58.f, 98.f, 96.f, 160.f, 210.f, 346.f, 248.f, 408.f};
static constexpr std::string_view equation180 = "bca,cd,dfe->bcd";
static constexpr std::array<int64_t, 3> shape180{2, 2, 2};
static constexpr std::array<float, 8> expected180{0.f, 22.f, 60.f, 330.f, 0.f, 198.f, 156.f, 858.f};
static constexpr std::string_view equation181 = "bca,cd,dfe->bcf";
static constexpr std::array<int64_t, 3> shape181{2, 2, 2};
static constexpr std::array<float, 8> expected181{9.f, 13.f, 145.f, 245.f, 81.f, 117.f, 377.f, 637.f};
static constexpr std::string_view equation182 = "bca,cd,dfe->bad";
static constexpr std::array<int64_t, 3> shape182{2, 2, 2};
static constexpr std::array<float, 8> expected182{24.f, 132.f, 36.f, 220.f, 72.f, 484.f, 84.f, 572.f};
static constexpr std::string_view equation183 = "bca,cd,dfe->baf";
static constexpr std::array<int64_t, 3> shape183{2, 2, 2};
static constexpr std::array<float, 8> expected183{58.f, 98.f, 96.f, 160.f, 210.f, 346.f, 248.f, 408.f};
static constexpr std::string_view equation184 = "bca,cd,edf->bce";
static constexpr std::array<int64_t, 3> shape184{2, 2, 2};
static constexpr std::array<float, 8> expected184{5.f, 13.f, 85.f, 285.f, 45.f, 117.f, 221.f, 741.f};
static constexpr std::string_view equation185 = "bca,cd,edf->bcd";
static constexpr std::array<int64_t, 3> shape185{2, 2, 2};
static constexpr std::array<float, 8> expected185{0.f, 18.f, 100.f, 270.f, 0.f, 162.f, 260.f, 702.f};
static constexpr std::string_view equation186 = "bca,cd,edf->bae";
static constexpr std::array<int64_t, 3> shape186{2, 2, 2};
static constexpr std::array<float, 8> expected186{34.f, 114.f, 56.f, 184.f, 122.f, 394.f, 144.f, 464.f};
static constexpr std::string_view equation187 = "bca,cd,edf->bad";
static constexpr std::array<int64_t, 3> shape187{2, 2, 2};
static constexpr std::array<float, 8> expected187{40.f, 108.f, 60.f, 180.f, 120.f, 396.f, 140.f, 468.f};
static constexpr std::string_view equation188 = "bca,cd,efd->bce";
static constexpr std::array<int64_t, 3> shape188{2, 2, 2};
static constexpr std::array<float, 8> expected188{4.f, 12.f, 80.f, 280.f, 36.f, 108.f, 208.f, 728.f};
static constexpr std::string_view equation189 = "bca,cd,efd->bcf";
static constexpr std::array<int64_t, 3> shape189{2, 2, 2};
static constexpr std::array<float, 8> expected189{6.f, 10.f, 130.f, 230.f, 54.f, 90.f, 338.f, 598.f};
static constexpr std::string_view equation190 = "bca,cd,efd->bae";
static constexpr std::array<int64_t, 3> shape190{2, 2, 2};
static constexpr std::array<float, 8> expected190{32.f, 112.f, 52.f, 180.f, 112.f, 384.f, 132.f, 452.f};
static constexpr std::string_view equation191 = "bca,cd,efd->baf";
static constexpr std::array<int64_t, 3> shape191{2, 2, 2};
static constexpr std::array<float, 8> expected191{52.f, 92.f, 84.f, 148.f, 180.f, 316.f, 212.f, 372.f};
static constexpr std::string_view equation192 = "bca,cd,fde->bcf";
static constexpr std::array<int64_t, 3> shape192{2, 2, 2};
static constexpr std::array<float, 8> expected192{5.f, 13.f, 85.f, 285.f, 45.f, 117.f, 221.f, 741.f};
static constexpr std::string_view equation193 = "bca,cd,fde->bcd";
static constexpr std::array<int64_t, 3> shape193{2, 2, 2};
static constexpr std::array<float, 8> expected193{0.f, 18.f, 100.f, 270.f, 0.f, 162.f, 260.f, 702.f};
static constexpr std::string_view equation194 = "bca,cd,fde->baf";
static constexpr std::array<int64_t, 3> shape194{2, 2, 2};
static constexpr std::array<float, 8> expected194{34.f, 114.f, 56.f, 184.f, 122.f, 394.f, 144.f, 464.f};
static constexpr std::string_view equation195 = "bca,cd,fde->bad";
static constexpr std::array<int64_t, 3> shape195{2, 2, 2};
static constexpr std::array<float, 8> expected195{40.f, 108.f, 60.f, 180.f, 120.f, 396.f, 140.f, 468.f};
static constexpr std::string_view equation196 = "bca,cd,fed->bcf";
static constexpr std::array<int64_t, 3> shape196{2, 2, 2};
static constexpr std::array<float, 8> expected196{4.f, 12.f, 80.f, 280.f, 36.f, 108.f, 208.f, 728.f};
static constexpr std::string_view equation197 = "bca,cd,fed->bce";
static constexpr std::array<int64_t, 3> shape197{2, 2, 2};
static constexpr std::array<float, 8> expected197{6.f, 10.f, 130.f, 230.f, 54.f, 90.f, 338.f, 598.f};
static constexpr std::string_view equation198 = "bca,cd,fed->baf";
static constexpr std::array<int64_t, 3> shape198{2, 2, 2};
static constexpr std::array<float, 8> expected198{32.f, 112.f, 52.f, 180.f, 112.f, 384.f, 132.f, 452.f};
static constexpr std::string_view equation199 = "bca,cd,fed->bae";
static constexpr std::array<int64_t, 3> shape199{2, 2, 2};
static constexpr std::array<float, 8> expected199{52.f, 92.f, 84.f, 148.f, 180.f, 316.f, 212.f, 372.f};
static constexpr std::string_view equation200 = "bca,dc,def->bcd";
static constexpr std::array<int64_t, 3> shape200{2, 2, 2};
static constexpr std::array<float, 8> expected200{0.f, 44.f, 30.f, 330.f, 0.f, 396.f, 78.f, 858.f};
static constexpr std::string_view equation201 = "bca,dc,def->bce";
static constexpr std::array<int64_t, 3> shape201{2, 2, 2};
static constexpr std::array<float, 8> expected201{18.f, 26.f, 140.f, 220.f, 162.f, 234.f, 364.f, 572.f};
static constexpr std::string_view equation202 = "bca,dc,def->bad";
static constexpr std::array<int64_t, 3> shape202{2, 2, 2};
static constexpr std::array<float, 8> expected202{12.f, 132.f, 18.f, 242.f, 36.f, 572.f, 42.f, 682.f};
static constexpr std::string_view equation203 = "bca,dc,def->bae";
static constexpr std::array<int64_t, 3> shape203{2, 2, 2};
static constexpr std::array<float, 8> expected203{56.f, 88.f, 102.f, 158.f, 240.f, 368.f, 286.f, 438.f};
static constexpr std::string_view equation204 = "bca,dc,dfe->bcd";
static constexpr std::array<int64_t, 3> shape204{2, 2, 2};
static constexpr std::array<float, 8> expected204{0.f, 44.f, 30.f, 330.f, 0.f, 396.f, 78.f, 858.f};
static constexpr std::string_view equation205 = "bca,dc,dfe->bcf";
static constexpr std::array<int64_t, 3> shape205{2, 2, 2};
static constexpr std::array<float, 8> expected205{18.f, 26.f, 140.f, 220.f, 162.f, 234.f, 364.f, 572.f};
static constexpr std::string_view equation206 = "bca,dc,dfe->bad";
static constexpr std::array<int64_t, 3> shape206{2, 2, 2};
static constexpr std::array<float, 8> expected206{12.f, 132.f, 18.f, 242.f, 36.f, 572.f, 42.f, 682.f};
static constexpr std::string_view equation207 = "bca,dc,dfe->baf";
static constexpr std::array<int64_t, 3> shape207{2, 2, 2};
static constexpr std::array<float, 8> expected207{56.f, 88.f, 102.f, 158.f, 240.f, 368.f, 286.f, 438.f};
static constexpr std::string_view equation208 = "bca,dc,edf->bce";
static constexpr std::array<int64_t, 3> shape208{2, 2, 2};
static constexpr std::array<float, 8> expected208{10.f, 26.f, 80.f, 240.f, 90.f, 234.f, 208.f, 624.f};
static constexpr std::string_view equation209 = "bca,dc,edf->bcd";
static constexpr std::array<int64_t, 3> shape209{2, 2, 2};
static constexpr std::array<float, 8> expected209{0.f, 36.f, 50.f, 270.f, 0.f, 324.f, 130.f, 702.f};
static constexpr std::string_view equation210 = "bca,dc,edf->bae";
static constexpr std::array<int64_t, 3> shape210{2, 2, 2};
static constexpr std::array<float, 8> expected210{32.f, 96.f, 58.f, 170.f, 136.f, 392.f, 162.f, 466.f};
static constexpr std::string_view equation211 = "bca,dc,edf->bad";
static constexpr std::array<int64_t, 3> shape211{2, 2, 2};
static constexpr std::array<float, 8> expected211{20.f, 108.f, 30.f, 198.f, 60.f, 468.f, 70.f, 558.f};
static constexpr std::string_view equation212 = "bca,dc,efd->bce";
static constexpr std::array<int64_t, 3> shape212{2, 2, 2};
static constexpr std::array<float, 8> expected212{8.f, 24.f, 70.f, 230.f, 72.f, 216.f, 182.f, 598.f};
static constexpr std::string_view equation213 = "bca,dc,efd->bcf";
static constexpr std::array<int64_t, 3> shape213{2, 2, 2};
static constexpr std::array<float, 8> expected213{12.f, 20.f, 110.f, 190.f, 108.f, 180.f, 286.f, 494.f};
static constexpr std::string_view equation214 = "bca,dc,efd->bae";
static constexpr std::array<int64_t, 3> shape214{2, 2, 2};
static constexpr std::array<float, 8> expected214{28.f, 92.f, 50.f, 162.f, 116.f, 372.f, 138.f, 442.f};
static constexpr std::string_view equation215 = "bca,dc,efd->baf";
static constexpr std::array<int64_t, 3> shape215{2, 2, 2};
static constexpr std::array<float, 8> expected215{44.f, 76.f, 78.f, 134.f, 180.f, 308.f, 214.f, 366.f};
static constexpr std::string_view equation216 = "bca,dc,fde->bcf";
static constexpr std::array<int64_t, 3> shape216{2, 2, 2};
static constexpr std::array<float, 8> expected216{10.f, 26.f, 80.f, 240.f, 90.f, 234.f, 208.f, 624.f};
static constexpr std::string_view equation217 = "bca,dc,fde->bcd";
static constexpr std::array<int64_t, 3> shape217{2, 2, 2};
static constexpr std::array<float, 8> expected217{0.f, 36.f, 50.f, 270.f, 0.f, 324.f, 130.f, 702.f};
static constexpr std::string_view equation218 = "bca,dc,fde->baf";
static constexpr std::array<int64_t, 3> shape218{2, 2, 2};
static constexpr std::array<float, 8> expected218{32.f, 96.f, 58.f, 170.f, 136.f, 392.f, 162.f, 466.f};
static constexpr std::string_view equation219 = "bca,dc,fde->bad";
static constexpr std::array<int64_t, 3> shape219{2, 2, 2};
static constexpr std::array<float, 8> expected219{20.f, 108.f, 30.f, 198.f, 60.f, 468.f, 70.f, 558.f};
static constexpr std::string_view equation220 = "bca,dc,fed->bcf";
static constexpr std::array<int64_t, 3> shape220{2, 2, 2};
static constexpr std::array<float, 8> expected220{8.f, 24.f, 70.f, 230.f, 72.f, 216.f, 182.f, 598.f};
static constexpr std::string_view equation221 = "bca,dc,fed->bce";
static constexpr std::array<int64_t, 3> shape221{2, 2, 2};
static constexpr std::array<float, 8> expected221{12.f, 20.f, 110.f, 190.f, 108.f, 180.f, 286.f, 494.f};
static constexpr std::string_view equation222 = "bca,dc,fed->baf";
static constexpr std::array<int64_t, 3> shape222{2, 2, 2};
static constexpr std::array<float, 8> expected222{28.f, 92.f, 50.f, 162.f, 116.f, 372.f, 138.f, 442.f};
static constexpr std::string_view equation223 = "bca,dc,fed->bae";
static constexpr std::array<int64_t, 3> shape223{2, 2, 2};
static constexpr std::array<float, 8> expected223{44.f, 76.f, 78.f, 134.f, 180.f, 308.f, 214.f, 366.f};
static constexpr std::string_view equation224 = "cab,cd,def->cad";
static constexpr std::array<int64_t, 3> shape224{2, 2, 2};
static constexpr std::array<float, 8> expected224{0.f, 22.f, 0.f, 110.f, 108.f, 594.f, 156.f, 858.f};
static constexpr std::string_view equation225 = "cab,cd,def->cae";
static constexpr std::array<int64_t, 3> shape225{2, 2, 2};
static constexpr std::array<float, 8> expected225{9.f, 13.f, 45.f, 65.f, 261.f, 441.f, 377.f, 637.f};
static constexpr std::string_view equation226 = "cab,cd,def->cbd";
static constexpr std::array<int64_t, 3> shape226{2, 2, 2};
static constexpr std::array<float, 8> expected226{0.f, 44.f, 0.f, 88.f, 120.f, 660.f, 144.f, 792.f};
static constexpr std::string_view equation227 = "cab,cd,def->cbe";
static constexpr std::array<int64_t, 3> shape227{2, 2, 2};
static constexpr std::array<float, 8> expected227{18.f, 26.f, 36.f, 52.f, 290.f, 490.f, 348.f, 588.f};
static constexpr std::string_view equation228 = "cab,cd,dfe->cad";
static constexpr std::array<int64_t, 3> shape228{2, 2, 2};
static constexpr std::array<float, 8> expected228{0.f, 22.f, 0.f, 110.f, 108.f, 594.f, 156.f, 858.f};
static constexpr std::string_view equation229 = "cab,cd,dfe->caf";
static constexpr std::array<int64_t, 3> shape229{2, 2, 2};
static constexpr std::array<float, 8> expected229{9.f, 13.f, 45.f, 65.f, 261.f, 441.f, 377.f, 637.f};
static constexpr std::string_view equation230 = "cab,cd,dfe->cbd";
static constexpr std::array<int64_t, 3> shape230{2, 2, 2};
static constexpr std::array<float, 8> expected230{0.f, 44.f, 0.f, 88.f, 120.f, 660.f, 144.f, 792.f};
static constexpr std::string_view equation231 = "cab,cd,dfe->cbf";
static constexpr std::array<int64_t, 3> shape231{2, 2, 2};
static constexpr std::array<float, 8> expected231{18.f, 26.f, 36.f, 52.f, 290.f, 490.f, 348.f, 588.f};
static constexpr std::string_view equation232 = "cab,cd,edf->cae";
static constexpr std::array<int64_t, 3> shape232{2, 2, 2};
static constexpr std::array<float, 8> expected232{5.f, 13.f, 25.f, 65.f, 153.f, 513.f, 221.f, 741.f};
static constexpr std::string_view equation233 = "cab,cd,edf->cad";
static constexpr std::array<int64_t, 3> shape233{2, 2, 2};
static constexpr std::array<float, 8> expected233{0.f, 18.f, 0.f, 90.f, 180.f, 486.f, 260.f, 702.f};
static constexpr std::string_view equation234 = "cab,cd,edf->cbe";
static constexpr std::array<int64_t, 3> shape234{2, 2, 2};
static constexpr std::array<float, 8> expected234{10.f, 26.f, 20.f, 52.f, 170.f, 570.f, 204.f, 684.f};
static constexpr std::string_view equation235 = "cab,cd,edf->cbd";
static constexpr std::array<int64_t, 3> shape235{2, 2, 2};
static constexpr std::array<float, 8> expected235{0.f, 36.f, 0.f, 72.f, 200.f, 540.f, 240.f, 648.f};
static constexpr std::string_view equation236 = "cab,cd,efd->cae";
static constexpr std::array<int64_t, 3> shape236{2, 2, 2};
static constexpr std::array<float, 8> expected236{4.f, 12.f, 20.f, 60.f, 144.f, 504.f, 208.f, 728.f};
static constexpr std::string_view equation237 = "cab,cd,efd->caf";
static constexpr std::array<int64_t, 3> shape237{2, 2, 2};
static constexpr std::array<float, 8> expected237{6.f, 10.f, 30.f, 50.f, 234.f, 414.f, 338.f, 598.f};
static constexpr std::string_view equation238 = "cab,cd,efd->cbe";
static constexpr std::array<int64_t, 3> shape238{2, 2, 2};
static constexpr std::array<float, 8> expected238{8.f, 24.f, 16.f, 48.f, 160.f, 560.f, 192.f, 672.f};
static constexpr std::string_view equation239 = "cab,cd,efd->cbf";
static constexpr std::array<int64_t, 3> shape239{2, 2, 2};
static constexpr std::array<float, 8> expected239{12.f, 20.f, 24.f, 40.f, 260.f, 460.f, 312.f, 552.f};
static constexpr std::string_view equation240 = "cab,cd,fde->caf";
static constexpr std::array<int64_t, 3> shape240{2, 2, 2};
static constexpr std::array<float, 8> expected240{5.f, 13.f, 25.f, 65.f, 153.f, 513.f, 221.f, 741.f};
static constexpr std::string_view equation241 = "cab,cd,fde->cad";
static constexpr std::array<int64_t, 3> shape241{2, 2, 2};
static constexpr std::array<float, 8> expected241{0.f, 18.f, 0.f, 90.f, 180.f, 486.f, 260.f, 702.f};
static constexpr std::string_view equation242 = "cab,cd,fde->cbf";
static constexpr std::array<int64_t, 3> shape242{2, 2, 2};
static constexpr std::array<float, 8> expected242{10.f, 26.f, 20.f, 52.f, 170.f, 570.f, 204.f, 684.f};
static constexpr std::string_view equation243 = "cab,cd,fde->cbd";
static constexpr std::array<int64_t, 3> shape243{2, 2, 2};
static constexpr std::array<float, 8> expected243{0.f, 36.f, 0.f, 72.f, 200.f, 540.f, 240.f, 648.f};
static constexpr std::string_view equation244 = "cab,cd,fed->caf";
static constexpr std::array<int64_t, 3> shape244{2, 2, 2};
static constexpr std::array<float, 8> expected244{4.f, 12.f, 20.f, 60.f, 144.f, 504.f, 208.f, 728.f};
static constexpr std::string_view equation245 = "cab,cd,fed->cae";
static constexpr std::array<int64_t, 3> shape245{2, 2, 2};
static constexpr std::array<float, 8> expected245{6.f, 10.f, 30.f, 50.f, 234.f, 414.f, 338.f, 598.f};
static constexpr std::string_view equation246 = "cab,cd,fed->cbf";
static constexpr std::array<int64_t, 3> shape246{2, 2, 2};
static constexpr std::array<float, 8> expected246{8.f, 24.f, 16.f, 48.f, 160.f, 560.f, 192.f, 672.f};
static constexpr std::string_view equation247 = "cab,cd,fed->cbe";
static constexpr std::array<int64_t, 3> shape247{2, 2, 2};
static constexpr std::array<float, 8> expected247{12.f, 20.f, 24.f, 40.f, 260.f, 460.f, 312.f, 552.f};
static constexpr std::string_view equation248 = "cab,dc,def->cad";
static constexpr std::array<int64_t, 3> shape248{2, 2, 2};
static constexpr std::array<float, 8> expected248{0.f, 44.f, 0.f, 220.f, 54.f, 594.f, 78.f, 858.f};
static constexpr std::string_view equation249 = "cab,dc,def->cae";
static constexpr std::array<int64_t, 3> shape249{2, 2, 2};
static constexpr std::array<float, 8> expected249{18.f, 26.f, 90.f, 130.f, 252.f, 396.f, 364.f, 572.f};
static constexpr std::string_view equation250 = "cab,dc,def->cbd";
static constexpr std::array<int64_t, 3> shape250{2, 2, 2};
static constexpr std::array<float, 8> expected250{0.f, 88.f, 0.f, 176.f, 60.f, 660.f, 72.f, 792.f};

static constexpr std::string_view equation251 = "cab,dc,def->cbe";
static constexpr std::array<int64_t, 3> shape251{2, 2, 2};
static constexpr std::array<float, 8> expected251{36.f, 52.f, 72.f, 104.f, 280.f, 440.f, 336.f, 528.f};
static constexpr std::string_view equation252 = "cab,dc,dfe->cad";
static constexpr std::array<int64_t, 3> shape252{2, 2, 2};
static constexpr std::array<float, 8> expected252{0.f, 44.f, 0.f, 220.f, 54.f, 594.f, 78.f, 858.f};
static constexpr std::string_view equation253 = "cab,dc,dfe->caf";
static constexpr std::array<int64_t, 3> shape253{2, 2, 2};
static constexpr std::array<float, 8> expected253{18.f, 26.f, 90.f, 130.f, 252.f, 396.f, 364.f, 572.f};
static constexpr std::string_view equation254 = "cab,dc,dfe->cbd";
static constexpr std::array<int64_t, 3> shape254{2, 2, 2};
static constexpr std::array<float, 8> expected254{0.f, 88.f, 0.f, 176.f, 60.f, 660.f, 72.f, 792.f};
static constexpr std::string_view equation255 = "cab,dc,dfe->cbf";
static constexpr std::array<int64_t, 3> shape255{2, 2, 2};
static constexpr std::array<float, 8> expected255{36.f, 52.f, 72.f, 104.f, 280.f, 440.f, 336.f, 528.f};
static constexpr std::string_view equation256 = "cab,dc,edf->cae";
static constexpr std::array<int64_t, 3> shape256{2, 2, 2};
static constexpr std::array<float, 8> expected256{10.f, 26.f, 50.f, 130.f, 144.f, 432.f, 208.f, 624.f};
static constexpr std::string_view equation257 = "cab,dc,edf->cad";
static constexpr std::array<int64_t, 3> shape257{2, 2, 2};
static constexpr std::array<float, 8> expected257{0.f, 36.f, 0.f, 180.f, 90.f, 486.f, 130.f, 702.f};
static constexpr std::string_view equation258 = "cab,dc,edf->cbe";
static constexpr std::array<int64_t, 3> shape258{2, 2, 2};
static constexpr std::array<float, 8> expected258{20.f, 52.f, 40.f, 104.f, 160.f, 480.f, 192.f, 576.f};
static constexpr std::string_view equation259 = "cab,dc,edf->cbd";
static constexpr std::array<int64_t, 3> shape259{2, 2, 2};
static constexpr std::array<float, 8> expected259{0.f, 72.f, 0.f, 144.f, 100.f, 540.f, 120.f, 648.f};
static constexpr std::string_view equation260 = "cab,dc,efd->cae";
static constexpr std::array<int64_t, 3> shape260{2, 2, 2};
static constexpr std::array<float, 8> expected260{8.f, 24.f, 40.f, 120.f, 126.f, 414.f, 182.f, 598.f};
static constexpr std::string_view equation261 = "cab,dc,efd->caf";
static constexpr std::array<int64_t, 3> shape261{2, 2, 2};
static constexpr std::array<float, 8> expected261{12.f, 20.f, 60.f, 100.f, 198.f, 342.f, 286.f, 494.f};
static constexpr std::string_view equation262 = "cab,dc,efd->cbe";
static constexpr std::array<int64_t, 3> shape262{2, 2, 2};
static constexpr std::array<float, 8> expected262{16.f, 48.f, 32.f, 96.f, 140.f, 460.f, 168.f, 552.f};
static constexpr std::string_view equation263 = "cab,dc,efd->cbf";
static constexpr std::array<int64_t, 3> shape263{2, 2, 2};
static constexpr std::array<float, 8> expected263{24.f, 40.f, 48.f, 80.f, 220.f, 380.f, 264.f, 456.f};
static constexpr std::string_view equation264 = "cab,dc,fde->caf";
static constexpr std::array<int64_t, 3> shape264{2, 2, 2};
static constexpr std::array<float, 8> expected264{10.f, 26.f, 50.f, 130.f, 144.f, 432.f, 208.f, 624.f};
static constexpr std::string_view equation265 = "cab,dc,fde->cad";
static constexpr std::array<int64_t, 3> shape265{2, 2, 2};
static constexpr std::array<float, 8> expected265{0.f, 36.f, 0.f, 180.f, 90.f, 486.f, 130.f, 702.f};
static constexpr std::string_view equation266 = "cab,dc,fde->cbf";
static constexpr std::array<int64_t, 3> shape266{2, 2, 2};
static constexpr std::array<float, 8> expected266{20.f, 52.f, 40.f, 104.f, 160.f, 480.f, 192.f, 576.f};
static constexpr std::string_view equation267 = "cab,dc,fde->cbd";
static constexpr std::array<int64_t, 3> shape267{2, 2, 2};
static constexpr std::array<float, 8> expected267{0.f, 72.f, 0.f, 144.f, 100.f, 540.f, 120.f, 648.f};
static constexpr std::string_view equation268 = "cab,dc,fed->caf";
static constexpr std::array<int64_t, 3> shape268{2, 2, 2};
static constexpr std::array<float, 8> expected268{8.f, 24.f, 40.f, 120.f, 126.f, 414.f, 182.f, 598.f};
static constexpr std::string_view equation269 = "cab,dc,fed->cae";
static constexpr std::array<int64_t, 3> shape269{2, 2, 2};
static constexpr std::array<float, 8> expected269{12.f, 20.f, 60.f, 100.f, 198.f, 342.f, 286.f, 494.f};
static constexpr std::string_view equation270 = "cab,dc,fed->cbf";
static constexpr std::array<int64_t, 3> shape270{2, 2, 2};
static constexpr std::array<float, 8> expected270{16.f, 48.f, 32.f, 96.f, 140.f, 460.f, 168.f, 552.f};
static constexpr std::string_view equation271 = "cab,dc,fed->cbe";
static constexpr std::array<int64_t, 3> shape271{2, 2, 2};
static constexpr std::array<float, 8> expected271{24.f, 40.f, 48.f, 80.f, 220.f, 380.f, 264.f, 456.f};
static constexpr std::string_view equation272 = "cba,cd,def->cbd";
static constexpr std::array<int64_t, 3> shape272{2, 2, 2};
static constexpr std::array<float, 8> expected272{0.f, 22.f, 0.f, 110.f, 108.f, 594.f, 156.f, 858.f};
static constexpr std::string_view equation273 = "cba,cd,def->cbe";
static constexpr std::array<int64_t, 3> shape273{2, 2, 2};
static constexpr std::array<float, 8> expected273{9.f, 13.f, 45.f, 65.f, 261.f, 441.f, 377.f, 637.f};
static constexpr std::string_view equation274 = "cba,cd,def->cad";
static constexpr std::array<int64_t, 3> shape274{2, 2, 2};
static constexpr std::array<float, 8> expected274{0.f, 44.f, 0.f, 88.f, 120.f, 660.f, 144.f, 792.f};
static constexpr std::string_view equation275 = "cba,cd,def->cae";
static constexpr std::array<int64_t, 3> shape275{2, 2, 2};
static constexpr std::array<float, 8> expected275{18.f, 26.f, 36.f, 52.f, 290.f, 490.f, 348.f, 588.f};
static constexpr std::string_view equation276 = "cba,cd,dfe->cbd";
static constexpr std::array<int64_t, 3> shape276{2, 2, 2};
static constexpr std::array<float, 8> expected276{0.f, 22.f, 0.f, 110.f, 108.f, 594.f, 156.f, 858.f};
static constexpr std::string_view equation277 = "cba,cd,dfe->cbf";
static constexpr std::array<int64_t, 3> shape277{2, 2, 2};
static constexpr std::array<float, 8> expected277{9.f, 13.f, 45.f, 65.f, 261.f, 441.f, 377.f, 637.f};
static constexpr std::string_view equation278 = "cba,cd,dfe->cad";
static constexpr std::array<int64_t, 3> shape278{2, 2, 2};
static constexpr std::array<float, 8> expected278{0.f, 44.f, 0.f, 88.f, 120.f, 660.f, 144.f, 792.f};
static constexpr std::string_view equation279 = "cba,cd,dfe->caf";
static constexpr std::array<int64_t, 3> shape279{2, 2, 2};
static constexpr std::array<float, 8> expected279{18.f, 26.f, 36.f, 52.f, 290.f, 490.f, 348.f, 588.f};
static constexpr std::string_view equation280 = "cba,cd,edf->cbe";
static constexpr std::array<int64_t, 3> shape280{2, 2, 2};
static constexpr std::array<float, 8> expected280{5.f, 13.f, 25.f, 65.f, 153.f, 513.f, 221.f, 741.f};
static constexpr std::string_view equation281 = "cba,cd,edf->cbd";
static constexpr std::array<int64_t, 3> shape281{2, 2, 2};
static constexpr std::array<float, 8> expected281{0.f, 18.f, 0.f, 90.f, 180.f, 486.f, 260.f, 702.f};
static constexpr std::string_view equation282 = "cba,cd,edf->cae";
static constexpr std::array<int64_t, 3> shape282{2, 2, 2};
static constexpr std::array<float, 8> expected282{10.f, 26.f, 20.f, 52.f, 170.f, 570.f, 204.f, 684.f};
static constexpr std::string_view equation283 = "cba,cd,edf->cad";
static constexpr std::array<int64_t, 3> shape283{2, 2, 2};
static constexpr std::array<float, 8> expected283{0.f, 36.f, 0.f, 72.f, 200.f, 540.f, 240.f, 648.f};
static constexpr std::string_view equation284 = "cba,cd,efd->cbe";
static constexpr std::array<int64_t, 3> shape284{2, 2, 2};
static constexpr std::array<float, 8> expected284{4.f, 12.f, 20.f, 60.f, 144.f, 504.f, 208.f, 728.f};
static constexpr std::string_view equation285 = "cba,cd,efd->cbf";
static constexpr std::array<int64_t, 3> shape285{2, 2, 2};
static constexpr std::array<float, 8> expected285{6.f, 10.f, 30.f, 50.f, 234.f, 414.f, 338.f, 598.f};
static constexpr std::string_view equation286 = "cba,cd,efd->cae";
static constexpr std::array<int64_t, 3> shape286{2, 2, 2};
static constexpr std::array<float, 8> expected286{8.f, 24.f, 16.f, 48.f, 160.f, 560.f, 192.f, 672.f};
static constexpr std::string_view equation287 = "cba,cd,efd->caf";
static constexpr std::array<int64_t, 3> shape287{2, 2, 2};
static constexpr std::array<float, 8> expected287{12.f, 20.f, 24.f, 40.f, 260.f, 460.f, 312.f, 552.f};
static constexpr std::string_view equation288 = "cba,cd,fde->cbf";
static constexpr std::array<int64_t, 3> shape288{2, 2, 2};
static constexpr std::array<float, 8> expected288{5.f, 13.f, 25.f, 65.f, 153.f, 513.f, 221.f, 741.f};
static constexpr std::string_view equation289 = "cba,cd,fde->cbd";
static constexpr std::array<int64_t, 3> shape289{2, 2, 2};
static constexpr std::array<float, 8> expected289{0.f, 18.f, 0.f, 90.f, 180.f, 486.f, 260.f, 702.f};
static constexpr std::string_view equation290 = "cba,cd,fde->caf";
static constexpr std::array<int64_t, 3> shape290{2, 2, 2};
static constexpr std::array<float, 8> expected290{10.f, 26.f, 20.f, 52.f, 170.f, 570.f, 204.f, 684.f};
static constexpr std::string_view equation291 = "cba,cd,fde->cad";
static constexpr std::array<int64_t, 3> shape291{2, 2, 2};
static constexpr std::array<float, 8> expected291{0.f, 36.f, 0.f, 72.f, 200.f, 540.f, 240.f, 648.f};
static constexpr std::string_view equation292 = "cba,cd,fed->cbf";
static constexpr std::array<int64_t, 3> shape292{2, 2, 2};
static constexpr std::array<float, 8> expected292{4.f, 12.f, 20.f, 60.f, 144.f, 504.f, 208.f, 728.f};
static constexpr std::string_view equation293 = "cba,cd,fed->cbe";
static constexpr std::array<int64_t, 3> shape293{2, 2, 2};
static constexpr std::array<float, 8> expected293{6.f, 10.f, 30.f, 50.f, 234.f, 414.f, 338.f, 598.f};
static constexpr std::string_view equation294 = "cba,cd,fed->caf";
static constexpr std::array<int64_t, 3> shape294{2, 2, 2};
static constexpr std::array<float, 8> expected294{8.f, 24.f, 16.f, 48.f, 160.f, 560.f, 192.f, 672.f};
static constexpr std::string_view equation295 = "cba,cd,fed->cae";
static constexpr std::array<int64_t, 3> shape295{2, 2, 2};
static constexpr std::array<float, 8> expected295{12.f, 20.f, 24.f, 40.f, 260.f, 460.f, 312.f, 552.f};
static constexpr std::string_view equation296 = "cba,dc,def->cbd";
static constexpr std::array<int64_t, 3> shape296{2, 2, 2};
static constexpr std::array<float, 8> expected296{0.f, 44.f, 0.f, 220.f, 54.f, 594.f, 78.f, 858.f};
static constexpr std::string_view equation297 = "cba,dc,def->cbe";
static constexpr std::array<int64_t, 3> shape297{2, 2, 2};
static constexpr std::array<float, 8> expected297{18.f, 26.f, 90.f, 130.f, 252.f, 396.f, 364.f, 572.f};
static constexpr std::string_view equation298 = "cba,dc,def->cad";
static constexpr std::array<int64_t, 3> shape298{2, 2, 2};
static constexpr std::array<float, 8> expected298{0.f, 88.f, 0.f, 176.f, 60.f, 660.f, 72.f, 792.f};
static constexpr std::string_view equation299 = "cba,dc,def->cae";
static constexpr std::array<int64_t, 3> shape299{2, 2, 2};
static constexpr std::array<float, 8> expected299{36.f, 52.f, 72.f, 104.f, 280.f, 440.f, 336.f, 528.f};
static constexpr std::string_view equation300 = "cba,dc,dfe->cbd";
static constexpr std::array<int64_t, 3> shape300{2, 2, 2};
static constexpr std::array<float, 8> expected300{0.f, 44.f, 0.f, 220.f, 54.f, 594.f, 78.f, 858.f};
static constexpr std::string_view equation301 = "cba,dc,dfe->cbf";
static constexpr std::array<int64_t, 3> shape301{2, 2, 2};
static constexpr std::array<float, 8> expected301{18.f, 26.f, 90.f, 130.f, 252.f, 396.f, 364.f, 572.f};
static constexpr std::string_view equation302 = "cba,dc,dfe->cad";
static constexpr std::array<int64_t, 3> shape302{2, 2, 2};
static constexpr std::array<float, 8> expected302{0.f, 88.f, 0.f, 176.f, 60.f, 660.f, 72.f, 792.f};
static constexpr std::string_view equation303 = "cba,dc,dfe->caf";
static constexpr std::array<int64_t, 3> shape303{2, 2, 2};
static constexpr std::array<float, 8> expected303{36.f, 52.f, 72.f, 104.f, 280.f, 440.f, 336.f, 528.f};
static constexpr std::string_view equation304 = "cba,dc,edf->cbe";
static constexpr std::array<int64_t, 3> shape304{2, 2, 2};
static constexpr std::array<float, 8> expected304{10.f, 26.f, 50.f, 130.f, 144.f, 432.f, 208.f, 624.f};
static constexpr std::string_view equation305 = "cba,dc,edf->cbd";
static constexpr std::array<int64_t, 3> shape305{2, 2, 2};
static constexpr std::array<float, 8> expected305{0.f, 36.f, 0.f, 180.f, 90.f, 486.f, 130.f, 702.f};
static constexpr std::string_view equation306 = "cba,dc,edf->cae";
static constexpr std::array<int64_t, 3> shape306{2, 2, 2};
static constexpr std::array<float, 8> expected306{20.f, 52.f, 40.f, 104.f, 160.f, 480.f, 192.f, 576.f};
static constexpr std::string_view equation307 = "cba,dc,edf->cad";
static constexpr std::array<int64_t, 3> shape307{2, 2, 2};
static constexpr std::array<float, 8> expected307{0.f, 72.f, 0.f, 144.f, 100.f, 540.f, 120.f, 648.f};
static constexpr std::string_view equation308 = "cba,dc,efd->cbe";
static constexpr std::array<int64_t, 3> shape308{2, 2, 2};
static constexpr std::array<float, 8> expected308{8.f, 24.f, 40.f, 120.f, 126.f, 414.f, 182.f, 598.f};
static constexpr std::string_view equation309 = "cba,dc,efd->cbf";
static constexpr std::array<int64_t, 3> shape309{2, 2, 2};
static constexpr std::array<float, 8> expected309{12.f, 20.f, 60.f, 100.f, 198.f, 342.f, 286.f, 494.f};
static constexpr std::string_view equation310 = "cba,dc,efd->cae";
static constexpr std::array<int64_t, 3> shape310{2, 2, 2};
static constexpr std::array<float, 8> expected310{16.f, 48.f, 32.f, 96.f, 140.f, 460.f, 168.f, 552.f};
static constexpr std::string_view equation311 = "cba,dc,efd->caf";
static constexpr std::array<int64_t, 3> shape311{2, 2, 2};
static constexpr std::array<float, 8> expected311{24.f, 40.f, 48.f, 80.f, 220.f, 380.f, 264.f, 456.f};
static constexpr std::string_view equation312 = "cba,dc,fde->cbf";
static constexpr std::array<int64_t, 3> shape312{2, 2, 2};
static constexpr std::array<float, 8> expected312{10.f, 26.f, 50.f, 130.f, 144.f, 432.f, 208.f, 624.f};
static constexpr std::string_view equation313 = "cba,dc,fde->cbd";
static constexpr std::array<int64_t, 3> shape313{2, 2, 2};
static constexpr std::array<float, 8> expected313{0.f, 36.f, 0.f, 180.f, 90.f, 486.f, 130.f, 702.f};
static constexpr std::string_view equation314 = "cba,dc,fde->caf";
static constexpr std::array<int64_t, 3> shape314{2, 2, 2};
static constexpr std::array<float, 8> expected314{20.f, 52.f, 40.f, 104.f, 160.f, 480.f, 192.f, 576.f};
static constexpr std::string_view equation315 = "cba,dc,fde->cad";
static constexpr std::array<int64_t, 3> shape315{2, 2, 2};
static constexpr std::array<float, 8> expected315{0.f, 72.f, 0.f, 144.f, 100.f, 540.f, 120.f, 648.f};
static constexpr std::string_view equation316 = "cba,dc,fed->cbf";
static constexpr std::array<int64_t, 3> shape316{2, 2, 2};
static constexpr std::array<float, 8> expected316{8.f, 24.f, 40.f, 120.f, 126.f, 414.f, 182.f, 598.f};
static constexpr std::string_view equation317 = "cba,dc,fed->cbe";
static constexpr std::array<int64_t, 3> shape317{2, 2, 2};
static constexpr std::array<float, 8> expected317{12.f, 20.f, 60.f, 100.f, 198.f, 342.f, 286.f, 494.f};
static constexpr std::string_view equation318 = "cba,dc,fed->caf";
static constexpr std::array<int64_t, 3> shape318{2, 2, 2};
static constexpr std::array<float, 8> expected318{16.f, 48.f, 32.f, 96.f, 140.f, 460.f, 168.f, 552.f};
static constexpr std::string_view equation319 = "cba,dc,fed->cae";
static constexpr std::array<int64_t, 3> shape319{2, 2, 2};
static constexpr std::array<float, 8> expected319{24.f, 40.f, 48.f, 80.f, 220.f, 380.f, 264.f, 456.f};
static constexpr std::array<EinsumTestCase, 288> case1 = {{{equation32, shape32, expected32},
                                                           {equation33, shape33, expected33},
                                                           {equation34, shape34, expected34},
                                                           {equation35, shape35, expected35},
                                                           {equation36, shape36, expected36},
                                                           {equation37, shape37, expected37},
                                                           {equation38, shape38, expected38},
                                                           {equation39, shape39, expected39},
                                                           {equation40, shape40, expected40},
                                                           {equation41, shape41, expected41},
                                                           {equation42, shape42, expected42},
                                                           {equation43, shape43, expected43},
                                                           {equation44, shape44, expected44},
                                                           {equation45, shape45, expected45},
                                                           {equation46, shape46, expected46},
                                                           {equation47, shape47, expected47},
                                                           {equation48, shape48, expected48},
                                                           {equation49, shape49, expected49},
                                                           {equation50, shape50, expected50},
                                                           {equation51, shape51, expected51},
                                                           {equation52, shape52, expected52},
                                                           {equation53, shape53, expected53},
                                                           {equation54, shape54, expected54},
                                                           {equation55, shape55, expected55},
                                                           {equation56, shape56, expected56},
                                                           {equation57, shape57, expected57},
                                                           {equation58, shape58, expected58},
                                                           {equation59, shape59, expected59},
                                                           {equation60, shape60, expected60},
                                                           {equation61, shape61, expected61},
                                                           {equation62, shape62, expected62},
                                                           {equation63, shape63, expected63},
                                                           {equation64, shape64, expected64},
                                                           {equation65, shape65, expected65},
                                                           {equation66, shape66, expected66},
                                                           {equation67, shape67, expected67},
                                                           {equation68, shape68, expected68},
                                                           {equation69, shape69, expected69},
                                                           {equation70, shape70, expected70},
                                                           {equation71, shape71, expected71},
                                                           {equation72, shape72, expected72},
                                                           {equation73, shape73, expected73},
                                                           {equation74, shape74, expected74},
                                                           {equation75, shape75, expected75},
                                                           {equation76, shape76, expected76},
                                                           {equation77, shape77, expected77},
                                                           {equation78, shape78, expected78},
                                                           {equation79, shape79, expected79},
                                                           {equation80, shape80, expected80},
                                                           {equation81, shape81, expected81},
                                                           {equation82, shape82, expected82},
                                                           {equation83, shape83, expected83},
                                                           {equation84, shape84, expected84},
                                                           {equation85, shape85, expected85},
                                                           {equation86, shape86, expected86},
                                                           {equation87, shape87, expected87},
                                                           {equation88, shape88, expected88},
                                                           {equation89, shape89, expected89},
                                                           {equation90, shape90, expected90},
                                                           {equation91, shape91, expected91},
                                                           {equation92, shape92, expected92},
                                                           {equation93, shape93, expected93},
                                                           {equation94, shape94, expected94},
                                                           {equation95, shape95, expected95},
                                                           {equation96, shape96, expected96},
                                                           {equation97, shape97, expected97},
                                                           {equation98, shape98, expected98},
                                                           {equation99, shape99, expected99},
                                                           {equation100, shape100, expected100},
                                                           {equation101, shape101, expected101},
                                                           {equation102, shape102, expected102},
                                                           {equation103, shape103, expected103},
                                                           {equation104, shape104, expected104},
                                                           {equation105, shape105, expected105},
                                                           {equation106, shape106, expected106},
                                                           {equation107, shape107, expected107},
                                                           {equation108, shape108, expected108},
                                                           {equation109, shape109, expected109},
                                                           {equation110, shape110, expected110},
                                                           {equation111, shape111, expected111},
                                                           {equation112, shape112, expected112},
                                                           {equation113, shape113, expected113},
                                                           {equation114, shape114, expected114},
                                                           {equation115, shape115, expected115},
                                                           {equation116, shape116, expected116},
                                                           {equation117, shape117, expected117},
                                                           {equation118, shape118, expected118},
                                                           {equation119, shape119, expected119},
                                                           {equation120, shape120, expected120},
                                                           {equation121, shape121, expected121},
                                                           {equation122, shape122, expected122},
                                                           {equation123, shape123, expected123},
                                                           {equation124, shape124, expected124},
                                                           {equation125, shape125, expected125},
                                                           {equation126, shape126, expected126},
                                                           {equation127, shape127, expected127},
                                                           {equation128, shape128, expected128},
                                                           {equation129, shape129, expected129},
                                                           {equation130, shape130, expected130},
                                                           {equation131, shape131, expected131},
                                                           {equation132, shape132, expected132},
                                                           {equation133, shape133, expected133},
                                                           {equation134, shape134, expected134},
                                                           {equation135, shape135, expected135},
                                                           {equation136, shape136, expected136},
                                                           {equation137, shape137, expected137},
                                                           {equation138, shape138, expected138},
                                                           {equation139, shape139, expected139},
                                                           {equation140, shape140, expected140},
                                                           {equation141, shape141, expected141},
                                                           {equation142, shape142, expected142},
                                                           {equation143, shape143, expected143},
                                                           {equation144, shape144, expected144},
                                                           {equation145, shape145, expected145},
                                                           {equation146, shape146, expected146},
                                                           {equation147, shape147, expected147},
                                                           {equation148, shape148, expected148},
                                                           {equation149, shape149, expected149},
                                                           {equation150, shape150, expected150},
                                                           {equation151, shape151, expected151},
                                                           {equation152, shape152, expected152},
                                                           {equation153, shape153, expected153},
                                                           {equation154, shape154, expected154},
                                                           {equation155, shape155, expected155},
                                                           {equation156, shape156, expected156},
                                                           {equation157, shape157, expected157},
                                                           {equation158, shape158, expected158},
                                                           {equation159, shape159, expected159},
                                                           {equation160, shape160, expected160},
                                                           {equation161, shape161, expected161},
                                                           {equation162, shape162, expected162},
                                                           {equation163, shape163, expected163},
                                                           {equation164, shape164, expected164},
                                                           {equation165, shape165, expected165},
                                                           {equation166, shape166, expected166},
                                                           {equation167, shape167, expected167},
                                                           {equation168, shape168, expected168},
                                                           {equation169, shape169, expected169},
                                                           {equation170, shape170, expected170},
                                                           {equation171, shape171, expected171},
                                                           {equation172, shape172, expected172},
                                                           {equation173, shape173, expected173},
                                                           {equation174, shape174, expected174},
                                                           {equation175, shape175, expected175},
                                                           {equation176, shape176, expected176},
                                                           {equation177, shape177, expected177},
                                                           {equation178, shape178, expected178},
                                                           {equation179, shape179, expected179},
                                                           {equation180, shape180, expected180},
                                                           {equation181, shape181, expected181},
                                                           {equation182, shape182, expected182},
                                                           {equation183, shape183, expected183},
                                                           {equation184, shape184, expected184},
                                                           {equation185, shape185, expected185},
                                                           {equation186, shape186, expected186},
                                                           {equation187, shape187, expected187},
                                                           {equation188, shape188, expected188},
                                                           {equation189, shape189, expected189},
                                                           {equation190, shape190, expected190},
                                                           {equation191, shape191, expected191},
                                                           {equation192, shape192, expected192},
                                                           {equation193, shape193, expected193},
                                                           {equation194, shape194, expected194},
                                                           {equation195, shape195, expected195},
                                                           {equation196, shape196, expected196},
                                                           {equation197, shape197, expected197},
                                                           {equation198, shape198, expected198},
                                                           {equation199, shape199, expected199},
                                                           {equation200, shape200, expected200},
                                                           {equation201, shape201, expected201},
                                                           {equation202, shape202, expected202},
                                                           {equation203, shape203, expected203},
                                                           {equation204, shape204, expected204},
                                                           {equation205, shape205, expected205},
                                                           {equation206, shape206, expected206},
                                                           {equation207, shape207, expected207},
                                                           {equation208, shape208, expected208},
                                                           {equation209, shape209, expected209},
                                                           {equation210, shape210, expected210},
                                                           {equation211, shape211, expected211},
                                                           {equation212, shape212, expected212},
                                                           {equation213, shape213, expected213},
                                                           {equation214, shape214, expected214},
                                                           {equation215, shape215, expected215},
                                                           {equation216, shape216, expected216},
                                                           {equation217, shape217, expected217},
                                                           {equation218, shape218, expected218},
                                                           {equation219, shape219, expected219},
                                                           {equation220, shape220, expected220},
                                                           {equation221, shape221, expected221},
                                                           {equation222, shape222, expected222},
                                                           {equation223, shape223, expected223},
                                                           {equation224, shape224, expected224},
                                                           {equation225, shape225, expected225},
                                                           {equation226, shape226, expected226},
                                                           {equation227, shape227, expected227},
                                                           {equation228, shape228, expected228},
                                                           {equation229, shape229, expected229},
                                                           {equation230, shape230, expected230},
                                                           {equation231, shape231, expected231},
                                                           {equation232, shape232, expected232},
                                                           {equation233, shape233, expected233},
                                                           {equation234, shape234, expected234},
                                                           {equation235, shape235, expected235},
                                                           {equation236, shape236, expected236},
                                                           {equation237, shape237, expected237},
                                                           {equation238, shape238, expected238},
                                                           {equation239, shape239, expected239},
                                                           {equation240, shape240, expected240},
                                                           {equation241, shape241, expected241},
                                                           {equation242, shape242, expected242},
                                                           {equation243, shape243, expected243},
                                                           {equation244, shape244, expected244},
                                                           {equation245, shape245, expected245},
                                                           {equation246, shape246, expected246},
                                                           {equation247, shape247, expected247},
                                                           {equation248, shape248, expected248},
                                                           {equation249, shape249, expected249},
                                                           {equation250, shape250, expected250},
                                                           {equation251, shape251, expected251},
                                                           {equation252, shape252, expected252},
                                                           {equation253, shape253, expected253},
                                                           {equation254, shape254, expected254},
                                                           {equation255, shape255, expected255},
                                                           {equation256, shape256, expected256},
                                                           {equation257, shape257, expected257},
                                                           {equation258, shape258, expected258},
                                                           {equation259, shape259, expected259},
                                                           {equation260, shape260, expected260},
                                                           {equation261, shape261, expected261},
                                                           {equation262, shape262, expected262},
                                                           {equation263, shape263, expected263},
                                                           {equation264, shape264, expected264},
                                                           {equation265, shape265, expected265},
                                                           {equation266, shape266, expected266},
                                                           {equation267, shape267, expected267},
                                                           {equation268, shape268, expected268},
                                                           {equation269, shape269, expected269},
                                                           {equation270, shape270, expected270},
                                                           {equation271, shape271, expected271},
                                                           {equation272, shape272, expected272},
                                                           {equation273, shape273, expected273},
                                                           {equation274, shape274, expected274},
                                                           {equation275, shape275, expected275},
                                                           {equation276, shape276, expected276},
                                                           {equation277, shape277, expected277},
                                                           {equation278, shape278, expected278},
                                                           {equation279, shape279, expected279},
                                                           {equation280, shape280, expected280},
                                                           {equation281, shape281, expected281},
                                                           {equation282, shape282, expected282},
                                                           {equation283, shape283, expected283},
                                                           {equation284, shape284, expected284},
                                                           {equation285, shape285, expected285},
                                                           {equation286, shape286, expected286},
                                                           {equation287, shape287, expected287},
                                                           {equation288, shape288, expected288},
                                                           {equation289, shape289, expected289},
                                                           {equation290, shape290, expected290},
                                                           {equation291, shape291, expected291},
                                                           {equation292, shape292, expected292},
                                                           {equation293, shape293, expected293},
                                                           {equation294, shape294, expected294},
                                                           {equation295, shape295, expected295},
                                                           {equation296, shape296, expected296},
                                                           {equation297, shape297, expected297},
                                                           {equation298, shape298, expected298},
                                                           {equation299, shape299, expected299},
                                                           {equation300, shape300, expected300},
                                                           {equation301, shape301, expected301},
                                                           {equation302, shape302, expected302},
                                                           {equation303, shape303, expected303},
                                                           {equation304, shape304, expected304},
                                                           {equation305, shape305, expected305},
                                                           {equation306, shape306, expected306},
                                                           {equation307, shape307, expected307},
                                                           {equation308, shape308, expected308},
                                                           {equation309, shape309, expected309},
                                                           {equation310, shape310, expected310},
                                                           {equation311, shape311, expected311},
                                                           {equation312, shape312, expected312},
                                                           {equation313, shape313, expected313},
                                                           {equation314, shape314, expected314},
                                                           {equation315, shape315, expected315},
                                                           {equation316, shape316, expected316},
                                                           {equation317, shape317, expected317},
                                                           {equation318, shape318, expected318},
                                                           {equation319, shape319, expected319}}};

TEST(Einsum, EinsumTransposeMatMulTwoInputsTestSuite) {
  std::vector<float> m1{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> m2{0.f, 1.f, 2.f, 3.f};
  for (const auto& tst : case0) {
    OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
    std::string s(tst.equation);
    test.AddAttribute<std::string>("equation", s);
    test.AddInput<float>("x", {2, 2, 2}, m1);
    test.AddInput<float>("y", {2, 2}, m2);

    std::vector<int64_t> v1(tst.shape.begin(), tst.shape.end());
    std::vector<float> v2(tst.expected.begin(), tst.expected.end());
    test.AddOutput<float>("o", v1, v2);
    test.Run();
  }
}

class EinsumTransposeMatMulThreeInputsTest : public testing::TestWithParam<EinsumTestCase> {
};

TEST_P(EinsumTransposeMatMulThreeInputsTest, EinsumTransposeMatMulThreeInputsTestSuite) {
  const auto& tst = GetParam();
  std::vector<float> m1{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> m2{0.f, 1.f, 2.f, 3.f};
  std::vector<float> m3{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  std::string s(tst.equation);
  test.AddAttribute<std::string>("equation", s);
  test.AddInput<float>("x", {2, 2, 2}, m1);
  test.AddInput<float>("y", {2, 2}, m2);
  test.AddInput<float>("z", {2, 2, 2}, m3);
  std::vector<int64_t> v1(tst.shape.begin(), tst.shape.end());
  std::vector<float> v2(tst.expected.begin(), tst.expected.end());
  test.AddOutput<float>("o", v1, v2);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

INSTANTIATE_TEST_SUITE_P(EinsumTransposeMatMulThreeInputsTests, EinsumTransposeMatMulThreeInputsTest, testing::ValuesIn(case1));

}  // namespace test
}  // namespace onnxruntime

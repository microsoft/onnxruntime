// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "core/framework/data_types.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

// Tests are aplit up "theme-wise" (i.e.) each kind of operation Einsum can be used for
// Within each theme we test "explicit" and "implicit" versions of the Einsum equation (wherever possible)
// Some operations are not possible with implicit notation (reordering, reduction, etc.)

// Theme: Deep copy / No-op

// Explicit
TEST(Einsum, ExplicitEinsumAsIdentity_1D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i->i");
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.Run();
}

// Implicit
TEST(Einsum, ImplicitEinsumAsIdentity_1D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i");
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.Run();
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
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsBatchedTransposeOp_3D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ji->...ij");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2, 2, 2}, {1.f, 3.f, 2.f, 4.f, 1.f, 3.f, 2.f, 4.f});
  test.Run();
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
  test.Run();
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
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsBatchedReduceOp_3D_input_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ji->...");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("y", {2}, {10.f, 10.f});
  test.Run();
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
  test.Run();
}

// Implicit
TEST(Einsum, ImplicitEinsumAsOuterProductOp_2D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i,j,k");
  test.AddInput<float>("x", {2}, {1.f, 2.f});
  test.AddInput<float>("y", {2}, {3.f, 4.f});
  test.AddInput<float>("z", {2}, {5.f, 6.f});
  test.AddOutput<float>("o", {2, 2, 2}, {15.f, 18.f, 20.f, 24.f, 30.f, 36.f, 40.f, 48.f});
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsOuterProductOp_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i,j,k");
  test.AddInput<float>("x", {2}, {1.f, 2.f});
  test.AddInput<float>("y", {2}, {3.f, 4.f});
  test.AddInput<float>("z", {2}, {5.f, 6.f});
  test.AddOutput<float>("o", {2, 2, 2}, {15.f, 18.f, 20.f, 24.f, 30.f, 36.f, 40.f, 48.f});
  test.Run();
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
  test.Run();
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
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsBatchedMatmulWithBroadcasting_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ij,bjk->...ik");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2, 2}, {14.f, 20.f, 30.f, 44.f, 14.f, 20.f, 30.f, 44.f});
  test.Run();
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
  test.Run();
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
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsMatmul_2) {
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
  test.Run();
}

// Theme: Diagonal parsing

// Explicit
TEST(Einsum, ExplicitEinsumAsDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ii->i");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2}, {1.f, 4.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsDiagonalOp_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iii->i");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2}, {1.f, 4.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithAxisReduced) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->j");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2}, {3.f, 7.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithAxisPreserved) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ij");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {1.f, 3.f, 2.f, 4.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithTranspose) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ji");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithTranspose_double) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ji");
  test.AddInput<double>("x", {2, 2, 2}, {1., 2., 3., 4., 1., 2., 3., 4.});
  test.AddOutput<double>("o", {2, 2}, {1., 2., 3., 4.});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithTranspose_int32) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ji");
  test.AddInput<int32_t>("x", {2, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4});
  test.AddOutput<int32_t>("o", {2, 2}, {1, 2, 3, 4});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsDiagonalOpWithTranspose_int64) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji->ji");
  test.AddInput<int64_t>("x", {2, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4});
  test.AddOutput<int64_t>("o", {2, 2}, {1, 2, 3, 4});
  test.Run();
}
TEST(Einsum, ExplicitEinsumAsBatchedDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ii->...i");
  test.AddInput<float>("x", {3, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {3, 2}, {1.f, 4.f, 1.f, 4.f, 1.f, 4.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsBatchedDiagonalOp_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...iij->...j");
  test.AddInput<float>("x", {2, 2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {4.f, 6.f, 4.f, 6.f});
  test.Run();
}

// Implicit (Implicit diagonal ops will sum up diagonal values)
TEST(Einsum, ImplicitEinsumAsDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ii");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {}, {5.f});
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsDiagonalOp_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iii");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {}, {5.f});
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsDiagonalOpWithAxisReduced) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "iji");
  test.AddInput<float>("x", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2}, {3.f, 7.f});
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsBatchedDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ii");
  test.AddInput<float>("x", {2, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 1}, {5.f, 5.f});
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsBatchedDiagonalOp_1) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...iij");
  test.AddInput<float>("x", {2, 2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {4.f, 6.f, 4.f, 6.f});
  test.Run();
}

// Theme: Scalar inputs and outputs

// Explicit
TEST(Einsum, ExplicitEinsumAsElementwiseMulOpWithOneScalar) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",...i->...i");
  test.AddInput<float>("x", {}, {10.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {10.f, 20.f, 30.f, 40.f});
  test.Run();
}

TEST(Einsum, ExplicitEinsumAsElementwiseMulOpWithTwoScalars_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",...i,->...i");
  test.AddInput<float>("x", {}, {10.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("z", {}, {10.f});
  test.AddOutput<float>("o", {2, 2}, {100.f, 200.f, 300.f, 400.f});
  test.Run();
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
  test.Run();
}

TEST(Einsum, ImplicitEinsumAsElementwiseMulOpWithThreeScalars_Multi_Input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", ",...i,,");
  test.AddInput<float>("a", {}, {10.f});
  test.AddInput<float>("b", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("c", {}, {10.f});
  test.AddInput<float>("d", {}, {10.f});
  test.AddOutput<float>("o", {2, 2}, {1000.f, 2000.f, 3000.f, 4000.f});
  test.Run();
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
  test.Run();
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

// Test each theme for half support
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
  test.Run();
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
  test.Run();
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

struct EinsumTestCase {
  std::string equation;
  std::vector<int64_t> shape;
  std::vector<float> expected;
  EinsumTestCase(const std::string& eq, const std::vector<int64_t>& sh, const std::vector<float>& exp) : equation(eq), shape(sh), expected(exp) {}
};

TEST(Einsum, EinsumTransposeMatMulN2) {
  std::vector<EinsumTestCase> test_cases{
      EinsumTestCase("abc,cd->abc", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 5.f, 2.f, 15.f, 4.f, 25.f, 6.f, 35.f}),
      EinsumTestCase("abc,cd->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{2.f, 3.f, 6.f, 11.f, 10.f, 19.f, 14.f, 27.f}),
      EinsumTestCase("abc,cd->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 2.f, 8.f, 12.f, 0.f, 10.f, 24.f, 36.f}),
      EinsumTestCase("abc,dc->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{1.f, 3.f, 3.f, 13.f, 5.f, 23.f, 7.f, 33.f}),
      EinsumTestCase("abc,dc->abc", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 4.f, 4.f, 12.f, 8.f, 20.f, 12.f, 28.f}),
      EinsumTestCase("abc,dc->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 4.f, 4.f, 12.f, 0.f, 20.f, 12.f, 36.f}),
      EinsumTestCase("acb,cd->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 1.f, 10.f, 15.f, 0.f, 9.f, 26.f, 39.f}),
      EinsumTestCase("acb,cd->abc", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 10.f, 1.f, 15.f, 4.f, 30.f, 5.f, 35.f}),
      EinsumTestCase("acb,cd->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 6.f, 6.f, 10.f, 12.f, 22.f, 14.f, 26.f}),
      EinsumTestCase("acb,dc->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 2.f, 5.f, 15.f, 0.f, 18.f, 13.f, 39.f}),
      EinsumTestCase("acb,dc->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{2.f, 6.f, 3.f, 11.f, 6.f, 26.f, 7.f, 31.f}),
      EinsumTestCase("acb,dc->abc", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 8.f, 2.f, 12.f, 8.f, 24.f, 10.f, 28.f}),
      EinsumTestCase("bac,cd->bac", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 5.f, 2.f, 15.f, 4.f, 25.f, 6.f, 35.f}),
      EinsumTestCase("bac,cd->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{2.f, 3.f, 6.f, 11.f, 10.f, 19.f, 14.f, 27.f}),
      EinsumTestCase("bac,cd->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 2.f, 8.f, 12.f, 0.f, 10.f, 24.f, 36.f}),
      EinsumTestCase("bac,dc->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{1.f, 3.f, 3.f, 13.f, 5.f, 23.f, 7.f, 33.f}),
      EinsumTestCase("bac,dc->bac", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 4.f, 4.f, 12.f, 8.f, 20.f, 12.f, 28.f}),
      EinsumTestCase("bac,dc->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 4.f, 4.f, 12.f, 0.f, 20.f, 12.f, 36.f}),
      EinsumTestCase("bca,cd->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 1.f, 10.f, 15.f, 0.f, 9.f, 26.f, 39.f}),
      EinsumTestCase("bca,cd->bac", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 10.f, 1.f, 15.f, 4.f, 30.f, 5.f, 35.f}),
      EinsumTestCase("bca,cd->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 6.f, 6.f, 10.f, 12.f, 22.f, 14.f, 26.f}),
      EinsumTestCase("bca,dc->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 2.f, 5.f, 15.f, 0.f, 18.f, 13.f, 39.f}),
      EinsumTestCase("bca,dc->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{2.f, 6.f, 3.f, 11.f, 6.f, 26.f, 7.f, 31.f}),
      EinsumTestCase("bca,dc->bac", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 8.f, 2.f, 12.f, 8.f, 24.f, 10.f, 28.f}),
      EinsumTestCase("cab,cd->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 1.f, 0.f, 5.f, 18.f, 27.f, 26.f, 39.f}),
      EinsumTestCase("cab,cd->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 2.f, 0.f, 4.f, 20.f, 30.f, 24.f, 36.f}),
      EinsumTestCase("cab,dc->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 2.f, 0.f, 10.f, 9.f, 27.f, 13.f, 39.f}),
      EinsumTestCase("cab,dc->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 4.f, 0.f, 8.f, 10.f, 30.f, 12.f, 36.f}),
      EinsumTestCase("cba,cd->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 1.f, 0.f, 5.f, 18.f, 27.f, 26.f, 39.f}),
      EinsumTestCase("cba,cd->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 2.f, 0.f, 4.f, 20.f, 30.f, 24.f, 36.f}),
      EinsumTestCase("cba,dc->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 2.f, 0.f, 10.f, 9.f, 27.f, 13.f, 39.f}),
      EinsumTestCase("cba,dc->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 4.f, 0.f, 8.f, 10.f, 30.f, 12.f, 36.f})};

  std::vector<float> m1{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> m2{0.f, 1.f, 2.f, 3.f};
  for (auto tst : test_cases) {
    OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
    test.AddAttribute<std::string>("equation", tst.equation);
    test.AddInput<float>("x", {2, 2, 2}, m1);
    test.AddInput<float>("y", {2, 2}, m2);
    test.AddOutput<float>("o", tst.shape, tst.expected);
    test.Run();
  }
}

TEST(Einsum, EinsumTransposeMatMulN3bug) {
  std::vector<EinsumTestCase> test_cases{
      EinsumTestCase("abc,cd,def->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f})};

  std::vector<float> m1{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> m2{0.f, 1.f, 2.f, 3.f};
  std::vector<float> m3{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  for (auto tst : test_cases) {
    OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
    test.AddAttribute<std::string>("equation", tst.equation);
    test.AddInput<float>("x", {2, 2, 2}, m1);
    test.AddInput<float>("y", {2, 2}, m2);
    test.AddInput<float>("z", {2, 2, 2}, m3);
    test.AddOutput<float>("o", tst.shape, tst.expected);
    test.Run();
  }
}

TEST(Einsum, EinsumTransposeMatMulN3) {
  std::vector<EinsumTestCase> test_cases{
      EinsumTestCase("abc,cd,def->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f}),
      EinsumTestCase("abc,cd,def->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{29.f, 49.f, 105.f, 173.f, 181.f, 297.f, 257.f, 421.f}),
      EinsumTestCase("abc,cd,def->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 48.f, 264.f, 0.f, 220.f, 144.f, 792.f}),
      EinsumTestCase("abc,cd,def->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 116.f, 196.f, 90.f, 130.f, 348.f, 588.f}),
      EinsumTestCase("abc,cd,dfe->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f}),
      EinsumTestCase("abc,cd,dfe->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{29.f, 49.f, 105.f, 173.f, 181.f, 297.f, 257.f, 421.f}),
      EinsumTestCase("abc,cd,dfe->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 48.f, 264.f, 0.f, 220.f, 144.f, 792.f}),
      EinsumTestCase("abc,cd,dfe->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 116.f, 196.f, 90.f, 130.f, 348.f, 588.f}),
      EinsumTestCase("abc,cd,edf->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{17.f, 57.f, 61.f, 197.f, 105.f, 337.f, 149.f, 477.f}),
      EinsumTestCase("abc,cd,edf->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 54.f, 60.f, 198.f, 100.f, 342.f, 140.f, 486.f}),
      EinsumTestCase("abc,cd,edf->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 68.f, 228.f, 50.f, 130.f, 204.f, 684.f}),
      EinsumTestCase("abc,cd,edf->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 80.f, 216.f, 0.f, 180.f, 240.f, 648.f}),
      EinsumTestCase("abc,cd,efd->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 56.f, 56.f, 192.f, 96.f, 328.f, 136.f, 464.f}),
      EinsumTestCase("abc,cd,efd->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{26.f, 46.f, 90.f, 158.f, 154.f, 270.f, 218.f, 382.f}),
      EinsumTestCase("abc,cd,efd->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 64.f, 224.f, 40.f, 120.f, 192.f, 672.f}),
      EinsumTestCase("abc,cd,efd->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 104.f, 184.f, 60.f, 100.f, 312.f, 552.f}),
      EinsumTestCase("abc,cd,fde->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{17.f, 57.f, 61.f, 197.f, 105.f, 337.f, 149.f, 477.f}),
      EinsumTestCase("abc,cd,fde->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 54.f, 60.f, 198.f, 100.f, 342.f, 140.f, 486.f}),
      EinsumTestCase("abc,cd,fde->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 68.f, 228.f, 50.f, 130.f, 204.f, 684.f}),
      EinsumTestCase("abc,cd,fde->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 80.f, 216.f, 0.f, 180.f, 240.f, 648.f}),
      EinsumTestCase("abc,cd,fed->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 56.f, 56.f, 192.f, 96.f, 328.f, 136.f, 464.f}),
      EinsumTestCase("abc,cd,fed->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{26.f, 46.f, 90.f, 158.f, 154.f, 270.f, 218.f, 382.f}),
      EinsumTestCase("abc,cd,fed->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 64.f, 224.f, 40.f, 120.f, 192.f, 672.f}),
      EinsumTestCase("abc,cd,fed->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 104.f, 184.f, 60.f, 100.f, 312.f, 552.f}),
      EinsumTestCase("abc,dc,def->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 66.f, 18.f, 286.f, 30.f, 506.f, 42.f, 726.f}),
      EinsumTestCase("abc,dc,def->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{28.f, 44.f, 120.f, 184.f, 212.f, 324.f, 304.f, 464.f}),
      EinsumTestCase("abc,dc,def->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 88.f, 24.f, 264.f, 0.f, 440.f, 72.f, 792.f}),
      EinsumTestCase("abc,dc,def->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{36.f, 52.f, 112.f, 176.f, 180.f, 260.f, 336.f, 528.f}),
      EinsumTestCase("abc,dc,dfe->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 66.f, 18.f, 286.f, 30.f, 506.f, 42.f, 726.f}),
      EinsumTestCase("abc,dc,dfe->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{28.f, 44.f, 120.f, 184.f, 212.f, 324.f, 304.f, 464.f}),
      EinsumTestCase("abc,dc,dfe->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 88.f, 24.f, 264.f, 0.f, 440.f, 72.f, 792.f}),
      EinsumTestCase("abc,dc,dfe->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{36.f, 52.f, 112.f, 176.f, 180.f, 260.f, 336.f, 528.f}),
      EinsumTestCase("abc,dc,edf->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 68.f, 196.f, 120.f, 344.f, 172.f, 492.f}),
      EinsumTestCase("abc,dc,edf->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 54.f, 30.f, 234.f, 50.f, 414.f, 70.f, 594.f}),
      EinsumTestCase("abc,dc,edf->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 52.f, 64.f, 192.f, 100.f, 260.f, 192.f, 576.f}),
      EinsumTestCase("abc,dc,edf->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 72.f, 40.f, 216.f, 0.f, 360.f, 120.f, 648.f}),
      EinsumTestCase("abc,dc,efd->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{14.f, 46.f, 58.f, 186.f, 102.f, 326.f, 146.f, 466.f}),
      EinsumTestCase("abc,dc,efd->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{22.f, 38.f, 90.f, 154.f, 158.f, 270.f, 226.f, 386.f}),
      EinsumTestCase("abc,dc,efd->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 56.f, 184.f, 80.f, 240.f, 168.f, 552.f}),
      EinsumTestCase("abc,dc,efd->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 40.f, 88.f, 152.f, 120.f, 200.f, 264.f, 456.f}),
      EinsumTestCase("abc,dc,fde->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 68.f, 196.f, 120.f, 344.f, 172.f, 492.f}),
      EinsumTestCase("abc,dc,fde->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 54.f, 30.f, 234.f, 50.f, 414.f, 70.f, 594.f}),
      EinsumTestCase("abc,dc,fde->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 52.f, 64.f, 192.f, 100.f, 260.f, 192.f, 576.f}),
      EinsumTestCase("abc,dc,fde->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 72.f, 40.f, 216.f, 0.f, 360.f, 120.f, 648.f}),
      EinsumTestCase("abc,dc,fed->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{14.f, 46.f, 58.f, 186.f, 102.f, 326.f, 146.f, 466.f}),
      EinsumTestCase("abc,dc,fed->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{22.f, 38.f, 90.f, 154.f, 158.f, 270.f, 226.f, 386.f}),
      EinsumTestCase("abc,dc,fed->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 56.f, 184.f, 80.f, 240.f, 168.f, 552.f}),
      EinsumTestCase("abc,dc,fed->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 40.f, 88.f, 152.f, 120.f, 200.f, 264.f, 456.f}),
      EinsumTestCase("acb,cd,def->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 22.f, 60.f, 330.f, 0.f, 198.f, 156.f, 858.f}),
      EinsumTestCase("acb,cd,def->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{9.f, 13.f, 145.f, 245.f, 81.f, 117.f, 377.f, 637.f}),
      EinsumTestCase("acb,cd,def->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 132.f, 36.f, 220.f, 72.f, 484.f, 84.f, 572.f}),
      EinsumTestCase("acb,cd,def->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{58.f, 98.f, 96.f, 160.f, 210.f, 346.f, 248.f, 408.f}),
      EinsumTestCase("acb,cd,dfe->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 22.f, 60.f, 330.f, 0.f, 198.f, 156.f, 858.f}),
      EinsumTestCase("acb,cd,dfe->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{9.f, 13.f, 145.f, 245.f, 81.f, 117.f, 377.f, 637.f}),
      EinsumTestCase("acb,cd,dfe->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 132.f, 36.f, 220.f, 72.f, 484.f, 84.f, 572.f}),
      EinsumTestCase("acb,cd,dfe->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{58.f, 98.f, 96.f, 160.f, 210.f, 346.f, 248.f, 408.f}),
      EinsumTestCase("acb,cd,edf->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{5.f, 13.f, 85.f, 285.f, 45.f, 117.f, 221.f, 741.f}),
      EinsumTestCase("acb,cd,edf->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 18.f, 100.f, 270.f, 0.f, 162.f, 260.f, 702.f}),
      EinsumTestCase("acb,cd,edf->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{34.f, 114.f, 56.f, 184.f, 122.f, 394.f, 144.f, 464.f}),
      EinsumTestCase("acb,cd,edf->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{40.f, 108.f, 60.f, 180.f, 120.f, 396.f, 140.f, 468.f}),
      EinsumTestCase("acb,cd,efd->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 12.f, 80.f, 280.f, 36.f, 108.f, 208.f, 728.f}),
      EinsumTestCase("acb,cd,efd->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 10.f, 130.f, 230.f, 54.f, 90.f, 338.f, 598.f}),
      EinsumTestCase("acb,cd,efd->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{32.f, 112.f, 52.f, 180.f, 112.f, 384.f, 132.f, 452.f}),
      EinsumTestCase("acb,cd,efd->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{52.f, 92.f, 84.f, 148.f, 180.f, 316.f, 212.f, 372.f}),
      EinsumTestCase("acb,cd,fde->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{5.f, 13.f, 85.f, 285.f, 45.f, 117.f, 221.f, 741.f}),
      EinsumTestCase("acb,cd,fde->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 18.f, 100.f, 270.f, 0.f, 162.f, 260.f, 702.f}),
      EinsumTestCase("acb,cd,fde->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{34.f, 114.f, 56.f, 184.f, 122.f, 394.f, 144.f, 464.f}),
      EinsumTestCase("acb,cd,fde->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{40.f, 108.f, 60.f, 180.f, 120.f, 396.f, 140.f, 468.f}),
      EinsumTestCase("acb,cd,fed->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 12.f, 80.f, 280.f, 36.f, 108.f, 208.f, 728.f}),
      EinsumTestCase("acb,cd,fed->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 10.f, 130.f, 230.f, 54.f, 90.f, 338.f, 598.f}),
      EinsumTestCase("acb,cd,fed->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{32.f, 112.f, 52.f, 180.f, 112.f, 384.f, 132.f, 452.f}),
      EinsumTestCase("acb,cd,fed->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{52.f, 92.f, 84.f, 148.f, 180.f, 316.f, 212.f, 372.f}),
      EinsumTestCase("acb,dc,def->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 30.f, 330.f, 0.f, 396.f, 78.f, 858.f}),
      EinsumTestCase("acb,dc,def->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 140.f, 220.f, 162.f, 234.f, 364.f, 572.f}),
      EinsumTestCase("acb,dc,def->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 132.f, 18.f, 242.f, 36.f, 572.f, 42.f, 682.f}),
      EinsumTestCase("acb,dc,def->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{56.f, 88.f, 102.f, 158.f, 240.f, 368.f, 286.f, 438.f}),
      EinsumTestCase("acb,dc,dfe->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 30.f, 330.f, 0.f, 396.f, 78.f, 858.f}),
      EinsumTestCase("acb,dc,dfe->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 140.f, 220.f, 162.f, 234.f, 364.f, 572.f}),
      EinsumTestCase("acb,dc,dfe->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 132.f, 18.f, 242.f, 36.f, 572.f, 42.f, 682.f}),
      EinsumTestCase("acb,dc,dfe->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{56.f, 88.f, 102.f, 158.f, 240.f, 368.f, 286.f, 438.f}),
      EinsumTestCase("acb,dc,edf->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 80.f, 240.f, 90.f, 234.f, 208.f, 624.f}),
      EinsumTestCase("acb,dc,edf->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 50.f, 270.f, 0.f, 324.f, 130.f, 702.f}),
      EinsumTestCase("acb,dc,edf->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{32.f, 96.f, 58.f, 170.f, 136.f, 392.f, 162.f, 466.f}),
      EinsumTestCase("acb,dc,edf->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 108.f, 30.f, 198.f, 60.f, 468.f, 70.f, 558.f}),
      EinsumTestCase("acb,dc,efd->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 70.f, 230.f, 72.f, 216.f, 182.f, 598.f}),
      EinsumTestCase("acb,dc,efd->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 110.f, 190.f, 108.f, 180.f, 286.f, 494.f}),
      EinsumTestCase("acb,dc,efd->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{28.f, 92.f, 50.f, 162.f, 116.f, 372.f, 138.f, 442.f}),
      EinsumTestCase("acb,dc,efd->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{44.f, 76.f, 78.f, 134.f, 180.f, 308.f, 214.f, 366.f}),
      EinsumTestCase("acb,dc,fde->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 80.f, 240.f, 90.f, 234.f, 208.f, 624.f}),
      EinsumTestCase("acb,dc,fde->acd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 50.f, 270.f, 0.f, 324.f, 130.f, 702.f}),
      EinsumTestCase("acb,dc,fde->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{32.f, 96.f, 58.f, 170.f, 136.f, 392.f, 162.f, 466.f}),
      EinsumTestCase("acb,dc,fde->abd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 108.f, 30.f, 198.f, 60.f, 468.f, 70.f, 558.f}),
      EinsumTestCase("acb,dc,fed->acf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 70.f, 230.f, 72.f, 216.f, 182.f, 598.f}),
      EinsumTestCase("acb,dc,fed->ace", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 110.f, 190.f, 108.f, 180.f, 286.f, 494.f}),
      EinsumTestCase("acb,dc,fed->abf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{28.f, 92.f, 50.f, 162.f, 116.f, 372.f, 138.f, 442.f}),
      EinsumTestCase("acb,dc,fed->abe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{44.f, 76.f, 78.f, 134.f, 180.f, 308.f, 214.f, 366.f}),
      EinsumTestCase("bac,cd,def->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f}),
      EinsumTestCase("bac,cd,def->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{29.f, 49.f, 105.f, 173.f, 181.f, 297.f, 257.f, 421.f}),
      EinsumTestCase("bac,cd,def->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 48.f, 264.f, 0.f, 220.f, 144.f, 792.f}),
      EinsumTestCase("bac,cd,def->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 116.f, 196.f, 90.f, 130.f, 348.f, 588.f}),
      EinsumTestCase("bac,cd,dfe->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 66.f, 36.f, 242.f, 60.f, 418.f, 84.f, 594.f}),
      EinsumTestCase("bac,cd,dfe->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{29.f, 49.f, 105.f, 173.f, 181.f, 297.f, 257.f, 421.f}),
      EinsumTestCase("bac,cd,dfe->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 48.f, 264.f, 0.f, 220.f, 144.f, 792.f}),
      EinsumTestCase("bac,cd,dfe->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 116.f, 196.f, 90.f, 130.f, 348.f, 588.f}),
      EinsumTestCase("bac,cd,edf->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{17.f, 57.f, 61.f, 197.f, 105.f, 337.f, 149.f, 477.f}),
      EinsumTestCase("bac,cd,edf->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 54.f, 60.f, 198.f, 100.f, 342.f, 140.f, 486.f}),
      EinsumTestCase("bac,cd,edf->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 68.f, 228.f, 50.f, 130.f, 204.f, 684.f}),
      EinsumTestCase("bac,cd,edf->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 80.f, 216.f, 0.f, 180.f, 240.f, 648.f}),
      EinsumTestCase("bac,cd,efd->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 56.f, 56.f, 192.f, 96.f, 328.f, 136.f, 464.f}),
      EinsumTestCase("bac,cd,efd->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{26.f, 46.f, 90.f, 158.f, 154.f, 270.f, 218.f, 382.f}),
      EinsumTestCase("bac,cd,efd->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 64.f, 224.f, 40.f, 120.f, 192.f, 672.f}),
      EinsumTestCase("bac,cd,efd->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 104.f, 184.f, 60.f, 100.f, 312.f, 552.f}),
      EinsumTestCase("bac,cd,fde->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{17.f, 57.f, 61.f, 197.f, 105.f, 337.f, 149.f, 477.f}),
      EinsumTestCase("bac,cd,fde->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 54.f, 60.f, 198.f, 100.f, 342.f, 140.f, 486.f}),
      EinsumTestCase("bac,cd,fde->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 68.f, 228.f, 50.f, 130.f, 204.f, 684.f}),
      EinsumTestCase("bac,cd,fde->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 80.f, 216.f, 0.f, 180.f, 240.f, 648.f}),
      EinsumTestCase("bac,cd,fed->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 56.f, 56.f, 192.f, 96.f, 328.f, 136.f, 464.f}),
      EinsumTestCase("bac,cd,fed->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{26.f, 46.f, 90.f, 158.f, 154.f, 270.f, 218.f, 382.f}),
      EinsumTestCase("bac,cd,fed->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 64.f, 224.f, 40.f, 120.f, 192.f, 672.f}),
      EinsumTestCase("bac,cd,fed->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 104.f, 184.f, 60.f, 100.f, 312.f, 552.f}),
      EinsumTestCase("bac,dc,def->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 66.f, 18.f, 286.f, 30.f, 506.f, 42.f, 726.f}),
      EinsumTestCase("bac,dc,def->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{28.f, 44.f, 120.f, 184.f, 212.f, 324.f, 304.f, 464.f}),
      EinsumTestCase("bac,dc,def->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 88.f, 24.f, 264.f, 0.f, 440.f, 72.f, 792.f}),
      EinsumTestCase("bac,dc,def->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{36.f, 52.f, 112.f, 176.f, 180.f, 260.f, 336.f, 528.f}),
      EinsumTestCase("bac,dc,dfe->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 66.f, 18.f, 286.f, 30.f, 506.f, 42.f, 726.f}),
      EinsumTestCase("bac,dc,dfe->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{28.f, 44.f, 120.f, 184.f, 212.f, 324.f, 304.f, 464.f}),
      EinsumTestCase("bac,dc,dfe->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 88.f, 24.f, 264.f, 0.f, 440.f, 72.f, 792.f}),
      EinsumTestCase("bac,dc,dfe->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{36.f, 52.f, 112.f, 176.f, 180.f, 260.f, 336.f, 528.f}),
      EinsumTestCase("bac,dc,edf->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 68.f, 196.f, 120.f, 344.f, 172.f, 492.f}),
      EinsumTestCase("bac,dc,edf->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 54.f, 30.f, 234.f, 50.f, 414.f, 70.f, 594.f}),
      EinsumTestCase("bac,dc,edf->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 52.f, 64.f, 192.f, 100.f, 260.f, 192.f, 576.f}),
      EinsumTestCase("bac,dc,edf->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 72.f, 40.f, 216.f, 0.f, 360.f, 120.f, 648.f}),
      EinsumTestCase("bac,dc,efd->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{14.f, 46.f, 58.f, 186.f, 102.f, 326.f, 146.f, 466.f}),
      EinsumTestCase("bac,dc,efd->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{22.f, 38.f, 90.f, 154.f, 158.f, 270.f, 226.f, 386.f}),
      EinsumTestCase("bac,dc,efd->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 56.f, 184.f, 80.f, 240.f, 168.f, 552.f}),
      EinsumTestCase("bac,dc,efd->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 40.f, 88.f, 152.f, 120.f, 200.f, 264.f, 456.f}),
      EinsumTestCase("bac,dc,fde->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 68.f, 196.f, 120.f, 344.f, 172.f, 492.f}),
      EinsumTestCase("bac,dc,fde->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 54.f, 30.f, 234.f, 50.f, 414.f, 70.f, 594.f}),
      EinsumTestCase("bac,dc,fde->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 52.f, 64.f, 192.f, 100.f, 260.f, 192.f, 576.f}),
      EinsumTestCase("bac,dc,fde->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 72.f, 40.f, 216.f, 0.f, 360.f, 120.f, 648.f}),
      EinsumTestCase("bac,dc,fed->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{14.f, 46.f, 58.f, 186.f, 102.f, 326.f, 146.f, 466.f}),
      EinsumTestCase("bac,dc,fed->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{22.f, 38.f, 90.f, 154.f, 158.f, 270.f, 226.f, 386.f}),
      EinsumTestCase("bac,dc,fed->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 56.f, 184.f, 80.f, 240.f, 168.f, 552.f}),
      EinsumTestCase("bac,dc,fed->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 40.f, 88.f, 152.f, 120.f, 200.f, 264.f, 456.f}),
      EinsumTestCase("bca,cd,def->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 22.f, 60.f, 330.f, 0.f, 198.f, 156.f, 858.f}),
      EinsumTestCase("bca,cd,def->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{9.f, 13.f, 145.f, 245.f, 81.f, 117.f, 377.f, 637.f}),
      EinsumTestCase("bca,cd,def->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 132.f, 36.f, 220.f, 72.f, 484.f, 84.f, 572.f}),
      EinsumTestCase("bca,cd,def->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{58.f, 98.f, 96.f, 160.f, 210.f, 346.f, 248.f, 408.f}),
      EinsumTestCase("bca,cd,dfe->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 22.f, 60.f, 330.f, 0.f, 198.f, 156.f, 858.f}),
      EinsumTestCase("bca,cd,dfe->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{9.f, 13.f, 145.f, 245.f, 81.f, 117.f, 377.f, 637.f}),
      EinsumTestCase("bca,cd,dfe->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 132.f, 36.f, 220.f, 72.f, 484.f, 84.f, 572.f}),
      EinsumTestCase("bca,cd,dfe->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{58.f, 98.f, 96.f, 160.f, 210.f, 346.f, 248.f, 408.f}),
      EinsumTestCase("bca,cd,edf->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{5.f, 13.f, 85.f, 285.f, 45.f, 117.f, 221.f, 741.f}),
      EinsumTestCase("bca,cd,edf->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 18.f, 100.f, 270.f, 0.f, 162.f, 260.f, 702.f}),
      EinsumTestCase("bca,cd,edf->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{34.f, 114.f, 56.f, 184.f, 122.f, 394.f, 144.f, 464.f}),
      EinsumTestCase("bca,cd,edf->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{40.f, 108.f, 60.f, 180.f, 120.f, 396.f, 140.f, 468.f}),
      EinsumTestCase("bca,cd,efd->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 12.f, 80.f, 280.f, 36.f, 108.f, 208.f, 728.f}),
      EinsumTestCase("bca,cd,efd->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 10.f, 130.f, 230.f, 54.f, 90.f, 338.f, 598.f}),
      EinsumTestCase("bca,cd,efd->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{32.f, 112.f, 52.f, 180.f, 112.f, 384.f, 132.f, 452.f}),
      EinsumTestCase("bca,cd,efd->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{52.f, 92.f, 84.f, 148.f, 180.f, 316.f, 212.f, 372.f}),
      EinsumTestCase("bca,cd,fde->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{5.f, 13.f, 85.f, 285.f, 45.f, 117.f, 221.f, 741.f}),
      EinsumTestCase("bca,cd,fde->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 18.f, 100.f, 270.f, 0.f, 162.f, 260.f, 702.f}),
      EinsumTestCase("bca,cd,fde->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{34.f, 114.f, 56.f, 184.f, 122.f, 394.f, 144.f, 464.f}),
      EinsumTestCase("bca,cd,fde->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{40.f, 108.f, 60.f, 180.f, 120.f, 396.f, 140.f, 468.f}),
      EinsumTestCase("bca,cd,fed->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 12.f, 80.f, 280.f, 36.f, 108.f, 208.f, 728.f}),
      EinsumTestCase("bca,cd,fed->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 10.f, 130.f, 230.f, 54.f, 90.f, 338.f, 598.f}),
      EinsumTestCase("bca,cd,fed->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{32.f, 112.f, 52.f, 180.f, 112.f, 384.f, 132.f, 452.f}),
      EinsumTestCase("bca,cd,fed->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{52.f, 92.f, 84.f, 148.f, 180.f, 316.f, 212.f, 372.f}),
      EinsumTestCase("bca,dc,def->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 30.f, 330.f, 0.f, 396.f, 78.f, 858.f}),
      EinsumTestCase("bca,dc,def->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 140.f, 220.f, 162.f, 234.f, 364.f, 572.f}),
      EinsumTestCase("bca,dc,def->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 132.f, 18.f, 242.f, 36.f, 572.f, 42.f, 682.f}),
      EinsumTestCase("bca,dc,def->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{56.f, 88.f, 102.f, 158.f, 240.f, 368.f, 286.f, 438.f}),
      EinsumTestCase("bca,dc,dfe->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 30.f, 330.f, 0.f, 396.f, 78.f, 858.f}),
      EinsumTestCase("bca,dc,dfe->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 140.f, 220.f, 162.f, 234.f, 364.f, 572.f}),
      EinsumTestCase("bca,dc,dfe->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 132.f, 18.f, 242.f, 36.f, 572.f, 42.f, 682.f}),
      EinsumTestCase("bca,dc,dfe->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{56.f, 88.f, 102.f, 158.f, 240.f, 368.f, 286.f, 438.f}),
      EinsumTestCase("bca,dc,edf->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 80.f, 240.f, 90.f, 234.f, 208.f, 624.f}),
      EinsumTestCase("bca,dc,edf->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 50.f, 270.f, 0.f, 324.f, 130.f, 702.f}),
      EinsumTestCase("bca,dc,edf->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{32.f, 96.f, 58.f, 170.f, 136.f, 392.f, 162.f, 466.f}),
      EinsumTestCase("bca,dc,edf->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 108.f, 30.f, 198.f, 60.f, 468.f, 70.f, 558.f}),
      EinsumTestCase("bca,dc,efd->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 70.f, 230.f, 72.f, 216.f, 182.f, 598.f}),
      EinsumTestCase("bca,dc,efd->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 110.f, 190.f, 108.f, 180.f, 286.f, 494.f}),
      EinsumTestCase("bca,dc,efd->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{28.f, 92.f, 50.f, 162.f, 116.f, 372.f, 138.f, 442.f}),
      EinsumTestCase("bca,dc,efd->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{44.f, 76.f, 78.f, 134.f, 180.f, 308.f, 214.f, 366.f}),
      EinsumTestCase("bca,dc,fde->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 80.f, 240.f, 90.f, 234.f, 208.f, 624.f}),
      EinsumTestCase("bca,dc,fde->bcd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 50.f, 270.f, 0.f, 324.f, 130.f, 702.f}),
      EinsumTestCase("bca,dc,fde->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{32.f, 96.f, 58.f, 170.f, 136.f, 392.f, 162.f, 466.f}),
      EinsumTestCase("bca,dc,fde->bad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 108.f, 30.f, 198.f, 60.f, 468.f, 70.f, 558.f}),
      EinsumTestCase("bca,dc,fed->bcf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 70.f, 230.f, 72.f, 216.f, 182.f, 598.f}),
      EinsumTestCase("bca,dc,fed->bce", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 110.f, 190.f, 108.f, 180.f, 286.f, 494.f}),
      EinsumTestCase("bca,dc,fed->baf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{28.f, 92.f, 50.f, 162.f, 116.f, 372.f, 138.f, 442.f}),
      EinsumTestCase("bca,dc,fed->bae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{44.f, 76.f, 78.f, 134.f, 180.f, 308.f, 214.f, 366.f}),
      EinsumTestCase("cab,cd,def->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 22.f, 0.f, 110.f, 108.f, 594.f, 156.f, 858.f}),
      EinsumTestCase("cab,cd,def->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{9.f, 13.f, 45.f, 65.f, 261.f, 441.f, 377.f, 637.f}),
      EinsumTestCase("cab,cd,def->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 0.f, 88.f, 120.f, 660.f, 144.f, 792.f}),
      EinsumTestCase("cab,cd,def->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 36.f, 52.f, 290.f, 490.f, 348.f, 588.f}),
      EinsumTestCase("cab,cd,dfe->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 22.f, 0.f, 110.f, 108.f, 594.f, 156.f, 858.f}),
      EinsumTestCase("cab,cd,dfe->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{9.f, 13.f, 45.f, 65.f, 261.f, 441.f, 377.f, 637.f}),
      EinsumTestCase("cab,cd,dfe->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 0.f, 88.f, 120.f, 660.f, 144.f, 792.f}),
      EinsumTestCase("cab,cd,dfe->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 36.f, 52.f, 290.f, 490.f, 348.f, 588.f}),
      EinsumTestCase("cab,cd,edf->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{5.f, 13.f, 25.f, 65.f, 153.f, 513.f, 221.f, 741.f}),
      EinsumTestCase("cab,cd,edf->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 18.f, 0.f, 90.f, 180.f, 486.f, 260.f, 702.f}),
      EinsumTestCase("cab,cd,edf->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 20.f, 52.f, 170.f, 570.f, 204.f, 684.f}),
      EinsumTestCase("cab,cd,edf->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 0.f, 72.f, 200.f, 540.f, 240.f, 648.f}),
      EinsumTestCase("cab,cd,efd->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 12.f, 20.f, 60.f, 144.f, 504.f, 208.f, 728.f}),
      EinsumTestCase("cab,cd,efd->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 10.f, 30.f, 50.f, 234.f, 414.f, 338.f, 598.f}),
      EinsumTestCase("cab,cd,efd->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 16.f, 48.f, 160.f, 560.f, 192.f, 672.f}),
      EinsumTestCase("cab,cd,efd->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 24.f, 40.f, 260.f, 460.f, 312.f, 552.f}),
      EinsumTestCase("cab,cd,fde->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{5.f, 13.f, 25.f, 65.f, 153.f, 513.f, 221.f, 741.f}),
      EinsumTestCase("cab,cd,fde->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 18.f, 0.f, 90.f, 180.f, 486.f, 260.f, 702.f}),
      EinsumTestCase("cab,cd,fde->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 20.f, 52.f, 170.f, 570.f, 204.f, 684.f}),
      EinsumTestCase("cab,cd,fde->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 0.f, 72.f, 200.f, 540.f, 240.f, 648.f}),
      EinsumTestCase("cab,cd,fed->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 12.f, 20.f, 60.f, 144.f, 504.f, 208.f, 728.f}),
      EinsumTestCase("cab,cd,fed->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 10.f, 30.f, 50.f, 234.f, 414.f, 338.f, 598.f}),
      EinsumTestCase("cab,cd,fed->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 16.f, 48.f, 160.f, 560.f, 192.f, 672.f}),
      EinsumTestCase("cab,cd,fed->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 24.f, 40.f, 260.f, 460.f, 312.f, 552.f}),
      EinsumTestCase("cab,dc,def->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 0.f, 220.f, 54.f, 594.f, 78.f, 858.f}),
      EinsumTestCase("cab,dc,def->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 90.f, 130.f, 252.f, 396.f, 364.f, 572.f}),
      EinsumTestCase("cab,dc,def->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 88.f, 0.f, 176.f, 60.f, 660.f, 72.f, 792.f}),
      EinsumTestCase("cab,dc,def->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{36.f, 52.f, 72.f, 104.f, 280.f, 440.f, 336.f, 528.f}),
      EinsumTestCase("cab,dc,dfe->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 0.f, 220.f, 54.f, 594.f, 78.f, 858.f}),
      EinsumTestCase("cab,dc,dfe->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 90.f, 130.f, 252.f, 396.f, 364.f, 572.f}),
      EinsumTestCase("cab,dc,dfe->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 88.f, 0.f, 176.f, 60.f, 660.f, 72.f, 792.f}),
      EinsumTestCase("cab,dc,dfe->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{36.f, 52.f, 72.f, 104.f, 280.f, 440.f, 336.f, 528.f}),
      EinsumTestCase("cab,dc,edf->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 50.f, 130.f, 144.f, 432.f, 208.f, 624.f}),
      EinsumTestCase("cab,dc,edf->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 0.f, 180.f, 90.f, 486.f, 130.f, 702.f}),
      EinsumTestCase("cab,dc,edf->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 52.f, 40.f, 104.f, 160.f, 480.f, 192.f, 576.f}),
      EinsumTestCase("cab,dc,edf->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 72.f, 0.f, 144.f, 100.f, 540.f, 120.f, 648.f}),
      EinsumTestCase("cab,dc,efd->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 40.f, 120.f, 126.f, 414.f, 182.f, 598.f}),
      EinsumTestCase("cab,dc,efd->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 60.f, 100.f, 198.f, 342.f, 286.f, 494.f}),
      EinsumTestCase("cab,dc,efd->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 32.f, 96.f, 140.f, 460.f, 168.f, 552.f}),
      EinsumTestCase("cab,dc,efd->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 40.f, 48.f, 80.f, 220.f, 380.f, 264.f, 456.f}),
      EinsumTestCase("cab,dc,fde->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 50.f, 130.f, 144.f, 432.f, 208.f, 624.f}),
      EinsumTestCase("cab,dc,fde->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 0.f, 180.f, 90.f, 486.f, 130.f, 702.f}),
      EinsumTestCase("cab,dc,fde->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 52.f, 40.f, 104.f, 160.f, 480.f, 192.f, 576.f}),
      EinsumTestCase("cab,dc,fde->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 72.f, 0.f, 144.f, 100.f, 540.f, 120.f, 648.f}),
      EinsumTestCase("cab,dc,fed->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 40.f, 120.f, 126.f, 414.f, 182.f, 598.f}),
      EinsumTestCase("cab,dc,fed->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 60.f, 100.f, 198.f, 342.f, 286.f, 494.f}),
      EinsumTestCase("cab,dc,fed->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 32.f, 96.f, 140.f, 460.f, 168.f, 552.f}),
      EinsumTestCase("cab,dc,fed->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 40.f, 48.f, 80.f, 220.f, 380.f, 264.f, 456.f}),
      EinsumTestCase("cba,cd,def->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 22.f, 0.f, 110.f, 108.f, 594.f, 156.f, 858.f}),
      EinsumTestCase("cba,cd,def->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{9.f, 13.f, 45.f, 65.f, 261.f, 441.f, 377.f, 637.f}),
      EinsumTestCase("cba,cd,def->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 0.f, 88.f, 120.f, 660.f, 144.f, 792.f}),
      EinsumTestCase("cba,cd,def->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 36.f, 52.f, 290.f, 490.f, 348.f, 588.f}),
      EinsumTestCase("cba,cd,dfe->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 22.f, 0.f, 110.f, 108.f, 594.f, 156.f, 858.f}),
      EinsumTestCase("cba,cd,dfe->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{9.f, 13.f, 45.f, 65.f, 261.f, 441.f, 377.f, 637.f}),
      EinsumTestCase("cba,cd,dfe->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 0.f, 88.f, 120.f, 660.f, 144.f, 792.f}),
      EinsumTestCase("cba,cd,dfe->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 36.f, 52.f, 290.f, 490.f, 348.f, 588.f}),
      EinsumTestCase("cba,cd,edf->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{5.f, 13.f, 25.f, 65.f, 153.f, 513.f, 221.f, 741.f}),
      EinsumTestCase("cba,cd,edf->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 18.f, 0.f, 90.f, 180.f, 486.f, 260.f, 702.f}),
      EinsumTestCase("cba,cd,edf->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 20.f, 52.f, 170.f, 570.f, 204.f, 684.f}),
      EinsumTestCase("cba,cd,edf->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 0.f, 72.f, 200.f, 540.f, 240.f, 648.f}),
      EinsumTestCase("cba,cd,efd->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 12.f, 20.f, 60.f, 144.f, 504.f, 208.f, 728.f}),
      EinsumTestCase("cba,cd,efd->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 10.f, 30.f, 50.f, 234.f, 414.f, 338.f, 598.f}),
      EinsumTestCase("cba,cd,efd->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 16.f, 48.f, 160.f, 560.f, 192.f, 672.f}),
      EinsumTestCase("cba,cd,efd->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 24.f, 40.f, 260.f, 460.f, 312.f, 552.f}),
      EinsumTestCase("cba,cd,fde->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{5.f, 13.f, 25.f, 65.f, 153.f, 513.f, 221.f, 741.f}),
      EinsumTestCase("cba,cd,fde->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 18.f, 0.f, 90.f, 180.f, 486.f, 260.f, 702.f}),
      EinsumTestCase("cba,cd,fde->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 20.f, 52.f, 170.f, 570.f, 204.f, 684.f}),
      EinsumTestCase("cba,cd,fde->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 0.f, 72.f, 200.f, 540.f, 240.f, 648.f}),
      EinsumTestCase("cba,cd,fed->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{4.f, 12.f, 20.f, 60.f, 144.f, 504.f, 208.f, 728.f}),
      EinsumTestCase("cba,cd,fed->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{6.f, 10.f, 30.f, 50.f, 234.f, 414.f, 338.f, 598.f}),
      EinsumTestCase("cba,cd,fed->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 16.f, 48.f, 160.f, 560.f, 192.f, 672.f}),
      EinsumTestCase("cba,cd,fed->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 24.f, 40.f, 260.f, 460.f, 312.f, 552.f}),
      EinsumTestCase("cba,dc,def->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 0.f, 220.f, 54.f, 594.f, 78.f, 858.f}),
      EinsumTestCase("cba,dc,def->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 90.f, 130.f, 252.f, 396.f, 364.f, 572.f}),
      EinsumTestCase("cba,dc,def->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 88.f, 0.f, 176.f, 60.f, 660.f, 72.f, 792.f}),
      EinsumTestCase("cba,dc,def->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{36.f, 52.f, 72.f, 104.f, 280.f, 440.f, 336.f, 528.f}),
      EinsumTestCase("cba,dc,dfe->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 44.f, 0.f, 220.f, 54.f, 594.f, 78.f, 858.f}),
      EinsumTestCase("cba,dc,dfe->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{18.f, 26.f, 90.f, 130.f, 252.f, 396.f, 364.f, 572.f}),
      EinsumTestCase("cba,dc,dfe->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 88.f, 0.f, 176.f, 60.f, 660.f, 72.f, 792.f}),
      EinsumTestCase("cba,dc,dfe->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{36.f, 52.f, 72.f, 104.f, 280.f, 440.f, 336.f, 528.f}),
      EinsumTestCase("cba,dc,edf->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 50.f, 130.f, 144.f, 432.f, 208.f, 624.f}),
      EinsumTestCase("cba,dc,edf->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 0.f, 180.f, 90.f, 486.f, 130.f, 702.f}),
      EinsumTestCase("cba,dc,edf->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 52.f, 40.f, 104.f, 160.f, 480.f, 192.f, 576.f}),
      EinsumTestCase("cba,dc,edf->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 72.f, 0.f, 144.f, 100.f, 540.f, 120.f, 648.f}),
      EinsumTestCase("cba,dc,efd->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 40.f, 120.f, 126.f, 414.f, 182.f, 598.f}),
      EinsumTestCase("cba,dc,efd->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 60.f, 100.f, 198.f, 342.f, 286.f, 494.f}),
      EinsumTestCase("cba,dc,efd->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 32.f, 96.f, 140.f, 460.f, 168.f, 552.f}),
      EinsumTestCase("cba,dc,efd->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 40.f, 48.f, 80.f, 220.f, 380.f, 264.f, 456.f}),
      EinsumTestCase("cba,dc,fde->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{10.f, 26.f, 50.f, 130.f, 144.f, 432.f, 208.f, 624.f}),
      EinsumTestCase("cba,dc,fde->cbd", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 36.f, 0.f, 180.f, 90.f, 486.f, 130.f, 702.f}),
      EinsumTestCase("cba,dc,fde->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{20.f, 52.f, 40.f, 104.f, 160.f, 480.f, 192.f, 576.f}),
      EinsumTestCase("cba,dc,fde->cad", std::vector<int64_t>{2, 2, 2}, std::vector<float>{0.f, 72.f, 0.f, 144.f, 100.f, 540.f, 120.f, 648.f}),
      EinsumTestCase("cba,dc,fed->cbf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{8.f, 24.f, 40.f, 120.f, 126.f, 414.f, 182.f, 598.f}),
      EinsumTestCase("cba,dc,fed->cbe", std::vector<int64_t>{2, 2, 2}, std::vector<float>{12.f, 20.f, 60.f, 100.f, 198.f, 342.f, 286.f, 494.f}),
      EinsumTestCase("cba,dc,fed->caf", std::vector<int64_t>{2, 2, 2}, std::vector<float>{16.f, 48.f, 32.f, 96.f, 140.f, 460.f, 168.f, 552.f}),
      EinsumTestCase("cba,dc,fed->cae", std::vector<int64_t>{2, 2, 2}, std::vector<float>{24.f, 40.f, 48.f, 80.f, 220.f, 380.f, 264.f, 456.f})};

  std::vector<float> m1{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> m2{0.f, 1.f, 2.f, 3.f};
  std::vector<float> m3{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  for (auto tst : test_cases) {
    OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
    test.AddAttribute<std::string>("equation", tst.equation);
    test.AddInput<float>("x", {2, 2, 2}, m1);
    test.AddInput<float>("y", {2, 2}, m2);
    test.AddInput<float>("z", {2, 2, 2}, m3);
    test.AddOutput<float>("o", tst.shape, tst.expected);
    test.Run();
  }
}

}  // namespace test
}  // namespace onnxruntime

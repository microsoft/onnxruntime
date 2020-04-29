// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
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

// Implicit
TEST(Einsum, ImplicitEinsumAsTensorContraction) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "abcd,ea");
  test.AddInput<float>("x", {2, 2, 2, 2}, {1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 1.f, 2.f});
  test.AddOutput<float>("o", {2, 2, 2, 2}, {3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f, 3.f, 3.f, 6.f, 6.f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/data_types.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

/*
TEST(Einsum, EinsumAsIdentityImplicit) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i");
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.Run();
}
*/
TEST(Einsum, ExplicitEinsumAsIdentity_1D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i->i");
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.Run();
}

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

TEST(Einsum, ExplicitEinsumAsTransposeOp_2D_input) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ji->ij");
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

TEST(Einsum, ExplicitEinsumAsMatmul) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ij,jk->ik");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddInput<float>("y", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 2}, {7.f, 10.f, 15.f, 22.f});
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

TEST(Einsum, ExplicitEinsumAsDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "ii->i");
  test.AddInput<float>("x", {2, 2}, {1.f, 2.f, 3.f, 4.f});
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
TEST(Einsum, ExplicitEinsumAsBatchedDiagonalOp) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "...ii->...i");
  test.AddInput<float>("x", {2, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f});
  test.AddOutput<float>("o", {2, 1, 2}, {1.f, 4.f, 1.f, 4.f});
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime

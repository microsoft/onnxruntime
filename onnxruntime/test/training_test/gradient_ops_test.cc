// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/gradient_checker.h"

namespace onnxruntime {
namespace test {

TEST(GradientCheckerTest, SigmoidGrad) {
  TensorShape shape({2, 3, 4});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "Sigmoid";

  EXPECT_THROW(gradient_checker.ComputeGradientError(op_name, {shape}, {shape}, &max_error), OnnxRuntimeException);
}

TEST(GradientCheckerTest, SinGrad) {
  TensorShape shape({2, 3, 4});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "Sin";

  gradient_checker.ComputeGradientError(op_name, {shape}, {shape}, &max_error);

  EXPECT_TRUE(max_error <= 1e-3);
}

TEST(GradientCheckerTest, AddGrad) {
  TensorShape shape({2, 6});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "Add";

  gradient_checker.ComputeGradientError(op_name, {shape, shape}, {shape}, &max_error);

  EXPECT_TRUE(max_error <= 1e-2);
}

TEST(GradientCheckerTest, SubGrad) {
  TensorShape shape({1});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "Sub";

  gradient_checker.ComputeGradientError(op_name, {shape, shape}, {shape}, &max_error);

  EXPECT_TRUE(max_error <= 1e-2);
}

// TODO: Enable this test once Powgrad is implemented completely.
//TEST(GradientCheckerTest, PowGrad) {
//  TensorShape shape({1});
//  float max_error;
//  GradientChecker<float, float, float> gradient_checker;
//  std::string op_name = "Pow";
//
//  gradient_checker.ComputeGradientError(op_name, {shape, shape}, {shape}, &max_error);
//
//  EXPECT_TRUE(max_error <= 1e-2);
//}

TEST(GradientCheckerTest, MatMulGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "MatMul";

  gradient_checker.ComputeGradientError(op_name, {{2, 4}, {4, 3}}, {{2, 3}}, &max_error);

  EXPECT_TRUE(max_error <= 1e-1);
}

#ifndef USE_CUDA
// There is a bug in the impl. Lets fix it and enable it
//TEST(GradientCheckerTest, GemmGrad) {
//  float max_error;
//  GradientChecker<float, float, float> gradient_checker;
//  std::string op_name = "Gemm";
//
//  gradient_checker.ComputeGradientError(op_name, {{1, 4}, {4, 3}, {1, 3}}, {{1, 3}}, &max_error);
//
//  EXPECT_TRUE(max_error <= 1e-2);
//}
#endif

TEST(GradientCheckerTest, ReduceMeanGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "ReduceMean";

  gradient_checker.ComputeGradientError(op_name, {{3, 5}}, {{1, 1}}, &max_error);

  EXPECT_TRUE(max_error <= 1e-2);
}

TEST(GradientCheckerTest, ReluGrad) {
  TensorShape shape({5, 6});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "Relu";

  gradient_checker.ComputeGradientError(op_name, {shape}, {shape}, &max_error);

  EXPECT_TRUE(max_error <= 1e-2);
}

TEST(GradientCheckerTest, SoftMaxGrad) {
  TensorShape shape({3, 4, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "Softmax";

  gradient_checker.ComputeGradientError(op_name, {shape}, {shape}, &max_error, {"axis"});

  EXPECT_TRUE(max_error <= 1e-2);
}

TEST(GradientCheckerTest, SplitGrad) {
  TensorShape shape({9, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  std::string op_name = "Split";

  gradient_checker.ComputeGradientError(op_name, {shape}, {{3, 5}, {3, 5}, {3, 5}}, &max_error, {"axis"});

  EXPECT_TRUE(max_error <= 1e-2);
}

}  // namespace test
}  // namespace onnxruntime

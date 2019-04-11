// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/gradient_checker.h"

// TODO: replace this with ONNX version of attr_proto_util.h when ONNX dependency version is updated
// TODO: update attributes type to AttributeProtoWrapper when ONNX version is ready
#include "core/training//attr_proto_util.h"

namespace onnxruntime {
namespace test {

using onnxruntime::training::MakeAttribute;
using training::OpDef;

TEST(GradientCheckerTest, SigmoidGrad) {
  TensorShape shape({2, 3, 4});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Sigmoid"};

  EXPECT_THROW(gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error), OnnxRuntimeException);
}

TEST(GradientCheckerTest, SinGrad) {
  TensorShape shape({2, 3, 4});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Sin"};

  gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error);

  EXPECT_TRUE(max_error <= 1e-3);
}

TEST(GradientCheckerTest, AddGrad) {
  TensorShape shape({2, 6});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Add"};

  gradient_checker.ComputeGradientError(op_def, {shape, shape}, {shape}, &max_error);
  EXPECT_TRUE(max_error <= 1e-2);
}

TEST(GradientCheckerTest, AddGrad_WithBroadcast) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Add"};

  {
    gradient_checker.ComputeGradientError(op_def, {{2, 6}, {6}}, {{2, 6}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  {
    gradient_checker.ComputeGradientError(op_def, {{3, 3}, {3, 1}}, {{3, 3}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  {
    gradient_checker.ComputeGradientError(op_def, {{}, {}}, {{}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  {
    gradient_checker.ComputeGradientError(op_def, {{}, {1}}, {{1}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  {
    gradient_checker.ComputeGradientError(op_def, {{1}, {}}, {{1}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  {
    gradient_checker.ComputeGradientError(op_def, {{1}, {1}}, {{1}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  {
    gradient_checker.ComputeGradientError(op_def, {{3, 2}, {3, 1}}, {{3, 2}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 4}, {1, 3, 1}}, {{2, 3, 4}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 1}, {3, 4}}, {{2, 3, 4}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }
}

TEST(GradientCheckerTest, SubGrad) {
  TensorShape shape({1});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Sub"};

  gradient_checker.ComputeGradientError(op_def, {shape, shape}, {shape}, &max_error);
  EXPECT_TRUE(max_error <= 1e-2);
}

// TODO: Enable this test once Powgrad is implemented completely.
TEST(GradientCheckerTest, DISABLED_PowGrad) {
  TensorShape shape({1});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Pow"};

  gradient_checker.ComputeGradientError(op_def, {shape, shape}, {shape}, &max_error);
  EXPECT_TRUE(max_error <= 1e-2);
}

TEST(GradientCheckerTest, MatMulGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"MatMul"};

  gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}}, {{2, 3}}, &max_error);

  EXPECT_TRUE(max_error <= 1e-1);
}

#ifndef USE_CUDA
// There is a bug in the impl. Lets fix it and enable it
TEST(GradientCheckerTest, DISABLED_GemmGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Gemm"};

  gradient_checker.ComputeGradientError(op_def, {{1, 4}, {4, 3}, {1, 3}}, {{1, 3}}, &max_error);
  EXPECT_TRUE(max_error <= 1e-2);
}
#endif

TEST(GradientCheckerTest, ReduceMeanGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"ReduceMean"};

  gradient_checker.ComputeGradientError(op_def, {{3, 5}}, {{1, 1}}, &max_error);
  EXPECT_TRUE(max_error <= 1e-2);
}

TEST(GradientCheckerTest, ReluGrad) {
  TensorShape shape({3, 4, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Relu"};

  gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error);
  EXPECT_TRUE(max_error <= 1e-3);
}

TEST(GradientCheckerTest, SoftMaxGrad) {
  TensorShape shape({3, 4, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Softmax"};

  // default_axis
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  // axis=0
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error, {MakeAttribute("axis", int64_t(0))});
    EXPECT_TRUE(max_error <= 1e-2);
  }

  // axis=2
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error, {MakeAttribute("axis", int64_t(2))});
    EXPECT_TRUE(max_error <= 1e-2);
  }
}  // namespace test

TEST(GradientCheckerTest, SplitGrad) {
  TensorShape shape({9, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Split"};

  gradient_checker.ComputeGradientError(op_def, {shape}, {{3, 5}, {3, 5}, {3, 5}}, &max_error, {MakeAttribute("axis", int64_t(0))});
  EXPECT_TRUE(max_error <= 1e-2);
}

TEST(GradientCheckerTest, MaxPoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"MaxPool"};
  const float error_tolerance = 1e-3f;

  //maxpool_1d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 2, 9}}, {{1, 2, 8}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //maxpool_2d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 5, 5}}, {{1, 3, 4, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{1, 1})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  // maxpool_2d_pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 1, 5, 5}}, {{1, 1, 7, 7}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{2, 2, 2, 2})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //maxpool_2d_strides
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 1, 32, 32}}, {{1, 1, 10, 10}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{5, 5}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //maxpool_3d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 1, 3, 3, 3}}, {{1, 1, 2, 2, 2}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }
}

TEST(GradientCheckerTest, GlobalAveragePoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GlobalAveragePool"};
  const float error_tolerance = 1e-3f;

  //globalaveragepool
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 5, 5}}, {{1, 3, 1, 1}}, &max_error);
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //globalaveragepool_precomputed
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 1, 3, 3}}, {{1, 1, 1, 1}}, &max_error);
    EXPECT_TRUE(max_error <= error_tolerance);
  }
}

TEST(GradientCheckerTest, DISABLED_ConvGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Conv"};

  //conv
  {
    TensorShape x_shape({1, 1, 5, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({1, 1, 5, 5});
    gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})});
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  //conv_with_strides
  {
    TensorShape x_shape({1, 1, 7, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({1, 1, 4, 3});
    gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1}),
                                           MakeAttribute("strides", std::vector<int64_t>{2, 2})});
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }
}

TEST(GradientCheckerTest, ConcatGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Concat"};

  //concat_1d
  {
    TensorShape x_shape({2});
    TensorShape y_shape({6});
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", int64_t(0))});
    EXPECT_TRUE(max_error <= 1e-2);
  }

  //concat_2d
  {
    TensorShape x_shape({2, 2});
    TensorShape y_shape({2, 6});
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", int64_t(1))});
    EXPECT_TRUE(max_error <= 1e-2);
  }

  //concat_3d
  {
    TensorShape x_shape({1, 2, 3});
    TensorShape y_shape({1, 2, 9});
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", int64_t(2))});
    EXPECT_TRUE(max_error <= 1e-2);
  }
}

// TODO: label should adds up to one, enable this when test framework accepts assigned test data
TEST(GradientCheckerTest, DISABLED_SoftmaxCrossEntropyGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropy", kMSDomain};
  const float error_tolerance = 1e-3f;

  {
    TensorShape input_shape({1, 100});
    gradient_checker.ComputeGradientError(op_def, {input_shape, {input_shape, false}}, {{1}}, &max_error);
    EXPECT_TRUE(max_error <= error_tolerance) << "max_error: " << max_error;
  }
}

TEST(GradientCheckerTest, AveragePoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"AveragePool"};
  const float error_tolerance = 1e-1f;

  //averagepool - 1D
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 8}}, {{1, 3, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                                           MakeAttribute("strides", std::vector<int64_t>{2})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //averagepool - 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 8, 8}}, {{1, 3, 7, 7}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{1, 1})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //averagepool - 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 8, 8, 8}}, {{1, 3, 4, 4, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //averagepool - 1D - With padding
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 8}}, {{1, 3, 3}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                                           MakeAttribute("strides", std::vector<int64_t>{3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 0})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  // averagepool - 2D - With padding - include pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 7, 8}}, {{1, 3, 3, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 2}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 0, 1, 0}),
                                           MakeAttribute("count_include_pad", int64_t(1))});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  // averagepool - 2D - With padding - exclude pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 7, 7}}, {{1, 3, 3, 3}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //averagepool - 3D - With padding
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 8, 8, 8}}, {{1, 3, 3, 3, 3}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 0, 0, 0})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //averagepool - 3D - With padding- exclude pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4, 7, 7, 7}}, {{1, 4, 3, 3, 3}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1}),
                                           MakeAttribute("count_include_pad", int64_t(1))});
    EXPECT_TRUE(max_error <= error_tolerance);
  }
}
}  // namespace test
}  // namespace onnxruntime

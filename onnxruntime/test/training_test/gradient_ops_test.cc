// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <random>
#include <algorithm>
#include <bitset>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/gradient_checker.h"
#include "test/providers/gradient_op_test_utils.h"
#include "test/random_seed.h"

// TODO: replace this with ONNX version of attr_proto_util.h when ONNX dependency version is updated
// TODO: update attributes type to AttributeProtoWrapper when ONNX version is ready
#include "core/graph/training/attr_proto_util.h"

namespace onnxruntime {
namespace test {

using onnxruntime::training::MakeAttribute;
using training::OpDef;

static bool IsErrorWithinTolerance(float error, float tolerance) {
  return !std::isnan(error) && !std::isnan(tolerance) && error <= tolerance;
}

#define EXPECT_IS_TINIER_THAN(max_error, tolerance)         \
  EXPECT_TRUE(IsErrorWithinTolerance(max_error, tolerance)) \
      << "max_error: " << max_error                         \
      << "; tolerance: " << tolerance                       \
      << "; ORT test random seed: " << GetStaticRandomSeed() << "; "

#define EXPECT_IS_TINY(max_error) \
  EXPECT_IS_TINIER_THAN(max_error, 1.5e-2f)

template <typename T>
void GenerateRandomDataWithOneHot(
    std::vector<std::vector<float>>& x_datas,
    std::vector<TensorShape> input_shapes,
    const std::unordered_set<int>& one_hot_input_indices) {
  for (int i = 0; i < 2; i++) {
    // TODO: Consider varying mean and variance
    float scale = 5.f;
    float mean = 0.f;
    const uint32_t seed = GetStaticRandomSeed();

    std::default_random_engine generator{gsl::narrow_cast<decltype(generator)::result_type>(seed)};
    std::normal_distribution<T> distribution{mean, scale};

    auto x_data_length = input_shapes[i].Size();
    x_datas[i].resize(x_data_length);

    if (one_hot_input_indices.count(i) > 0 && input_shapes[i].NumDimensions() > 1) {
      int64_t N = input_shapes[i].SizeToDimension(input_shapes[i].NumDimensions() - 1);
      int64_t D = input_shapes[i][input_shapes[i].NumDimensions() - 1];

      std::fill(x_datas[i].begin(), x_datas[i].end(), (T)0);
      for (int64_t k = 0; k < N; k++)
        x_datas[i][k * D + (seed % D)] = (T)1;
    } else {
      std::generate(x_datas[i].begin(), x_datas[i].end(), [&] { return distribution(generator); });
    }
  }
}

void UnaryOpGradientTest(const std::string& op_type) {
  TensorShape shape({2, 3, 4});
  float max_error;
  float error_tolerance = 1e-3f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op_type};

  gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error);

  EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
}

TEST(GradientCheckerTest, ErfGrad) {
  UnaryOpGradientTest("Erf");
}

TEST(GradientCheckerTest, SqrtGrad) {
  TensorShape shape({2, 3, 4});

  std::function<float(float)> transformer = [](float x) { return std::fabs(x) + 1; };
  TensorInfo x_info{shape, true, &transformer};

  float max_error;
  float error_tolerance = 1e-3f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Sqrt"};

  gradient_checker.ComputeGradientError(op_def, {x_info}, {shape}, &max_error);

  EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
}

void TestBroadcastableBinaryOpGrad(const std::string& op_type,
                                   std::function<float(float)>* transformer = nullptr) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op_type};

  //shape(A) = (2, 3, 4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo B_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo B_info{{}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (,), shape(B) = (2, 3, 4, 5), i.e. A is a scalar ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{}, true, transformer};
    TensorInfo B_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo B_info{{5}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{4, 5}, true, transformer};
    TensorInfo B_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (1, 4, 5), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{1, 4, 5}, true, transformer};
    TensorInfo B_info{{2, 3, 1, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (3, 4, 5), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{3, 4, 5}, true, transformer};
    TensorInfo B_info{{2, 1, 1, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (2, 1, 1, 5), shape(B) = (1, 3, 4, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{2, 1, 1, 5}, true, transformer};
    TensorInfo B_info{{1, 3, 4, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, AddGrad) {
  TestBroadcastableBinaryOpGrad("Add");
}

TEST(GradientCheckerTest, SubGrad) {
  TestBroadcastableBinaryOpGrad("Sub");
}

TEST(GradientCheckerTest, MulGrad) {
  TestBroadcastableBinaryOpGrad("Mul");
}

#ifdef USE_CUDA
TEST(GradientCheckerTest, DivGrad) {
  std::function<float(float)> transformer = [](float x) { return x > 0 ? x + 0.2f : x - 0.2f; };
  TestBroadcastableBinaryOpGrad("Div", &transformer);
}
#endif

// TODO: Powgrad Test doesn't cover exponent
TEST(GradientCheckerTest, PowGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Pow"};

  std::function<float(float)> x_transformer = [](float x) { return std::max(-2.f, std::min(2.f, x)); };
  TensorInfo x_info{{2, 3, 4}, true, &x_transformer};
  TensorInfo y_info{2, 3, 4};

  // square
  {
    std::function<float(float)> two = [](float) { return 2.0f; };
    TensorInfo exponent_info{{1}, false, &two};
    gradient_checker.ComputeGradientError(op_def, {x_info, exponent_info}, {y_info}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // cube
  {
    std::function<float(float)> three = [](float) { return 3.0f; };
    TensorInfo exponent_info{{1}, false, &three};
    gradient_checker.ComputeGradientError(op_def, {x_info, exponent_info}, {y_info}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, 1e-1f);
  }
}

TEST(GradientCheckerTest, MatMulGrad) {
  float max_error;
  const float error_tolerance = 1e-1f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"MatMul"};

  // 2D x 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}}, {{2, 3}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D x 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4}, {2, 4, 3}}, {{2, 3, 3}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D x 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4}, {4, 3}}, {{2, 3, 3}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D x 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{3, 4}, {2, 4, 3}}, {{2, 3, 3}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 4D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {2, 3, 5, 4}}, {{2, 3, 4, 4}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {5, 4}}, {{2, 3, 4, 4}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {3, 5, 4}}, {{2, 3, 4, 4}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 4D with broadcast
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 4, 5}, {1, 3, 5, 4}}, {{2, 3, 4, 4}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, SinGrad) {
  UnaryOpGradientTest("Sin");
}

TEST(GradientCheckerTest, TanhGrad) {
  UnaryOpGradientTest("Tanh");
}

TEST(GradientCheckerTest, GemmGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Gemm"};

  // Single Batch with Scalar Bias
  // TODO!!!! :	following test case is failing due to a bug in ReduceSum cuda
  /*
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4}, {4, 3}, {}}, {{1, 3}}, &max_error);
    ASSERT_IS_TINY(max_error);
  }
  */

  // Single Batch with Vector Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4}, {4, 3}, {3}}, {{1, 3}}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // Non-Single Batch with Scalar Bias
  // TODO!!!! :	following test case is failing due to a bug in ReduceSum cuda
  /*
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {}}, {{2, 3}}, &max_error);
    ASSERT_IS_TINY(max_error);
  }
  */

  // Non-Single Batch with Vector Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {3}}, {{2, 3}}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // Non-Single Batch with Broadcast Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {1, 3}}, {{2, 3}}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // Non-Single Batch with Non-BroadcastBias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {2, 3}}, {{2, 3}}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // TransA
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 2}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transA", int64_t(1))});
    EXPECT_IS_TINY(max_error);
  }

  // TransB
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {3, 4}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transB", int64_t(1))});
    EXPECT_IS_TINY(max_error);
  }

  // TransA and TransB
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 2}, {3, 4}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transA", int64_t(1)),
                                           MakeAttribute("transB", int64_t(1))});
    EXPECT_IS_TINY(max_error);
  }

  // alpha and beta + no_broadcast
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {2, 3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("alpha", 0.7f),
                                           MakeAttribute("beta", 5.0f)});
    EXPECT_IS_TINY(max_error);
  }

  // alpha and beta + broadcast
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("alpha", 0.7f),
                                           MakeAttribute("beta", 5.0f)});
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ReduceMeanGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"ReduceMean"};

  // default
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 3, 2}}, {{1, 1, 1}}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // TODO: Fix forward kernel behavior for default axes
  // default axes, keepdims = 0
  /*
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 3, 2}}, {{}}, &max_error,
                                          {MakeAttribute("keepdims", int64_t(0))});
    EXPECT_IS_TINY(max_error);
  }
  */

  // axes = [0, 1, 2], keepdims = 0
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 3, 2}}, {{}}, &max_error,
                                          {MakeAttribute("axes", std::vector<int64_t>{0, 1, 2}),
                                           MakeAttribute("keepdims", int64_t(0))});
    EXPECT_IS_TINY(max_error);
  }

  // axes = [0, 2], keepdims = 1
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 3, 2}}, {{1, 3, 1}}, &max_error,
                                          {MakeAttribute("axes", std::vector<int64_t>{0, 2})});
    EXPECT_IS_TINY(max_error);
  }

  // axes = [0, 1], keepdims = 0
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 3, 2}}, {{2}}, &max_error,
                                          {MakeAttribute("axes", std::vector<int64_t>{0, 1}),
                                           MakeAttribute("keepdims", int64_t(0))});
    EXPECT_IS_TINY(max_error);
  }

  // axes = [1], keepdims = 1
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 3, 2}}, {{4, 1, 2}}, &max_error,
                                          {MakeAttribute("axes", std::vector<int64_t>{1}),
                                           MakeAttribute("keepdims", int64_t(1))});
    EXPECT_IS_TINY(max_error);
  }

  // axes = [2], keepdims = 0
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 3, 2}}, {{4, 3}}, &max_error,
                                          {MakeAttribute("axes", std::vector<int64_t>{2}),
                                           MakeAttribute("keepdims", int64_t(0))});
    EXPECT_IS_TINY(max_error);
  }
}

#ifndef USE_CUDA
TEST(GradientCheckerTest, CastGrad) {
  // A dummy test that cast float to float
  // TODO: add more test here
  {
    TensorShape shape({2, 3, 4});
    float max_error;
    float error_tolerance = 1e-3f;
    GradientChecker<float, float, float> gradient_checker;
    OpDef op_def{"Cast"};

    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error,
                                          {MakeAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, ReluGrad) {
  UnaryOpGradientTest("Relu");
}

TEST(GradientCheckerTest, SplitGrad) {
  TensorShape shape({9, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Split"};

  gradient_checker.ComputeGradientError(op_def, {shape}, {{3, 5}, {3, 5}, {3, 5}}, &max_error,
                                        {MakeAttribute("axis", int64_t(0))});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, MaxPoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"MaxPool"};
  const float error_tolerance = 1e-3f;

  //maxpool_1d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 2, 9}}, {{2, 2, 8}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  //maxpool_2d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 5, 5}}, {{2, 3, 4, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{1, 1})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // maxpool_2d_pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 1, 5, 5}}, {{1, 1, 7, 7}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{2, 2, 2, 2})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  //maxpool_2d_strides
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 1, 32, 32}}, {{1, 1, 10, 10}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{5, 5}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  //maxpool_3d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 3, 3, 3}}, {{2, 1, 2, 2, 2}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, GlobalAveragePoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GlobalAveragePool"};
  const float error_tolerance = 1e-3f;

  //globalaveragepool
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 5, 5}}, {{2, 3, 1, 1}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  //globalaveragepool_precomputed
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 3, 3}}, {{2, 1, 1, 1}}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, ConvGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Conv"};
  float error_tolerance = 1e-1f;

  //conv
  {
    TensorShape x_shape({2, 1, 5, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5, 5});
    gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                                          // TODO: ConvGrad does not handle the case where W does not have gradient.
                                          // Check for not has_gradient need to be disabled to pass this test.
                                          false);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  //conv_with_strides
  {
    TensorShape x_shape({2, 1, 7, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 4, 3});
    gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1}),
                                           MakeAttribute("strides", std::vector<int64_t>{2, 2})},
                                          // TODO: ConvGrad does not handle the case where W does not have gradient.
                                          // Check for not has_gradient need to be disabled to pass this test.
                                          false);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
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
    EXPECT_IS_TINY(max_error);
  }

  //concat_2d
  {
    TensorShape x_shape({2, 2});
    TensorShape y_shape({2, 6});
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", int64_t(1))});
    EXPECT_IS_TINY(max_error);
  }

  //concat_3d
  {
    TensorShape x_shape({1, 2, 3});
    TensorShape y_shape({1, 2, 9});
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", int64_t(2))});
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, AveragePoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"AveragePool"};

  //averagepool - 1D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 8}}, {{2, 3, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                                           MakeAttribute("strides", std::vector<int64_t>{2})});
    EXPECT_IS_TINY(max_error);
  }

  //averagepool - 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 8, 8}}, {{2, 3, 7, 7}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{1, 1})});
    EXPECT_IS_TINY(max_error);
  }

  //averagepool - 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 8, 8, 8}}, {{2, 3, 4, 4, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})});
    EXPECT_IS_TINY(max_error);
  }

  //averagepool - 1D - With padding
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 8}}, {{1, 3, 3}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                                           MakeAttribute("strides", std::vector<int64_t>{3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 0})});
    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 2D - With padding - include pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 7, 8}}, {{1, 3, 3, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 2}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 0, 1, 0}),
                                           MakeAttribute("count_include_pad", int64_t(1))});
    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 2D - With padding - exclude pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 7, 7}}, {{1, 3, 3, 3}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})});
    EXPECT_IS_TINY(max_error);
  }

  //averagepool - 3D - With padding
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 3, 8, 8, 8}}, {{1, 3, 3, 3, 3}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 0, 0, 0})});
    EXPECT_IS_TINY(max_error);
  }

  //averagepool - 3D - With padding- exclude pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4, 7, 7, 7}}, {{1, 4, 3, 3, 3}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1}),
                                           MakeAttribute("count_include_pad", int64_t(1))});
    EXPECT_IS_TINY(max_error);
  }
}
#endif

TEST(GradientCheckerTest, TransposeGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Transpose"};
  float error_tolerance = 1e-3f;

  // default
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({4, 3, 2});
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 012
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({2, 3, 4});
    std::vector<int64_t> perm{0, 1, 2};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 021
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({2, 4, 3});
    std::vector<int64_t> perm{0, 2, 1};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 102
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({3, 2, 4});
    std::vector<int64_t> perm{1, 0, 2};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 120
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({3, 4, 2});
    std::vector<int64_t> perm{1, 2, 0};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 201
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({4, 2, 3});
    std::vector<int64_t> perm{2, 0, 1};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 210
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({4, 3, 2});
    std::vector<int64_t> perm{2, 1, 0};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, UnsqueezeGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Unsqueeze"};
  float error_tolerance = 1e-3f;

  {
    TensorShape x_shape({2, 3});
    TensorShape y_shape({1, 2, 3, 1});
    std::vector<int64_t> axes{0, 3};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({2, 3});
    TensorShape y_shape({1, 1, 2, 3});
    std::vector<int64_t> axes{0, 1};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({2, 3});
    TensorShape y_shape({1, 2, 1, 3, 1});
    std::vector<int64_t> axes{0, 2, 4};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, SqueezeGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Squeeze"};
  float error_tolerance = 1e-3f;

  {
    TensorShape x_shape({1, 2, 3, 1});
    TensorShape y_shape({2, 3});
    std::vector<int64_t> axes{0, 3};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({1, 1, 2, 3, 4});
    TensorShape y_shape({2, 3, 4});
    std::vector<int64_t> axes{0, 1};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({1, 2, 1, 3, 1});
    TensorShape y_shape({2, 3});
    std::vector<int64_t> axes{0, 2, 4};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  {
    TensorShape x_shape({1, 2, 1, 3, 1});
    TensorShape y_shape({1, 2, 3, 1});
    std::vector<int64_t> axes{2};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("axes", axes)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  /* TODO: enable test with no axis when squeeze kernel is fixed (separate bug filed)
  {
    TensorShape x_shape({1, 2, 1, 3, 1});
    TensorShape y_shape({2, 3});
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
  */
}

// TODO: Reshape missing

TEST(GradientCheckerTest, SoftMaxGrad) {
  TensorShape shape({3, 4, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Softmax"};

  // default_axis
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // axis=0
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error, {MakeAttribute("axis", int64_t(0))});
    EXPECT_IS_TINY(max_error);
  }

  // axis=2
  {
    gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error, {MakeAttribute("axis", int64_t(2))});
    EXPECT_IS_TINY(max_error);
  }
}

TEST(OptimizerTest, SGDOptimizerTest) {
  OpTester test("SGDOptimizer", 9, onnxruntime::kOnnxDomain);
  test.AddInput<float>("ETA", {}, {0.5f});
  test.AddInput<float>("W", {3}, {1, 2, 3});
  test.AddInput<float>("G", {3}, {4, 5, 6});
  test.AddOutput<float>("W_New", {3}, {-1.f, -0.5f, 0.f});
  test.Run();
}

void TestSoftmaxCrossEntropyGrad(const TensorShape& input_shape, const std::string& reduction) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropy"};

  std::vector<std::vector<float>> x_datas(2);
  GenerateRandomDataWithOneHot<float>(x_datas, {input_shape, input_shape}, {1});

  gradient_checker.ComputeGradientError(op_def, {input_shape, {input_shape, false}},
                                        {{1}, {input_shape, false}}, &max_error, x_datas,
                                        {MakeAttribute("reduction", reduction)});
  EXPECT_IS_TINY(max_error);
}
TEST(GradientCheckerTest, SoftmaxCrossEntropyGrad) {
  TestSoftmaxCrossEntropyGrad({5, 11}, "mean");
  TestSoftmaxCrossEntropyGrad({5, 11}, "sum");
  TestSoftmaxCrossEntropyGrad({2, 3, 2, 11}, "mean");
  TestSoftmaxCrossEntropyGrad({2, 3, 2, 11}, "sum");
}

void TestSparseSoftmaxCrossEntropyGrad(const TensorShape& index_shape, const std::string& reduction) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SparseSoftmaxCrossEntropy"};

  const int64_t D = 7;
  std::function<float(float)> transformer_index = [](float x) { return std::fmod(std::fabs(x) * 5.0f, 7.0f); };
  std::function<float(float)> transformer_weight = [](float x) { return std::fmod(std::fabs(x), 2.0f); };

  // without weight
  {
    TensorShape logit_shape(index_shape);
    logit_shape.emplace_back(D);

    TensorInfo x_info({logit_shape});
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info},
                                          {{1}, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }

  // with weight
  {
    TensorShape logit_shape(index_shape);
    logit_shape.emplace_back(D);

    TensorInfo x_info({logit_shape});
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo weight_info(index_shape, false, &transformer_weight);

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info, weight_info},
                                          {{1}, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, SparseSoftmaxCrossEntropyGrad) {
  TestSparseSoftmaxCrossEntropyGrad({5}, "mean");
  TestSparseSoftmaxCrossEntropyGrad({5}, "sum");
  TestSparseSoftmaxCrossEntropyGrad({2, 3, 2}, "mean");
  TestSparseSoftmaxCrossEntropyGrad({2, 3, 2}, "sum");
}

TEST(GradientCheckerTest, GeluGrad) {
  UnaryOpGradientTest("Gelu");
}

TEST(OptimizerTest, AdamOptimizerTest) {
  OpTester test("AdamOptimizer", 9, onnxruntime::kOnnxDomain);
  test.AddInput<float>("ETA", {}, {0.5f});
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, {1, 2, 3});
  test.AddInput<float>("G", {3}, {4, 5, 6});
  test.AddInput<float>("Moment_1", {3}, {0.1f, 0.2f, 0.3f});
  test.AddInput<float>("Moment_2", {3}, {0.4f, 0.5f, 0.6f});

  // Verify AdamOptimizer outputs
  test.AddOutput<float>("W_Out", {3}, {0.9232284f, 1.9051629f, 2.8897603f});
  test.AddOutput<float>("Moment_1_Out", {3}, {0.49f, 0.68f, 0.87f});
  test.AddOutput<float>("Moment_2_Out", {3}, {0.4156f, 0.5245f, 0.6354f});
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});

  test.Run();
}

#ifdef USE_CUDA

TEST(GradientCheckerTest, GatherGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Gather"};

  TensorInfo x_info({5, 4, 3, 2});
  std::function<float(float)> transformer = [](float x) { return std::fmod(7 * std::fabs(x), 5.0f); };

  // gather_0 without duplicated indices
  {
    int num_indices = 2;
    TensorInfo indices_info({num_indices}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 0;
    y_shape[axis] = num_indices;

    gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", axis)});
    EXPECT_IS_TINY(max_error);
  }

  // gather_0 with duplicated indices
  {
    int num_indices = 10;
    TensorInfo indices_info({num_indices}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 0;
    y_shape[axis] = num_indices;

    gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", axis)});
    EXPECT_IS_TINY(max_error);
  }

  // gather_1
  {
    int num_indices = 8;
    std::function<float(float)> transformer = [](float x) { return std::fmod(7 * std::fabs(x), 4.0f); };
    TensorInfo indices_info({num_indices}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 1;
    y_shape[axis] = num_indices;

    gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", axis)});
    EXPECT_IS_TINY(max_error);
  }

  // 2D Indices
  {
    TensorInfo indices_info({2, 3}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{2, 3, 4, 3, 2};

    gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                          {MakeAttribute("axis", int64_t(0))});
    EXPECT_IS_TINY(max_error);
  }
}

void TestDropoutOp(float ratio, TensorShape& x_shape, bool default_ratio = true) {
  OpTester test("TrainableDropout", 9, kOnnxDomain, false);
  if (default_ratio)
    ratio = 0.5f;
  float input_constant = 3.0f;
  std::vector<float> x_data(x_shape.Size(), input_constant);
  std::vector<float> y_data(x_shape.Size(), 3.0f);
  std::vector<float> ratio_data(1, ratio);

  test.AddInput<float>("x", x_shape.GetDims(), x_data);
  if (!default_ratio)
    test.AddInput<float>("ratio", {1}, ratio_data);
  test.AddOutput<float>("y", x_shape.GetDims(), y_data);
  test.AddOutput<bool>("mask", x_shape.GetDims(), {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true});
  test.Run();

  //Check output
  auto fwd_output = test.GetFetches();
  for (size_t idx = 0; idx < x_data.size() / 8; ++idx) {
    //convert the binary to bool
    if (ratio > 0) {
      std::bitset<8> mask(fwd_output[1].Get<Tensor>().Data<bool>()[idx]);
      for (size_t i = 0; i < 8; ++i) {
        auto output = fwd_output[0].Get<Tensor>().Data<float>()[idx * 8 + i];
        if (mask[i] == 0) {
          EXPECT_EQ(0, output);
        } else {
          EXPECT_IS_TINY(output - input_constant / (1.0f - ratio));
        }
      }
    } else {
      for (size_t i = 0; i < 8; ++i) {
        auto output = fwd_output[0].Get<Tensor>().Data<float>()[idx * 8 + i];
        EXPECT_EQ(output, input_constant);
      }
    }
  }
}

void TestDropoutGradOp(float ratio, TensorShape& x_shape, bool default_ratio = true) {
  OpTester test("TrainableDropoutGrad", 9, kOnnxDomain, true);
  if (default_ratio)
    ratio = 0.5;
  float input_constant = 3;

  std::vector<float> dy_data(x_shape.Size(), input_constant);
  std::vector<float> ratio_data(1, ratio);

  float output_constant = input_constant / (1 - ratio);
  std::vector<float> dx_data({output_constant, output_constant, output_constant, 0,
                              output_constant, 0, output_constant, 0,
                              output_constant, 0, output_constant, 0,
                              output_constant, 0, output_constant, 0});

  test.AddInput<float>("dy", x_shape.GetDims(), dy_data);

  test.AddInput<bool>("mask", x_shape.GetDims(), {true, true, true, false,   //
                                                  true, false, true, false,  //
                                                  true, false, true, false,  //
                                                  true, false, true, false});
  if (!default_ratio) {
    test.AddInput<float>("ratio", {1}, ratio_data);
  }

  test.AddOutput<float>("dx", x_shape.GetDims(), dx_data);

  test.Run();
}

TEST(GradientCheckerTest, DISABLED_TrainableDropout) {
  {
    //Ratio 0
    TensorShape x_shape({2, 2, 2, 2});
    TestDropoutOp(0.0f, x_shape, false);
  }
  //Ratio 0.2, 3D
  {
    TensorShape x_shape({4, 2, 2});
    TestDropoutOp(0.2f, x_shape, false);
  }
  //Ratio 0.4, 2D
  {
    TensorShape x_shape({4, 4});
    TestDropoutOp(0.4f, x_shape, false);
  }

  //Default ratio, 1D
  {
    TensorShape x_shape({16});
    TestDropoutOp(0.2f, x_shape, true);
  }
}

TEST(GradientCheckerTest, DISABLED_TrainableDropoutGrad) {
  {
    //Ratio 0
    TensorShape x_shape({8, 2});
    TestDropoutGradOp(0.0f, x_shape);
  }

  //Ratio 0.2, 1D
  {
    TensorShape x_shape({16});
    TestDropoutGradOp(0.2f, x_shape, false);
  }

  //Ratio 0.3, 2D
  {
    TensorShape x_shape({8, 2});
    TestDropoutGradOp(0.3f, x_shape, false);
  }

  //Ratio 0.4, 3D
  {
    TensorShape x_shape({2, 4, 2});
    TestDropoutGradOp(0.4f, x_shape, false);
  }

  //default Ratio, 4D
  {
    TensorShape x_shape({2, 4, 2});
    TestDropoutGradOp(0.6f, x_shape);
  }
}

void TestCurandDropoutOp(float ratio, TensorShape& x_shape, bool default_ratio = true) {
  OpTester test("TrainableDropout", 9, kOnnxDomain, false);
  if (default_ratio)
    ratio = 0.5f;
  float input_constant = 3.0f;
  std::vector<float> x_data(x_shape.Size(), input_constant);
  std::vector<float> y_data(x_shape.Size(), 3.0f);
  std::vector<float> ratio_data(1, ratio);

  test.AddInput<float>("x", x_shape.GetDims(), x_data);
  if (!default_ratio)
    test.AddInput<float>("ratio", {1}, ratio_data);
  test.AddOutput<float>("y", x_shape.GetDims(), y_data);
  test.AddOutput<bool>("mask", x_shape.GetDims(), {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true});
  test.Run();

  //Check output
  auto fwd_output = test.GetFetches();

  const float* output = fwd_output[0].Get<Tensor>().Data<float>();
  const bool* mask = fwd_output[1].Get<Tensor>().Data<bool>();

  if (ratio > 0) {
    for (size_t idx = 0; idx < x_data.size(); ++idx) {
      if (mask[idx] == false) {
        EXPECT_EQ(0, output[idx]);
      } else {
        EXPECT_IS_TINY(output[idx] - input_constant / (1.0f - ratio));
      }
    }
  } else {
    for (size_t idx = 0; idx < x_data.size(); ++idx) {
      EXPECT_EQ(output[idx], input_constant);
    }
  }
}

void TestCurandDropoutGradOp(float ratio, TensorShape& x_shape, bool default_ratio = true) {
  OpTester test("TrainableDropoutGrad", 9, kOnnxDomain, true);
  if (default_ratio)
    ratio = 0.5;
  float input_constant = 3;

  std::vector<float> dy_data(x_shape.Size(), input_constant);
  std::vector<float> ratio_data(1, ratio);

  float output_constant = input_constant / (1 - ratio);
  std::vector<float> dx_data({output_constant, output_constant, output_constant, 0,
                              output_constant, 0, output_constant, 0,
                              output_constant, 0, output_constant, 0,
                              output_constant, 0, output_constant, 0});

  test.AddInput<float>("dy", x_shape.GetDims(), dy_data);

  test.AddInput<bool>("mask", x_shape.GetDims(), {true, true, true, false,   //
                                                  true, false, true, false,  //
                                                  true, false, true, false,  //
                                                  true, false, true, false});
  if (!default_ratio)
    test.AddInput<float>("ratio", {1}, ratio_data);
  test.AddOutput<float>("dx", x_shape.GetDims(), dx_data);

  test.Run();
}
TEST(GradientCheckerTest, TrainableDropout_CuRand) {
  {
    //Ratio 0
    TensorShape x_shape({2, 2, 2, 2});
    TestCurandDropoutOp(0.0f, x_shape, false);
  }
  //Ratio 0.2, 3D
  {
    TensorShape x_shape({4, 2, 2});
    TestCurandDropoutOp(0.2f, x_shape, false);
  }
  //Ratio 0.4, 2D
  {
    TensorShape x_shape({4, 4});
    TestCurandDropoutOp(0.4f, x_shape, false);
  }

  //Default ratio, 1D
  {
    TensorShape x_shape({16});
    TestCurandDropoutOp(0.2f, x_shape, true);
  }
}

TEST(GradientCheckerTest, TrainableDropoutGrad_Curand) {
  {
    //Ratio 0
    TensorShape x_shape({8, 2});
    TestCurandDropoutGradOp(0.0f, x_shape);
  }

  //Ratio 0.2, 1D
  {
    TensorShape x_shape({16});
    TestCurandDropoutGradOp(0.2f, x_shape, false);
  }

  //Ratio 0.3, 2D
  {
    TensorShape x_shape({8, 2});
    TestCurandDropoutGradOp(0.3f, x_shape, false);
  }

  //Ratio 0.4, 3D
  {
    TensorShape x_shape({2, 4, 2});
    TestCurandDropoutGradOp(0.4f, x_shape, false);
  }

  //default Ratio, 4D
  {
    TensorShape x_shape({2, 4, 2});
    TestCurandDropoutGradOp(0.6f, x_shape);
  }
}

TEST(GradientCheckerTest, GatherNDGrad_int64_indice_repeat_float_data) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherND"};

  TensorInfo x_info({2, 2}, true);
  TensorInfo indice_info({2, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3}, {1, 1, 1, 1}};

  TensorInfo y_info({2}, true);
  int64_t axis = 0;

  gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherNDGrad_int64_indice_unique_float_data) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherND"};

  TensorInfo x_info({2, 2}, true);
  TensorInfo indice_info({2, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3}, {0, 1, 1, 0}};

  TensorInfo y_info({2}, true);
  int64_t axis = 0;

  gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherNDGrad_int32_indice_unique_float_data) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherND"};

  TensorInfo x_info({2, 2, 3}, true);
  TensorInfo indice_info({2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int32_t>());
  std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 0}};

  TensorInfo y_info({2, 3}, true);
  int64_t axis = 1;

  gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherNDGrad_int32_indice_unique_float_data_axis_2) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherND"};

  TensorInfo x_info({2, 2, 3}, true);
  TensorInfo indice_info({2, 2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int32_t>());
  std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 0, 2, 1}};

  TensorInfo y_info({2, 2}, true);
  int64_t axis = 2;

  gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}

// This helper function is a CPU-based LAMB optimizer
// implementation. It mainly focuses on readability.
void compute_lamb(
    const std::vector<int64_t> shape,
    /* weights */ const std::vector<float>& w,
    /* gradient */ const std::vector<float>& g,
    /* momentum */ const std::vector<float>& m,
    /* 2nd-order momentum */ const std::vector<float>& v,
    const float eta,
    const float lambda,
    const float alpha,
    const float beta,
    const float epsilon,
    /* updated weights */ std::vector<float>& w_new,
    /* updated momentum */ std::vector<float>& m_new,
    /* updated 2nd-order momentum */ std::vector<float>& v_new) {
  // Element counts of all vector-typed arguments.
  const int64_t size = std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());

  // Buffer to store update direction.
  std::vector<float> r(size, 0.0f);

  // Compute new 1st-, 2nd-order momentums, and the update direction.
  for (int i = 0; i < size; ++i) {
    m_new[i] = alpha * m[i] + (1.0f - alpha) * g[i];
    v_new[i] = beta * v[i] + (1.0f - beta) * g[i] * g[i];
    r[i] = m_new[i] / (std::sqrt(v_new[i]) + epsilon) + lambda * w[i];
  }

  // Compute squared sum of all elements. Note that Eigen sqrt could lead to significant
  // numerical error so we use std::sqrt. The std::inner_product produces wrong result
  // when std::inner_product(r.begin(), r.end(), r.begin(), 0) so we just use a loop below. 
  float r_norm = 0.0f;
  float w_norm = 0.0f;
  for (int i = 0; i < size; ++i)
  {
    r_norm += r[i] * r[i];
    w_norm += w[i] * w[i];
  }
  r_norm = std::sqrt(r_norm);
  w_norm = std::sqrt(w_norm);

  // Compute the new weight.
  for (int64_t i = 0; i < size; ++i)
    w_new[i] = w[i] - eta * w_norm / r_norm * r[i];
}

void run_lamb_test(
  const std::vector<int64_t> &shape,
  const std::vector<float> &w,
  const std::vector<float> &g,
  const std::vector<float> &m,
  const std::vector<float> &v,
  const float eta,
  const float lambda,
  const float alpha,
  const float beta,
  const float epsilon) {
  OpTester test("LambOptimizer", 9, onnxruntime::kOnnxDomain, true);

  // Output buffers of the optimizer.
  std::vector<float> w_new(w.size(), 0);
  std::vector<float> m_new(w.size(), 0);
  std::vector<float> v_new(v.size(), 0);

  // Invoke LAMB's reference implementation to compute output.
  compute_lamb(
    shape, w, g, m, v,
    eta, lambda, alpha, beta, epsilon,
    w_new, m_new, v_new);

  // Create test to make sure the output is correct.
  test.AddInput<float>("ETA", {}, {eta});
  test.AddInput<float>("W", shape, w);
  test.AddInput<float>("G", shape, g);
  test.AddInput<float>("Moment_1", shape, m);
  test.AddInput<float>("Moment_2", shape, v);

  test.AddAttribute<float>("alpha", alpha);
  test.AddAttribute<float>("beta", beta);
  test.AddAttribute<float>("lambda", lambda);
  test.AddAttribute<float>("epsilon", epsilon);

  test.AddOutput<float>("W_Out", shape, w_new);
  test.AddOutput<float>("Moment_1_Out", shape, m_new);
  test.AddOutput<float>("Moment_2_Out", shape, v_new);

  test.Run();
}

// A optimizer test with an 2-element vector.
TEST(OptimizerTest, LambOptimizerTestVector) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float eta = 0.5f;
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_lamb_test(shape, w, g, m, v, eta, lambda, alpha, beta, epsilon);
}

// A optimizer test with an 2-by-1-by-1-by-1 tensor.
TEST(OptimizerTest, LambOptimizerTest4DTensor) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 1, 1, 1};
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float eta = 0.5f;
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_lamb_test(shape, w, g, m, v, eta, lambda, alpha, beta, epsilon);
}

// A optimizer test with an 2-by-3 tensor.
TEST(OptimizerTest, LambOptimizerTest2by3Tensor) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> w = {1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f, 2.0f, 1.0f, 1.0f, -2.0f};
  const std::vector<float> v = {1.0f, 1.0f, 5.0f, 5.0f, 6.0f, 6.0f};
  const float eta = 0.5f;
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_lamb_test(shape, w, g, m, v, eta, lambda, alpha, beta, epsilon);
}

// A optimizer test with an 1-element tensor.
TEST(OptimizerTest, LambOptimizerTestScalar) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {(int64_t)1};
  const std::vector<float> w = {1.0f};
  const std::vector<float> g = {3.0f};
  const std::vector<float> m = {-10.0f};
  const std::vector<float> v = {1.0f};
  const float eta = 0.5f;
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  // Intermediate and output buffers of the optimizer.
  std::vector<float> m_new = {0.0f};
  std::vector<float> v_new = {0.0f};
  std::vector<float> w_new = {0.0f};

  run_lamb_test(shape, w, g, m, v, eta, lambda, alpha, beta, epsilon);
}

TEST(OptimizerTest, LambOptimizerTestExternalBaseline) {
  OpTester test("LambOptimizer", 9, onnxruntime::kOnnxDomain, true);

  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 5};
  const std::vector<float> w = {
    0.01379026f, 0.15308191f, -0.24356517f, -0.21798165f, -0.13770047f, 0.09694599f,
    -0.02223516f, 0.2664228f, -0.01177993f, 0.06832688f};
  const std::vector<float> g = {
    -6.048543f, 10.569487f, -9.207029f, -0.57407373f,
    5.884985f, -0.21047728f, 3.539946f, -5.957566f, -9.343748f, 1.1502024f};
  const std::vector<float> m = {
    -5.9078765f, 9.673933f, -8.731428f, -0.6227454f, 5.284312f, -0.27138948f,
    3.443532f, -5.681713f, -8.72421f, 1.1441823f};
  const std::vector<float> v = {
    4.2659229e+01f, 1.1438165e+02f, 9.3179581e+01f, 4.7399229e-01f, 3.4129276e+01f,
    9.0019435e-02f, 1.4493006e+01f, 3.9455612e+01f, 9.3025581e+01f, 1.6000764e+0f};

  const float eta = 0.1f;
  const float lambda = 0.1f;
  const float alpha = 0.1f;
  const float beta = 0.01f;
  const float epsilon = 0.1f;

  std::vector<float> w_new = {
    0.02979828f, 0.13677707f, -0.22708717f, -0.20361158f, -0.15338624f, 0.1081504f,
    -0.03804127f, 0.28198114f, 0.00430069f, 0.05319814f};
  std::vector<float> m_new = {
    -6.0344763f, 10.479931f, -9.15947f, -0.57894087f, 5.824918f, -0.2165685f,
    3.5303047f, -5.9299808f, -9.281795f, 1.1496004f};
  std::vector<float> v_new = {
    3.6645618e+01f, 1.1174072e+02f, 8.4853485e+01f, 3.3100498e-01f, 3.4628010e+01f,
    4.4757873e-02f, 1.2550836e+01f, 3.5532223e+01f, 8.7362823e+01f, 1.3257366e+00f};

  test.AddInput<float>("ETA", {}, {eta});
  test.AddInput<float>("W", shape, w);
  test.AddInput<float>("G", shape, g);
  test.AddInput<float>("Moment_1", shape, m);
  test.AddInput<float>("Moment_2", shape, v);

  test.AddAttribute<float>("alpha", alpha);
  test.AddAttribute<float>("beta", beta);
  test.AddAttribute<float>("lambda", lambda);
  test.AddAttribute<float>("epsilon", epsilon);

  test.AddOutput<float>("W_Out", shape, w_new);
  test.AddOutput<float>("Moment_1_Out", shape, m_new);
  test.AddOutput<float>("Moment_2_Out", shape, v_new);

  test.Run();
}

static void TestLayerNormGradient(const std::vector<int64_t>& X_dims,
                                  const std::vector<int64_t>& scale_dims,
                                  const std::vector<int64_t>& B_dims,
                                  const std::vector<int64_t>& Y_dims,
                                  const std::vector<int64_t>& mean_dims,
                                  const std::vector<int64_t>& var_dims,
                                  optional<float> epsilon,
                                  int64_t axis = -1,
                                  int64_t keep_dims = 1) {
  OpTester test("LayerNormalization", 9, onnxruntime::kOnnxDomain, false);
  test.AddAttribute("axis", axis);
  test.AddAttribute("keep_dims", keep_dims);
  if (epsilon.has_value()) {
    test.AddAttribute("epsilon", epsilon.value());
  }

  int64_t X_size = std::accumulate(X_dims.cbegin(), X_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t scale_size = std::accumulate(scale_dims.cbegin(), scale_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t B_size = std::accumulate(B_dims.cbegin(), B_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t Y_size = std::accumulate(Y_dims.cbegin(), Y_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t mean_size = std::accumulate(mean_dims.cbegin(), mean_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t var_size = std::accumulate(var_dims.cbegin(), var_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});

  // create rand inputs
  std::vector<float> X_data(X_size, 1.0f);
  std::vector<float> scale_data(scale_size, 1.0f);
  std::vector<float> B_data(B_size, 2.0f);
  std::vector<float> Y_data(Y_size);
  std::vector<float> mean_data(mean_size);
  std::vector<float> var_data(var_size);

  FillRandom<float>(X_data, 0.0f, 1.0f);
  FillRandom<float>(scale_data, 0.0f, 1.0f);
  FillRandom<float>(B_data, 0.0f, 1.0f);

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("scale", scale_dims, scale_data, true);
  test.AddInput<float>("B", B_dims, B_data, true);

  test.AddOutput<float>("output", Y_dims, Y_data);
  test.AddOutput<float>("mean", mean_dims, mean_data);
  test.AddOutput<float>("var", var_dims, var_data);
  test.Run();
}

TEST(LayerNormGradientTest, BERTLayerNorm) {
  float epsilon = 1e-05f;

  std::vector<int64_t> X_dims{4, 512, 128};
  std::vector<int64_t> scale_dims{128};
  std::vector<int64_t> B_dims{128};
  std::vector<int64_t> Y_dims{4, 512, 128};
  std::vector<int64_t> mean_dims{4, 512, 1};
  std::vector<int64_t> var_dims{4, 512, 1};

  TestLayerNormGradient(X_dims, scale_dims, B_dims, Y_dims, mean_dims, var_dims, epsilon);
}

TEST(GradientCheckerTest, LayerNormGrad) {
  GradientChecker<float, float, float> gradient_checker;
  {
    TensorShape shape({2, 3, 4});
    TensorInfo x_info{shape, true};
    TensorInfo scale_info{{4}, true};
    TensorInfo B_info{{4}, true};
    TensorInfo mean_info{{2, 3, 1}, false};
    TensorInfo var_info{{2, 3, 1}, false};

    float max_error;
    float error_tolerance = 1e-2f;

    OpDef op_def{"LayerNormalization"};
    gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, B_info}, {shape, mean_info, var_info}, &max_error);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime

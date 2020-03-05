// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <bitset>
#include <cmath>
#include <random>

#include "gtest/gtest.h"
#include "core/framework/random_seed.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "orttraining/test/gradient/gradient_checker.h"
#include "orttraining/test/gradient/gradient_op_test_utils.h"

#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace test {

using ONNX_NAMESPACE::MakeAttribute;
using training::OpDef;

static bool IsErrorWithinTolerance(float error, float tolerance) {
  return !std::isnan(error) && !std::isnan(tolerance) && error <= tolerance;
}

#define EXPECT_IS_TINIER_THAN(max_error, tolerance)         \
  EXPECT_TRUE(IsErrorWithinTolerance(max_error, tolerance)) \
      << "max_error: " << max_error                         \
      << "; tolerance: " << tolerance                       \
      << "; ORT test random seed: " << utils::GetStaticRandomSeed() << "; "

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
    const uint32_t seed = utils::GetStaticRandomSeed();

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

void UnaryOpGradientTest(const std::string& op_type, const std::string& domain = kOnnxDomain, const int opset_version = 9) {
  TensorShape shape({2, 3, 4});
  float max_error;
  float error_tolerance = 1e-3f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op_type, domain, opset_version};

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
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4}, {4, 3}, {}}, {{1, 3}}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // Single Batch with Vector Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4}, {4, 3}, {3}}, {{1, 3}}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

  // Non-Single Batch with Scalar Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {}}, {{2, 3}}, &max_error);
    EXPECT_IS_TINY(max_error);
  }

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

template <typename T>
static std::vector<std::vector<T>> GetRandomValuesForMaxPool(const std::vector<TensorInfo>& infos) {
  std::vector<std::vector<T>> datas(infos.size());
  for (size_t i = 0; i < infos.size(); i++) {
    const TensorInfo& info = infos[i];

    // First add an increasing sequence of values with reasonable
    // differences (larger than the Jacobian delta).
    T value = 0;
    for (int64_t n = 0; n < info.shape.Size(); n++) {
      datas[i].push_back(value);
      value += T(1e-2);
    }

    // Next, shuffle the sequence.
    std::random_shuffle(datas[i].begin(), datas[i].end());
  }

  return datas;
}

TEST(GradientCheckerTest, MaxPoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"MaxPool"};
  const float error_tolerance = 1e-3f;

  //maxpool_1d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 2, 9}}, {{2, 2, 8}}, &max_error,
                                          GetRandomValuesForMaxPool<float>({{2, 2, 9}}),
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  //maxpool_2d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 5, 5}}, {{2, 3, 4, 4}}, &max_error,
                                          GetRandomValuesForMaxPool<float>({{2, 3, 5, 5}}),
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{1, 1})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // maxpool_2d_pads
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 1, 5, 5}}, {{1, 1, 7, 7}}, &max_error,
                                          GetRandomValuesForMaxPool<float>({{1, 1, 5, 5}}),
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{2, 2, 2, 2})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  //maxpool_2d_strides
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 1, 32, 32}}, {{1, 1, 10, 10}}, &max_error,
                                          GetRandomValuesForMaxPool<float>({{1, 1, 32, 32}}),
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{5, 5}),
                                           MakeAttribute("strides", std::vector<int64_t>{3, 3})});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  //maxpool_3d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 3, 3, 3}}, {{2, 1, 2, 2, 2}}, &max_error,
                                          GetRandomValuesForMaxPool<float>({{2, 1, 3, 3, 3}}),
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

#ifdef USE_CUDA
TEST(GradientCheckerTest, BatchNormalizationGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"BatchNormalization"};
  float error_tolerance = 1e-2f;
  float epsilon = 1e-05f;
  float momentum = 0.1f;

  // image data example where input dimensions are (N X C X H X W)
  {
    int channel_dim = 3;
    TensorShape in_out_shape({3, channel_dim, 2, 4});
    TensorShape channel_shape({channel_dim});
    // inputs
    TensorInfo x_info{in_out_shape, true};
    TensorInfo scale_info{channel_shape, true};
    TensorInfo bias_info{channel_shape, true};
    TensorInfo mean_info(channel_shape, false);
    TensorInfo var_info(channel_shape, false);
    // outputs
    TensorInfo y_info{in_out_shape, true};
    TensorInfo running_mean_info(channel_shape, false);
    TensorInfo running_var_info(channel_shape, false);
    TensorInfo saved_mean_info(channel_shape, false);
    TensorInfo saved_var_info(channel_shape, false);

    gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, bias_info, mean_info, var_info}, {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
                                          {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // channel_size = 1
  {
    int channel_dim = 1;
    TensorShape in_out_shape({3, channel_dim, 2, 4});
    TensorShape channel_shape({channel_dim});
    // inputs
    TensorInfo x_info{in_out_shape, true};
    TensorInfo scale_info{channel_shape, true};
    TensorInfo bias_info{channel_shape, true};
    TensorInfo mean_info(channel_shape, false);
    TensorInfo var_info(channel_shape, false);
    // outputs
    TensorInfo y_info{in_out_shape, true};
    TensorInfo running_mean_info(channel_shape, false);
    TensorInfo running_var_info(channel_shape, false);
    TensorInfo saved_mean_info(channel_shape, false);
    TensorInfo saved_var_info(channel_shape, false);

    gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, bias_info, mean_info, var_info}, {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
                                          {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // batch_size (N) = 1
  {
    int channel_dim = 4;
    TensorShape in_out_shape({1, channel_dim, 2});
    TensorShape channel_shape({channel_dim});
    // inputs
    TensorInfo x_info{in_out_shape, true};
    TensorInfo scale_info{channel_shape, true};
    TensorInfo bias_info{channel_shape, true};
    TensorInfo mean_info(channel_shape, false);
    TensorInfo var_info(channel_shape, false);
    // outputs
    TensorInfo y_info{in_out_shape, true};
    TensorInfo running_mean_info(channel_shape, false);
    TensorInfo running_var_info(channel_shape, false);
    TensorInfo saved_mean_info(channel_shape, false);
    TensorInfo saved_var_info(channel_shape, false);

    gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, bias_info, mean_info, var_info}, {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
                                          {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // case with epsilon not explicitly provided (default value should be used)
  {
    int channel_dim = 4;
    TensorShape in_out_shape({1, channel_dim, 2});
    TensorShape channel_shape({channel_dim});
    // inputs
    TensorInfo x_info{in_out_shape, true};
    TensorInfo scale_info{channel_shape, true};
    TensorInfo bias_info{channel_shape, true};
    TensorInfo mean_info(channel_shape, false);
    TensorInfo var_info(channel_shape, false);
    // outputs
    TensorInfo y_info{in_out_shape, true};
    TensorInfo running_mean_info(channel_shape, false);
    TensorInfo running_var_info(channel_shape, false);
    TensorInfo saved_mean_info(channel_shape, false);
    TensorInfo saved_var_info(channel_shape, false);

    gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, bias_info, mean_info, var_info}, {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
                                          {MakeAttribute("momentum", momentum)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // case for larger multi-dimensional X
  {
    int channel_dim = 5;
    TensorShape in_out_shape({6, channel_dim, 1, 3, 2, 4});
    TensorShape channel_shape({channel_dim});
    // inputs
    TensorInfo x_info{in_out_shape, true};
    TensorInfo scale_info{channel_shape, true};
    TensorInfo bias_info{channel_shape, true};
    TensorInfo mean_info(channel_shape, false);
    TensorInfo var_info(channel_shape, false);
    // outputs
    TensorInfo y_info{in_out_shape, true};
    TensorInfo running_mean_info(channel_shape, false);
    TensorInfo running_var_info(channel_shape, false);
    TensorInfo saved_mean_info(channel_shape, false);
    TensorInfo saved_var_info(channel_shape, false);

    gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, bias_info, mean_info, var_info}, {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
                                          {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  /* // single dimension input where C should be assumed to be 1 (does not seem to be currently supported by op)
  {
    int channel_dim = 1;
    TensorShape in_out_shape({5});
    TensorShape channel_shape({channel_dim});
    // inputs
    TensorInfo x_info{in_out_shape, true};
    TensorInfo scale_info{channel_shape, true};
    TensorInfo bias_info{channel_shape, true};
    TensorInfo mean_info(channel_shape, false);
    TensorInfo var_info(channel_shape, false);
    // outputs
    TensorInfo y_info{in_out_shape, true};
    TensorInfo running_mean_info(channel_shape, false);
    TensorInfo running_var_info(channel_shape, false);
    TensorInfo saved_mean_info(channel_shape, false);
    TensorInfo saved_var_info(channel_shape, false);

    gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, bias_info, mean_info, var_info}, {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
                                          {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)});
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
  */
}
#endif

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

TEST(OptimizerTest, SGDOptimizerTest_Gradient) {
  OpTester test("SGDOptimizer", 9, onnxruntime::kOnnxDomain);
  test.AddInput<float>("ETA", {}, {0.5f});
  test.AddInput<float>("W", {3}, {1, 2, 3});
  test.AddInput<float>("G", {3}, {4, 5, 6});
  test.AddMissingOptionalOutput<float>();
  test.AddOutput<float>("G_New", {3}, {-2.f, -2.5f, -3.f});
  test.Run();
}

void TestSoftmaxCrossEntropyGrad(const TensorShape& input_shape, const std::string& reduction) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropy", kMSDomain, 1};

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
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    logit_shape.emplace_back(D);

    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info},
                                          {{1}, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }

  // with weight
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    logit_shape.emplace_back(D);

    TensorInfo x_info(logit_shape);
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
  UnaryOpGradientTest("Gelu", kMSDomain, 1);
}

TEST(GradientCheckerTest, FastGeluGrad) {
  UnaryOpGradientTest("FastGelu", kMSDomain, 1);
}

struct AdamOptimizerInputOutput {
  AdamOptimizerInputOutput() {
    eta_half.resize(eta.size());
    g_half.resize(g.size());
    m1_half.resize(m1.size());
    m2_half.resize(m2.size());
    w_half.resize(w.size());
    ConvertFloatToMLFloat16(eta.data(), eta_half.data(), int(eta.size()));
    ConvertFloatToMLFloat16(g.data(), g_half.data(), int(g.size()));
    ConvertFloatToMLFloat16(m1.data(), m1_half.data(), int(m1.size()));
    ConvertFloatToMLFloat16(m2.data(), m2_half.data(), int(m2.size()));
    ConvertFloatToMLFloat16(w.data(), w_half.data(), int(w.size()));

    m1_new_half.resize(m1_new.size());
    m2_new_half.resize(m2_new.size());
    w_new_half.resize(w_new.size());
    g_new_half.resize(g_new.size());
    ConvertFloatToMLFloat16(m1_new.data(), m1_new_half.data(), int(m1_new.size()));
    ConvertFloatToMLFloat16(m2_new.data(), m2_new_half.data(), int(m2_new.size()));
    ConvertFloatToMLFloat16(w_new.data(), w_new_half.data(), int(w_new.size()));
    ConvertFloatToMLFloat16(g_new.data(), g_new_half.data(), int(g_new.size()));
  }

  // Fp32 Inputs
  std::vector<float> eta = {0.5f};
  std::vector<float> w = {1.0f, 2.0f, 3.0f};
  std::vector<float> g = {4.0f, 5.0f, 6.0f};
  std::vector<float> m1 = {0.1f, 0.2f, 0.3f};
  std::vector<float> m2 = {0.4f, 0.5f, 0.6f};

  // Fp16 Inputs
  std::vector<MLFloat16> eta_half;
  std::vector<MLFloat16> w_half;
  std::vector<MLFloat16> g_half;
  std::vector<MLFloat16> m1_half;
  std::vector<MLFloat16> m2_half;

  // FP32 Outptus
  std::vector<float> m1_new = {0.49f, 0.68f, 0.87f};
  std::vector<float> m2_new = {0.4156f, 0.5245f, 0.6354f};
  std::vector<float> w_new = {0.6199609f, 1.5305318f, 2.4542853f};
  std::vector<float> g_new = {-0.3800391f, -0.4694682f, -0.5457147f};

  // FP16 Outptus
  std::vector<MLFloat16> m1_new_half;
  std::vector<MLFloat16> m2_new_half;
  std::vector<MLFloat16> w_new_half;
  std::vector<MLFloat16> g_new_half;
};

TEST(OptimizerTest, AdamOptimizerTest) {
  OpTester test("AdamOptimizer", 9, onnxruntime::kOnnxDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, data.eta);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<float>("G", {3}, data.g);
  test.AddInput<float>("Moment_1", {3}, data.m1);
  test.AddInput<float>("Moment_2", {3}, data.m2);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<float>("Moment_1_Out", {3}, data.m1_new);
  test.AddOutput<float>("Moment_2_Out", {3}, data.m2_new);
  test.AddOutput<float>("W_Out", {3}, data.w_new);

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerTest_Gradient) {
  OpTester test("AdamOptimizer", 9, onnxruntime::kOnnxDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, data.eta);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<float>("G", {3}, data.g);
  test.AddInput<float>("Moment_1", {3}, data.m1);
  test.AddInput<float>("Moment_2", {3}, data.m2);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<float>("Moment_1_Out", {3}, data.m1_new);
  test.AddOutput<float>("Moment_2_Out", {3}, data.m2_new);
  test.AddMissingOptionalOutput<float>();
  test.AddOutput<float>("G_Out", {3}, data.g_new);

  test.Run();
}

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
    std::function<float(float)> transformer2 = [](float x) { return std::fmod(7 * std::fabs(x), 4.0f); };
    TensorInfo indices_info({num_indices}, false, &transformer2, DataTypeImpl::GetTensorType<int64_t>());

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
  OpTester test("Dropout", 12, kOnnxDomain, false);
  if (default_ratio)
    ratio = 0.5f;
  float input_constant = 3.0f;
  std::vector<float> x_data(x_shape.Size(), input_constant);
  std::vector<float> y_data(x_shape.Size(), 3.0f);

  test.AddInput<float>("x", x_shape.GetDims(), x_data);
  if (!default_ratio)
    test.AddInput<float>("ratio", {}, {ratio});
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
  OpTester test("DropoutGrad", 1, kMSDomain, true);
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

#ifdef USE_CUDA

TEST(GradientCheckerTest, DISABLED_Dropout) {
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

TEST(GradientCheckerTest, DISABLED_DropoutGrad) {
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

TEST(OptimizerTest, AdamOptimizerMixPrecisionTest) {
  OpTester test("AdamOptimizer", 9, onnxruntime::kOnnxDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddOutput<float>("W_Out", {3}, data.w_new);

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecision_FP16Weight_Test) {
  OpTester test("AdamOptimizer", 9, onnxruntime::kOnnxDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
  test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddOutput<float>("W_Out", {3}, data.w_new);
  test.AddMissingOptionalOutput<MLFloat16>();
  test.AddOutput<MLFloat16>("FP16_W_Out", {3}, data.w_new_half);

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecision_FP16Weight_SkipUpdate_Test) {
  OpTester test("AdamOptimizer", 9, onnxruntime::kOnnxDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
  test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);
  test.AddInput<float>("loss_scale", {1}, {1.0f});
  // grad clipping should not take effect
  test.AddInput<float>("grad_norm", {1}, {0.01f});
  test.AddInput<bool>("DoUpdate", {1}, {false});

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {3});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_half);
  test.AddOutput<float>("W_Out", {3}, data.w);
  test.AddMissingOptionalOutput<MLFloat16>();
  test.AddOutput<MLFloat16>("FP16_W_Out", {3}, data.w_half);

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecisionTestFloatEta) {
  OpTester test("AdamOptimizer", 9, onnxruntime::kOnnxDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, data.eta);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddOutput<float>("W_Out", {3}, data.w_new);

  test.Run();
}

TEST(OptimizerTest, AdamOptimizerMixPrecisionTest_Gradient) {
  OpTester test("AdamOptimizer", 9, onnxruntime::kOnnxDomain);
  AdamOptimizerInputOutput data;

  test.AddInput<float>("ETA", {}, data.eta);
  test.AddInput<int64_t>("Update_Count", {}, {3});
  test.AddInput<float>("W", {3}, data.w);
  test.AddInput<MLFloat16>("G", {3}, data.g_half);
  test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
  test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

  // Verify AdamOptimizer outputs
  test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
  test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
  test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
  test.AddMissingOptionalOutput<float>();
  test.AddOutput<MLFloat16>("G_Out", {3}, data.g_new_half);

  test.Run();
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
    const float loss_scale,
    const float g_norm,
    const float lambda,
    const float alpha,
    const float beta,
    const float epsilon,
    /* updated weights */ std::vector<float>& w_new,
    /* updated gradients */ std::vector<float>& g_new,
    /* updated momentum */ std::vector<float>& m_new,
    /* updated 2nd-order momentum */ std::vector<float>& v_new,
    const int64_t step = 0) {
  // Element counts of all vector-typed arguments.
  const int64_t size = std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());

  // Buffer to store update direction.
  std::vector<float> r(size, 0.0f);

  float g_scale = loss_scale;
  if (g_norm > loss_scale) {
    g_scale *= g_norm / loss_scale;
  }

  const float alpha_correction = step > 0 ?
    1.f - std::pow(alpha, static_cast<float>(step)) : 1.f;
  const float beta_correction = step > 0 ?
    1.f - std::pow(beta, static_cast<float>(step)) : 1.f;

  // Compute new 1st-, 2nd-order momentums, and the update direction.
  for (int i = 0; i < size; ++i) {
    const float g_scaled = g[i] / g_scale;
    m_new[i] = alpha * m[i] + (1.0f - alpha) * g_scaled;
    v_new[i] = beta * v[i] + (1.0f - beta) * g_scaled * g_scaled;
    const float m_new_tmp = m_new[i] / alpha_correction;
    const float v_new_tmp = v_new[i] / beta_correction;
    r[i] = lambda * w[i] + m_new_tmp / (std::sqrt(v_new_tmp) + epsilon);
  }

  // Compute squared sum of all elements. Note that Eigen sqrt could lead to significant
  // numerical error so we use std::sqrt. The std::inner_product produces wrong result
  // when std::inner_product(r.begin(), r.end(), r.begin(), 0) so we just use a loop below.
  float r_norm = 0.0f;
  float w_norm = 0.0f;
  for (int i = 0; i < size; ++i) {
    r_norm += r[i] * r[i];
    w_norm += w[i] * w[i];
  }

  r_norm = std::sqrt(r_norm);
  w_norm = std::sqrt(w_norm);

  const float ratio = (w_norm != 0.0f && r_norm != 0.0f) ? eta * w_norm / r_norm : eta;

  // Compute the new weight.
  for (int64_t i = 0; i < size; ++i) {
    g_new[i] = -ratio * r[i];
    w_new[i] = w[i] + g_new[i];
  }
}

template <typename T1, typename T2, typename T3, typename T4>
void run_lamb_test_with_baseline(
    const std::vector<int64_t>& shape,
    const std::vector<T1>& eta,
    const std::vector<T2>& w,
    const std::vector<T3>& g,
    const std::vector<T4>& m,
    const std::vector<T4>& v,
    const float alpha,
    const float beta,
    const float lambda,
    const float epsilon,
    const std::vector<T2>& w_new,
    const std::vector<T3>& g_new,
    const std::vector<T4>& m_new,
    const std::vector<T4>& v_new,
    const std::vector<MLFloat16>& w_half = {},
    const std::vector<MLFloat16>& w_new_half = {},
    const bool do_update = true,
    const int64_t step = 0) {
  OpTester test("LambOptimizer", 9, onnxruntime::kOnnxDomain, true);

  test.AddInput<bool>("update_signal", {1}, {do_update});
  test.AddMissingOptionalInput<T2>();
  test.AddMissingOptionalInput<T2>();
  test.AddInput<T1>("ETA", {1}, eta);
  if (step > 0) {
    test.AddInput<int64_t>("Step", {}, {step});
  } else {
    test.AddMissingOptionalInput<int64_t>();
  }
  test.AddInput<T2>("W", shape, w);
  test.AddInput<T3>("G", shape, g);
  test.AddInput<T4>("Moment_1", shape, m);
  test.AddInput<T4>("Moment_2", shape, v);
  if (!w_half.empty()) {
    test.AddInput<MLFloat16>("FP16_W", shape, w_half);
  } else {
    test.AddMissingOptionalInput<MLFloat16>();
  }

  test.AddAttribute("alpha", std::vector<float>(1, alpha));
  test.AddAttribute("beta", std::vector<float>(1, beta));
  test.AddAttribute("lambda", std::vector<float>(1, lambda));
  test.AddAttribute("epsilon", std::vector<float>(1, epsilon));

  if (step > 0) {
    test.AddOutput<int64_t>("Step_Out", {}, {do_update ? step + 1 : step});
  } else {
    test.AddMissingOptionalOutput<int64_t>();
  }
  if (!w_new.empty()) {
    test.AddOutput<T2>("W_Out", shape, w_new);
  } else {
    test.AddMissingOptionalOutput<T2>();
  }
  if (!g_new.empty()) {
    test.AddOutput<T3>("G_Out", shape, g_new);
  } else {
    test.AddMissingOptionalOutput<T3>();
  }
  test.AddOutput<T4>("Moment_1_Out", shape, m_new);
  test.AddOutput<T4>("Moment_2_Out", shape, v_new);
  if (!w_new_half.empty()) {
    test.AddOutput<MLFloat16>("FP16_W_Out", shape, w_new_half);
  } else {
    test.AddMissingOptionalOutput<MLFloat16>();
  }

  test.Run();
}

template <typename T1, typename T2, typename T3, typename T4>
void run_multi_tensor_lamb_test_with_baseline(
    const std::vector<std::vector<int64_t>>& shapes,
    const T1 eta,
    const T1 loss_scale,
    const T1 g_norm,
    const std::vector<std::vector<T2>>& ws,
    const std::vector<std::vector<T3>>& gs,
    const std::vector<std::vector<T4>>& ms,
    const std::vector<std::vector<T4>>& vs,
    const std::vector<float>& alphas,
    const std::vector<float>& betas,
    const std::vector<float>& lambdas,
    const std::vector<float>& epsilons,
    const std::vector<std::vector<T2>>& w_news,
    const std::vector<std::vector<T3>>& g_news,
    const std::vector<std::vector<T4>>& m_news,
    const std::vector<std::vector<T4>>& v_news,
    const std::vector<std::vector<MLFloat16>>& w_halfs = {},
    const std::vector<std::vector<MLFloat16>>& w_new_halfs = {},
    const bool do_update = true,
    const int64_t step = 0) {
  OpTester test("LambOptimizer", 9, onnxruntime::kOnnxDomain, true);

  ORT_ENFORCE(shapes.size() == ws.size());
  ORT_ENFORCE(shapes.size() == gs.size());
  ORT_ENFORCE(shapes.size() == ms.size());
  ORT_ENFORCE(shapes.size() == vs.size());
  ORT_ENFORCE(shapes.size() == alphas.size());
  ORT_ENFORCE(shapes.size() == betas.size());
  ORT_ENFORCE(shapes.size() == lambdas.size());
  ORT_ENFORCE(shapes.size() == epsilons.size());
  if (!w_news.empty()) {
    ORT_ENFORCE(shapes.size() == w_news.size());
  }
  if (!g_news.empty()) {
    ORT_ENFORCE(shapes.size() == g_news.size());
  }
  ORT_ENFORCE(shapes.size() == m_news.size());
  ORT_ENFORCE(shapes.size() == v_news.size());
  if (!w_halfs.empty()) {
    ORT_ENFORCE(shapes.size() == w_halfs.size());
  }
  if (!w_new_halfs.empty()) {
    ORT_ENFORCE(shapes.size() == w_new_halfs.size());
  }

  const int group_count = static_cast<int>(ws.size());

  test.AddInput<bool>("update_signal", {}, {do_update});
  test.AddInput<T1>("loss_scale", {}, {loss_scale});
  test.AddInput<T1>("gradient_norm", {}, {g_norm});
  test.AddInput<T1>("ETA", {}, {eta});
  if (step > 0) {
    test.AddInput<int64_t>("Step", {}, {step});
    test.AddOutput<int64_t>("Step_Out", {}, {do_update ? step + 1 : step});
  } else {
    test.AddMissingOptionalInput<int64_t>();
    test.AddMissingOptionalOutput<int64_t>();
  }
  for (int i = 0; i < group_count; ++i) {
    std::string w_name = "W_" + std::to_string(i);
    std::string g_name = "G_" + std::to_string(i);
    std::string m1_name = "Moment_1_" + std::to_string(i);
    std::string m2_name = "Moment_2_" + std::to_string(i);
    std::string w_fp16_name = "FP16_W_" + std::to_string(i);
    std::string w_new_name = "W_Out_" + std::to_string(i);
    std::string g_new_name = "G_Out_" + std::to_string(i);
    std::string m1_new_name = "Moment_1_Out_" + std::to_string(i);
    std::string m2_new_name = "Moment_2_Out_" + std::to_string(i);
    std::string w_fp16_new_name = "FP16_W_Out_" + std::to_string(i);

    test.AddInput<T2>(w_name.c_str(), shapes[i], ws[i]);
    test.AddInput<T3>(g_name.c_str(), shapes[i], gs[i]);
    test.AddInput<T4>(m1_name.c_str(), shapes[i], ms[i]);
    test.AddInput<T4>(m2_name.c_str(), shapes[i], vs[i]);
    if (!w_halfs.empty() && !w_halfs[i].empty()) {
      test.AddInput<MLFloat16>(w_fp16_name.c_str(), shapes[i], w_halfs[i]);
    } else {
      test.AddMissingOptionalInput<MLFloat16>();
    }

    if (!w_news.empty() && !w_news[i].empty()) {
      test.AddOutput<T2>(w_new_name.c_str(), shapes[i], w_news[i]);
    } else {
      test.AddMissingOptionalOutput<T2>();
    }
    if (!g_news.empty() && !g_news[i].empty()) {
      test.AddOutput<T3>(g_new_name.c_str(), shapes[i], g_news[i]);
    } else {
      test.AddMissingOptionalOutput<T3>();
    }
    test.AddOutput<T4>(m1_new_name.c_str(), shapes[i], m_news[i]);
    test.AddOutput<T4>(m2_new_name.c_str(), shapes[i], v_news[i]);
    if (!w_new_halfs.empty() && !w_new_halfs[i].empty()) {
      test.AddOutput<MLFloat16>(w_fp16_new_name.c_str(), shapes[i], w_new_halfs[i]);
    } else {
      test.AddMissingOptionalOutput<MLFloat16>();
    }
  }

  test.AddAttribute("alpha", alphas);
  test.AddAttribute("beta", betas);
  test.AddAttribute("lambda", lambdas);
  test.AddAttribute("epsilon", epsilons);

  test.Run();
}

// Lamb test without baseline. This function computes
// baseline via an internal function and then invoke
// run_lamb_test_with_baseline(...) to check the result.
void run_multi_tensor_lamb_test(
    const std::vector<std::vector<int64_t>> shapes,
    const float eta,
    const float loss_scale,
    const float g_norm,
    const std::vector<std::vector<float>> ws,
    const std::vector<std::vector<float>> gs,
    const std::vector<std::vector<float>> ms,
    const std::vector<std::vector<float>> vs,
    const std::vector<float> lambdas,
    const std::vector<float> alphas,
    const std::vector<float> betas,
    const std::vector<float> epsilons,
    const int64_t step = 0) {
  // Check if parallel vectors have the same length.
  ORT_ENFORCE(shapes.size() == ws.size());
  ORT_ENFORCE(shapes.size() == gs.size());
  ORT_ENFORCE(shapes.size() == ms.size());
  ORT_ENFORCE(shapes.size() == vs.size());
  ORT_ENFORCE(shapes.size() == alphas.size());
  ORT_ENFORCE(shapes.size() == betas.size());
  ORT_ENFORCE(shapes.size() == lambdas.size());
  ORT_ENFORCE(shapes.size() == epsilons.size());

  const int group_count = static_cast<int>(ws.size());

  // Output buffers of the optimizer.
  std::vector<std::vector<float>> w_news(group_count);
  std::vector<std::vector<float>> g_news(group_count);
  std::vector<std::vector<float>> m_news(group_count);
  std::vector<std::vector<float>> v_news(group_count);

  for (int i = 0; i < group_count; ++i) {
    w_news[i] = std::vector<float>(ws[i].size(), 0.f);
    g_news[i] = std::vector<float>(gs[i].size(), 0.f);
    m_news[i] = std::vector<float>(ms[i].size(), 0.f);
    v_news[i] = std::vector<float>(vs[i].size(), 0.f);

    // Invoke LAMB's reference implementation to compute baseline output.
    compute_lamb(
        shapes[i], ws[i], gs[i], ms[i], vs[i],
        eta, loss_scale, g_norm,
        lambdas[i], alphas[i], betas[i], epsilons[i],
        w_news[i], g_news[i], m_news[i], v_news[i], step);
  }

  // Create tests to make sure the output is correct.

  // Output new weights.
  run_multi_tensor_lamb_test_with_baseline(
      shapes, eta, loss_scale, g_norm,
      ws, gs, ms, vs,
      alphas, betas, lambdas, epsilons,
      w_news, {}, m_news, v_news, {}, {}, true, step);

    // Output new gradients.
    run_multi_tensor_lamb_test_with_baseline(
      shapes, eta, loss_scale, g_norm,
      ws, gs, ms, vs,
      alphas, betas, lambdas, epsilons,
      {}, g_news, m_news, v_news, {}, {}, true, step);
}

void run_lamb_mix_precision_test(
    const std::vector<int64_t>& shape,
    const std::vector<float>& eta,
    const std::vector<float>& w,
    const std::vector<float>& g,
    const std::vector<float>& m,
    const std::vector<float>& v,
    const float lambda,
    const float alpha,
    const float beta,
    const float epsilon,
    const int64_t step = 0) {
  std::vector<float> w_new(w.size(), 0);
  std::vector<float> g_new(g.size(), 0);
  std::vector<float> m_new(m.size(), 0);
  std::vector<float> v_new(v.size(), 0);

  // Invoke LAMB's reference implementation to compute output.
  compute_lamb(
      shape, w, g, m, v,
      eta[0], 1.f, 1.f, lambda, alpha, beta, epsilon,
      w_new, g_new, m_new, v_new, step);

  std::vector<MLFloat16> eta_half(eta.size());
  std::vector<MLFloat16> g_half(w.size());
  std::vector<MLFloat16> m_half(w.size());
  std::vector<MLFloat16> v_half(w.size());
  std::vector<MLFloat16> w_half(w.size());
  ConvertFloatToMLFloat16(eta.data(), eta_half.data(), int(eta.size()));
  ConvertFloatToMLFloat16(g.data(), g_half.data(), int(g.size()));
  ConvertFloatToMLFloat16(m.data(), m_half.data(), int(m.size()));
  ConvertFloatToMLFloat16(v.data(), v_half.data(), int(v.size()));
  ConvertFloatToMLFloat16(w.data(), w_half.data(), int(w.size()));

  std::vector<MLFloat16> m_new_half(m_new.size());
  std::vector<MLFloat16> v_new_half(v_new.size());
  std::vector<MLFloat16> w_new_half(w_new.size());
  std::vector<MLFloat16> g_new_half(g_new.size());
  ConvertFloatToMLFloat16(m_new.data(), m_new_half.data(), int(m_new.size()));
  ConvertFloatToMLFloat16(v_new.data(), v_new_half.data(), int(v_new.size()));
  ConvertFloatToMLFloat16(w_new.data(), w_new_half.data(), int(w_new.size()));
  ConvertFloatToMLFloat16(g_new.data(), g_new_half.data(), int(g_new.size()));

  // Half momentums, without fp16 weight
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m_half, v_half, alpha, beta, lambda, epsilon, w_new, {}, m_new_half, v_new_half, {}, {}, true, step);

  // Float momentums, without fp16 weight
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, w_new, {}, m_new, v_new, {}, {}, true, step);

  // Half momentums, with fp16 weight
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m_half, v_half, alpha, beta, lambda, epsilon, w_new, {}, m_new_half, v_new_half, {}, {}, true, step);

  // Float momentums, with fp16 weight
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, w_new, {}, m_new, v_new, w_half, w_new_half, true, step);

  // Half momentums, with fp16 weight, skip weight update
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m_half, v_half, alpha, beta, lambda, epsilon, w, {}, m_half, v_half, w_half, w_half, false, step);

  // Float momentums, with fp16 weight, skip weight update
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, w, {}, m, v, w_half, w_half, false, step);

  // Float eta, float momentums, with fp16 weight
  run_lamb_test_with_baseline(
      shape, eta, w, g_half, m, v, alpha, beta, lambda, epsilon, w_new, {}, m_new, v_new, w_half, w_new_half, true, step);

  // Float eta, float momentums, with fp16 weight, skip weight update
  run_lamb_test_with_baseline(
      shape, eta, w, g_half, m, v, alpha, beta, lambda, epsilon, w, {}, m, v, w_half, w_half, false, step);

  // Float momentums, without fp16 weight, output gradients only
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, {}, g_new_half, m_new, v_new, {}, {}, true, step);

  // Float momentums, with fp16 weight, output gradients only
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, {}, g_new_half, m_new, v_new, w_half, {}, true, step);

  // Float momentums, with fp16 weight, output gradients only, skip weight update
  run_lamb_test_with_baseline(
      shape, eta_half, w, g_half, m, v, alpha, beta, lambda, epsilon, {}, g_half, m, v, w_half, {}, false, step);
}

// A optimizer test with an 2-element vector.
TEST(OptimizerTest, LambOptimizerTestVector) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      1.f,
      1.f,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon});
}

TEST(OptimizerTest, LambOptimizerTestVectorWithZeroWeight) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {0.0f, 0.0f};
  const std::vector<float> g = {1.0f, -1.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      1.f,
      1.f,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon});
}

TEST(OptimizerTest, LambOptimizerTestBiasCorrectionFirst) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_multi_tensor_lamb_test(
    {shape},
    eta,
    1.f,
    1.f,
    {w},
    {g},
    {m},
    {v},
    {lambda},
    {alpha},
    {beta},
    {epsilon},
    1);
}

TEST(OptimizerTest, LambOptimizerTestBiasCorrectionThird) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_multi_tensor_lamb_test(
    {shape},
    eta,
    1.f,
    1.f,
    {w},
    {g},
    {m},
    {v},
    {lambda},
    {alpha},
    {beta},
    {epsilon},
    3);
}

// A optimizer test with an 2-by-1-by-1-by-1 tensor.
TEST(OptimizerTest, LambOptimizerTest4DTensor) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 1, 1, 1};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f};
  const std::vector<float> v = {2.0f, 1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      1.f,
      1.f,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon});
}

// A optimizer test with an 2-by-3 tensor.
TEST(OptimizerTest, LambOptimizerTest2by3Tensor) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 3};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f};
  const std::vector<float> g = {3.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f};
  const std::vector<float> m = {-1.0f, -2.0f, 2.0f, 1.0f, 1.0f, -2.0f};
  const std::vector<float> v = {1.0f, 1.0f, 5.0f, 5.0f, 6.0f, 6.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      1.f,
      1.f,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon});
}

// A optimizer test with an 1-element tensor.
TEST(OptimizerTest, LambOptimizerTestScalar) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {(int64_t)1};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f};
  const std::vector<float> g = {3.0f};
  const std::vector<float> m = {-10.0f};
  const std::vector<float> v = {1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  // Intermediate and output buffers of the optimizer.
  std::vector<float> m_new = {0.0f};
  std::vector<float> v_new = {0.0f};
  std::vector<float> w_new = {0.0f};

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      1.f,
      1.f,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon});
}

TEST(OptimizerTest, LambOptimizerTestScalarScaling) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {(int64_t)1};
  const float eta = 0.5f;
  const std::vector<float> w = {1.0f};
  const std::vector<float> g = {3.0f};
  const std::vector<float> m = {-10.0f};
  const std::vector<float> v = {1.0f};
  const float lambda = 0.5f;
  const float alpha = 0.2f;
  const float beta = 0.8f;
  const float epsilon = 1e-6f;

  // Intermediate and output buffers of the optimizer.
  std::vector<float> m_new = {0.0f};
  std::vector<float> v_new = {0.0f};
  std::vector<float> w_new = {0.0f};

  run_multi_tensor_lamb_test(
      {shape},
      eta,
      8.f,
      4.f,
      {w},
      {g},
      {m},
      {v},
      {lambda},
      {alpha},
      {beta},
      {epsilon});
}

TEST(OptimizerTest, LambOptimizerTestExternalBaseline) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 5};
  const std::vector<float> eta = {0.1f};
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

  const float lambda = 0.1f;
  const float alpha = 0.1f;
  const float beta = 0.01f;
  const float epsilon = 0.1f;

  std::vector<float> w_new = {
      0.02979828f, 0.13677707f, -0.22708717f, -0.20361158f, -0.15338624f, 0.1081504f,
      -0.03804127f, 0.28198114f, 0.00430069f, 0.05319814f};
  std::vector<float> g_new = {
      0.01600802f, -0.01630484f, 0.01647800f, 0.01437007f, -0.01568577f, 0.01120441f,
      -0.01580611f, 0.01555834f, 0.01608062f, -0.01512874f};
  std::vector<float> m_new = {
      -6.0344763f, 10.479931f, -9.15947f, -0.57894087f, 5.824918f, -0.2165685f,
      3.5303047f, -5.9299808f, -9.281795f, 1.1496004f};
  std::vector<float> v_new = {
      3.6645618e+01f, 1.1174072e+02f, 8.4853485e+01f, 3.3100498e-01f, 3.4628010e+01f,
      4.4757873e-02f, 1.2550836e+01f, 3.5532223e+01f, 8.7362823e+01f, 1.3257366e+00f};

  // Output new weights
  run_lamb_test_with_baseline(
      shape, eta, w, g, m, v, alpha, beta, lambda, epsilon, w_new, {}, m_new, v_new);

  // Output new gradients
  run_lamb_test_with_baseline(
      shape, eta, w, g, m, v, alpha, beta, lambda, epsilon, {}, g_new, m_new, v_new);
}

TEST(OptimizerTest, LambOptimizerTestExternalBaselineDouble) {
  // Input tensors and attributes.
  const std::vector<int64_t> shape = {2, 5};
  const std::vector<double> eta = {0.1f};
  const std::vector<double> w = {
      0.01379026, 0.15308191, -0.24356517, -0.21798165, -0.13770047, 0.09694599,
      -0.02223516, 0.2664228, -0.01177993, 0.06832688};
  const std::vector<double> g = {
      -6.048543, 10.569487, -9.207029, -0.57407373,
      5.884985, -0.21047728, 3.539946, -5.957566, -9.343748, 1.1502024};
  const std::vector<double> m = {
      -5.9078765, 9.673933, -8.731428, -0.6227454, 5.284312, -0.27138948,
      3.443532, -5.681713, -8.72421, 1.1441823};
  const std::vector<double> v = {
      4.2659229e+01, 1.1438165e+02, 9.3179581e+01, 4.7399229e-01, 3.4129276e+01,
      9.0019435e-02, 1.4493006e+01, 3.9455612e+01, 9.3025581e+01, 1.6000764e+0};

  const float lambda = 0.1f;
  const float alpha = 0.1f;
  const float beta = 0.01f;
  const float epsilon = 0.1f;

  std::vector<double> w_new = {
      0.02979828, 0.13677707, -0.22708717, -0.20361158, -0.15338624, 0.1081504,
      -0.03804127, 0.28198114, 0.00430069, 0.05319814};
  std::vector<double> g_new = {
      0.01600802, -0.01630484, 0.016478, 0.01437007, -0.01568577, 0.01120441,
      -0.01580611, 0.01555834, 0.01608062, -0.01512874};
  std::vector<double> m_new = {
      -6.0344763, 10.479931, -9.15947, -0.57894087, 5.824918, -0.2165685,
      3.5303047, -5.9299808, -9.281795, 1.1496004};
  std::vector<double> v_new = {
      3.6645618e+01, 1.1174072e+02, 8.4853485e+01, 3.3100498e-01, 3.4628010e+01,
      4.4757873e-02, 1.2550836e+01, 3.5532223e+01, 8.7362823e+01, 1.3257366e+00};

  // Output new weights
  run_lamb_test_with_baseline(
      shape, eta, w, g, m, v, alpha, beta, lambda, epsilon, w_new, {}, m_new, v_new);

  // Output new gradients
  run_lamb_test_with_baseline(
      shape, eta, w, g, m, v, alpha, beta, lambda, epsilon, {}, g_new, m_new, v_new);
}

TEST(OptimizerTest, LambOptimizerTest5DTensorMixPrecision32_16) {
  const std::vector<int64_t> shape = {2, 2, 2, 1, 1};
  const std::vector<float> eta = {0.5f};
  const std::vector<float> w = {1.0f, 2.0f, 2.5f, 1.5f, 1.0f, 2.0f, 2.0f, 1.5f};
  const std::vector<float> g = {-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 0.8f};
  const std::vector<float> m = {1.0f, 2.0f, -0.25f, 1.1f, 1.0f, 2.0f, -0.21f, 1.1f};
  const std::vector<float> v = {1.5f, 1.0f, 1.1f, 0.76f, 1.5f, 1.0f, 1.5f, 0.76f};

  const float lambda = 1.5f;
  const float alpha = 1.5f;
  const float beta = 1.5f;
  const float epsilon = 1.0f;

  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon);
}

TEST(OptimizerTest, LambOptimizerTestSimpleBaselineMixPrecision32_16) {
  const std::vector<int64_t> shape = {2, 1};
  const std::vector<float> eta = {1.0f};
  const std::vector<float> w = {1.0f, 1.0f};
  const std::vector<float> g = {-1.0f, 1.0f};
  const std::vector<float> m = {1.0f, 1.0f};
  const std::vector<float> v = {0.0f, 0.0f};

  const float lambda = 0.0f;
  const float alpha = 1.0f;
  const float beta = 1.0f;
  const float epsilon = 1.0f;

  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon);
}

TEST(OptimizerTest, LambOptimizerTestBaselineMixPrecision32_16) {
  const std::vector<int64_t> shape = {2, 1};
  const std::vector<float> eta = {0.1f};
  const std::vector<float> w = {-1.5f, 2.4f};
  const std::vector<float> g = {-0.75f, 1.2f};
  const std::vector<float> m = {0.87f, -0.94f};
  const std::vector<float> v = {0.12f, 0.28f};

  const float lambda = 0.25f;
  const float alpha = 0.9f;
  const float beta = 0.95f;
  const float epsilon = 0.33f;

  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon);
}

TEST(OptimizerTest, LambOptimizerTestScalarMixPrecision32_16) {
  const std::vector<int64_t> shape = {1};
  const std::vector<float> eta = {0.1f};
  const std::vector<float> w = {-1.5f};
  const std::vector<float> g = {-0.75f};
  const std::vector<float> m = {0.87f};
  const std::vector<float> v = {0.12f};

  const float lambda = 0.25f;
  const float alpha = 0.9f;
  const float beta = 0.95f;
  const float epsilon = 0.33f;

  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon);

  run_lamb_mix_precision_test(
      shape, eta, w, g, m, v,
      lambda, alpha, beta, epsilon, 2);
}

TEST(OptimizerTest, LambOptimizerTestLarge) {
  // Input tensors and attributes.
  for (const auto& size : {55667, 1944006, 3907584}) {
    const std::vector<int64_t> shape = {static_cast<int64_t>(size)};
    const float eta = 0.5f;
    std::vector<float> w(size);
    std::vector<float> g(size);
    std::vector<float> m(size);
    std::vector<float> v(size);

    std::random_device random_device;
    std::mt19937 random_engine(0);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    for (int i = 0; i < size; ++i) {
      w[i] = dist(random_engine);
      g[i] = dist(random_engine);
      m[i] = dist(random_engine);
      v[i] = dist(random_engine);
    }

    const float lambda = 0.5f;
    const float alpha = 0.2f;
    const float beta = 0.8f;
    const float epsilon = 1e-6f;

    run_multi_tensor_lamb_test(
        {shape},
        eta,
        1.f,
        1.f,
        {w},
        {g},
        {m},
        {v},
        {lambda},
        {alpha},
        {beta},
        {epsilon});
  }
}

TEST(OptimizerTest, LambOptimizerMultiTensor6) {
  const int group_count = 127;
  std::random_device random_device;
  std::mt19937 random_engine(0);
  std::uniform_real_distribution<float> dist(0.1f, 1.0f);
  std::uniform_int_distribution<int64_t> dist_int(1, 1228);

  std::vector<int64_t> sizes(group_count);
  std::vector<std::vector<int64_t>> shapes(group_count);

  std::vector<std::vector<float>> ws(group_count);
  std::vector<std::vector<float>> gs(group_count);
  std::vector<std::vector<float>> ms(group_count);
  std::vector<std::vector<float>> vs(group_count);
  std::vector<float> alphas(group_count);
  std::vector<float> betas(group_count);
  std::vector<float> lambdas(group_count);
  std::vector<float> epsilons(group_count);

  const float eta = dist(random_engine);

  for (int64_t i = 0; i < group_count; ++i) {
    const auto size = dist_int(random_engine);
    sizes[i] = size;
    shapes[i] = std::vector<int64_t>(1, size);

    ws[i] = std::vector<float>(sizes[i]);
    gs[i] = std::vector<float>(sizes[i]);
    ms[i] = std::vector<float>(sizes[i]);
    vs[i] = std::vector<float>(sizes[i]);

    for (int64_t j = 0; j < sizes[i]; ++j) {
      ws[i][j] = dist(random_engine);
      gs[i][j] = dist(random_engine);
      ms[i][j] = dist(random_engine);
      vs[i][j] = dist(random_engine);
    }

    alphas[i] = dist(random_engine);
    betas[i] = dist(random_engine);
    lambdas[i] = dist(random_engine);
    epsilons[i] = dist(random_engine);
  }

  run_multi_tensor_lamb_test(
      shapes, eta, 1.f, 1.f,
      ws, gs, ms, vs,
      lambdas, alphas, betas, epsilons);
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

TEST(GradientUtilsTest, InPlaceAccumulatorFloat32) {
  OpTester test("InPlaceAccumulator", 9, onnxruntime::kOnnxDomain);

  test.AddInput<float>("old_sum", {3}, {1, 2, 3});
  test.AddInput<float>("value", {3}, {4, 5, 6});

  test.AddOutput<float>("new_sum", {3}, {5, 7, 9});

  test.Run();
}

#ifdef USE_CUDA
TEST(GradientUtilsTest, InPlaceAccumulatorFloat16) {
  OpTester test("InPlaceAccumulator", 9, onnxruntime::kOnnxDomain);

  std::vector<float> old_sum = {1.0f, 2.0f, 3.0f};
  std::vector<float> value = {4.0f, 5.0f, 6.0f};
  std::vector<float> new_sum = {5.0f, 7.0f, 9.0f};

  std::vector<MLFloat16> value_half(3);
  ConvertFloatToMLFloat16(value.data(), value_half.data(), 3);

  test.AddInput<float>("old_sum", {3}, old_sum);
  test.AddInput<MLFloat16>("value", {3}, value_half);
  test.AddOutput<float>("new_sum", {3}, new_sum);

  // Didn't implement mixed precision InPlaceAccumulator in CPU
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}
#endif

TEST(GradientUtilsTest, ZeroGradientFloat32) {
  OpTester test("ZeroGradient", 9, onnxruntime::kOnnxDomain);

  test.AddInput<float>("old_gradient", {3}, {1, 2, 3});
  test.AddInput<float>("reset_signal", {3}, {1, 10, 100});

  test.AddOutput<float>("zero_gradient", {3}, {0, 0, 0});

  test.Run();
}

#ifdef USE_CUDA
TEST(GradientUtilsTest, ZeroGradientFloat16) {
  OpTester test("ZeroGradient", 9, onnxruntime::kOnnxDomain);

  std::vector<float> old_gradient = {1.0f, 2.0f, 3.0f};
  std::vector<float> zero_gradient = {0.0f, 0.0f, 0.0f};

  std::vector<MLFloat16> old_gradient_half(3);
  std::vector<MLFloat16> zero_gradient_half(3);

  ConvertFloatToMLFloat16(old_gradient.data(), old_gradient_half.data(), 3);
  ConvertFloatToMLFloat16(zero_gradient.data(), zero_gradient_half.data(), 3);

  test.AddInput<MLFloat16>("old_gradient", {3}, old_gradient_half);
  test.AddInput<float>("reset_signal", {3}, {1, 10, 100});

  test.AddOutput<MLFloat16>("zero_gradient", {3}, zero_gradient_half);

  test.Run();
}

#endif

TEST(GradientCheckerTest, SliceGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Slice", kOnnxDomain, 10};

  // default values for optional tensors like axes and steps.
  {
    TensorInfo x_info({2, 4}, true);
    TensorInfo start_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo end_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 0}, {2, 3}};

    TensorInfo y_info({1, 3}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, start_info, end_info}, {y_info}, &max_error, x_datas);
    EXPECT_IS_TINY(max_error);
  }

  // all input tensors have some value and slice end out of bound.
  {
    TensorInfo x_info({2, 4}, true);
    TensorInfo start_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo end_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo axes_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo steps_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 0}, {2, 3}, {0, 1}, {1, 2}};

    TensorInfo y_info({1, 2}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, start_info, end_info, axes_info, steps_info},
        {y_info}, &max_error, x_datas);

    EXPECT_IS_TINY(max_error);
  }

  // 3-D tensor
  {
    TensorInfo x_info({2, 4, 2}, true);
    TensorInfo start_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo end_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo axes_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo steps_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8}, {1, 0}, {2, 3},
      {0, 1}, {1, 2}};

    TensorInfo y_info({1, 2, 2}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, start_info, end_info, axes_info, steps_info}, {y_info},
        &max_error, x_datas);

    EXPECT_IS_TINY(max_error);
  }
}

}  // namespace test
}  // namespace onnxruntime

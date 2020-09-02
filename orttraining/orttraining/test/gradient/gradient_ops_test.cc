﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef NDEBUG  // disable for debug builds because some of these tests are slow

#include <algorithm>
#include <bitset>
#include <cmath>
#include <random>
#include <thread>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/test_random_seed.h"
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
      << "; ORT test random seed: " << GetTestRandomSeed() << "; "

#define EXPECT_IS_TINY(max_error) \
  EXPECT_IS_TINIER_THAN(max_error, 1.5e-2f)

static void RunReductionTests(const OpDef& op_def) {
  TestDataVector test_data(
      // Input X
      {
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
          {{4, 3, 2}},
      },
      // Input Y
      {
          {{1, 1, 1}},
          {{}},
          {{1, 3, 1}},
          {{2}},
          {{4, 1, 2}},
          {{4, 3}},
          {{4, 1, 2}},
          {{4}}},
      // Attributes
      {
          // default
          {},
          // axes = [0, 1, 2], keepdims = 0
          {MakeAttribute("axes", std::vector<int64_t>{0, 1, 2}),
           MakeAttribute("keepdims", int64_t(0))},
          // axes = [0, 2], keepdims = 1
          {MakeAttribute("axes", std::vector<int64_t>{0, 2})},
          // axes = [0, 1], keepdims = 0
          {MakeAttribute("axes", std::vector<int64_t>{0, 1}),
           MakeAttribute("keepdims", int64_t(0))},
          // axes = [1], keepdims = 1
          {MakeAttribute("axes", std::vector<int64_t>{1}),
           MakeAttribute("keepdims", int64_t(1))},
          // axes = [2], keepdims = 0
          {MakeAttribute("axes", std::vector<int64_t>{2}),
           MakeAttribute("keepdims", int64_t(0))},
          // axes = [-2], keepdims = 1
          {MakeAttribute("axes", std::vector<int64_t>{-2}),
           MakeAttribute("keepdims", int64_t(1))},
          // axes = [-2, -1], keepdims = 0
          {MakeAttribute("axes", std::vector<int64_t>{-2, -1}),
           MakeAttribute("keepdims", int64_t(0))}});

  GradientChecker<float, float, float> gradient_checker;

  float max_error;

  for (size_t i = 0; i < std::get<0>(test_data).size(); i++) {
    max_error = 0;
    gradient_checker.ComputeGradientError(op_def, std::get<0>(test_data)[i],
                                          std::get<1>(test_data)[i], &max_error,
                                          std::get<2>(test_data)[i]);
    EXPECT_IS_TINY(max_error);
  }
}

template <typename T>
void GenerateRandomDataWithOneHot(
    std::vector<std::vector<float>>& x_datas,
    std::vector<TensorShape> input_shapes,
    const std::unordered_set<int>& one_hot_input_indices) {
  for (int i = 0; i < 2; i++) {
    // TODO: Consider varying mean and variance
    float scale = 5.f;
    float mean = 0.f;
    const uint32_t seed = GetTestRandomSeed();

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
                                   std::function<float(float)>* transformer = nullptr,
                                   bool check_not_have_shape_inferencing = true) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op_type};
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  //shape(A) = (2, 3, 4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo B_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo B_info{{}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (,), shape(B) = (2, 3, 4, 5), i.e. A is a scalar ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{}, true, transformer};
    TensorInfo B_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo B_info{{5}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{4, 5}, true, transformer};
    TensorInfo B_info{{2, 3, 4, 5}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (1, 4, 5), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{1, 4, 5}, true, transformer};
    TensorInfo B_info{{2, 3, 1, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (3, 4, 5), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{3, 4, 5}, true, transformer};
    TensorInfo B_info{{2, 1, 1, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //shape(A) = (2, 1, 1, 5), shape(B) = (1, 3, 4, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    TensorInfo A_info{{2, 1, 1, 5}, true, transformer};
    TensorInfo B_info{{1, 3, 4, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 4, 5}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  // symbolic broadcast
  // shape(A) = (4, 2, 1, "seq(3)"), shape(B) = (4, 2, 1, 1), ==> shape(result) = (4, 2, 1, 3)
  {
    TensorInfo A_info{{4, 2, 1, 3}, true, transformer, DataTypeImpl::GetTensorType<float>(), {"4", "2", "1", "seq"}};
    TensorInfo B_info{{4, 2, 1, 1}, true, transformer, DataTypeImpl::GetTensorType<float>(), {"4", "2", "1", "1"}};
    TensorInfo Y_info{{4, 2, 1, 3}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }
  // symbolic broadcast + numeric broadcast
  // shape(A) = ("batch(4)", 2, "seq(3)", "seq(3)"), shape(B) = ("batch(4)", 1, "seq(3)", "seq(3)"), ==> shape(result) = (4, 2, 3, 3)
  {
    TensorInfo A_info{{4, 2, 3, 3}, true, transformer, DataTypeImpl::GetTensorType<float>(), {"batch", "2", "seq", "seq"}};
    TensorInfo B_info{{4, 1, 1, 3}, true, transformer, DataTypeImpl::GetTensorType<float>(), {"batch", "1", "1", "seq"}};
    TensorInfo Y_info{{4, 2, 3, 3}};

    gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error,
                                          attributes, true, check_not_have_shape_inferencing);
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
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  // 2D x 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}}, {{2, 3}}, &max_error,
                                          attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D x 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4}, {2, 4, 3}}, {{2, 3, 3}}, &max_error,
                                          attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D x 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4}, {4, 3}}, {{2, 3, 3}}, &max_error,
                                          attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D x 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{3, 4}, {2, 4, 3}}, {{2, 3, 3}}, &max_error,
                                          attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 4D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {2, 3, 5, 4}}, {{2, 3, 4, 4}}, &max_error,
                                          attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {5, 4}}, {{2, 3, 4, 4}}, &max_error,
                                          attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {3, 5, 4}}, {{2, 3, 4, 4}}, &max_error,
                                          attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 4D with broadcast
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 4, 5}, {1, 3, 5, 4}}, {{2, 3, 4, 4}}, &max_error,
                                          attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, SinGrad) {
  UnaryOpGradientTest("Sin");
}

TEST(GradientCheckerTest, NegGrad) {
  UnaryOpGradientTest("Neg");
}

TEST(GradientCheckerTest, LogGrad) {
  TensorShape shape({2,5,6});

  std::function<float(float)> transformer = [](float x) { return std::fabs(x) + 1e-1f; };
  TensorInfo x_info{shape, true, &transformer};

  float max_error;
  float error_tolerance = 1e-3f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Log"};

  gradient_checker.ComputeGradientError(op_def, {x_info}, {shape}, &max_error);

  EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
}

TEST(GradientCheckerTest, TanhGrad) {
  UnaryOpGradientTest("Tanh");
}

// TODO fix flaky test
// failing random seed with error_tolerance of 1.5e-2f: 322298223
TEST(GradientCheckerTest, GemmGrad) {
  float max_error;
  const float error_tolerance = 2e-2f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Gemm"};
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  // Single Batch with Scalar Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4}, {4, 3}, {}}, {{1, 3}}, &max_error, attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Single Batch with Vector Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4}, {4, 3}, {3}}, {{1, 3}}, &max_error, attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Non-Single Batch with Scalar Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {}}, {{2, 3}}, &max_error, attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Non-Single Batch with Vector Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {3}}, {{2, 3}}, &max_error, attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Non-Single Batch with Broadcast Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {1, 3}}, {{2, 3}}, &max_error, attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Non-Single Batch with Non-BroadcastBias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {2, 3}}, {{2, 3}}, &max_error, attributes, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // TransA
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 2}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transA", int64_t(1))}, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // TransB
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {3, 4}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transB", int64_t(1))}, true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // TransA and TransB
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 2}, {3, 4}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transA", int64_t(1)),
                                           MakeAttribute("transB", int64_t(1))},
                                          true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // alpha and beta + no_broadcast
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {2, 3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("alpha", 0.7f),
                                           MakeAttribute("beta", 5.0f)},
                                          true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // alpha and beta + broadcast
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("alpha", 0.7f),
                                           MakeAttribute("beta", 5.0f)},
                                          true, true);
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, ReduceMeanGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def{"ReduceMean", kOnnxDomain, 11};

  RunReductionTests(op_def);
}

TEST(GradientCheckerTest, ReduceSumGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def{"ReduceSum", kOnnxDomain, 11};

  RunReductionTests(op_def);
}

TEST(GradientCheckerTest, ReduceLogSumExpGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def{"ReduceLogSumExp", kOnnxDomain, 11};

  RunReductionTests(op_def);
}

TEST(GradientCheckerTest, ReluGrad) {
  UnaryOpGradientTest("Relu");
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

static void TestConcatOpGrad(const std::string& op_type,
                      const std::string& domain = kOnnxDomain,
                      int opset_version = 9,
                      bool check_not_have_shape_inferencing = false) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  const bool extra_input = op_type == "ConcatTraining";
  OpDef op_def{op_type, domain, opset_version};

  //concat_1d
  {
    TensorShape x_shape({2});
    TensorShape y_shape({6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape},
                                                                        output, &max_error,
                                                                        {MakeAttribute("axis", int64_t(0))}, true,
                                                                        check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //concat_2d
  {
    TensorShape x_shape({2, 2});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(1))}, true, 
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //concat_3d
  {
    TensorShape x_shape({1, 2, 3});
    TensorShape y_shape({1, 2, 9});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(2))}, true, 
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //concat_different_shape
  {
    TensorShape x1_shape({2, 2});
    TensorShape x2_shape({2, 4});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x1_shape, x2_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(1))}, true, 
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }

  //concat_different_shape_and_negative_axis
  {
    TensorShape x1_shape({2, 2});
    TensorShape x2_shape({2, 4});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    gradient_checker.ComputeGradientError(op_def, {x1_shape, x2_shape},
                                          output, &max_error,
                                          {MakeAttribute("axis", int64_t(-1))}, true, 
                                          check_not_have_shape_inferencing);
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ConcatGrad) {
  TestConcatOpGrad("Concat");
}

TEST(GradientCheckerTest, ConcatTrainingGrad) { /*also test w/o shape inferencing */
  TestConcatOpGrad("ConcatTraining", kMSDomain, 1, true);
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
    const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};
    gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error,
                                          attributes, true, true /*also test w/o shape inferencing */);
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
// TODO fix flaky test
// failing random seed: 4133818171
TEST(GradientCheckerTest, DISABLED_BatchNormalizationGrad) {
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

TEST(GradientCheckerTest, SigmoidGrad) {
  UnaryOpGradientTest("Sigmoid");
}

void GradientCheckerSoftmaxGradHelper(bool is_log_softmax) {
  TensorShape shape({3, 4, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;

  const std::string op = is_log_softmax? "LogSoftmax" : "Softmax";
  OpDef op_def{op};

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

TEST(GradientCheckerTest, SoftMaxGrad) {
  GradientCheckerSoftmaxGradHelper(false);
}

TEST(GradientCheckerTest, LogSoftMaxGrad) {
  GradientCheckerSoftmaxGradHelper(true);
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

void TestSoftmaxCrossEntropyLossGrad(const TensorShape& index_shape,  //label_shape
                                     const std::string& reduction,
                                     int64_t ignore_index = 0,
                                     int64_t D = 2 /* num_class*/) {
  float max_error;
  bool include_ignore_index = false;
  bool insert_ignore_index = false;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropyLoss", kOnnxDomain, 12};
  std::function<float(float)> transformer_index = [D, &include_ignore_index, &insert_ignore_index, ignore_index](float x) {
    if (include_ignore_index) {
      if (insert_ignore_index) {
        insert_ignore_index = false;
        return static_cast<float>(ignore_index);
      } else {
        insert_ignore_index = true;
        return std::fmod(std::fabs(x) * 5.0f, D * 1.0f);
      }
    } else {
      return std::fmod(std::fabs(x) * 5.0f, D * 1.0f);
    }
  };

  std::function<float(float)> transformer_weight = [](float x) { return std::fmod(std::fabs(x), 2.0f); };

  // without weight and ignore_index
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = true;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info},
                                          {loss_info, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }

  // with weight and no ignore_index
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = false;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo weight_info({logit_shape[1]}, false, &transformer_weight);

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info, weight_info},
                                          {loss_info, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction)});
    EXPECT_IS_TINY(max_error);
  }

  // without weight and ignore index
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = true;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info},
                                          {loss_info, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction), MakeAttribute("ignore_index", ignore_index)});
    EXPECT_IS_TINY(max_error);
  }

  // with weight and ignore_index
  {
    std::vector<int64_t> logit_shape(index_shape.GetDims());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = true;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo weight_info({logit_shape[1]}, false, &transformer_weight);

    gradient_checker.ComputeGradientError(op_def, {x_info, index_info, weight_info},
                                          {loss_info, {logit_shape, false}}, &max_error,
                                          {MakeAttribute("reduction", reduction), MakeAttribute("ignore_index", ignore_index)});
    EXPECT_IS_TINY(max_error);
  }
}

// TODO fix flaky test
// failing random seed: 1
TEST(GradientCheckerTest, DISABLED_SoftmaxCrossEntropyLossGrad) {
  TestSoftmaxCrossEntropyLossGrad({5}, "mean");
  TestSoftmaxCrossEntropyLossGrad({5}, "sum");
  TestSoftmaxCrossEntropyLossGrad({2}, "none");
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "mean");
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "sum");
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "none");
  TestSoftmaxCrossEntropyLossGrad({5}, "mean", -1);
  TestSoftmaxCrossEntropyLossGrad({5}, "sum", -1);
  TestSoftmaxCrossEntropyLossGrad({2}, "none", -1);
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "mean", -1);
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "sum", -1);
  TestSoftmaxCrossEntropyLossGrad({2, 3, 2}, "none", -1);
}

TEST(GradientCheckerTest, GeluGrad) {
  UnaryOpGradientTest("Gelu", kMSDomain, 1);
}

TEST(GradientCheckerTest, FastGeluGrad) {
  UnaryOpGradientTest("FastGelu", kMSDomain, 1);
}

// used for BiasGelu and FastGelu
void TestBiasGeluGrad(const std::string& op_type, const std::string& domain, int opset_version) {
  const TensorShape input_shape({2, 3, 4});
  const TensorShape bias_shape({4});

  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op_type, domain, opset_version};
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  float max_error;
  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
      op_def, {input_shape, bias_shape}, {input_shape}, &max_error,
      attributes, true, true));

  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, FastGeluGrad_Bias) {
  TestBiasGeluGrad("FastGelu", kMSDomain, 1);
}

TEST(GradientCheckerTest, BiasGeluGrad) {
  TestBiasGeluGrad("BiasGelu", kMSDomain, 1);
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
  } else {
    test.AddMissingOptionalInput<float>();
  }

  test.AddInput("training_mode", {}, {true});

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

TEST(GradientCheckerTest, GatherNDGrad_repeat_float_data) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherND", kOnnxDomain, 12};

  TensorInfo x_info({2, 2}, true);
  TensorInfo indice_info({2, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3}, {1, 1, 1, 1}};

  TensorInfo y_info({2}, true);
  int64_t batch_dims = 0;

  gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("batch_dims", batch_dims)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherNDGrad_unique_float_data) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherND", kOnnxDomain, 12};

  {
    TensorInfo x_info({2, 2}, true);
    TensorInfo indice_info({2, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3}, {0, 1, 1, 0}};

    TensorInfo y_info({2}, true);
    int64_t batch_dims = 0;

    gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("batch_dims", batch_dims)});
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 2, 3}, true);
    TensorInfo indice_info({2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 0}};

    TensorInfo y_info({2, 3}, true);
    int64_t batch_dims = 1;

    gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("batch_dims", batch_dims)});
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 2, 3}, true);
    TensorInfo indice_info({2, 2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 0, 2, 1}};

    TensorInfo y_info({2, 2}, true);
    int64_t batch_dims = 2;

    gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas, {MakeAttribute("batch_dims", batch_dims)});
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, GatherElementsGradWithDuplicateUpdate) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherElements", kOnnxDomain, 11};

  TensorInfo data_info({3, 3}, true);
  TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 0, 2, 0, 0}};

  TensorInfo y_info({2, 3}, true);
  int64_t axis = 0;

  gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error, x_datas,
                                        {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherElementsGradWithoutDuplicateUpdate) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherElements", kOnnxDomain, 11};

  TensorInfo data_info({3, 3}, true);
  TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 2, 2, 2}};

  TensorInfo y_info({2, 3}, true);
  int64_t axis = 0;

  gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error, x_datas,
                                        {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, GatherElementsGradAxisWithDuplicateUpdate) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherElements", kOnnxDomain, 11};

  TensorInfo data_info({3, 3}, true);
  TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 1}};

  TensorInfo y_info({2, 3}, true);
  int64_t axis = 1;

  gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error, x_datas,
                                        {MakeAttribute("axis", axis)});
  EXPECT_IS_TINY(max_error);
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
  OpTester test("InPlaceAccumulator", 1, onnxruntime::kMSDomain);

  test.AddInput<float>("old_sum", {3}, {1, 2, 3});
  test.AddInput<float>("value", {3}, {4, 5, 6});

  test.AddOutput<float>("new_sum", {3}, {5, 7, 9});

  test.Run();
}

#ifdef USE_CUDA
TEST(GradientUtilsTest, InPlaceAccumulatorFloat16) {
  OpTester test("InPlaceAccumulator", 1, onnxruntime::kMSDomain);

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
  OpTester test("ZeroGradient", 1, onnxruntime::kMSDomain);

  test.AddInput<float>("old_gradient", {3}, {1, 2, 3});
  test.AddInput<float>("reset_signal", {3}, {1, 10, 100});

  test.AddOutput<float>("zero_gradient", {3}, {0, 0, 0});

  test.Run();
}

#ifdef USE_CUDA
TEST(GradientUtilsTest, ZeroGradientFloat16) {
  OpTester test("ZeroGradient", 1, onnxruntime::kMSDomain);

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

TEST(GradientCheckerTest, WhereGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Where"};

  std::vector<int64_t> shape{4, 3, 2};
  TensorInfo x_info(shape), y_info(shape);
  std::function<float(float)> transformer = [](float x) {
    return static_cast<float>(std::fmod(std::fabs(x), 1.0f) > 0.5f);
  };
  TensorInfo condition_info(shape, false, &transformer, DataTypeImpl::GetTensorType<bool>());

  TensorShape output_shape{shape};
  gradient_checker.ComputeGradientError(op_def, {condition_info, x_info, y_info}, {output_shape}, &max_error);
  EXPECT_IS_TINY(max_error);
}

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
    std::vector<std::vector<float>> x_datas = {{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8}, {1, 0}, {2, 3}, {0, 1}, {1, 2}};

    TensorInfo y_info({1, 2, 2}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, start_info, end_info, axes_info, steps_info}, {y_info},
                                          &max_error, x_datas);

    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ExpandGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Expand"};
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  //input_shape = (2, 3, 1), target_shape = (2, 3, 4) ==> shape(result) = (2, 3, 4)
  {
    TensorInfo x_info({2, 3, 1}, true);
    TensorInfo shape_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {2, 3, 4}};

    TensorInfo y_info({2, 3, 4}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (2, 3, 1), target_shape = (1, 1, 4) ==> shape(result) = (2, 3, 4)
  {
    TensorInfo x_info({2, 3, 1}, true);
    TensorInfo shape_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {1, 1, 4}};

    TensorInfo y_info({2, 3, 4}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (2, 3, 1), target_shape = (4) ==> shape(result) = (2, 3, 4)
  {
    TensorInfo x_info({2, 3, 1}, true);
    TensorInfo shape_info({1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {4}};

    TensorInfo y_info({2, 3, 4}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (2, 3, 1), target_shape = (1, 1) ==> shape(result) = (2, 3, 1)
  {
    TensorInfo x_info({2, 3, 1}, true);
    TensorInfo shape_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {1, 1}};

    TensorInfo y_info({2, 3, 1}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (2, 3), target_shape = (4, 5, 2, 3) ==> shape(result) = (4, 5, 2, 3)
  {
    TensorInfo x_info({2, 3}, true);
    TensorInfo shape_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {4, 5, 2, 3}};

    TensorInfo y_info({4, 5, 2, 3}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }

  //input_shape = (1, 2, 3), target_shape = (4, 5, 1, 1) ==> shape(result) = (4, 5, 2, 3)
  {
    TensorInfo x_info({1, 2, 3}, true);
    TensorInfo shape_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {4, 5, 1, 1}};

    TensorInfo y_info({4, 5, 2, 3}, true);

    gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas, attributes, true, true);
    EXPECT_IS_TINY(max_error);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // NDEBUG

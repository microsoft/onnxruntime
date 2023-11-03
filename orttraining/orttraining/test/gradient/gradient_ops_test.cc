// Copyright (c) Microsoft Corporation. All rights reserved.
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
#include "test/util/include/default_providers.h"
#include "test/common/cuda_op_test_utils.h"

#include "onnx/defs/attr_proto_util.h"

/**
 * The GradientChecker will compute numeric Jacobian and theoretical Jacobian for comparison during the test.
 * The nemeric Jacobian is computed on every single input element by running the forward graph using OpTester twice.
 * The theoretical Jacobian is computed on every single output element by running both forward and backward graph
 * using OpTester. I.E., if there are M elements in inputs and N elements in outputs, it will run forward graph
 * (M * 2 + N) times and backward graph N times using OpTester, which is super time consuming. So please keep the
 * size of inputs and outputs small in the tests.
 */

namespace onnxruntime {
namespace test {

using ONNX_NAMESPACE::MakeAttribute;
using training::OpDef;

static bool IsErrorWithinTolerance(float error, float tolerance) {
  return !std::isnan(error) && !std::isnan(tolerance) && error <= tolerance;
}

#define EXPECT_IS_TINIER_THAN(max_error, tolerance)                 \
  EXPECT_TRUE(IsErrorWithinTolerance(max_error, tolerance))         \
      << "max_error: " << max_error << "; tolerance: " << tolerance \
      << "; ORT test random seed: " << GetTestRandomSeed() << "; "

#define EXPECT_IS_TINY(max_error) EXPECT_IS_TINIER_THAN(max_error, 1.5e-2f)

static void RunReductionTests(const OpDef& op_def, bool axes_as_input = false,
                              bool check_not_have_shape_inferencing = false) {
  std::vector<std::vector<int64_t>> x_shapes = {
      {4, 3, 2},
      {4, 3, 2},
      {4, 3, 2},
      {4, 3, 2},
      {4, 3, 2},
      {4, 3, 2},
      {4, 3, 2},
      {4, 3, 2},
  };
  std::vector<std::vector<int64_t>> y_shapes = {
      {1, 1, 1},
      {},
      {1, 3, 1},
      {2},
      {4, 1, 2},
      {4, 3},
      {4, 1, 2},
      {4},
  };
  std::vector<std::vector<int64_t>> axes_vec = {
      {},  // default case
      {0, 1, 2},
      {0, 2},
      {0, 1},
      {1},
      {2},
      {-2},
      {-2, -1},
  };
  std::vector<int64_t> keepdims_ip = {
      -1,  // default case
      0,
      1,
      0,
      1,
      0,
      1,
      0,
  };

  GradientChecker<float, float, float> gradient_checker;

  float max_error;
  for (size_t i = 0; i < x_shapes.size(); i++) {
    max_error = 0;
    TensorShape x_shape(gsl::make_span(x_shapes[i]));
    TensorShape y_shape(gsl::make_span(y_shapes[i]));
    std::vector<int64_t> axes = axes_vec[i];
    std::vector<std::vector<float>> x_datas;
    RandomValueGenerator random{};
    x_datas.push_back(random.Gaussian<float>(x_shapes[i], 0.f, 5.f));
    std::vector<TensorInfo> input = {x_shape};
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};
    if (keepdims_ip[i] != -1) attributes.push_back(MakeAttribute("keepdims", keepdims_ip[i]));
    if (axes_as_input) {
      std::vector<float> axes_float;
      axes_float.reserve(axes.size());
      std::transform(std::begin(axes), std::end(axes), std::back_inserter(axes_float),
                     [](int64_t i) { return static_cast<float>(i); });
      TensorInfo axes_info({static_cast<int64_t>(axes.size())}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
      input.push_back(axes_info);
      x_datas.push_back(axes_float);
    } else {
      if (axes.size() > 0) attributes.push_back(MakeAttribute("axes", axes));
    }

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, input, {y_shape}, &max_error, x_datas, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }
}

template <typename T>
void GenerateRandomDataWithOneHot(std::vector<std::vector<float>>& x_datas, std::vector<TensorShape> input_shapes,
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
      for (int64_t k = 0; k < N; k++) x_datas[i][k * D + (seed % D)] = (T)1;
    } else {
      std::generate(x_datas[i].begin(), x_datas[i].end(), [&] { return distribution(generator); });
    }
  }
}

void UnaryOpGradientTest(const std::string& op_type, const std::string& domain = kOnnxDomain,
                         const int opset_version = 9,
                         std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
                         std::function<float(float)>* transformer = nullptr,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes = {},
                         float error_tolerance = 1e-3f) {
  TensorShape shape({2, 3, 4});
  TensorInfo x_info{shape, true, transformer};
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op_type, domain, opset_version};

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {shape}, &max_error, attributes, true, false,
                                                         execution_providers));

  EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
}

TEST(GradientCheckerTest, ErfGrad) { UnaryOpGradientTest("Erf"); }

TEST(GradientCheckerTest, SqrtGrad) {
  TensorShape shape({2, 3, 4});

  std::function<float(float)> transformer = [](float x) { return std::fabs(x) + 1; };
  TensorInfo x_info{shape, true, &transformer};

  float max_error;
  float error_tolerance = 1e-3f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Sqrt"};

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {shape}, &max_error));

  EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
}

void RunBroadcastableBinaryOpGradTests(const OpDef& op_def, std::function<float(float)>* transformer,
                                       bool check_not_have_shape_inferencing) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  // shape(A) = (2, 3, 2, 3), shape(B) = (2, 3, 2, 3), ==> shape(result) = (2, 3, 2, 3)
  {
    TensorInfo A_info{{2, 3, 2, 3}, true, transformer};
    TensorInfo B_info{{2, 3, 2, 3}, true, transformer};
    TensorInfo Y_info{{2, 3, 2, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // shape(A) = (2, 3, 2, 3), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 2, 3)
  {
    TensorInfo A_info{{2, 3, 2, 3}, true, transformer};
    TensorInfo B_info{{}, true, transformer};
    TensorInfo Y_info{{2, 3, 2, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // shape(A) = (,), shape(B) = (2, 3, 2, 3), i.e. A is a scalar ==> shape(result) = (2, 3, 2, 3)
  {
    TensorInfo A_info{{}, true, transformer};
    TensorInfo B_info{{2, 3, 2, 3}, true, transformer};
    TensorInfo Y_info{{2, 3, 2, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // shape(A) = (2, 3, 2, 3), shape(B) = (3,), ==> shape(result) = (2, 3, 2, 3)
  {
    TensorInfo A_info{{2, 3, 2, 3}, true, transformer};
    TensorInfo B_info{{3}, true, transformer};
    TensorInfo Y_info{{2, 3, 2, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // shape(A) = (2, 3), shape(B) = (2, 3, 2, 3), ==> shape(result) = (2, 3, 2, 3)
  {
    TensorInfo A_info{{2, 3}, true, transformer};
    TensorInfo B_info{{2, 3, 2, 3}, true, transformer};
    TensorInfo Y_info{{2, 3, 2, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // shape(A) = (1, 2, 3), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3, 2, 3)
  {
    TensorInfo A_info{{1, 2, 3}, true, transformer};
    TensorInfo B_info{{2, 3, 1, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 2, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // shape(A) = (3, 2, 3), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3, 2, 3)
  {
    TensorInfo A_info{{3, 2, 3}, true, transformer};
    TensorInfo B_info{{2, 1, 1, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 2, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // shape(A) = (2, 1, 1, 3), shape(B) = (1, 3, 2, 1), ==> shape(result) = (2, 3, 2, 3)
  {
    TensorInfo A_info{{2, 1, 1, 3}, true, transformer};
    TensorInfo B_info{{1, 3, 2, 1}, true, transformer};
    TensorInfo Y_info{{2, 3, 2, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // symbolic broadcast
  // shape(A) = (3, 2, 1, "seq(3)"), shape(B) = (3, 2, 1, 1), ==> shape(result) = (3, 2, 1, 3)
  {
    TensorInfo A_info{{3, 2, 1, 3}, true, transformer, DataTypeImpl::GetTensorType<float>(), {"3", "2", "1", "seq"}};
    TensorInfo B_info{{3, 2, 1, 1}, true, transformer, DataTypeImpl::GetTensorType<float>(), {"3", "2", "1", "1"}};
    TensorInfo Y_info{{3, 2, 1, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }
  // symbolic broadcast + numeric broadcast
  // shape(A) = ("batch(3)", 2, "seq(3)", "seq(3)"), shape(B) = ("batch(3)", 1, 1, "seq(3)"), ==> shape(result) =
  // (3, 2, 3, 3)
  {
    TensorInfo A_info{
        {3, 2, 3, 3}, true, transformer, DataTypeImpl::GetTensorType<float>(), {"batch", "2", "seq", "seq"}};
    TensorInfo B_info{
        {3, 1, 1, 3}, true, transformer, DataTypeImpl::GetTensorType<float>(), {"batch", "1", "1", "seq"}};
    TensorInfo Y_info{{3, 2, 3, 3}};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {A_info, B_info}, {Y_info}, &max_error, attributes,
                                                           true, check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }
}

void TestBroadcastableBinaryOpGrad(const std::string& op_type, std::function<float(float)>* transformer = nullptr,
                                   bool check_not_have_shape_inferencing = true) {
  OpDef op_def_opset11{op_type, kOnnxDomain, 11};
  RunBroadcastableBinaryOpGradTests(op_def_opset11, transformer, check_not_have_shape_inferencing);
  OpDef op_def_opset13{op_type, kOnnxDomain, 13};
  RunBroadcastableBinaryOpGradTests(op_def_opset13, transformer, check_not_have_shape_inferencing);
}

TEST(GradientCheckerTest, AddGrad) { TestBroadcastableBinaryOpGrad("Add"); }

TEST(GradientCheckerTest, SubGrad) { TestBroadcastableBinaryOpGrad("Sub"); }

// flaky
TEST(GradientCheckerTest, DISABLED_MulGrad) { TestBroadcastableBinaryOpGrad("Mul"); }

TEST(GradientCheckerTest, DivGrad) {
  std::function<float(float)> transformer = [](float x) { return x > 0 ? x + 0.2f : x - 0.2f; };
  TestBroadcastableBinaryOpGrad("Div", &transformer);
}

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
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, exponent_info}, {y_info}, &max_error));
    EXPECT_IS_TINY(max_error);
  }

  // cube
  {
    std::function<float(float)> three = [](float) { return 3.0f; };
    TensorInfo exponent_info{{1}, false, &three};
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, exponent_info}, {y_info}, &max_error));
    EXPECT_IS_TINIER_THAN(max_error, 1e-1f);
  }
}

void RunMatMulGradTests(const OpDef& op_def) {
  float max_error;
  constexpr float error_tolerance = 1e-1f;
  GradientChecker<float, float, float> gradient_checker;
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  // 2D x 2D
  {
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {{2, 2}, {2, 3}}, {{2, 3}}, &max_error, attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D x 3D
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 3, 2}, {2, 2, 3}}, {{2, 3, 3}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D x 2D
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 3, 2}, {2, 3}}, {{2, 3, 3}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D x 3D
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{3, 2}, {2, 2, 3}}, {{2, 3, 3}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 4D
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 3, 3, 2}, {2, 3, 2, 3}}, {{2, 3, 3, 3}},
                                                           &max_error, attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 2D
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 3, 3, 2}, {2, 3}}, {{2, 3, 3, 3}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 3D
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 3, 3, 2}, {3, 2, 3}}, {{2, 3, 3, 3}},
                                                           &max_error, attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 4D x 4D with broadcast
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 1, 3, 2}, {1, 3, 2, 3}}, {{2, 3, 3, 3}},
                                                           &max_error, attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, MatMulGrad) {
  OpDef op_def_opset11{"MatMul", kOnnxDomain, 11};
  RunMatMulGradTests(op_def_opset11);
  OpDef op_def_opset13{"MatMul", kOnnxDomain, 13};
  RunMatMulGradTests(op_def_opset13);
}

TEST(GradientCheckerTest, SinGrad) { UnaryOpGradientTest("Sin"); }

TEST(GradientCheckerTest, CosGrad) { UnaryOpGradientTest("Cos"); }

TEST(GradientCheckerTest, NegGrad) { UnaryOpGradientTest("Neg"); }

TEST(GradientCheckerTest, AbsGrad) {
  // Exclude input data at 0, since Abs is not smooth at 0.
  std::function<float(float)> transformer = [](float x) { return x > 0 ? x + 0.2f : x - 0.2f; };
  UnaryOpGradientTest("Abs", kOnnxDomain, 9, nullptr, &transformer);
}

TEST(GradientCheckerTest, LogGrad) {
  TensorShape shape({2, 3, 4});

  std::function<float(float)> transformer = [](float x) { return std::fabs(x) + 1e-1f; };
  TensorInfo x_info{shape, true, &transformer};

  float max_error;
#ifdef USE_DNNL
  float error_tolerance = 4e-3f;
#else
  float error_tolerance = 1e-3f;
#endif
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Log"};

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {shape}, &max_error));

  EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
}

TEST(GradientCheckerTest, ExpGrad) {
  // Define input data with a narrower distribution than the default GradientChecker, to avoid
  // precision issues.
  TensorShape shape({2, 3, 4});
  std::vector<std::vector<float>> x_datas(1);
  const auto seed = GetTestRandomSeed();
  std::default_random_engine generator{gsl::narrow_cast<decltype(generator)::result_type>(seed)};
  std::uniform_real_distribution<float> distribution{-1.0, 1.0};
  x_datas[0].resize(shape.Size());
  std::generate(x_datas[0].begin(), x_datas[0].end(), [&] { return distribution(generator); });

  float max_error;
  float error_tolerance = 1e-3f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Exp"};

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error, x_datas));

  EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
}

TEST(GradientCheckerTest, FlattenGrad) {
  TensorShape shape({2, 3, 2});
  float max_error;
  float error_tolerance = 1e-3f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Flatten", kOnnxDomain, 11};

  const std::vector<std::pair<int, TensorShape>> axis_to_shape = {
      {-3, {1, 12}}, {-2, {2, 6}}, {-1, {6, 2}}, {0, {1, 12}}, {1, {2, 6}}, {2, {6, 2}}, {3, {12, 1}}};

  for (auto& pair : axis_to_shape) {
    int axis = pair.first;
    const TensorShape& output_shape = pair.second;
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {shape}, {output_shape}, &max_error,
                                                           {MakeAttribute("axis", int64_t(axis))}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, TanhGrad) { UnaryOpGradientTest("Tanh"); }

// TODO fix flaky test
// failing random seed with error_tolerance of 1.5e-2f: 322298223
void RunGemmGradTests(const OpDef& op_def) {
  float max_error;
  constexpr float error_tolerance = 2e-2f;
  GradientChecker<float, float, float> gradient_checker;
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  // Single Batch no third input
  {
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {{1, 3}, {3, 2}}, {{1, 2}}, &max_error, attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Single Batch with Scalar Bias
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{1, 3}, {3, 2}, {}}, {{1, 2}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Single Batch with Vector Bias
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{1, 3}, {3, 2}, {2}}, {{1, 2}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Non-Single Batch with Scalar Bias
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {}}, {{2, 3}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Non-Single Batch with Vector Bias
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Non-Single Batch with Broadcast Bias
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {1, 3}}, {{2, 3}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // Non-Single Batch with Non-BroadcastBias
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {2, 3}}, {{2, 3}}, &max_error,
                                                           attributes, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // TransA
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{4, 2}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                                           {MakeAttribute("transA", int64_t(1))}, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // TransB
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 4}, {3, 4}, {3}}, {{2, 3}}, &max_error,
                                                           {MakeAttribute("transB", int64_t(1))}, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // TransA and TransB
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {{4, 2}, {3, 4}, {3}}, {{2, 3}}, &max_error,
        {MakeAttribute("transA", int64_t(1)), MakeAttribute("transB", int64_t(1))}, true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // alpha and beta + no_broadcast
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {2, 3}}, {{2, 3}}, &max_error,
                                                           {MakeAttribute("alpha", 0.7f), MakeAttribute("beta", 5.0f)},
                                                           true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // alpha and beta + broadcast
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                                           {MakeAttribute("alpha", 0.7f), MakeAttribute("beta", 5.0f)},
                                                           true, true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, GemmGrad) {
  OpDef op_def_opset11{"Gemm", kOnnxDomain, 11};
  RunGemmGradTests(op_def_opset11);
  OpDef op_def_opset13{"Gemm", kOnnxDomain, 13};
  RunGemmGradTests(op_def_opset13);
}

TEST(GradientCheckerTest, ReduceMeanGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def_opset11{"ReduceMean", kOnnxDomain, 11};
  RunReductionTests(op_def_opset11);

  OpDef op_def_opset13{"ReduceMean", kOnnxDomain, 13};
  RunReductionTests(op_def_opset13);
}

TEST(GradientCheckerTest, ReduceSumGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def_11{"ReduceSum", kOnnxDomain, 11};

  RunReductionTests(op_def_11, false, true);

  // axes is input from opset 13.
  OpDef op_def_13{"ReduceSum", kOnnxDomain, 13};

  RunReductionTests(op_def_13, true, true);
}

TEST(GradientCheckerTest, ReduceL2Grad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def{"ReduceL2", kOnnxDomain, 11};

  RunReductionTests(op_def);

  // Y with 0 elements case.
  {
    float max_error;
    GradientChecker<float, float, float> gradient_checker;

    TensorInfo x_info({4, 2}, true);
    std::vector<std::vector<float>> x_datas = {{1, 1, 0, 0, 3, 0, 0, 0}};

    TensorInfo y_info({4, 1}, true);
    std::vector<int64_t> axes{-1};
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {y_info}, &max_error, x_datas,
                                                           {MakeAttribute("axes", axes)}));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ReduceLogSumExpGrad) {
  // Attribute axes supports negative values from opset 11.
  OpDef op_def{"ReduceLogSumExp", kOnnxDomain, 11};

  RunReductionTests(op_def);
}

TEST(GradientCheckerTest, ReluGrad) {
  TensorShape shape({2, 3, 4});
  float max_error;
  float error_tolerance = 1e-3f;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Relu"};

  // Exclude input data at 0, since Relu is not smooth at 0
  std::function<float(float)> transformer = [](float x) { return x > 0 ? x + 0.2f : x - 0.2f; };
  TensorInfo x_info(shape, true, &transformer);
  TensorInfo y_info(shape);

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {y_info}, &max_error));

  EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
}

TEST(GradientCheckerTest, CastGrad) {
  // A dummy test that cast float to float
  // TODO: add more test here
  {
    TensorShape shape({2, 3, 4});
    float max_error;
    float error_tolerance = 1e-3f;
    GradientChecker<float, float, float> gradient_checker;
    OpDef op_def{"Cast"};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {shape}, {shape}, &max_error, {MakeAttribute("to", int64_t(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, SplitGrad) {
  TensorShape shape({9, 5});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Split"};

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {shape}, {{3, 5}, {3, 5}, {3, 5}}, &max_error,
                                                         {MakeAttribute("axis", int64_t(0))}));
  EXPECT_IS_TINY(max_error);

  // opset13 test
  OpDef op_def_13{"Split", kOnnxDomain, 13};
  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def_13, {shape}, {{3, 5}, {3, 5}, {3, 5}}, &max_error,
                                                         {MakeAttribute("axis", int64_t(0))}));
  EXPECT_IS_TINY(max_error);
}

template <typename T>
static std::vector<std::vector<T>> GetRandomValuesForMaxPool(const std::vector<TensorInfo>& infos) {
  std::vector<std::vector<T>> datas(infos.size());
  const uint32_t seed = GetTestRandomSeed();

  std::default_random_engine generator{gsl::narrow_cast<decltype(generator)::result_type>(seed)};
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
    std::shuffle(datas[i].begin(), datas[i].end(), generator);
  }

  return datas;
}

TEST(GradientCheckerTest, MaxPoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"MaxPool"};
  constexpr float error_tolerance = 1e-3f;
  // maxpool_1d_default
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 2, 9}}, {{2, 2, 8}}, &max_error,
                                                           GetRandomValuesForMaxPool<float>({{2, 2, 9}}),
                                                           {MakeAttribute("kernel_shape", std::vector<int64_t>{2})}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // maxpool_2d_default
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 3, 5, 5}}, {{2, 3, 4, 4}}, &max_error,
                                                           GetRandomValuesForMaxPool<float>({{2, 3, 5, 5}}),
                                                           {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                                            MakeAttribute("strides", std::vector<int64_t>{1, 1})}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // maxpool_2d_pads
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{1, 1, 5, 5}}, {{1, 1, 7, 7}}, &max_error,
                                                           GetRandomValuesForMaxPool<float>({{1, 1, 5, 5}}),
                                                           {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                                            MakeAttribute("pads", std::vector<int64_t>{2, 2, 2, 2})}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // maxpool_2d_strides
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{1, 1, 32, 32}}, {{1, 1, 10, 10}}, &max_error,
                                                           GetRandomValuesForMaxPool<float>({{1, 1, 32, 32}}),
                                                           {MakeAttribute("kernel_shape", std::vector<int64_t>{5, 5}),
                                                            MakeAttribute("strides", std::vector<int64_t>{3, 3})}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // maxpool_3d_default
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {{2, 1, 3, 3, 3}}, {{2, 1, 2, 2, 2}}, &max_error, GetRandomValuesForMaxPool<float>({{2, 1, 3, 3, 3}}),
        {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2})}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, GlobalAveragePoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GlobalAveragePool"};
  constexpr float error_tolerance = 1e-3f;

  // globalaveragepool
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 3, 5, 5}}, {{2, 3, 1, 1}}, &max_error, {},
                                                           /*check_not_have_gradient*/ true,
                                                           /*check_not_have_shape_inferencing*/ true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // globalaveragepool_precomputed
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 1, 3, 3}}, {{2, 1, 1, 1}}, &max_error, {},
                                                           /*check_not_have_gradient*/ true,
                                                           /*check_not_have_shape_inferencing*/ true));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

void ConvGradientCheckerTest(std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Conv"};

  // TODO: revisit the tol when ConvGrad impl is completed
  float error_tolerance = 3e-1f;

  // 1D convolution
  {
    TensorShape x_shape({2, 2, 5});
    TensorShape w_shape({2, 2, 3});
    TensorShape b_shape({2});
    TensorShape y_shape({2, 2, 5});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3}), MakeAttribute("pads", std::vector<int64_t>{1, 1})},
        // TODO: ConvGrad does not handle the case where W does not have gradient.
        // Check for not has_gradient need to be disabled to pass this test.
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 1D strided convolution
  {
    TensorShape x_shape({2, 1, 7});
    TensorShape w_shape({1, 1, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 4});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3}), MakeAttribute("pads", std::vector<int64_t>{1, 1}),
         MakeAttribute("strides", std::vector<int64_t>{2})},
        // TODO: ConvGrad does not handle the case where W does not have gradient.
        // Check for not has_gradient need to be disabled to pass this test.
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 1D pointwise convolution (with padding)
  {
    TensorShape x_shape({2, 1, 5});
    TensorShape w_shape({1, 1, 1});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 7});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{1}), MakeAttribute("pads", std::vector<int64_t>{1, 1})},
        // TODO: ConvGrad does not handle the case where W does not have gradient.
        // Check for not has_gradient need to be disabled to pass this test.
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 1D pointwise convolution (no padding)
  {
    TensorShape x_shape({2, 1, 5});
    TensorShape w_shape({1, 1, 1});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{1}), MakeAttribute("pads", std::vector<int64_t>{0, 0})},
        // TODO: ConvGrad does not handle the case where W does not have gradient.
        // Check for not has_gradient need to be disabled to pass this test.
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D convolution
  {
    TensorShape x_shape({1, 1, 3, 3});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({1, 1, 3, 3});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                                              // TODO: ConvGrad does not handle the case where W does not have gradient.
                                              // Check for not has_gradient need to be disabled to pass this test.
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D convolution
  {
    TensorShape x_shape({2, 1, 5, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5, 5});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                                              // TODO: ConvGrad does not handle the case where W does not have gradient.
                                              // Check for not has_gradient need to be disabled to pass this test.
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D pointwise convolution (with padding)
  {
    TensorShape x_shape({1, 1, 1, 1});
    TensorShape w_shape({1, 1, 1, 1});
    TensorShape b_shape({1});
    TensorShape y_shape({1, 1, 3, 3});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                                              // TODO: ConvGrad does not handle the case where W does not have gradient.
                                              // Check for not has_gradient need to be disabled to pass this test.
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D pointwise convolution (no padding)
  {
    TensorShape x_shape({1, 1, 1, 1});
    TensorShape w_shape({1, 1, 1, 1});
    TensorShape b_shape({1});
    TensorShape y_shape({1, 1, 1, 1});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                               MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0})},
                                              // TODO: ConvGrad does not handle the case where W does not have gradient.
                                              // Check for not has_gradient need to be disabled to pass this test.
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D strided convolution
  {
    TensorShape x_shape({2, 1, 7, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 4, 3});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
         MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1}), MakeAttribute("strides", std::vector<int64_t>{2, 2})},
        // TODO: ConvGrad does not handle the case where W does not have gradient.
        // Check for not has_gradient need to be disabled to pass this test.
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D dilated convolution (no padding)
  {
    TensorShape x_shape({2, 1, 5, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 1, 1});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                                               MakeAttribute("dilations", std::vector<int64_t>{2, 2})},
                                              // TODO: ConvGrad does not handle the case where W does not have gradient.
                                              // Check for not has_gradient need to be disabled to pass this test.
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D dilated convolution (with padding)
  {
    TensorShape x_shape({2, 1, 7, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5, 3});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1}),
                                               MakeAttribute("dilations", std::vector<int64_t>{2, 2})},
                                              // TODO: ConvGrad does not handle the case where W does not have gradient.
                                              // Check for not has_gradient need to be disabled to pass this test.
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D convolution
  {
    TensorShape x_shape({2, 1, 5, 5, 5});
    TensorShape w_shape({1, 1, 3, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5, 5, 5});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1})},
                                              // TODO: ConvGrad does not handle the case where W does not have gradient.
                                              // Check for not has_gradient need to be disabled to pass this test.
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D strided convolution
  {
    TensorShape x_shape({2, 1, 7, 5, 5});
    TensorShape w_shape({1, 1, 3, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 4, 3, 3});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1}),
                                               MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})},
                                              // TODO: ConvGrad does not handle the case where W does not have gradient.
                                              // Check for not has_gradient need to be disabled to pass this test.
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, ConvGrad) {
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_DNNL
  // Dnnl EP does not run for ConvGrad unless it is pushed first.
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif

  execution_providers.push_back(DefaultCpuExecutionProvider());

  if (HasCudaEnvironment(700)) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }

  ConvGradientCheckerTest(&execution_providers);
}

static void TestConcatOpGrad(const std::string& op_type, const std::string& domain = kOnnxDomain, int opset_version = 9,
                             bool check_not_have_shape_inferencing = false) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  const bool extra_input = op_type == "ConcatTraining";
  OpDef op_def{op_type, domain, opset_version};

  // concat_1d
  {
    TensorShape x_shape({2});
    TensorShape y_shape({6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape}, output, &max_error,
                                                           {MakeAttribute("axis", int64_t(0))}, true,
                                                           check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // concat_2d
  {
    TensorShape x_shape({2, 2});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape}, output, &max_error,
                                                           {MakeAttribute("axis", int64_t(1))}, true,
                                                           check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // concat_3d
  {
    TensorShape x_shape({1, 2, 3});
    TensorShape y_shape({1, 2, 9});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_shape, x_shape, x_shape}, output, &max_error,
                                                           {MakeAttribute("axis", int64_t(2))}, true,
                                                           check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // concat_different_shape
  {
    TensorShape x1_shape({2, 2});
    TensorShape x2_shape({2, 4});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x1_shape, x2_shape}, output, &max_error,
                                                           {MakeAttribute("axis", int64_t(1))}, true,
                                                           check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }

  // concat_different_shape_and_negative_axis
  {
    TensorShape x1_shape({2, 2});
    TensorShape x2_shape({2, 4});
    TensorShape y_shape({2, 6});
    std::vector<TensorInfo> output = {y_shape};
    if (extra_input) output.push_back(TensorInfo({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>()));
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x1_shape, x2_shape}, output, &max_error,
                                                           {MakeAttribute("axis", int64_t(-1))}, true,
                                                           check_not_have_shape_inferencing));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ConcatGrad) {
  // Concat's gradient uses Split, and Split Op move "split" attribute to input since OpSet13.
  TestConcatOpGrad("Concat");
  TestConcatOpGrad("Concat", kOnnxDomain, 13);
}

TEST(GradientCheckerTest, ConcatTrainingGrad) { /*also test w/o shape inferencing */
  TestConcatOpGrad("ConcatTraining", kMSDomain, 1, true);
}

void AveragepoolGradientCheckerTest(std::vector<std::unique_ptr<IExecutionProvider>>* execution_provider) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"AveragePool"};

  // averagepool - 1D
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {{2, 3, 8}}, {{2, 3, 4}}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{2}), MakeAttribute("strides", std::vector<int64_t>{2})},
        true, false, execution_provider));

    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 2D
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {{2, 3, 8, 8}}, {{2, 3, 7, 7}}, &max_error,
                                                           {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                                            MakeAttribute("strides", std::vector<int64_t>{1, 1})},
                                                           true, false, execution_provider));

    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 3D
  {
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {{2, 3, 8, 8, 8}}, {{2, 3, 4, 4, 4}}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                                               MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})},
                                              true, false, execution_provider));

    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 1D - With padding
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {{1, 3, 8}}, {{1, 3, 3}}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3}), MakeAttribute("strides", std::vector<int64_t>{3}),
         MakeAttribute("pads", std::vector<int64_t>{1, 0})},
        true, false, execution_provider));

    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 2D - With padding - include pads
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {{1, 3, 7, 8}}, {{1, 3, 3, 4}}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 2}),
         MakeAttribute("strides", std::vector<int64_t>{3, 2}), MakeAttribute("pads", std::vector<int64_t>{1, 0, 1, 0}),
         MakeAttribute("count_include_pad", int64_t(1))},
        true, false, execution_provider));

    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 2D - With padding - exclude pads
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {{1, 3, 7, 7}}, {{1, 3, 3, 3}}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
         MakeAttribute("strides", std::vector<int64_t>{3, 3}), MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
        true, false, execution_provider));

    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 3D - With padding
  {
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {{1, 3, 8, 8, 8}}, {{1, 3, 3, 3, 3}}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                               MakeAttribute("strides", std::vector<int64_t>{3, 3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 0, 0, 0})},
                                              true, false, execution_provider));

    EXPECT_IS_TINY(max_error);
  }

  // averagepool - 3D - With padding- exclude pads
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {{1, 4, 7, 7, 7}}, {{1, 4, 3, 3, 3}}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
         MakeAttribute("strides", std::vector<int64_t>{3, 3, 3}),
         MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1}), MakeAttribute("count_include_pad", int64_t(1))},
        true, false, execution_provider));

    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, AveragePoolGrad) {
  AveragepoolGradientCheckerTest(nullptr);

#ifdef USE_DNNL
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultDnnlExecutionProvider());
  AveragepoolGradientCheckerTest(&execution_providers);
#endif  // USE_DNNL
}

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
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, attributes, true,
                                                           true /*also test w/o shape inferencing */));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 012
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({2, 3, 4});
    std::vector<int64_t> perm{0, 1, 2};
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 021
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({2, 4, 3});
    std::vector<int64_t> perm{0, 2, 1};
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 102
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({3, 2, 4});
    std::vector<int64_t> perm{1, 0, 2};
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 120
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({3, 4, 2});
    std::vector<int64_t> perm{1, 2, 0};
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 201
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({4, 2, 3});
    std::vector<int64_t> perm{2, 0, 1};
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // perm 210
  {
    TensorShape x_shape({2, 3, 4});
    TensorShape y_shape({4, 3, 2});
    std::vector<int64_t> perm{2, 1, 0};
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape}, {y_shape}, &max_error, {MakeAttribute("perm", perm)}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

static void RunSqueezeUnsqueezeTests(const OpDef& op_def, std::vector<std::vector<int64_t>> x_shapes,
                                     std::vector<std::vector<int64_t>> y_shapes,
                                     std::vector<std::vector<int64_t>> axes_ip, bool axes_input = false) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  float error_tolerance = 1e-3f;

  for (size_t i = 0; i < x_shapes.size(); i++) {
    TensorShape x_shape(gsl::make_span(x_shapes[i]));
    TensorShape y_shape(gsl::make_span(y_shapes[i]));
    std::vector<int64_t> axes = axes_ip[i];
    std::vector<std::vector<float>> x_datas;
    RandomValueGenerator random{};
    x_datas.push_back(random.Gaussian<float>(x_shapes[i], 0.f, 5.f));
    std::vector<TensorInfo> input = {x_shape};
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

    // Test case w/o axes attribute/input, only valid for Squeeze Op.
    if (op_def.type == "Squeeze") {
      ASSERT_STATUS_OK(
          gradient_checker.ComputeGradientError(op_def, input, {y_shape}, &max_error, x_datas, attributes));
      EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
    }

    // test case w/ axes attribute/input.
    if (axes_input) {
      std::vector<float> axes_float;
      std::transform(begin(axes), end(axes), std::back_inserter(axes_float),
                     [](int64_t i) { return static_cast<float>(i); });
      TensorInfo axes_info({static_cast<int64_t>(axes.size())}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
      input.push_back(axes_info);
      x_datas.push_back(axes_float);
    } else {
      attributes.push_back(MakeAttribute("axes", axes));
    }

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, input, {y_shape}, &max_error, x_datas, attributes));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, SqueezeGrad) {
  /* TODO: enable test with no axis when squeeze kernel is fixed (separate bug filed)
    TensorShape x_shape({1, 2, 1, 3, 1});
    TensorShape y_shape({2, 3});
  */
  std::vector<std::vector<int64_t>> x_shapes = {
      {1, 2, 3, 1}, {1, 1, 2, 3, 2}, {1, 2, 1, 3, 1}, {1, 2, 1, 3, 1},
      // {1, 2, 1, 3, 1},
  };
  std::vector<std::vector<int64_t>> y_shapes = {
      {2, 3}, {2, 3, 2}, {2, 3}, {1, 2, 3, 1},
      // {2, 3},
  };
  std::vector<std::vector<int64_t>> axes_ip = {
      {0, 3}, {0, 1}, {0, 2, 4}, {2},
      // {}
  };

  OpDef op_def{"Squeeze"};
  RunSqueezeUnsqueezeTests(op_def, x_shapes, y_shapes, axes_ip);

  // axes as input from opset 13
  OpDef op_def_2{"Squeeze", kOnnxDomain, 13};
  RunSqueezeUnsqueezeTests(op_def_2, x_shapes, y_shapes, axes_ip, true);
}

TEST(GradientCheckerTest, UnsqueezeGrad) {
  std::vector<std::vector<int64_t>> x_shapes = {
      {2, 3},
      {2, 3},
      {2, 3},
  };
  std::vector<std::vector<int64_t>> y_shapes = {
      {1, 2, 3, 1},
      {1, 1, 2, 3},
      {1, 2, 1, 3, 1},
  };
  std::vector<std::vector<int64_t>> axes_ip = {
      {0, 3},
      {0, 1},
      {0, 2, 4},
  };

  OpDef op_def{"Unsqueeze"};
  RunSqueezeUnsqueezeTests(op_def, x_shapes, y_shapes, axes_ip);

  // axes as input from opset 13
  OpDef op_def_2{"Unsqueeze", kOnnxDomain, 13};
  RunSqueezeUnsqueezeTests(op_def_2, x_shapes, y_shapes, axes_ip, true);
}

// TODO: Reshape missing

TEST(GradientCheckerTest, BatchNormalizationGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"BatchNormInternal", kMSDomain, 1};
  float error_tolerance = 2e-2f;
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, scale_info, bias_info, mean_info, var_info},
        {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
        {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)}));
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, scale_info, bias_info, mean_info, var_info},
        {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
        {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)}));
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, scale_info, bias_info, mean_info, var_info},
        {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
        {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)}));
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, scale_info, bias_info, mean_info, var_info},
        {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
        {MakeAttribute("momentum", momentum)}));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // case for larger multi-dimensional X
  {
    int channel_dim = 5;
    TensorShape in_out_shape({6, channel_dim, 3, 2, 4});
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, scale_info, bias_info, mean_info, var_info},
        {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error,
        {MakeAttribute("epsilon", epsilon), MakeAttribute("momentum", momentum)}));
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, bias_info, mean_info, var_info},
  {y_info, running_mean_info, running_var_info, saved_mean_info, saved_var_info}, &max_error, {MakeAttribute("epsilon",
  epsilon), MakeAttribute("momentum", momentum)})); EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
  */
}

TEST(GradientCheckerTest, SigmoidGrad) { UnaryOpGradientTest("Sigmoid"); }

TEST(GradientCheckerTest, QuickGeluGrad) {
  // Default alpha = 1.702, relax the tolerance due failure on Win for some seed.
  { UnaryOpGradientTest("QuickGelu", kMSDomain, 1, nullptr, nullptr, {}, 5e-2f); }

  // Silu, alpha = 1.0.
  {
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {MakeAttribute("alpha", 1.0f)};
    UnaryOpGradientTest("QuickGelu", kMSDomain, 1, nullptr, nullptr, attributes, 5e-2f);
  }

  // Negative alpha.
  {
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {MakeAttribute("alpha", -1.702f)};
    UnaryOpGradientTest("QuickGelu", kMSDomain, 1, nullptr, nullptr, attributes, 5e-2f);
  }
}

void GradientCheckerSoftmaxGradHelper(bool is_log_softmax, int version = 11) {
  TensorShape shape({2, 3, 4});
  float max_error;
  GradientChecker<float, float, float> gradient_checker;

  const std::string op = is_log_softmax ? "LogSoftmax" : "Softmax";
  OpDef op_def{op, kOnnxDomain, version};

  // default_axis
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error));
    EXPECT_IS_TINY(max_error);
  }

  // axis=0
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error,
                                                           {MakeAttribute("axis", int64_t(0))}));
    EXPECT_IS_TINY(max_error);
  }

  // axis=1
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error,
                                                           {MakeAttribute("axis", int64_t(1))}));
    EXPECT_IS_TINY(max_error);
  }

  // axis=2
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {shape}, {shape}, &max_error,
                                                           {MakeAttribute("axis", int64_t(2))}));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, SoftMaxGrad) {
  GradientCheckerSoftmaxGradHelper(false);
  GradientCheckerSoftmaxGradHelper(false, 13);
}

TEST(GradientCheckerTest, LogSoftMaxGrad) {
  GradientCheckerSoftmaxGradHelper(true);
  GradientCheckerSoftmaxGradHelper(true, 13);
}

void TestSoftmaxCrossEntropyGrad(const TensorShape& input_shape, const std::string& reduction) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropy", kMSDomain, 1};

  std::vector<std::vector<float>> x_datas(2);
  GenerateRandomDataWithOneHot<float>(x_datas, {input_shape, input_shape}, {1});

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {input_shape, {input_shape, false}},
                                                         {{}, {input_shape, false}}, &max_error, x_datas,
                                                         {MakeAttribute("reduction", reduction)}));
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

  constexpr int64_t D = 7;
  std::function<float(float)> transformer_index = [](float x) { return std::fmod(std::fabs(x) * 5.0f, 7.0f); };
  std::function<float(float)> transformer_weight = [](float x) { return std::fmod(std::fabs(x), 2.0f); };

  // without weight
  {
    auto logit_shape(index_shape.AsShapeVector());
    logit_shape.emplace_back(D);

    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, index_info}, {{}, {logit_shape, false}},
                                                           &max_error, {MakeAttribute("reduction", reduction)}));
    EXPECT_IS_TINY(max_error);
  }

  // with weight
  {
    auto logit_shape(index_shape.AsShapeVector());
    logit_shape.emplace_back(D);

    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo weight_info(index_shape, false, &transformer_weight);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, index_info, weight_info},
                                                           {{}, {logit_shape, false}}, &max_error,
                                                           {MakeAttribute("reduction", reduction)}));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, SparseSoftmaxCrossEntropyGrad) {
  TestSparseSoftmaxCrossEntropyGrad({5}, "mean");
  TestSparseSoftmaxCrossEntropyGrad({5}, "sum");
  TestSparseSoftmaxCrossEntropyGrad({2, 3, 2}, "mean");
  TestSparseSoftmaxCrossEntropyGrad({2, 3, 2}, "sum");
}

void TestSoftmaxCrossEntropyLossGrad(const TensorShape& index_shape,  // label_shape
                                     const std::string& reduction, int64_t ignore_index = 0,
                                     int64_t D = 2 /* num_class*/) {
  float max_error;
  bool include_ignore_index = false;
  bool insert_ignore_index = false;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropyLoss", kOnnxDomain, 12};
  std::function<float(float)> transformer_index = [D, &include_ignore_index, &insert_ignore_index,
                                                   ignore_index](float x) {
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
    TensorShapeVector logit_shape(index_shape.AsShapeVector());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = true;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, index_info},
                                                           {loss_info, {logit_shape, false}}, &max_error,
                                                           {MakeAttribute("reduction", reduction)}));
    EXPECT_IS_TINY(max_error);
  }

  // with weight and no ignore_index
  {
    TensorShapeVector logit_shape(index_shape.AsShapeVector());
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, index_info, weight_info},
                                                           {loss_info, {logit_shape, false}}, &max_error,
                                                           {MakeAttribute("reduction", reduction)}));
    EXPECT_IS_TINY(max_error);
  }

  // without weight and ignore index
  {
    TensorShapeVector logit_shape(index_shape.AsShapeVector());
    auto it = logit_shape.begin() + 1;
    logit_shape.insert(it, D);
    TensorInfo loss_info = {};
    if (reduction == "none") {
      loss_info = {TensorInfo(index_shape.GetDims())};
    }

    include_ignore_index = true;
    TensorInfo x_info(logit_shape);
    TensorInfo index_info(index_shape, false, &transformer_index, DataTypeImpl::GetTensorType<int64_t>());

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, index_info}, {loss_info, {logit_shape, false}}, &max_error,
        {MakeAttribute("reduction", reduction), MakeAttribute("ignore_index", ignore_index)}));
    EXPECT_IS_TINY(max_error);
  }

  // with weight and ignore_index
  {
    TensorShapeVector logit_shape(index_shape.AsShapeVector());
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, index_info, weight_info}, {loss_info, {logit_shape, false}}, &max_error,
        {MakeAttribute("reduction", reduction), MakeAttribute("ignore_index", ignore_index)}));
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

TEST(GradientCheckerTest, GeluGrad) { UnaryOpGradientTest("Gelu", kMSDomain, 1); }

TEST(GradientCheckerTest, FastGeluGrad) { UnaryOpGradientTest("FastGelu", kMSDomain, 1); }

// used for BiasGelu and FastGelu
void TestBiasGeluGrad(const std::string& op_type, const std::string& domain, int opset_version) {
  const TensorShape input_shape({2, 3, 4});
  const TensorShape bias_shape({4});

  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op_type, domain, opset_version};
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  float max_error;
  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {input_shape, bias_shape}, {input_shape}, &max_error,
                                                         attributes, true, true));

  EXPECT_IS_TINY(max_error);
}

TEST(GradientCheckerTest, FastGeluGrad_Bias) { TestBiasGeluGrad("FastGelu", kMSDomain, 1); }

TEST(GradientCheckerTest, BiasGeluGrad) { TestBiasGeluGrad("BiasGelu", kMSDomain, 1); }

TEST(GradientCheckerTest, GatherGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Gather"};

  TensorInfo x_info({4, 2, 3, 2});
  std::function<float(float)> transformer = [](float x) { return std::fmod(7 * std::fabs(x), 4.0f); };

  // gather_0 without duplicated indices
  {
    int num_indices = 2;
    TensorInfo indices_info({num_indices}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 0;
    y_shape[axis] = num_indices;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                                           {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }

  // gather_0 with duplicated indices
  {
    int num_indices = 8;
    TensorInfo indices_info({num_indices}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 0;
    y_shape[axis] = num_indices;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                                           {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }

  // gather_1
  {
    int num_indices = 3;
    std::function<float(float)> transformer2 = [](float x) { return std::fmod(7 * std::fabs(x), 2.0f); };
    TensorInfo indices_info({num_indices}, false, &transformer2, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{x_info.shape};
    int64_t axis = 1;
    y_shape[axis] = num_indices;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                                           {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }

  // 2D Indices
  {
    TensorInfo indices_info({2, 3}, false, &transformer, DataTypeImpl::GetTensorType<int64_t>());

    TensorShape y_shape{2, 3, 2, 3, 2};

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indices_info}, {y_shape}, &max_error,
                                                           {MakeAttribute("axis", int64_t(0))}));
    EXPECT_IS_TINY(max_error);
  }

  // negative indices
  {
    TensorInfo x_info_2({4, 2});
    TensorInfo indices_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {-1, 0, -2}};

    TensorShape y_shape{x_info_2.shape};

    int64_t axis = 0;
    y_shape[axis] = 3;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info_2, indices_info}, {y_shape}, &max_error,
                                                           x_datas, {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }
}

void TestDropoutOp(float ratio, TensorShape& x_shape, bool default_ratio = true) {
  OpTester test("Dropout", 12, kOnnxDomain, false);
  if (default_ratio) ratio = 0.5f;
  float input_constant = 3.0f;
  std::vector<float> x_data(x_shape.Size(), input_constant);
  std::vector<float> y_data(x_shape.Size(), 3.0f);

  test.AddInput<float>("x", x_shape.AsShapeVector(), x_data);
  if (!default_ratio) test.AddInput<float>("ratio", {}, {ratio});
  test.AddOutput<float>("y", x_shape.AsShapeVector(), y_data);
  test.AddOutput<bool>(
      "mask", x_shape.AsShapeVector(),
      {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true});
  test.Run();

  // Check output
  auto fwd_output = test.GetFetches();
  for (size_t idx = 0; idx < x_data.size() / 8; ++idx) {
    // convert the binary to bool
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
  if (default_ratio) ratio = 0.5;
  float input_constant = 3;

  std::vector<float> dy_data(x_shape.Size(), input_constant);
  std::vector<float> ratio_data(1, ratio);

  float output_constant = input_constant / (1 - ratio);
  std::vector<float> dx_data({output_constant, output_constant, output_constant, 0, output_constant, 0, output_constant,
                              0, output_constant, 0, output_constant, 0, output_constant, 0, output_constant, 0});

  test.AddInput<float>("dy", x_shape.AsShapeVector(), dy_data);

  test.AddInput<bool>("mask", x_shape.AsShapeVector(),
                      {true, true, true, false,   //
                       true, false, true, false,  //
                       true, false, true, false,  //
                       true, false, true, false});
  if (!default_ratio) {
    test.AddInput<float>("ratio", {1}, ratio_data);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  test.AddInput("training_mode", {}, {true});

  test.AddOutput<float>("dx", x_shape.AsShapeVector(), dx_data);

  test.Run();
}

#ifdef USE_CUDA
TEST(GradientCheckerTest, DISABLED_Dropout) {
  {
    // Ratio 0
    TensorShape x_shape({2, 2, 2, 2});
    TestDropoutOp(0.0f, x_shape, false);
  }
  // Ratio 0.2, 3D
  {
    TensorShape x_shape({4, 2, 2});
    TestDropoutOp(0.2f, x_shape, false);
  }
  // Ratio 0.4, 2D
  {
    TensorShape x_shape({4, 4});
    TestDropoutOp(0.4f, x_shape, false);
  }

  // Default ratio, 1D
  {
    TensorShape x_shape({16});
    TestDropoutOp(0.2f, x_shape, true);
  }
}

TEST(GradientCheckerTest, DISABLED_DropoutGrad) {
  {
    // Ratio 0
    TensorShape x_shape({8, 2});
    TestDropoutGradOp(0.0f, x_shape);
  }

  // Ratio 0.2, 1D
  {
    TensorShape x_shape({16});
    TestDropoutGradOp(0.2f, x_shape, false);
  }

  // Ratio 0.3, 2D
  {
    TensorShape x_shape({8, 2});
    TestDropoutGradOp(0.3f, x_shape, false);
  }

  // Ratio 0.4, 3D
  {
    TensorShape x_shape({2, 4, 2});
    TestDropoutGradOp(0.4f, x_shape, false);
  }

  // default Ratio, 4D
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

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas,
                                                         {MakeAttribute("batch_dims", batch_dims)}));
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas,
                                                           {MakeAttribute("batch_dims", batch_dims)}));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 2, 3}, true);
    TensorInfo indice_info({2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 0}};

    TensorInfo y_info({2, 3}, true);
    int64_t batch_dims = 1;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas,
                                                           {MakeAttribute("batch_dims", batch_dims)}));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 2, 3}, true);
    TensorInfo indice_info({2, 2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 0, 2, 1}};

    TensorInfo y_info({2, 2}, true);
    int64_t batch_dims = 2;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indice_info}, {y_info}, &max_error, x_datas,
                                                           {MakeAttribute("batch_dims", batch_dims)}));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, LayerNormGrad) {
  // Seems the CPU kernel of LayerNorm/LayerNormGrad has some issue so that for some random seed this test will fail.
  // So we pass in the CUDA EP to calculate both the numeric and theoretical Jacobian on CUDA.
  GradientChecker<float, float, float> gradient_checker;
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultCudaExecutionProvider());

  {
    TensorShape shape({2, 3, 4});
    TensorInfo x_info{shape, true};
    TensorInfo scale_info{{4}, true};
    TensorInfo B_info{{4}, true};
    TensorInfo mean_info{{2, 3, 1}, false};
    TensorInfo var_info{{2, 3, 1}, false};

    float max_error;
    float error_tolerance = 3e-2f;

    OpDef op_def{"LayerNormalization"};
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, scale_info, B_info},
                                                           {shape, mean_info, var_info}, &max_error, {}, true, false,
                                                           &execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, SimplifiedLayerNormGrad) {
  GradientChecker<float, float, float> gradient_checker;
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(DefaultCudaExecutionProvider());

  {
    TensorShape shape({2, 3, 8});
    TensorInfo x_info{shape, true};
    TensorInfo scale_info{{8}, true};
    TensorInfo var_info{{2, 3, 1}, false};

    float max_error;
    float error_tolerance = 3e-2f;

    OpDef op_def{"SimplifiedLayerNormalization"};
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, scale_info}, {shape, var_info}, &max_error,
                                                           {}, true, false, &execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}
#endif  // USE_CUDA

TEST(GradientUtilsTest, InPlaceAccumulatorFloat32) {
  OpTester test("InPlaceAccumulator", 1, onnxruntime::kMSDomain);

  test.AddInput<float>("old_sum", {3}, {1.f, 2.f, 3.f});
  test.AddInput<float>("value", {3}, {4.f, 5.f, 6.f});

  test.AddOutput<float>("new_sum", {3}, {5.f, 7.f, 9.f});

  test.Run();
}

void TestInPlaceAccumulatorV2(
    const std::vector<int64_t>& tensor_dim,
    const std::unordered_set<std::string>& excluded_providers,
    std::vector<std::unique_ptr<IExecutionProvider>>& providers,
    bool* need_override) {
  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> buffer_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
  std::vector<float> grad_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);

  size_t num_of_elem = tensor_dim[0];
  for (size_t i = 1; i < tensor_dim.size(); ++i) {
    num_of_elem *= tensor_dim[i];
  }

  bool override = (need_override != nullptr) && *need_override;
  std::vector<float> updated_buffer_data(num_of_elem);
  for (size_t i = 0; i < num_of_elem; ++i) {
    updated_buffer_data[i] = *(buffer_data.data() + i) + (override ? 0 : *(grad_data.data() + i));
  }

  OpTester test("InPlaceAccumulatorV2", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("old_sum", tensor_dim, buffer_data);
  test.AddInput<float>("value", tensor_dim, grad_data);
  if (need_override != nullptr) {
    test.AddInput<bool>("overwrite", {1}, {*need_override});
  }
  test.AddOutput<bool>("updated", {1}, {true});
  test.AddOutput<float>("new_sum", tensor_dim, updated_buffer_data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers, nullptr, &providers);
}

TEST(GradientUtilsTest, InPlaceAccumulatorV2_CPU) {
  std::vector<int64_t> test_dim{3};
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());
  TestInPlaceAccumulatorV2(test_dim, {}, providers, nullptr);
}

TEST(GradientUtilsTest, InPlaceAccumulatorV2Overwrite) {
  OpTester test("InPlaceAccumulatorV2", 1, onnxruntime::kMSDomain);

  test.AddInput<float>("old_sum", {3}, {1.f, 2.f, 3.f});
  test.AddInput<float>("value", {3}, {4.f, 5.f, 6.f});
  test.AddInput<bool>("overwrite", {1}, {true});
  test.AddOutput<bool>("updated", {1}, {true});
  test.AddOutput<float>("new_sum", {3}, {4.f, 5.f, 6.f});

  // This test can run on all EPs back to back because
  // the input buffer (accumulation buffer) is overwritten and
  // not used to compute the output for that run.
  test.Run();
}

#if defined(USE_CUDA)
// TODO: Add rocm kernel defs
TEST(GradientUtilsTest, InPlaceAccumulatorV2_GPU) {
  std::vector<std::vector<int64_t>> test_dims{
      {768},
      {32},
      {3072},
      {514, 768},
      {768, 768},
      {1024, 768},
      {2048, 768},
      {3072, 768},
      {4096, 768},
      {8192, 768},
      {16384, 768},
      {32768, 768},
      {65536, 768},
      {131072, 768},
      {250002, 768},
      {500004, 768},
  };

  for (const auto& test_dim : test_dims) {
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.emplace_back(DefaultCudaExecutionProvider());
    TestInPlaceAccumulatorV2(test_dim, {}, providers, nullptr);
  }
}

TEST(GradientUtilsTest, InPlaceAccumulatorV2_Float16) {
  OpTester test("InPlaceAccumulatorV2", 1, onnxruntime::kMSDomain);

  std::vector<float> old_sum = {1.0f, 2.0f, 3.0f};
  std::vector<float> value = {4.0f, 5.0f, 6.0f};
  std::vector<float> new_sum = {4.0f, 5.0f, 6.0f};

  std::vector<MLFloat16> value_half(3);
  ConvertFloatToMLFloat16(value.data(), value_half.data(), 3);

  test.AddInput<float>("old_sum", {3}, old_sum);
  test.AddInput<MLFloat16>("value", {3}, value_half);
  test.AddInput<bool>("overwrite", {1}, {true});
  test.AddOutput<bool>("updated", {1}, {true});
  test.AddOutput<float>("new_sum", {3}, new_sum);

  // Didn't implement mixed precision InPlaceAccumulatorV2 in CPU
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
#endif

#if defined(USE_CUDA) || defined(USE_ROCM)
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
#endif  // defined(USE_CUDA) || defined(USE_ROCM)

TEST(GradientUtilsTest, ZeroGradientFloat32) {
  OpTester test("ZeroGradient", 1, onnxruntime::kMSDomain);

  test.AddInput<float>("old_gradient", {3}, {1, 2, 3});
  test.AddInput<float>("reset_signal", {3}, {1, 10, 100});

  test.AddOutput<float>("zero_gradient", {3}, {0, 0, 0});

  test.Run();
}

#if defined(USE_CUDA) || defined(USE_ROCM)
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
#endif  // defined(USE_CUDA) || defined(USE_ROCM)

TEST(GradientCheckerTest, WhereGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Where"};

  TensorShapeVector shape{4, 3, 2};
  TensorInfo x_info(shape), y_info(shape);
  std::function<float(float)> transformer = [](float x) {
    return static_cast<float>(std::fmod(std::fabs(x), 1.0f) > 0.5f);
  };
  TensorInfo condition_info(shape, false, &transformer, DataTypeImpl::GetTensorType<bool>());

  TensorShape output_shape{shape};
  ASSERT_STATUS_OK(
      gradient_checker.ComputeGradientError(op_def, {condition_info, x_info, y_info}, {output_shape}, &max_error));
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

    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_info, start_info, end_info}, {y_info}, &max_error, x_datas));
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

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, start_info, end_info, axes_info, steps_info}, {y_info}, &max_error, x_datas));

    EXPECT_IS_TINY(max_error);
  }

  // 3-D tensor
  {
    TensorInfo x_info({2, 4, 2}, true);
    TensorInfo start_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo end_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo axes_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo steps_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {
        {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8}, {1, 0}, {2, 3}, {0, 1}, {1, 2}};

    TensorInfo y_info({1, 2, 2}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_info, start_info, end_info, axes_info, steps_info}, {y_info}, &max_error, x_datas));

    EXPECT_IS_TINY(max_error);
  }
}

void RunExpandGradTests(const OpDef& op_def) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  // input_shape = (2, 2, 1), target_shape = (2, 2, 3) ==> shape(result) = (2, 2, 3)
  {
    TensorInfo x_info({2, 2, 1}, true);
    TensorInfo shape_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4}, {2, 2, 3}};

    TensorInfo y_info({2, 2, 3}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas,
                                                           attributes, true, true));
    EXPECT_IS_TINY(max_error);
  }

  // input_shape = (2, 2, 1), target_shape = (1, 1, 3) ==> shape(result) = (2, 2, 3)
  {
    TensorInfo x_info({2, 2, 1}, true);
    TensorInfo shape_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4}, {1, 1, 3}};

    TensorInfo y_info({2, 2, 3}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas,
                                                           attributes, true, true));
    EXPECT_IS_TINY(max_error);
  }

  // input_shape = (2, 2, 1), target_shape = (3) ==> shape(result) = (2, 2, 3)
  {
    TensorInfo x_info({2, 2, 1}, true);
    TensorInfo shape_info({1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4}, {3}};

    TensorInfo y_info({2, 2, 3}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas,
                                                           attributes, true, true));
    EXPECT_IS_TINY(max_error);
  }

  // input_shape = (2, 2, 1), target_shape = (1, 1) ==> shape(result) = (2, 2, 1)
  {
    TensorInfo x_info({2, 2, 1}, true);
    TensorInfo shape_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4}, {1, 1}};

    TensorInfo y_info({2, 2, 1}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas,
                                                           attributes, true, true));
    EXPECT_IS_TINY(max_error);
  }

  // input_shape = (2, 3), target_shape = (3, 2, 2, 3) ==> shape(result) = (3, 2, 2, 3)
  {
    TensorInfo x_info({2, 3}, true);
    TensorInfo shape_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {3, 2, 2, 3}};

    TensorInfo y_info({3, 2, 2, 3}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas,
                                                           attributes, true, true));
    EXPECT_IS_TINY(max_error);
  }

  // input_shape = (1, 2, 3), target_shape = (3, 2, 1, 1) ==> shape(result) = (3, 2, 2, 3)
  {
    TensorInfo x_info({1, 2, 3}, true);
    TensorInfo shape_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {3, 2, 1, 1}};

    TensorInfo y_info({3, 2, 2, 3}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, shape_info}, {y_info}, &max_error, x_datas,
                                                           attributes, true, true));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ExpandGrad) {
  OpDef op_def_opset11{"Expand", kOnnxDomain, 11};
  RunExpandGradTests(op_def_opset11);
  OpDef op_def_opset13{"Expand", kOnnxDomain, 13};
  RunExpandGradTests(op_def_opset13);
}

TEST(GradientCheckerTest, GatherElementsGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"GatherElements", kOnnxDomain, 11};

  {
    // GatherElementsGradWithDuplicateUpdate
    TensorInfo data_info({3, 3}, true);
    TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 0, 2, 0, 0}};

    TensorInfo y_info({2, 3}, true);
    int64_t axis = 0;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error,
                                                           x_datas, {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }

  {
    // GatherElementsGradWithoutDuplicateUpdate
    TensorInfo data_info({3, 3}, true);
    TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 2, 2, 2}};

    TensorInfo y_info({2, 3}, true);
    int64_t axis = 0;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error,
                                                           x_datas, {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }

  {
    // GatherElementsGradAxisWithDuplicateUpdate
    TensorInfo data_info({3, 3}, true);
    TensorInfo indice_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 1}};

    TensorInfo y_info({2, 3}, true);
    int64_t axis = 1;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error,
                                                           x_datas, {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }

  {
    // GatherElementsGradWithAxisInMiddle
    TensorInfo data_info({2, 2, 2}, true);
    TensorInfo indice_info({2, 1, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 1, 1}};

    TensorInfo y_info({2, 1, 2}, true);
    int64_t axis = 1;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error,
                                                           x_datas, {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }

  {
    // GatherElementsGradWithLargerIndiceOnAxis
    TensorInfo data_info({2, 2}, true);
    TensorInfo indice_info({2, 4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4}, {1, 1, 1, 1, 1, 1, 1, 1}};

    TensorInfo y_info({2, 4}, true);
    int64_t axis = 1;

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indice_info}, {y_info}, &max_error,
                                                           x_datas, {MakeAttribute("axis", axis)}));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, TopKGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"TopK", kOnnxDomain, 11};

  {
    TensorInfo x_info({2, 2, 2}, true);
    TensorInfo k_info({1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {1}};
    TensorInfo y1_info({2, 2, 1}, true);
    TensorInfo y2_info({2, 2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, k_info}, {y1_info, y2_info}, &max_error,
                                                           x_datas, {}, true, true));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 2, 2}, true);
    TensorInfo k_info({1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {1}};
    TensorInfo y1_info({2, 1, 2}, true);
    TensorInfo y2_info({2, 1, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, k_info}, {y1_info, y2_info}, &max_error,
                                                           x_datas, {MakeAttribute("axis", int64_t(-2))}, true, true));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({3, 3}, true);
    TensorInfo k_info({1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {2}};
    TensorInfo y1_info({3, 2}, true);
    TensorInfo y2_info({3, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, k_info}, {y1_info, y2_info}, &max_error,
                                                           x_datas, {}, true, true));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ClipGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Clip", kOnnxDomain, 12};

  {
    TensorInfo x_info({2, 2, 2}, true);
    TensorInfo min_info({}, false);
    TensorInfo max_info({}, false);
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {2.8f}, {7.2f}};
    TensorInfo y_info({2, 2, 2}, true);
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_info, min_info, max_info}, {y_info}, &max_error, x_datas));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 2, 2}, true);
    TensorInfo min_info({}, false);
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {3.8f}};
    TensorInfo y_info({2, 2, 2}, true);
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, min_info}, {y_info}, &max_error, x_datas));
    EXPECT_IS_TINY(max_error);
  }

  // Should have a case with Op(x, null, max), but current ComputeGradientError doesn't support doing this.

  {
    TensorInfo x_info({2, 2, 2}, true);
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}};
    TensorInfo y_info({2, 2, 2}, true);
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {y_info}, &max_error, x_datas));
    EXPECT_IS_TINY(max_error);
  }
}

void GradientCheckerMinMaxGradHelper(const std::string op) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{op, kOnnxDomain, 11};
  // Ensure the gap between tensors is greater than 1e-3f, otherwise the result of NumericJacobian
  // will be incorrect. This also excludes equal inputs case, where Min/Max is not smooth.
  std::function<float(float)> x1_transformer = [](float x) { return (int)(x * 100) / 100.f; };
  std::function<float(float)> x2_transformer = [](float x) { return (int)(x * 100) / 100.f + 0.002f; };
  std::function<float(float)> x3_transformer = [](float x) { return (int)(x * 100) / 100.f + 0.004f; };
  std::function<float(float)> x4_transformer = [](float x) { return (int)(x * 100) / 100.f + 0.006f; };
  TensorInfo x1_info({2, 3}, true, &x1_transformer);
  TensorInfo y_info({2, 3}, true);

  {
    TensorInfo x_info({2, 3}, true);
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {y_info}, &max_error));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x2_info({2, 3}, true, &x2_transformer);
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x1_info, x2_info}, {y_info}, &max_error));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x2_info({3}, true, &x2_transformer);
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x1_info, x2_info}, {y_info}, &max_error));
    EXPECT_IS_TINY(max_error);
  }

  // More than 2 inputs.
  {
    TensorInfo x2_info({2, 3}, true, &x2_transformer);
    TensorInfo x3_info({2, 3}, true, &x3_transformer);
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x1_info, x2_info, x3_info}, {y_info}, &max_error));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x2_info({3}, true, &x2_transformer);
    TensorInfo x3_info({2, 1}, true, &x3_transformer);
    TensorInfo x4_info({2, 3}, true, &x4_transformer);
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x1_info, x2_info, x3_info, x4_info}, {y_info}, &max_error));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, MinGrad) { GradientCheckerMinMaxGradHelper("Min"); }

TEST(GradientCheckerTest, MaxGrad) { GradientCheckerMinMaxGradHelper("Max"); }

TEST(GradientCheckerTest, TileGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Tile", kOnnxDomain, 11};

  // 2D input
  {
    TensorInfo x_info({2, 3}, true);
    TensorInfo repeat_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6}, {2, 2}};

    TensorInfo y_info({4, 6}, true);

    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_info, repeat_info}, {y_info}, &max_error, x_datas));
    EXPECT_IS_TINY(max_error);
  }

  // 1D input
  {
    TensorInfo x_info({2}, true);
    TensorInfo repeat_info({1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2}, {2}};

    TensorInfo y_info({4}, true);

    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_info, repeat_info}, {y_info}, &max_error, x_datas));
    EXPECT_IS_TINY(max_error);
  }

  // 3D input
  {
    TensorInfo x_info({2, 2, 3}, true);
    TensorInfo repeat_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2}};

    TensorInfo y_info({4, 6, 6}, true);

    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_info, repeat_info}, {y_info}, &max_error, x_datas));
    EXPECT_IS_TINY(max_error);
  }

  // 3D input - repeating 1s
  {
    TensorInfo x_info({2, 2, 3}, true);
    TensorInfo repeat_info({3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 1, 1}};

    TensorInfo y_info({2, 2, 3}, true);

    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_info, repeat_info}, {y_info}, &max_error, x_datas));
    EXPECT_IS_TINY(max_error);
  }
}

#ifdef USE_CUDA
TEST(GradientCheckerTest, PadGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Pad", kOnnxDomain, 11};

  {
    TensorInfo x_info({2, 4}, true);
    TensorInfo pads_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {0, 2, 2, 0}};

    TensorInfo y_info({4, 6}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, pads_info}, {y_info}, &max_error, x_datas));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 4}, true);
    TensorInfo pads_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {0, -2, 2, 0}};

    TensorInfo y_info({4, 2}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, pads_info}, {y_info}, &max_error, x_datas,
                                                           {MakeAttribute("mode", "constant")}));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo x_info({2, 4}, true);
    TensorInfo pads_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {0, 2, 2, 0}};

    TensorInfo y_info({4, 6}, true);

    bool has_error = false;
    try {
      ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, pads_info}, {y_info}, &max_error, x_datas,
                                                             {MakeAttribute("mode", "reflect")}));
    } catch (const std::exception& ex) {
      auto ret = std::string(ex.what()).find("Pad gradient currently supports constant mode only.");
      ASSERT_TRUE(ret != std::string::npos);
      has_error = true;
    }

    ASSERT_TRUE(has_error);
  }

  {
    TensorInfo x_info({2, 4}, true);
    TensorInfo pads_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    std::vector<std::vector<float>> x_datas = {{1, 2, 3, 4, 5, 6, 7, 8}, {0, 2, 2, 0}};

    TensorInfo y_info({4, 6}, true);

    bool has_error = false;
    try {
      ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, pads_info}, {y_info}, &max_error, x_datas,
                                                             {MakeAttribute("mode", "edge")}));
    } catch (const std::exception& ex) {
      auto ret = std::string(ex.what()).find("Pad gradient currently supports constant mode only.");
      ASSERT_TRUE(ret != std::string::npos);
      has_error = true;
    }

    ASSERT_TRUE(has_error);
  }
}
#endif  // USE_CUDA

TEST(GradientCheckerTest, ScatterNDGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"ScatterND", kOnnxDomain, 11};

  {
    TensorInfo data_info({8}, true);
    TensorInfo indices_info({4, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({4}, true);
    std::vector<std::vector<float>> input_datas = {{0, 1, 2, 3, 4, 5, 6, 7}, {4, 3, 1, 7}, {8, 9, 10, 11}};

    TensorInfo output_info({8}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo data_info({2, 2}, true);
    TensorInfo indices_info({2, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({2}, true);
    std::vector<std::vector<float>> input_datas = {{0, 1, 2, 3}, {0, 0, 1, 1}, {4, 5}};

    TensorInfo output_info({2, 2}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo data_info({2, 2}, true);
    TensorInfo indices_info({2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({2, 2}, true);
    std::vector<std::vector<float>> input_datas = {{0, 1, 2, 3}, {1, 0}, {4, 5, 6, 7}};

    TensorInfo output_info({2, 2}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo data_info({2, 2, 2}, true);
    TensorInfo indices_info({2, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({2, 2}, true);
    std::vector<std::vector<float>> input_datas = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 1, 0}, {8, 9, 10, 11}};

    TensorInfo output_info({2, 2, 2}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo data_info({2, 2, 2}, true);
    TensorInfo indices_info({2, 1, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({2, 1, 2}, true);
    std::vector<std::vector<float>> input_datas = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 1, 0}, {8, 9, 10, 11}};

    TensorInfo output_info({2, 2, 2}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas));
    EXPECT_IS_TINY(max_error);
  }

  {
    TensorInfo data_info({2, 2, 2}, true);
    TensorInfo indices_info({2, 1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({2, 2, 2}, true);
    std::vector<std::vector<float>> input_datas = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1}, {8, 9, 10, 11, 12, 13, 14, 15}};

    TensorInfo output_info({2, 2, 2}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, ScatterElementsGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"ScatterElements", kOnnxDomain, 13};

  {  // without axis
    TensorInfo data_info({3, 3}, true);
    TensorInfo indices_info({2, 3}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({2, 3}, true);
    std::vector<std::vector<float>> input_datas = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                                                   {1, 0, 2, 0, 2, 1},
                                                   {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}};

    TensorInfo output_info({3, 3}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas));
    EXPECT_IS_TINY(max_error);
  }

  {  // with axis
    TensorInfo data_info({1, 5}, true);
    TensorInfo indices_info({1, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({1, 2}, true);
    std::vector<std::vector<float>> input_datas = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {1, 3}, {1.1f, 2.1f}};

    TensorInfo output_info({1, 5}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas,
                                                           {MakeAttribute("axis", static_cast<int64_t>(1))}));
    EXPECT_IS_TINY(max_error);
  }

  {  // with -ve axis
    TensorInfo data_info({1, 5}, true);
    TensorInfo indices_info({1, 2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
    TensorInfo updates_info({1, 2}, true);
    std::vector<std::vector<float>> input_datas = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {1, 3}, {1.1f, 2.1f}};

    TensorInfo output_info({1, 5}, true);

    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {data_info, indices_info, updates_info},
                                                           {output_info}, &max_error, input_datas,
                                                           {MakeAttribute("axis", static_cast<int64_t>(-1))}));
    EXPECT_IS_TINY(max_error);
  }
}

TEST(GradientCheckerTest, TriluGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Trilu", kMSDomain, 1};
  constexpr int M = 3;
  constexpr int N = 4;
  TensorShape shape = {M, N};
  TensorInfo x_info(shape);
  TensorInfo k_info({1}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  TensorInfo y_info(shape);
  std::vector<float> x_data = {};
  // Initialize input data
  float f = 1.0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++, f++) {
      x_data.push_back(f);
    }
  }

  // Test without optional input and without attribute
  {
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {y_info}, &max_error, {x_data}));
    EXPECT_IS_TINY(max_error);
  }
  {
    // Test without optional input and with attribute upper=1
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {y_info}, &max_error, {x_data},
                                                           {MakeAttribute("upper", int64_t(1))}));
    EXPECT_IS_TINY(max_error);
  }
  {
    // Test without optional input and with attribute upper=0
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info}, {y_info}, &max_error, {x_data},
                                                           {MakeAttribute("upper", int64_t(0))}));
    EXPECT_IS_TINY(max_error);
  }
  for (int64_t k = -M; k <= M; k++) {
    std::vector<float> k_data = {static_cast<float>(k)};

    // Test with optional input and without attribute
    {
      ASSERT_STATUS_OK(
          gradient_checker.ComputeGradientError(op_def, {x_info, k_info}, {y_info}, &max_error, {x_data, k_data}));
      EXPECT_IS_TINY(max_error);
    }
    {
      // Test with optional input and with attribute upper=1
      ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, k_info}, {y_info}, &max_error,
                                                             {x_data, k_data}, {MakeAttribute("upper", int64_t(1))}));
      EXPECT_IS_TINY(max_error);
    }
    {
      // Test with optional input and with attribute upper=0
      ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, k_info}, {y_info}, &max_error,
                                                             {x_data, k_data}, {MakeAttribute("upper", int64_t(0))}));
      EXPECT_IS_TINY(max_error);
    }
  }
}

// TODO (enable once found why it fails on ROCM)
#if defined(USE_CUDA)
TEST(GradientCheckerTest, PadAndUnflattenGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"PadAndUnflatten", kMSDomain, 1};
  TensorInfo shape_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  TensorInfo indices_info({4}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());
  TensorInfo x_info({4, 3});
  std::vector<std::vector<float>> x_datas = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {3, 5, 0, 1}, {5, 2}};

  TensorInfo padded_out_info({5, 2, 3}, true);
  TensorInfo out_shape_info({2}, false, nullptr, DataTypeImpl::GetTensorType<int64_t>());

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.emplace_back(DefaultRocmExecutionProvider());
#endif

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, indices_info, shape_info},
                                                         {padded_out_info, out_shape_info}, &max_error,
                                                         x_datas, {}, true, false, &execution_providers));
  EXPECT_IS_TINY(max_error);
}
#endif

TEST(GradientCheckerTest, ReciprocalGrad) {
  // Avoid division by 0 by using the transformer.
  std::function<float(float)> transformer = [](float x) { return x > 0 ? x + 0.2f : x - 0.2f; };
  UnaryOpGradientTest("Reciprocal", kOnnxDomain, 12, nullptr, &transformer);
}

TEST(GradientCheckerTest, LeakyReluGrad) {
  // Gradient is non continuous at 0, so we need to avoid it.
  std::function<float(float)> transformer = [](float x) { return x > 0 ? x + 0.2f : x - 0.2f; };
  UnaryOpGradientTest("LeakyRelu", kOnnxDomain, 16, nullptr, &transformer);
}

#ifdef USE_CUDA
void ConvTransposeGradientCheckerTest(std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"ConvTranspose"};

  float error_tolerance = 3e-1f;

  // 1D convolution
  {
    TensorShape x_shape({2, 2, 5});
    TensorShape w_shape({2, 2, 3});
    TensorShape b_shape({2});
    TensorShape y_shape({2, 2, 5});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3}), MakeAttribute("pads", std::vector<int64_t>{1, 1})},
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 1D strided convolution
  {
    TensorShape x_shape({2, 1, 7});
    TensorShape w_shape({1, 1, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 13});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3}), MakeAttribute("pads", std::vector<int64_t>{1, 1}),
         MakeAttribute("strides", std::vector<int64_t>{2})},
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 1D pointwise convolution (with padding)
  {
    TensorShape x_shape({2, 1, 5});
    TensorShape w_shape({1, 1, 1});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 3});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{1}), MakeAttribute("pads", std::vector<int64_t>{1, 1})},
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 1D pointwise convolution (no padding)
  {
    TensorShape x_shape({2, 1, 5});
    TensorShape w_shape({1, 1, 1});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{1}), MakeAttribute("pads", std::vector<int64_t>{0, 0})},
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D convolution
  {
    TensorShape x_shape({1, 1, 3, 3});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({1, 1, 3, 3});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D convolution
  {
    TensorShape x_shape({2, 1, 5, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5, 5});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D pointwise convolution (with padding)
  {
    TensorShape x_shape({1, 1, 3, 3});
    TensorShape w_shape({1, 1, 1, 1});
    TensorShape b_shape({1});
    TensorShape y_shape({1, 1, 1, 1});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})},
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D pointwise convolution (no padding)
  {
    TensorShape x_shape({1, 1, 3, 3});
    TensorShape w_shape({1, 1, 1, 1});
    TensorShape b_shape({1});
    TensorShape y_shape({1, 1, 3, 3});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                               MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0})},
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D strided convolution
  {
    TensorShape x_shape({2, 1, 7, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 13, 9});
    ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(
        op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
        {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
         MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1}), MakeAttribute("strides", std::vector<int64_t>{2, 2})},
        false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D dilated convolution (no padding)
  {
    TensorShape x_shape({2, 1, 5, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 9, 9});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                                               MakeAttribute("dilations", std::vector<int64_t>{2, 2})},
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 2D dilated convolution (with padding)
  {
    TensorShape x_shape({2, 1, 7, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 9, 7});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1}),
                                               MakeAttribute("dilations", std::vector<int64_t>{2, 2})},
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D convolution
  {
    TensorShape x_shape({2, 1, 5, 5, 5});
    TensorShape w_shape({1, 1, 3, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5, 5, 5});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1})},
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }

  // 3D strided convolution
  {
    TensorShape x_shape({2, 1, 7, 5, 5});
    TensorShape w_shape({1, 1, 3, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 13, 9, 9});
    ASSERT_STATUS_OK(
        gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                              {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                               MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1, 1, 1}),
                                               MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})},
                                              false, false, execution_providers));
    EXPECT_IS_TINIER_THAN(max_error, error_tolerance);
  }
}

TEST(GradientCheckerTest, ConvTransposeGrad) {
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  ConvTransposeGradientCheckerTest(&execution_providers);
}

// TODO: Enable test for ROCM
TEST(GradientCheckerTest, ResizeGrad) {
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  const std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {
      MakeAttribute("coordinate_transformation_mode", "half_pixel"),
      MakeAttribute("cubic_coeff_a", -0.75f),
      MakeAttribute("exclude_outside", static_cast<int64_t>(0)),
      MakeAttribute("extrapolation_value", 0.0f),
      MakeAttribute("mode", "linear"),
      MakeAttribute("nearest_mode", "floor")};

  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Resize", kOnnxDomain, 18};

  TensorInfo x_info({1, 2, 4, 4}, true);
  TensorInfo roi_info({4}, false, nullptr, DataTypeImpl::GetTensorType<float>());
  TensorInfo scales_info({4}, false, nullptr, DataTypeImpl::GetTensorType<float>());

  TensorInfo y_info({1, 2, 8, 8}, true);

  std::vector<std::vector<float>> x_datas = {{0.2f, 0.4f, 0.6f, 0.8f, 0.2f, 0.4f, 0.6f, 0.8f,
                                              0.2f, 0.4f, 0.6f, 0.8f, 0.2f, 0.4f, 0.6f, 0.8f,
                                              0.2f, 0.4f, 0.6f, 0.8f, 0.2f, 0.4f, 0.6f, 0.8f,
                                              0.2f, 0.4f, 0.6f, 0.8f, 0.2f, 0.4f, 0.6f, 0.8f},
                                             {1.0f, 1.0f, 1.0f, 1.0f},
                                             {1.0f, 1.0f, 2.0f, 2.0f}};

  ASSERT_STATUS_OK(gradient_checker.ComputeGradientError(op_def, {x_info, roi_info, scales_info},
                                                         {y_info}, &max_error, x_datas, attributes, true, false, &execution_providers));
  EXPECT_IS_TINY(max_error);
}

#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime

#endif  // NDEBUG

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/gradient_checker.h"
#include <random>

// TODO: replace this with ONNX version of attr_proto_util.h when ONNX dependency version is updated
// TODO: update attributes type to AttributeProtoWrapper when ONNX version is ready
#include "core/training//attr_proto_util.h"

namespace onnxruntime {
namespace test {

using onnxruntime::training::MakeAttribute;
using training::OpDef;

template <typename T>
void GenerateRandomData(
    std::vector<std::vector<float>>& x_datas,
    std::vector<TensorShape> input_shapes,
    const std::unordered_set<int>& one_hot_input_indices) {
  for (int i = 0; i < 2; i++) {
    // TODO: Consider varying mean and variance
    float scale = 5.f;
    float mean = 0.f;
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
    std::normal_distribution<T> distribution{mean, scale};

    auto x_data_length = input_shapes[i].Size();
    x_datas[i].resize(x_data_length);

    if (one_hot_input_indices.count(i) > 0 && input_shapes[i].NumDimensions() == 2) {  //only 2 dims supported for now
      std::fill(x_datas[i].begin(), x_datas[i].end(), (T)0);
      auto data_count_per_batch = input_shapes[i].GetDims()[1];
      for (int64_t k = 0; k < input_shapes[i].GetDims()[0]; k++)
        x_datas[i][k * data_count_per_batch + (seed % data_count_per_batch)] = (T)1;
    } else {
      std::generate(x_datas[i].begin(), x_datas[i].end(), [&] { return distribution(generator); });
    }
  }
}

#ifndef USE_CUDA
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
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Add"};

  //shape(A) = (2, 3, 4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {2, 3, 4, 5}}, {{2, 3, 4, 5}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }
  //shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {}}, {{2, 3, 4, 5}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  //shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 4, 5}, {5}}, {{2, 3, 4, 5}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  //shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 5}, {2, 3, 4, 5}}, {{2, 3, 4, 5}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  //shape(A) = (1, 4, 5), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4, 5}, {2, 3, 1, 1}}, {{2, 3, 4, 5}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  //shape(A) = (3, 4, 5), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    gradient_checker.ComputeGradientError(op_def, {{3, 4, 5}, {2, 1, 1, 1}}, {{2, 3, 4, 5}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2);
  }

  //shape(A) = (2, 1, 1, 5), shape(B) = (1, 3, 4, 1), ==> shape(result) = (2, 3, 4, 5)
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 1, 5}, {1, 3, 4, 1}}, {{2, 3, 4, 5}}, &max_error);
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
}

TEST(GradientCheckerTest, SubGrad) {
  TensorShape shape({2, 3, 4, 5});
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

TEST(GradientCheckerTest, GemmGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Gemm"};

  // Single Batch with Scalar Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{1, 4}, {4, 3}, {3}}, {{1, 3}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  // Non-Single Batch with Scalar Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {3}}, {{2, 3}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  // Non-Single Batch with Broadcast Bias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {1, 3}}, {{2, 3}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  // Non-Single Batch with Non-BroadcastBias
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {2, 3}}, {{2, 3}}, &max_error);
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  // TransA
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 2}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transA", int64_t(1))});
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  // TransB
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {3, 4}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transB", int64_t(1))});
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  // TransA and TransB
  {
    gradient_checker.ComputeGradientError(op_def, {{4, 2}, {3, 4}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("transA", int64_t(1)),
                                           MakeAttribute("transB", int64_t(1))});
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  // alpha and beta + no_broadcast
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {2, 3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("alpha", 0.7f),
                                           MakeAttribute("beta", 5.0f)});
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }

  // alpha and beta + broadcast
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 4}, {4, 3}, {3}}, {{2, 3}}, &max_error,
                                          {MakeAttribute("alpha", 0.7f),
                                           MakeAttribute("beta", 5.0f)});
    EXPECT_TRUE(max_error <= 1e-2) << "max_error: " << max_error;
  }
}

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
}

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
    gradient_checker.ComputeGradientError(op_def, {{2, 2, 9}}, {{2, 2, 8}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //maxpool_2d_default
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 5, 5}}, {{2, 3, 4, 4}}, &max_error,
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
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 3, 3, 3}}, {{2, 1, 2, 2, 2}}, &max_error,
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
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 5, 5}}, {{2, 3, 1, 1}}, &max_error);
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //globalaveragepool_precomputed
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 1, 3, 3}}, {{2, 1, 1, 1}}, &max_error);
    EXPECT_TRUE(max_error <= error_tolerance);
  }
}

TEST(GradientCheckerTest, ConvGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"Conv"};

  //conv
  {
    TensorShape x_shape({2, 1, 5, 5});
    TensorShape w_shape({1, 1, 3, 3});
    TensorShape b_shape({1});
    TensorShape y_shape({2, 1, 5, 5});
    gradient_checker.ComputeGradientError(op_def, {x_shape, w_shape, b_shape}, {y_shape}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                           MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1})});
    EXPECT_TRUE(max_error <= 1e-1) << "max_error: " << max_error;
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
                                           MakeAttribute("strides", std::vector<int64_t>{2, 2})});
    EXPECT_TRUE(max_error <= 1e-1) << "max_error: " << max_error;
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

TEST(GradientCheckerTest, AveragePoolGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"AveragePool"};
  const float error_tolerance = 1e-1f;

  //averagepool - 1D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 8}}, {{2, 3, 4}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2}),
                                           MakeAttribute("strides", std::vector<int64_t>{2})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //averagepool - 2D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 8, 8}}, {{2, 3, 7, 7}}, &max_error,
                                          {MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                           MakeAttribute("strides", std::vector<int64_t>{1, 1})});
    EXPECT_TRUE(max_error <= error_tolerance);
  }

  //averagepool - 3D
  {
    gradient_checker.ComputeGradientError(op_def, {{2, 3, 8, 8, 8}}, {{2, 3, 4, 4, 4}}, &max_error,
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

// TODO: Reshape missing
#endif

TEST(GradientCheckerTest, SoftmaxCrossEntropyGrad) {
  float max_error;
  GradientChecker<float, float, float> gradient_checker;
  OpDef op_def{"SoftmaxCrossEntropy", kMSDomain};
  const float error_tolerance = 1e-2f;

  {
    TensorShape input_shape({1, 100});

    std::vector<std::vector<float>> x_datas(2);
    GenerateRandomData<float>(x_datas, {input_shape, input_shape}, {1});

    gradient_checker.ComputeGradientError(op_def, {input_shape, {input_shape, false}}, {{1}}, &max_error, x_datas);
    EXPECT_TRUE(max_error <= error_tolerance) << "max_error: " << max_error;
  }
}

#ifdef USE_CUDA
TEST(OptimizerTest, SGDTest) {
  OpTester test("SGDOptimizer", 9, onnxruntime::kOnnxDomain, false);
  test.AddInput<float>("ETA", {}, {0.5f});
  test.AddInput<float>("W", {3}, {1, 2, 3});
  test.AddInput<float>("G", {3}, {4, 5, 6});
  test.AddOutput<float>("W_New", {3}, {-1.f, -0.5f, 0.f});
  test.Run();
}

#endif

}  // namespace test
}  // namespace onnxruntime

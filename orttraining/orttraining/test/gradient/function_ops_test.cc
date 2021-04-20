// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/graph/contrib_ops/contrib_defs.h"
#include "orttraining/core/graph/training_op_defs.h"

#include "test/providers/function_test_util.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace test {

static void RegisterSchemas() {
  static bool registered = false;
  if (!registered) {
    onnxruntime::training::RegisterTrainingOpSchemas();
    registered = true;
  }
}

static void InitSoftmaxGradTestCase(FunctionTestCase& testCase, std::vector<int64_t> shape) {
  testCase.AddInput<float>("dY", shape);
  testCase.AddInput<float>("Y", shape);
  testCase.AddOutput("dX");
}

TEST(SoftmaxGradExpansionTest, DefaultAxis) {
  RegisterSchemas();
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, NegativeAxis) {
  RegisterSchemas();
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.AddAttribute("axis", -1);
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, PositiveAxis) {
  RegisterSchemas();
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.AddAttribute("axis", 1);
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, 3D) {
  RegisterSchemas();
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2, 2});
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, SymbolicShape) {
  RegisterSchemas();
  FunctionTestCase testCase("SoftmaxGrad");
  std::vector<int64_t> shape{3, 2, 2};
  std::vector<std::string> sym_shape{"BatchSize", "SeqSize", "2"};
  int size = 12;
  std::vector<float> value(size);
  for (int64_t i = 0; i < size; i++)
    value[i] = float(i);

  testCase.AddInput("dY", shape, value, sym_shape);
  testCase.AddInput("Y", shape, value, sym_shape);
  testCase.AddOutput("dX");
  testCase.RunTest();
}

// Test (unexpanded) versions for both opset 12 and opset 13 models to ensure
// function-schema does not impact handling of opset 12 models. The current
// expansion requires opset 13, and no expansion should happen in opset 12
// models. Test is required since ORT currently generates function-expansion
// even when op is dispatched to a kernel.

TEST(SoftmaxGradExpansionTest, OpsetTest) {
  RegisterSchemas();
  FunctionTestCase testCase("SoftmaxGrad");
  testCase.opsets[kOnnxDomain] = 12;
  testCase.opsets[kMSDomain] = 1;
  InitSoftmaxGradTestCase(testCase, {3, 2, 2});

  auto model1 = testCase.CreateModel();
  auto results1 = FunctionTestCase::Run(*model1, testCase.input_value_map, testCase.output_names);

  testCase.opsets[kOnnxDomain] = 13;
  testCase.opsets[kMSDomain] = 1;

  auto model2 = testCase.CreateModel();
  auto results2 = FunctionTestCase::Run(*model1, testCase.input_value_map, testCase.output_names);

  FunctionTestCase::AssertEqual(results1, results2);
}

template <typename T>
void DropoutGradWithoutRatio() {
  FunctionTestCase testCase("DropoutGrad");
  std::vector<int64_t> shape{16, 4, 4};
  testCase.AddInput<T>("dY", shape);
  testCase.AddInput<bool>("mask", shape);
  testCase.AddOutput("dX");
  testCase.RunTest();
}

TEST(DropoutGradExpansionTest, WithoutRatio) {
  RegisterSchemas();
  DropoutGradWithoutRatio<float>();
  DropoutGradWithoutRatio<double>();
}

template <typename T>
void DropoutGradWithRatio() {
  FunctionTestCase testCase("DropoutGrad");
  std::vector<int64_t> shape{16, 4, 4};
  testCase.AddInput<T>("dY", shape);
  testCase.AddInput<bool>("mask", shape);
  testCase.AddInput("ratio", {}, {0.5f});
  testCase.AddOutput("dX");
  testCase.RunTest();
}

TEST(DropoutGradExpansionTest, WithRatio) {
  RegisterSchemas();
  DropoutGradWithRatio<float>();
  DropoutGradWithRatio<double>();
}

template <typename T>
void CheckDropoutGradWithoutRatio(bool inline_call) {
  FunctionTestCase testCase("DropoutGrad");
  std::vector<int64_t> shape{16, 4, 4};
  testCase.AddInput<T, false>("dY", shape);
  testCase.AddInput<bool, false>("mask", shape);
  testCase.AddOutput("dX");
  auto model = testCase.CreateModel(inline_call);
  if (!inline_call) {
    auto& node = *model->MainGraph().Nodes().begin();
    auto* fnbody = node.GetFunctionBody(true);
    EXPECT_EQ(fnbody, nullptr);
  }
}

TEST(CheckDropoutGradExpansionTest, WithoutRatio) {
  RegisterSchemas();
  // bfloat16 not yet supported by ONNX op Where
  CheckDropoutGradWithoutRatio<BFloat16>(false);
  CheckDropoutGradWithoutRatio<MLFloat16>(true);
}

template <typename T>
void CheckDropoutGradWithRatio(bool inline_call) {
  FunctionTestCase testCase("DropoutGrad");
  std::vector<int64_t> shape{16, 4, 4};
  testCase.AddInput<T, false>("dY", shape);
  testCase.AddInput<bool, false>("mask", shape);
  testCase.AddInput("ratio", {}, {0.5f});
  testCase.AddOutput("dX");
  testCase.CreateModel(inline_call);
}

TEST(CheckDropoutGradExpansionTest, WithRatio) {
  RegisterSchemas();
  // bfloat16 not yet supported by ONNX op Where
  CheckDropoutGradWithRatio<BFloat16>(false);
  CheckDropoutGradWithRatio<MLFloat16>(true);
}

TEST(GeluGradExpansionTest, 2D) {
  RegisterSchemas();
  FunctionTestCase testCase("GeluGrad");
  std::vector<int64_t> shape{16, 4};
  testCase.AddInput<float>("dY", shape);
  testCase.AddInput<float>("X", shape);
  testCase.AddOutput("dX");
  testCase.RunTest();
}

template <typename T>
void CheckGeluGrad() {
  // Tests only expanded model creation and checking.
  FunctionTestCase testCase("GeluGrad");
  std::vector<int64_t> shape{16, 4};
  testCase.AddInput<T, false>("dY", shape);
  testCase.AddInput<T, false>("X", shape);
  testCase.AddOutput("dX");
  testCase.CreateModel(true);
}

TEST(CheckGeluGradExpansionTest, HalfPrecision) {
  RegisterSchemas();
  CheckGeluGrad<BFloat16>();
  CheckGeluGrad<MLFloat16>();
}

}  // namespace test
}  // namespace onnxruntime
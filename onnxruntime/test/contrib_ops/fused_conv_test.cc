// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) && !defined(DISABLE_CONTRIB_OPS)
using namespace std;

struct ConvOpAndTestAttributes {
  string auto_pad;
  vector<int64_t> dilations;
  int64_t group;
  vector<int64_t> kernel_shape;
  vector<int64_t> pads;
  vector<int64_t> strides;
  string activation;
};

static std::unordered_set<std::string> excluded_providers = {
  kCpuExecutionProvider,
  kDnnlExecutionProvider,
  kOpenVINOExecutionProvider,
  kNupharExecutionProvider,
  kVitisAIExecutionProvider,
  kTensorrtExecutionProvider,
  kNnapiExecutionProvider,
  kRknpuExecutionProvider,
  kDmlExecutionProvider,
  kMIGraphXExecutionProvider,
  kAclExecutionProvider,
  kArmNNExecutionProvider,
  kRocmExecutionProvider};

void TestConvOp(const ConvOpAndTestAttributes& attributes, const vector<vector<float>>& inputs, const vector<vector<int64_t>>& input_shapes, const std::initializer_list<float>& expected_output, const vector<int64_t>& expected_output_shape, bool weight_is_initializer = false, OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess, const std::string& err_str = "") {
  OpTester test("FusedConv", 1, onnxruntime::kMSDomain);
  test.AddAttribute("group", attributes.group);
  test.AddAttribute("kernel_shape", attributes.kernel_shape);

  if (!attributes.dilations.empty()) {
    test.AddAttribute("dilations", attributes.dilations);
  }

  // Only one of pads / auto_pad can be present
  if (!attributes.pads.empty()) {
    test.AddAttribute("pads", attributes.pads);
  } else {
    test.AddAttribute("auto_pad", attributes.auto_pad);
  }

  if (!attributes.strides.empty()) {
    test.AddAttribute("strides", attributes.strides);
  }

  ORT_ENFORCE(!attributes.activation.empty(), "activation must be set");
  test.AddAttribute("activation", attributes.activation);

  const char* szNames[] = {"X", "W", "B", "Z"};
  test.AddInput<float>(szNames[0], input_shapes[0], inputs[0]);
  test.AddInput<float>(szNames[1], input_shapes[1], inputs[1], weight_is_initializer);
  if (inputs.size() >= 3)
    test.AddInput<float>(szNames[2], input_shapes[2], inputs[2]);
  if (inputs.size() >= 4)
    test.AddInput<float>(szNames[3], input_shapes[3], inputs[3]);
  test.AddOutput<float>("Y", expected_output_shape, expected_output);
  test.Run(expect_result, err_str, excluded_providers);
}

TEST(FusedConvTest, Conv2D_Relu) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      "Relu"                        // activation
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 2, 2};
  auto expected_vals = {12.0f, 16.0f, 24.0f, 28.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  TestConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(FusedConvTest, Conv2D_Bias_Relu) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      "Relu"                        // activation
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 2, 2};
  vector<float> B = {1.0f, -1.0f};
  vector<int64_t> B_shape = {2};
  auto expected_vals = {13.0f, 17.0f, 25.0f, 29.0f, 11.0f, 15.0f, 23.0f, 27.0f};
  TestConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(FusedConvTest, Conv2D_Bias_Z_Relu) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      "Relu"                        // activation
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 2, 2};
  vector<float> B = {1.0f, -1.0f};
  vector<int64_t> B_shape = {2};
  vector<float> Z = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  vector<int64_t> Z_shape = {1, 2, 2, 2};
  auto expected_vals = {12.0f, 17.0f, 25.0f, 29.0f, 11.0f, 15.0f, 23.0f, 28.0f};
  TestConvOp(attrs, {X, W, B, Z}, {X_shape, W_shape, B_shape, Z_shape}, expected_vals, Y_shape);
}
#endif

}  // namespace test
}  // namespace onnxruntime

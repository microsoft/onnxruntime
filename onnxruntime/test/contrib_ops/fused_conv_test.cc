// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)
using namespace std;

struct ConvOpAndTestAttributes {
  string auto_pad;
  vector<int64_t> dilations;
  int64_t group;
  vector<int64_t> kernel_shape;
  vector<int64_t> pads;
  vector<int64_t> strides;
  string activation;
  vector<float> activation_parameters = {};
};

void TestConvOp(const ConvOpAndTestAttributes& attributes,
                const vector<vector<float>>& inputs,
                const vector<vector<int64_t>>& input_shapes,
                const std::initializer_list<float>& expected_output,
                const vector<int64_t>& expected_output_shape,
                bool disable_cpu = false,
                bool disable_cuda = false,
                bool disable_rocm = false,
                bool use_float16 = false,
                bool weight_is_initializer = false) {
  bool enable_cuda = HasCudaEnvironment(0) && !use_float16 && !disable_cuda;
  // Only ROCm EP supports float16.
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !disable_rocm;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  if (enable_cuda || enable_rocm || enable_cpu) {
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

    if (!attributes.activation_parameters.empty()) {
      test.AddAttribute("activation_params", attributes.activation_parameters);
    }

    const char* szNames[] = {"X", "W", "B", "Z"};

    if (use_float16) {
      test.AddInput<MLFloat16>(szNames[0], input_shapes[0], ToFloat16(inputs[0]));
      test.AddInput<MLFloat16>(szNames[1], input_shapes[1], ToFloat16(inputs[1]), weight_is_initializer);
      if (inputs.size() >= 3)
        test.AddInput<MLFloat16>(szNames[2], input_shapes[2], ToFloat16(inputs[2]));
      if (inputs.size() >= 4)
        test.AddInput<MLFloat16>(szNames[3], input_shapes[3], ToFloat16(inputs[3]));
      test.AddOutput<MLFloat16>("Y", expected_output_shape, ToFloat16(expected_output));
    } else {
      test.AddInput<float>(szNames[0], input_shapes[0], inputs[0]);
      test.AddInput<float>(szNames[1], input_shapes[1], inputs[1], weight_is_initializer);
      if (inputs.size() >= 3)
        test.AddInput<float>(szNames[2], input_shapes[2], inputs[2]);
      if (inputs.size() >= 4)
        test.AddInput<float>(szNames[3], input_shapes[3], inputs[3]);
      test.AddOutput<float>("Y", expected_output_shape, expected_output);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    if (enable_cuda) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    }

    if (enable_rocm) {
      execution_providers.push_back(DefaultRocmExecutionProvider());
    }

    if (enable_cpu) {
      execution_providers.push_back(DefaultCpuExecutionProvider());
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

void RunConvOp(const ConvOpAndTestAttributes& attributes,
               const vector<vector<float>>& inputs,
               const vector<vector<int64_t>>& input_shapes,
               const std::initializer_list<float>& expected_output,
               const vector<int64_t>& expected_output_shape,
               bool disable_cpu = false,
               bool disable_cuda = false,
               bool disable_rocm = false) {
  bool weight_is_initializer = false;
  bool use_float16 = false;
  TestConvOp(attributes, inputs, input_shapes, expected_output, expected_output_shape,
             disable_cpu, disable_cuda, disable_rocm, use_float16, weight_is_initializer);

  use_float16 = true;
  TestConvOp(attributes, inputs, input_shapes, expected_output, expected_output_shape,
             disable_cpu, disable_cuda, disable_rocm, use_float16, weight_is_initializer);
}

TEST(FusedConvTest, Conv2D_HardSigmoid) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      "HardSigmoid",                // activation
      vector<float>{0.2f, 0.5f}     // activation_parameters
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {0.125f, 0.125f, 0.125f, 0.125f, -0.125f, -0.125f, -0.125f, -0.125f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 2, 2};
  auto expected_vals = {0.8f, 0.9f, 1.0f, 1.0f, 0.2f, 0.1f, 0.0f, 0.0f};
  RunConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, false, true, true);
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
  RunConvOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
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
  RunConvOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

#if defined(USE_CUDA) || defined(USE_ROCM)

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
  RunConvOp(attrs, {X, W, B, Z}, {X_shape, W_shape, B_shape, Z_shape}, expected_vals, Y_shape, true, false, false);
}

#endif

TEST(FusedConvTest, Cpu_Conv2D_Bias_Z_Relu) {
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
  RunConvOp(attrs, {X, W, B, Z}, {X_shape, W_shape, B_shape, Z_shape}, expected_vals, Y_shape, false, true, true);
}

#endif

}  // namespace test
}  // namespace onnxruntime

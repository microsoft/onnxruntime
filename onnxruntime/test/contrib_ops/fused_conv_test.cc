// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <array>

#include "core/common/narrow.h"
#include "core/framework/kernel_registry.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#if defined(USE_KLEIDIAI) && defined(MLAS_TARGET_ARM64)
#include "core/mlas/lib/mlasi.h"
#endif

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
                bool disable_webgpu = false,
                bool use_float16 = false,
                bool weight_is_initializer = false) {
  bool enable_cuda = HasCudaEnvironment(0) && !use_float16 && !disable_cuda;
  bool enable_webgpu = (nullptr != DefaultWebGpuExecutionProvider().get()) && !disable_webgpu;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  if (enable_cuda || enable_cpu || enable_webgpu) {
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

    if (enable_webgpu) {
      execution_providers.push_back(DefaultWebGpuExecutionProvider());
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
               bool disable_webgpu = false) {
  bool weight_is_initializer = false;
  bool use_float16 = false;
  TestConvOp(attributes, inputs, input_shapes, expected_output, expected_output_shape,
             disable_cpu, disable_cuda, disable_webgpu, use_float16, weight_is_initializer);

  use_float16 = true;
  TestConvOp(attributes, inputs, input_shapes, expected_output, expected_output_shape,
             disable_cpu, disable_cuda, disable_webgpu, use_float16, weight_is_initializer);
}

#ifdef USE_KLEIDIAI
namespace {

#if defined(MLAS_TARGET_ARM64)
bool HasFloatNhwcFusedConvKernel() {
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (cpu_ep == nullptr) {
    return false;
  }

  auto kernel_registry = cpu_ep->GetKernelRegistry();
  if (!kernel_registry) {
    return false;
  }

  KernelRegistry::TypeConstraintMap type_constraints{
      {"T", DataTypeImpl::GetTensorType<float>()},
  };

  const KernelCreateInfo* kernel_create_info{};
  const auto status = kernel_registry->TryFindKernel(
      kCpuExecutionProvider,
      "NhwcFusedConv",
      kMSDomain,
      1,
      type_constraints,
      DefaultLoggingManager().DefaultLogger(),
      &kernel_create_info);

  return status.IsOK() && kernel_create_info != nullptr;
}

bool HasFloatNhwcNoTransposeSupport(const vector<int64_t>& input_shape,
                                    const vector<int64_t>& weight_shape,
                                    const vector<int64_t>& pads,
                                    const vector<int64_t>& strides,
                                    int64_t group) {
  if (!HasFloatNhwcFusedConvKernel() || !MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME()) {
    return false;
  }

  if (group <= 0 || input_shape.size() != 4 || weight_shape.size() != 4 ||
      pads.size() != 4 || strides.size() != 2 ||
      weight_shape[0] <= 0 || weight_shape[0] % group != 0) {
    return false;
  }

  std::array<size_t, 2> input_spatial_shape{
      narrow<size_t>(input_shape[1]),
      narrow<size_t>(input_shape[2]),
  };
  std::array<size_t, 2> kernel_spatial_shape{
      narrow<size_t>(weight_shape[2]),
      narrow<size_t>(weight_shape[3]),
  };
  std::array<size_t, 2> dilations{1, 1};
  std::array<size_t, 2> strides_size_t{
      narrow<size_t>(strides[0]),
      narrow<size_t>(strides[1]),
  };
  std::array<size_t, 4> pads_size_t{
      narrow<size_t>(pads[0]),
      narrow<size_t>(pads[1]),
      narrow<size_t>(pads[2]),
      narrow<size_t>(pads[3]),
  };

  return MlasConvSupportsSymmetricChannelsLast2DFloatKernel(
      /*Dimensions*/ 2,
      narrow<size_t>(input_shape[0]),
      narrow<size_t>(group),
      input_spatial_shape.data(),
      kernel_spatial_shape.data(),
      dilations.data(),
      pads_size_t.data(),
      strides_size_t.data(),
      narrow<size_t>(weight_shape[0] / group),
      /*Beta*/ 0.0f);
}
#endif

}  // namespace

void TestNhwcFusedConvFloatOp(const ConvOpAndTestAttributes& attributes,
                              const vector<vector<float>>& inputs,
                              const vector<vector<int64_t>>& input_shapes,
                              const std::initializer_list<float>& expected_output,
                              const vector<int64_t>& expected_output_shape,
                              bool weight_is_initializer = false) {
  auto cpu_ep = DefaultCpuExecutionProvider();
  if (cpu_ep == nullptr) {
    return;
  }

  OpTester test("NhwcFusedConv", 1, onnxruntime::kMSDomain);
  test.AddAttribute("group", attributes.group);
  test.AddAttribute("kernel_shape", attributes.kernel_shape);
  test.AddAttribute("activation", attributes.activation);

  if (!attributes.dilations.empty()) {
    test.AddAttribute("dilations", attributes.dilations);
  }

  if (!attributes.pads.empty()) {
    test.AddAttribute("pads", attributes.pads);
  } else {
    test.AddAttribute("auto_pad", attributes.auto_pad);
  }

  if (!attributes.strides.empty()) {
    test.AddAttribute("strides", attributes.strides);
  }

  if (!attributes.activation_parameters.empty()) {
    test.AddAttribute("activation_params", attributes.activation_parameters);
  }

  const char* szNames[] = {"X", "W", "B", "Z"};
  test.AddInput<float>(szNames[0], input_shapes[0], inputs[0]);
  test.AddInput<float>(szNames[1], input_shapes[1], inputs[1], weight_is_initializer);
  if (inputs.size() >= 3) {
    test.AddInput<float>(szNames[2], input_shapes[2], inputs[2]);
  }
  if (inputs.size() >= 4) {
    test.AddInput<float>(szNames[3], input_shapes[3], inputs[3]);
  }
  test.AddOutput<float>("Y", expected_output_shape, expected_output);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

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

#if defined(USE_CUDA)

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
  RunConvOp(attrs, {X, W, B, Z}, {X_shape, W_shape, B_shape, Z_shape}, expected_vals, Y_shape, true, false);
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

#ifdef USE_KLEIDIAI
TEST(FusedConvTest, Cpu_NhwcConv2D_Bias_Z_Relu) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      "Relu"                        // activation
  };

  vector<float> X = {1.0f, 2.0f, 3.0f,
                     4.0f, 5.0f, 6.0f,
                     7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 3, 3, 1};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 2, 2};
  vector<float> B = {1.0f, -1.0f};
  vector<int64_t> B_shape = {2};
  vector<float> Z = {-1.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 1.0f};
  vector<int64_t> Z_shape = {1, 2, 2, 2};
  auto expected_vals = {12.0f, 11.0f, 17.0f, 15.0f, 25.0f, 23.0f, 29.0f, 28.0f};
  TestNhwcFusedConvFloatOp(attrs, {X, W, B, Z}, {X_shape, W_shape, B_shape, Z_shape}, expected_vals, Y_shape);
  TestNhwcFusedConvFloatOp(attrs, {X, W, B, Z}, {X_shape, W_shape, B_shape, Z_shape}, expected_vals, Y_shape, true);
}

TEST(FusedConvTest, Cpu_NhwcConv2D_Z_Relu_Batch2) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      "Relu"                        // activation
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,
                     1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> X_shape = {2, 2, 2, 1};
  vector<float> W = {1.0f};
  vector<int64_t> W_shape = {1, 1, 1, 1};
  vector<float> B = {0.0f};
  vector<int64_t> B_shape = {1};
  vector<float> Z = {0.0f, 0.0f, 0.0f, 0.0f,
                     -2.0f, -3.0f, -4.0f, -5.0f};
  vector<int64_t> Z_shape = {2, 2, 2, 1};
  vector<int64_t> Y_shape = {2, 2, 2, 1};
  auto expected_vals = {1.0f, 2.0f, 3.0f, 4.0f,
                        0.0f, 0.0f, 0.0f, 0.0f};

  TestNhwcFusedConvFloatOp(attrs, {X, W, B, Z}, {X_shape, W_shape, B_shape, Z_shape}, expected_vals, Y_shape);
  TestNhwcFusedConvFloatOp(attrs, {X, W, B, Z}, {X_shape, W_shape, B_shape, Z_shape}, expected_vals, Y_shape, true);
}

TEST(FusedConvTest, Cpu_NhwcConv2D_AutoPadSameUpper) {
  ConvOpAndTestAttributes attrs = {
      "SAME_UPPER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      {},                     // pads
      vector<int64_t>{1, 1},  // strides
      "Relu"                  // activation
  };

  vector<float> X(25, 1.0f);
  vector<int64_t> X_shape = {1, 5, 5, 1};
  vector<float> W = {0.0f, 1.0f, 2.0f,
                     3.0f, 4.0f, 5.0f,
                     6.0f, 7.0f, 8.0f};
  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<int64_t> Y_shape = {1, 5, 5, 1};
  auto expected_vals = {24.0f, 33.0f, 33.0f, 33.0f, 20.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        12.0f, 15.0f, 15.0f, 15.0f, 8.0f};
  TestNhwcFusedConvFloatOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestNhwcFusedConvFloatOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

TEST(FusedConvTest, Cpu_NhwcDepthwiseConv2D_SymmetricPadding) {
#if !defined(MLAS_TARGET_ARM64)
  GTEST_SKIP() << "Float NHWC depthwise fast-path requires Arm64.";
#else
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      2,                            // group
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{1, 1, 1, 1},  // pads
      vector<int64_t>{1, 1},        // strides
      "Relu"                        // activation
  };

  vector<int64_t> X_shape = {1, 3, 3, 2};
  vector<float> X = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f,
                     4.0f, 40.0f, 5.0f, 50.0f, 6.0f, 60.0f,
                     7.0f, 70.0f, 8.0f, 80.0f, 9.0f, 90.0f};
  vector<int64_t> W_shape = {2, 1, 3, 3};
  vector<float> W = {1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f,
                     1.0f, 0.0f, 1.0f,
                     0.0f, 2.0f, 0.0f,
                     1.0f, 0.0f, 1.0f};
  vector<int64_t> Y_shape = {1, 3, 3, 2};
  auto expected_vals = {12.0f, 70.0f, 21.0f, 140.0f, 16.0f, 110.0f,
                        27.0f, 180.0f, 45.0f, 300.0f, 33.0f, 220.0f,
                        24.0f, 190.0f, 39.0f, 260.0f, 28.0f, 230.0f};

  if (!HasFloatNhwcNoTransposeSupport(X_shape, W_shape, attrs.pads, attrs.strides, attrs.group)) {
    GTEST_SKIP() << "Float NHWC depthwise fast-path is not available on this configuration.";
  }

  TestNhwcFusedConvFloatOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestNhwcFusedConvFloatOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
#endif
}

TEST(FusedConvTest, Cpu_NhwcDepthwiseConv2D_Relu_NegativePreActivation) {
#if !defined(MLAS_TARGET_ARM64)
  GTEST_SKIP() << "Float NHWC depthwise fast-path requires Arm64.";
#else
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      2,                            // group
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{1, 1, 1, 1},  // pads
      vector<int64_t>{1, 1},        // strides
      "Relu"                        // activation
  };

  vector<int64_t> X_shape = {1, 3, 3, 2};
  vector<float> X = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f,
                     4.0f, 40.0f, 5.0f, 50.0f, 6.0f, 60.0f,
                     7.0f, 70.0f, 8.0f, 80.0f, 9.0f, 90.0f};
  vector<int64_t> W_shape = {2, 1, 3, 3};
  vector<float> W = {-1.0f, -1.0f, -1.0f,
                     -1.0f, -1.0f, -1.0f,
                     -1.0f, -1.0f, -1.0f,
                     1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f};
  vector<int64_t> Y_shape = {1, 3, 3, 2};
  auto expected_vals = {0.0f, 120.0f, 0.0f, 210.0f, 0.0f, 160.0f,
                        0.0f, 270.0f, 0.0f, 450.0f, 0.0f, 330.0f,
                        0.0f, 240.0f, 0.0f, 390.0f, 0.0f, 280.0f};

  if (!HasFloatNhwcNoTransposeSupport(X_shape, W_shape, attrs.pads, attrs.strides, attrs.group)) {
    GTEST_SKIP() << "Float NHWC depthwise fast-path is not available on this configuration.";
  }

  TestNhwcFusedConvFloatOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestNhwcFusedConvFloatOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
#endif
}
#endif

TEST(FusedConvTest, Cpu_Conv3D_Batched_Relu) {
  constexpr size_t batch_count = 4;
  constexpr size_t input_channels = 1;
  constexpr size_t input_depth = 8;
  constexpr size_t input_height = 8;
  constexpr size_t input_width = 8;
  constexpr size_t filter_count = 6;
  constexpr size_t kernel_depth = 7;
  constexpr size_t kernel_height = 7;
  constexpr size_t kernel_width = 7;

  OpTester test("FusedConv", 1, onnxruntime::kMSDomain);
  test.AddAttribute("group", static_cast<int64_t>(1));
  test.AddAttribute("kernel_shape", vector<int64_t>{7, 7, 7});
  test.AddAttribute("pads", vector<int64_t>{3, 3, 3, 3, 3, 3});
  test.AddAttribute("strides", vector<int64_t>{1, 1, 1});
  test.AddAttribute("dilations", vector<int64_t>{1, 1, 1});
  test.AddAttribute("activation", string("Relu"));

  const vector<int64_t> X_shape = {static_cast<int64_t>(batch_count),
                                   static_cast<int64_t>(input_channels),
                                   static_cast<int64_t>(input_depth),
                                   static_cast<int64_t>(input_height),
                                   static_cast<int64_t>(input_width)};
  const vector<int64_t> W_shape = {static_cast<int64_t>(filter_count),
                                   static_cast<int64_t>(input_channels),
                                   static_cast<int64_t>(kernel_depth),
                                   static_cast<int64_t>(kernel_height),
                                   static_cast<int64_t>(kernel_width)};
  const vector<int64_t> Y_shape = {static_cast<int64_t>(batch_count),
                                   static_cast<int64_t>(filter_count),
                                   static_cast<int64_t>(input_depth),
                                   static_cast<int64_t>(input_height),
                                   static_cast<int64_t>(input_width)};

  vector<float> X(batch_count * input_channels * input_depth * input_height * input_width, 1.0f);
  vector<float> W(filter_count * input_channels * kernel_depth * kernel_height * kernel_width, 1.0f);

  // With X = 1, W = 1, no bias, and a single input channel, the pre-activation output at
  // [b][f][d][h][w] equals the number of valid kernel positions that fall inside the input
  // volume at (d, h, w). For a kernel of size K with stride 1 and pad = K/2, the per-axis
  // valid count at position p in a dimension of length L is:
  //     count(p, L, K) = min(K - 1, L - 1 - p + K/2) - max(0, K/2 - p) + 1.
  // The post-Relu output is the product of the per-axis counts (all values are positive).
  auto valid_count = [](int64_t pos, int64_t dim, int64_t kernel) -> int64_t {
    const int64_t pad = kernel / 2;
    const int64_t lo = std::max<int64_t>(0, pad - pos);
    const int64_t hi = std::min<int64_t>(kernel - 1, dim - 1 - pos + pad);
    return hi - lo + 1;
  };

  vector<float> Y(batch_count * filter_count * input_depth * input_height * input_width);
  for (size_t b = 0; b < batch_count; ++b) {
    for (size_t f = 0; f < filter_count; ++f) {
      for (size_t d = 0; d < input_depth; ++d) {
        const int64_t cd = valid_count(static_cast<int64_t>(d),
                                       static_cast<int64_t>(input_depth),
                                       static_cast<int64_t>(kernel_depth));
        for (size_t h = 0; h < input_height; ++h) {
          const int64_t ch = valid_count(static_cast<int64_t>(h),
                                         static_cast<int64_t>(input_height),
                                         static_cast<int64_t>(kernel_height));
          for (size_t w = 0; w < input_width; ++w) {
            const int64_t cw = valid_count(static_cast<int64_t>(w),
                                           static_cast<int64_t>(input_width),
                                           static_cast<int64_t>(kernel_width));
            const size_t idx = ((b * filter_count + f) * input_depth + d) * input_height * input_width +
                               h * input_width + w;
            Y[idx] = static_cast<float>(cd * ch * cw);
          }
        }
      }
    }
  }

  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W, true);
  test.AddOutput<float>("Y", Y_shape, Y);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif

}  // namespace test
}  // namespace onnxruntime

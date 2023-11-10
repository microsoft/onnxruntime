// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Creates the graph:
//                                  _______________________
//               input_u8 -> DQ -> |                       |
// scale_u8 (initializer) -> DQ -> |                       |
// bias_u8 (initializer)  -> DQ -> |  BatchNormalization   | -> Q -> output_u8
// mean_u8 (initializer)  -> DQ -> |                       |
// var_u8 (initializer)   -> DQ -> |_______________________|
//
// Currently used to test QNN EP.
template <typename InputQType, typename ScaleQType, typename BiasQType>
GetQDQTestCaseFn BuildQDQBatchNormTestCase(const std::vector<int64_t>& input_shape) {
  return [input_shape](ModelTestBuilder& builder) {
    const int64_t num_channels = input_shape[1];
    const InputQType quant_zero_point = 0;
    const float quant_scale = 1.0f;

    auto* input = builder.MakeInput<InputQType>(input_shape, std::numeric_limits<InputQType>::min(),
                                                std::numeric_limits<InputQType>::max());
    auto* dq_input = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputQType>(input, 0.0039f, quant_zero_point, dq_input);

    auto* dq_scale_output = builder.MakeIntermediate();
    auto* scale = builder.MakeInitializer<ScaleQType>({num_channels}, static_cast<ScaleQType>(1), static_cast<ScaleQType>(127));
    builder.AddDequantizeLinearNode<ScaleQType>(scale, 0.0028f, quant_zero_point, dq_scale_output);

    auto* dq_bias_output = builder.MakeIntermediate();
    auto* bias = builder.MakeInitializer<BiasQType>({num_channels}, std::vector<BiasQType>(num_channels));
    builder.AddDequantizeLinearNode<BiasQType>(bias, quant_scale, quant_zero_point, dq_bias_output);

    auto* dq_mean_output = builder.MakeIntermediate();
    auto* mean = builder.MakeInitializer<InputQType>({num_channels}, std::vector<InputQType>(num_channels));
    builder.AddDequantizeLinearNode<InputQType>(mean, quant_scale, quant_zero_point, dq_mean_output);

    auto* dq_var_output = builder.MakeIntermediate();
    auto* var = builder.MakeInitializer<InputQType>({num_channels}, std::vector<InputQType>(num_channels, 255));
    builder.AddDequantizeLinearNode<InputQType>(var, 0.003921f, 0, dq_var_output);

    auto* batchnorm_output = builder.MakeIntermediate();
    builder.AddNode("BatchNormalization", {dq_input, dq_scale_output, dq_bias_output, dq_mean_output, dq_var_output}, {batchnorm_output});

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InputQType>(batchnorm_output, 0.00377f, quant_zero_point, q_output);

    auto* final_output = builder.MakeOutput();
    builder.AddDequantizeLinearNode<InputQType>(q_output, 0.00377f,
                                                quant_zero_point,
                                                final_output);
  };
}

/**
 * Runs an BatchNormalization model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_shape The input's shape.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
static void RunBatchNormQDQTest(const std::vector<int64_t>& input_shape,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQBatchNormTestCase<uint8_t, uint8_t, uint8_t>(input_shape),
                  provider_options,
                  11,
                  expected_ep_assignment);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQBatchNorm1D) {
  RunBatchNormQDQTest({1, 2, 3}, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, TestQDQBatchNorm2D) {
  RunBatchNormQDQTest({2, 3, 4, 5}, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 5. QNN BatchNormalization doesn't support 5D on HTP
TEST_F(QnnHTPBackendTests, TestQDQBatchNorm3D) {
  RunBatchNormQDQTest({1, 2, 3, 4, 5}, ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
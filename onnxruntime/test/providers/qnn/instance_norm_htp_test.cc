// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Creates the graph:
//                                  _______________________
//               input_u8 -> DQ -> |                       | -> Q -> output_u8
// scale_u8 (initializer) -> DQ -> | InstanceNormalization |
// bias_u8 (initializer)  -> DQ -> |_______________________|
//
// Currently used to test QNN EP.
template <typename QuantType>
GetQDQTestCaseFn BuildQDQInstanceNormTestCase(const TestInputDef<QuantType>& input_def,
                                              const TestInputDef<QuantType>& scale_def,
                                              const TestInputDef<int32_t>& bias_def,
                                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, scale_def, bias_def, attrs](ModelTestBuilder& builder) {
    const QuantType quant_zero_point = 0;
    const float quant_scale = 1.0f;

    auto* dq_scale_output = builder.MakeIntermediate();
    auto* scale = MakeTestInput<QuantType>(builder, scale_def);
    builder.AddDequantizeLinearNode<QuantType>(scale, quant_scale, quant_zero_point, dq_scale_output);

    // Add bias (initializer) -> DQ ->
    auto* dq_bias_output = builder.MakeIntermediate();
    auto* bias = MakeTestInput<int32_t>(builder, bias_def);
    builder.AddDequantizeLinearNode<int32_t>(bias, 1.0f, 0, dq_bias_output);

    // Add input_u8 -> DQ ->
    auto* input_u8 = MakeTestInput<QuantType>(builder, input_def);
    auto* dq_input_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_u8, quant_scale, quant_zero_point, dq_input_output);

    // Add dq_input_output -> InstanceNormalization ->
    auto* instance_norm_output = builder.MakeIntermediate();
    Node& inst_norm_node = builder.AddNode("InstanceNormalization", {dq_input_output, dq_scale_output, dq_bias_output},
                                           {instance_norm_output});
    for (const auto& attr : attrs) {
      inst_norm_node.AddAttributeProto(attr);
    }

    // Add instance_norm_output -> Q -> output_u8
    auto* output_u8 = builder.MakeOutput();
    builder.AddQuantizeLinearNode<QuantType>(instance_norm_output, quant_scale, quant_zero_point, output_u8);
  };
}

/**
 * Runs an InstanceNormalization model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_def The test input's definition (shape, is_initializer, data).
 * \param scale_def The scale input's definition. Correct shapes must be 1D [num_input_channels].
 * \param bias_def The bias input's definition. Correct shapes must be 1D [num_input_channels].
 * \param attrs The node's attributes. The only valid attribute for InstanceNormalization is 'epsilon'.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
template <typename QuantType = uint8_t>
static void RunInstanceNormQDQTest(const TestInputDef<QuantType>& input_def,
                                   const TestInputDef<QuantType>& scale_def,
                                   const TestInputDef<int32_t>& bias_def,
                                   const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                   ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQInstanceNormTestCase<QuantType>(input_def, scale_def, bias_def, attrs),
                  provider_options,
                  18,
                  expected_ep_assignment);
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8) {
  RunInstanceNormQDQTest(TestInputDef<uint8_t>({1, 2, 3, 3}, false, 0, 255),
                         TestInputDef<uint8_t>({2}, true, 0, 127),
                         TestInputDef<int32_t>({2}, true, 0, 10),
                         {},
                         ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8Rank3) {
  RunInstanceNormQDQTest(TestInputDef<uint8_t>({1, 2, 3}, false, {6, 4, 2, 6, 8, 2}),
                         TestInputDef<uint8_t>({2}, true, {1, 2}),
                         TestInputDef<int32_t>({2}, true, {1, 3}),
                         {},
                         ExpectedEPNodeAssignment::All);
}

// TODO: This test now fails in QNN SDK version 2.12.0 (windows arm64 and linux x86_64).
// This worked in QNN SDK version 2.10.0. Need to determine the severity of this inaccuracy.
//
// Exepcted output: 2 6 2 42 42 0
// Actual output: 2 6 2 43 43 0
TEST_F(QnnHTPBackendTests, DISABLED_TestQDQInstanceNormU8Rank3_QnnSdk_2_12_Regression) {
  RunInstanceNormQDQTest(TestInputDef<uint8_t>({1, 2, 3}, false, {3, 4, 3, 9, 9, 8}),
                         TestInputDef<uint8_t>({2}, true, {2, 57}),
                         TestInputDef<int32_t>({2}, true, {3, 2}),
                         {},
                         ExpectedEPNodeAssignment::All);
}

// Check that QNN InstanceNorm operator does not handle inputs with rank > 4.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8Rank5) {
  RunInstanceNormQDQTest(TestInputDef<uint8_t>({1, 2, 3, 3, 3}, false, 0, 255),
                         TestInputDef<uint8_t>({2}, true, 0, 127),
                         TestInputDef<int32_t>({2}, true, 0, 10),
                         {},
                         ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
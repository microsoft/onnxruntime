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

// Function that builds a float32 model with an InstanceNormalization operator.
GetTestModelFn BuildInstanceNormTestCase(const TestInputDef<float>& input_def,
                                         const TestInputDef<float>& scale_def,
                                         const TestInputDef<float>& bias_def,
                                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, scale_def, bias_def, attrs](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput(builder, input_def);
    NodeArg* scale = MakeTestInput(builder, scale_def);
    NodeArg* bias = MakeTestInput(builder, bias_def);

    NodeArg* output = builder.MakeOutput();
    Node& op_node = builder.AddNode("InstanceNormalization", {input, scale, bias}, {output});

    for (const auto& attr : attrs) {
      op_node.AddAttributeProto(attr);
    }
  };
}

// Function that builds a QDQ model with an InstanceNormalization operator.
template <typename QuantType>
static GetTestQDQModelFn<QuantType> BuildQDQInstanceNormTestCase(const TestInputDef<float>& input_def,
                                                                 const TestInputDef<float>& scale_def,
                                                                 const TestInputDef<float>& bias_def,
                                                                 const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, scale_def, bias_def, attrs](ModelTestBuilder& builder,
                                                 std::vector<QuantParams<QuantType>>& output_qparams) {
    // input => Q => DQ =>
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams(input_def);
    NodeArg* input_qdq = AddQDQNodePair(builder, input, input_qparams.scale, input_qparams.zero_point);

    // scale => Q => DQ =>
    NodeArg* scale = MakeTestInput(builder, scale_def);
    QuantParams<QuantType> scale_qparams = GetTestInputQuantParams(scale_def);
    NodeArg* scale_qdq = AddQDQNodePair(builder, scale, scale_qparams.scale, scale_qparams.zero_point);

    // bias (as int32) => DQ =>
    NodeArg* bias_qdq = MakeTestQDQBiasInput(builder, bias_def, input_qparams.scale * scale_qparams.scale);

    // InstanceNormalization operator.
    auto* instance_norm_output = builder.MakeIntermediate();
    Node& inst_norm_node = builder.AddNode("InstanceNormalization", {input_qdq, scale_qdq, bias_qdq},
                                           {instance_norm_output});
    for (const auto& attr : attrs) {
      inst_norm_node.AddAttributeProto(attr);
    }

    // Add instance_norm_output -> Q -> output_u8
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, instance_norm_output, output_qparams[0].scale, output_qparams[0].zero_point);
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
static void RunInstanceNormQDQTest(const TestInputDef<float>& input_def,
                                   const TestInputDef<float>& scale_def,
                                   const TestInputDef<float>& bias_def,
                                   const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                   ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  TestQDQModelAccuracy(BuildInstanceNormTestCase(input_def, scale_def, bias_def, attrs),
                       BuildQDQInstanceNormTestCase<QuantType>(input_def, scale_def, bias_def, attrs),
                       provider_options,
                       18,
                       expected_ep_assignment,
                       1e-5f);
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, InstanceNormU8) {
  RunInstanceNormQDQTest(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),
                         TestInputDef<float>({2}, true, -2.0f, 2.0f),
                         TestInputDef<float>({2}, true, -3.0f, 3.0f),
                         {},
                         ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, InstanceNormU8Rank3) {
  RunInstanceNormQDQTest(TestInputDef<float>({1, 2, 3}, false, {6.0f, 4.0f, 2.0f, 6.0f, 8.0f, 2.0f}),
                         TestInputDef<float>({2}, true, {1.0f, 2.0f}),
                         TestInputDef<float>({2}, true, {1.0f, 3.0f}),
                         {},
                         ExpectedEPNodeAssignment::All);
}

// Check that QNN InstanceNorm operator does not handle inputs with rank > 4.
TEST_F(QnnHTPBackendTests, InstanceNormU8Rank5) {
  RunInstanceNormQDQTest(TestInputDef<float>({1, 2, 3, 3, 3}, false, -10.0f, 10.0f),
                         TestInputDef<float>({2}, true, -2.0f, 2.0f),
                         TestInputDef<float>({2}, true, -3.0f, 3.0f),
                         {},
                         ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
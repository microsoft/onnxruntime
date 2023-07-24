// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <filesystem>
#include <variant>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

using UInt8Limits = std::numeric_limits<uint8_t>;

// Creates the graph:
//                       _______________________
//                      |                       |
//    input_u8 -> DQ -> |       SimpleOp        | -> Q -> output_u8
//                      |_______________________|
//
// Currently used to test QNN EP.
template <typename InputQType>
GetQDQTestCaseFn BuildQDQSingleInputOpTestCase(const TestInputDef<InputQType>& input_def,
                                               const std::string& op_type,
                                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs = {},
                                               const std::string& domain = kOnnxDomain) {
  return [input_def, op_type, attrs, domain](ModelTestBuilder& builder) {
    const InputQType quant_zero_point = 0;
    const float quant_scale = 1.0f;

    auto* input = MakeTestInput<InputQType>(builder, input_def);
    auto* dq_input = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputQType>(input, quant_scale, quant_zero_point, dq_input);

    auto* op_output = builder.MakeIntermediate();
    auto& op_node = builder.AddNode(op_type, {dq_input}, {op_output}, domain);

    for (const auto& attr : attrs) {
      op_node.AddAttributeProto(attr);
    }

    auto* q_output = builder.MakeOutput();
    builder.AddQuantizeLinearNode<InputQType>(op_output, quant_scale, quant_zero_point, q_output);
  };
}

template <typename InputType = float, typename InputQType = uint8_t>
static GetTestModelFn BuildQDQBinaryOpTestCase(const std::string& op_type, const TestInputDef<InputType>& input0_def,
                                               const TestInputDef<InputType>& input1_def) {
  return [op_type, input0_def, input1_def](ModelTestBuilder& builder) {
    const InputQType zero_point = std::numeric_limits<InputQType>::max() / 2;
    constexpr float qdq_scale = 0.0004f;

    NodeArg* input0 = MakeTestInput(builder, input0_def);
    NodeArg* input1 = MakeTestInput(builder, input1_def);
    NodeArg* output = builder.MakeOutput();

    // input -> Q -> DQ -> Op
    auto* qdq0_output = AddQDQNodePair<InputQType>(builder, input0, qdq_scale, zero_point);
    auto* qdq1_output = AddQDQNodePair<InputQType>(builder, input1, qdq_scale, zero_point);

    // Op -> op_output
    auto* op_output = builder.MakeIntermediate();
    builder.AddNode(op_type, {qdq0_output, qdq1_output}, {op_output});

    // op_output -> Q -> DQ -> output
    auto* op_q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InputQType>(op_output, qdq_scale, zero_point, op_q_output);
    builder.AddDequantizeLinearNode<InputQType>(op_q_output, qdq_scale, zero_point, output);
  };
}

template <typename InputType = float, typename InputQType = uint8_t>
static void RunQDQBinaryOpTest(const std::string& op_type, const TestInputDef<InputType>& input0_def,
                               const TestInputDef<InputType>& input1_def,
                               int opset_version,
                               ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQBinaryOpTestCase<InputType, InputQType>(op_type, input0_def, input1_def),
                  provider_options,
                  opset_version,
                  expected_ep_assignment);
}

/**
 * Runs an Simple Op model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_shape The input's shape.
 * \param test_description Description of the test for error reporting.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param num_modes_in_graph The number of expected nodes in the graph.
 */
template <typename InputQType = uint8_t>
static void RunQDQSingleInputOpTest(const TestInputDef<InputQType>& input_def, const std::string& op_type,
                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                    int opset_version,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    const std::string& domain = kOnnxDomain) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> Op -> Q and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQSingleInputOpTestCase<InputQType>(input_def, op_type, attrs, domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment);
}

// Check that QNN compiles DQ -> Gelu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQGeluTest) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Gelu", {}, 11, ExpectedEPNodeAssignment::All, kMSDomain);
}

// Check that QNN compiles DQ -> Elu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQEluTest) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Elu", {}, 11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> HardSwish -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQHardSwishTest) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "HardSwish", {}, 14, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Atan -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQAtanTest) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Atan", {}, 11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Asin -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQAsinTest) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, 0, 1),  // input range 0 ~ 1
                          "Asin", {}, 11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Sign -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQSignTest) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Sign", {}, 11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Sign -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQSinTest) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Sin", {}, 11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that the default axis (-1) for SoftMax opset 13 works.
TEST_F(QnnHTPBackendTests, TestQDQSoftmax13_DefaultAxis) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Softmax",
                          {},  // Uses default axis of -1 for opset 13
                          13, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that an axis != -1 is not supported.
TEST_F(QnnHTPBackendTests, TestQDQSoftmax13_UnsupportedAxis) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Softmax",
                          {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                          13, ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that the default axis (1) for SoftMax opset < 13 does not work.
TEST_F(QnnHTPBackendTests, TestQDQSoftmax11_DefaultAxisFails) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Softmax",
                          {},  // Uses default axis of 1 for opset < 13.
                          11, ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that setting an axis value of -1 works for Softmax opset < 13.
TEST_F(QnnHTPBackendTests, TestQDQSoftmax11_SetValidAxis) {
  RunQDQSingleInputOpTest(TestInputDef<uint8_t>({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max()),
                          "Softmax",
                          {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                          11, ExpectedEPNodeAssignment::All);
}

// Run QDQ model on HTP twice
// 1st run will generate the Qnn context cache binary file
// 2nd run will load and run from Qnn context cache binary file
TEST_F(QnnHTPBackendTests, ContextBinaryCacheTest) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["qnn_context_cache_enable"] = "1";
  const std::string context_binary_file = "./qnn_context_binary_test.bin";
  provider_options["qnn_context_cache_path"] = context_binary_file;

  const TestInputDef<uint8_t> input_def({1, 2, 3}, false, UInt8Limits::min(), UInt8Limits::max());

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  RunQnnModelTest(BuildQDQSingleInputOpTestCase<uint8_t>(input_def, "Atan"),
                  provider_options,
                  11,
                  ExpectedEPNodeAssignment::All);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  // 2nd run will load and run from Qnn context cache binary file
  RunQnnModelTest(BuildQDQSingleInputOpTestCase<uint8_t>(input_def, "Atan"),
                  provider_options,
                  11,
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, TestSub4D_SmallInputs) {
  RunQDQBinaryOpTest<float, uint8_t>("Sub", TestInputDef<float>({1, 3, 8, 8}, false, -1.0f, 1.0f),
                                     TestInputDef<float>({1, 3, 8, 8}, false, -1.0f, 1.0f),
                                     17, ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
// Enable when this is fixed.
TEST_F(QnnHTPBackendTests, DISABLED_TestSub4D_LargeInputs) {
  RunQDQBinaryOpTest<float, uint8_t>("Sub", TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                                     TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                                     17, ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
// Enable when this is fixed.
TEST_F(QnnHTPBackendTests, DISABLED_TestSub4D_Broadcast) {
  RunQDQBinaryOpTest<float, uint8_t>("Sub", TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                                     TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f}),
                                     17, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, TestDiv4D_SmallInputs) {
  RunQDQBinaryOpTest<float, uint8_t>("Div", TestInputDef<float>({1, 3, 8, 8}, false, -1.0f, 1.0f),
                                     TestInputDef<float>({1, 3, 8, 8}, false, -1.0f, 1.0f),
                                     17, ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
// Enable when this is fixed.
TEST_F(QnnHTPBackendTests, DISABLED_TestDiv4D_LargeInputs) {
  RunQDQBinaryOpTest<float, uint8_t>("Div", TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                                     TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                                     17, ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
// Enable when this is fixed.
// Fails accuracy when input0 has dims [1,3,768,768]
TEST_F(QnnHTPBackendTests, DISABLED_TestDiv4D_Broadcast) {
  RunQDQBinaryOpTest<float, uint8_t>("Div", TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                                     TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f}),
                                     17, ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
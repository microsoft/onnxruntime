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
GetQDQTestCaseFn BuildQDQSingleInputOpTestCase(const TestInputDef<float>& input_def,
                                               const std::string& op_type,
                                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                               QuantParams<InputQType> output_qparams,
                                               const std::string& domain = kOnnxDomain) {
  return [input_def, op_type, attrs, output_qparams, domain](ModelTestBuilder& builder) {
    auto* input = MakeTestInput(builder, input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams(input_def);
    auto* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    auto* op_output = builder.MakeIntermediate();
    auto& op_node = builder.AddNode(op_type, {input_qdq}, {op_output}, domain);

    for (const auto& attr : attrs) {
      op_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, op_output, output_qparams.scale, output_qparams.zero_point);
  };
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
static void RunQDQSingleInputOpTest(const TestInputDef<float>& input_def, const std::string& op_type,
                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                    float output_min, float output_max,
                                    int opset_version,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    const std::string& domain = kOnnxDomain) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  QuantParams<InputQType> output_qparams = QuantParams<InputQType>::Compute(output_min, output_max);

  // NOTE: Because of rounding differences, quantized QNN values may differ by 1 quantized unit, which
  // corresponds to a potential float32 difference of (1 * scale).
  //
  // For example, this happens during quantization when the expression (real_val/scale) is near x.5 (e.g., 20.4999).
  // If the HTP backend computes floating-point values at a higher precision, then 20.4999 is rounded to 20. However,
  // ORT uses 32-bit floats, which cannot represent 20.4999 exactly. Instead, ORT uses 20.5 which rounds to 21.
  const float f32_abs_err = 2.0001f * output_qparams.scale;

  // Runs model with DQ-> Op -> Q and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQSingleInputOpTestCase<InputQType>(input_def, op_type, attrs, output_qparams, domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  f32_abs_err);
}

template <typename InputQType = uint8_t>
static GetTestModelFn BuildQDQBinaryOpTestCase(const std::string& op_type, const TestInputDef<float>& input0_def,
                                               const TestInputDef<float>& input1_def,
                                               QuantParams<InputQType> output_qparams) {
  return [op_type, input0_def, input1_def, output_qparams](ModelTestBuilder& builder) {
    NodeArg* input0 = MakeTestInput(builder, input0_def);
    NodeArg* input1 = MakeTestInput(builder, input1_def);

    // input -> Q -> DQ -> Op
    QuantParams<InputQType> input0_qparams = GetTestInputQuantParams(input0_def);
    auto* qdq0_output = AddQDQNodePair<InputQType>(builder, input0, input0_qparams.scale, input0_qparams.zero_point);

    QuantParams<InputQType> input1_qparams = GetTestInputQuantParams(input1_def);
    auto* qdq1_output = AddQDQNodePair<InputQType>(builder, input1, input1_qparams.scale, input1_qparams.zero_point);

    // Op -> op_output
    auto* op_output = builder.MakeIntermediate();
    builder.AddNode(op_type, {qdq0_output, qdq1_output}, {op_output});

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, op_output, output_qparams.scale, output_qparams.zero_point);
  };
}

template <typename InputQType = uint8_t>
static void RunQDQBinaryOpTest(const std::string& op_type, const TestInputDef<float>& input0_def,
                               const TestInputDef<float>& input1_def, float output_min, float output_max,
                               int opset_version,
                               ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  QuantParams<InputQType> output_qparams = QuantParams<InputQType>::Compute(output_min, output_max);

  // NOTE: Because of rounding differences, quantized QNN values may differ by 1 quantized unit, which
  // corresponds to a potential float32 difference of (1 * scale).
  //
  // For example, this happens during quantization when the expression (real_val/scale) is near x.5 (e.g., 20.4999).
  // If the HTP backend computes floating-point values at a higher precision, then 20.4999 is rounded to 20. However,
  // ORT uses 32-bit floats, which cannot represent 20.4999 exactly. Instead, ORT uses 20.5 which rounds to 21.
  const float f32_abs_err = 2.0001f * output_qparams.scale;

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQBinaryOpTestCase<InputQType>(op_type, input0_def, input1_def, output_qparams),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  f32_abs_err);
}

template <typename InputType = float>
static GetTestModelFn BuildBinaryOpTestCase(const std::string& op_type, const TestInputDef<InputType>& input0_def,
                                            const TestInputDef<InputType>& input1_def) {
  return [op_type, input0_def, input1_def](ModelTestBuilder& builder) {
    NodeArg* input0 = MakeTestInput(builder, input0_def);
    NodeArg* input1 = MakeTestInput(builder, input1_def);

    auto* output = builder.MakeOutput();
    builder.AddNode(op_type, {input0, input1}, {output});
  };
}

template <typename InputType = float>
static void RunBinaryOpTest(const std::string& op_type, const TestInputDef<InputType>& input0_def,
                            const TestInputDef<InputType>& input1_def, int opset_version,
                            ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildBinaryOpTestCase<InputType>(op_type, input0_def, input1_def),
                  provider_options,
                  opset_version,
                  expected_ep_assignment);
}

// Check that QNN compiles DQ -> Gelu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQGeluTest) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),  // Input range [-10.0, 10.0f]
                          "Gelu",
                          {},
                          -0.4f, 10.0f,  // Output range
                          11,
                          ExpectedEPNodeAssignment::All,
                          kMSDomain);  // GeLu is a contrib op.
}

// Check that QNN compiles DQ -> Elu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQEluTest) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),  // Input range [-10.0, 10.0f]
                          "Elu",
                          {},
                          -1.0f, 10.0f,  // Output range
                          11,
                          ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> HardSwish -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQHardSwishTest) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),  // Input range [-10.0, 10.0f]
                          "HardSwish",
                          {},
                          -0.3f, 10.0f,  // Output range
                          14,
                          ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Atan -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQAtanTest) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),  // Input range [-10.0, 10.0f]
                          "Atan",
                          {},
                          -1.48f, 1.48f,  // Output range
                          14,
                          ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that the default axis (-1) for SoftMax opset 13 works.
TEST_F(QnnHTPBackendTests, TestQDQSoftmax13_DefaultAxis) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f),
                          "Softmax",
                          {},          // Uses default axis of -1 for opset 13
                          0.0f, 1.0f,  // Output range
                          13, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that an axis != -1 is not supported.
TEST_F(QnnHTPBackendTests, TestQDQSoftmax13_UnsupportedAxis) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f),
                          "Softmax",
                          {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                          0.0f, 1.0f,  // Output range
                          13, ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that the default axis (1) for SoftMax opset < 13 does not work.
TEST_F(QnnHTPBackendTests, TestQDQSoftmax11_DefaultAxisFails) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f),
                          "Softmax",
                          {},          // Uses default axis of 1 for opset < 13.
                          0.0f, 1.0f,  // Output range
                          11, ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that setting an axis value of -1 works for Softmax opset < 13.
TEST_F(QnnHTPBackendTests, TestQDQSoftmax11_SetValidAxis) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f),
                          "Softmax",
                          {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                          0.0f, 1.0f,  // Output range
                          11, ExpectedEPNodeAssignment::All);
}

// Test QDQ Abs op.
TEST_F(QnnHTPBackendTests, TestQDQAbs) {
  RunQDQSingleInputOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),
                          "Abs",
                          {},
                          0.0f, 10.0f,  // Output range
                          13, ExpectedEPNodeAssignment::All);
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

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  constexpr float atan_min = -1.48f;
  constexpr float atan_max = 1.48f;
  const QuantParams<uint8_t> output_qparams = QuantParams<uint8_t>::Compute(atan_min, atan_max);

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  RunQnnModelTest(BuildQDQSingleInputOpTestCase<uint8_t>(input_def, "Atan", {}, output_qparams),
                  provider_options,
                  14,
                  ExpectedEPNodeAssignment::All);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  // 2nd run will load and run from Qnn context cache binary file
  RunQnnModelTest(BuildQDQSingleInputOpTestCase<uint8_t>(input_def, "Atan", {}, output_qparams),
                  provider_options,
                  14,
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, QuantAccuracyTest) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Note: a graph input -> Q -> DQ -> is optimized by Qnn to have a perfectly accurate output.
  // ORT's CPU EP, on the otherhand, actually quantizes and dequantizes the input, which leads to different outputs.
  auto builder_func = [](ModelTestBuilder& builder) {
    const TestInputDef<float> input0_def({1, 2, 3}, false, {1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f});

    // input -> Q -> Transpose -> DQ -> output
    NodeArg* input0 = MakeTestInput(builder, input0_def);
    QuantParams<uint8_t> qparams = GetTestInputQuantParams(input0_def);

    auto* quant_input = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input0, qparams.scale, qparams.zero_point, quant_input);

    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("Transpose", {quant_input}, {op_output});

    NodeArg* output = builder.MakeOutput();
    builder.AddDequantizeLinearNode<uint8_t>(op_output, qparams.scale, qparams.zero_point, output);
  };

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  RunQnnModelTest(builder_func,
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::All);
}

// Test QDQ Add
TEST_F(QnnHTPBackendTests, BinaryOp_Add4D) {
  RunQDQBinaryOpTest<uint8_t>("Add", TestInputDef<float>({1, 2, 2, 2}, false, -100.0f, 100.0f),
                              TestInputDef<float>({1, 2, 2, 2}, false, -100.0f, 100.0f),
                              -200.0f, 200.0f,  // Output range
                              17, ExpectedEPNodeAssignment::All);
}

// Test QDQ Sub
TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D) {
  RunQDQBinaryOpTest<uint8_t>("Sub", TestInputDef<float>({1, 3, 8, 8}, false, -100.0f, 100.0f),
                              TestInputDef<float>({1, 3, 8, 8}, false, -100.0f, 100.0f),
                              -200.0f, 200.0f,  // Output range
                              17, ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
// Enable when this is fixed.
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Sub4D_LargeInputs) {
  RunQDQBinaryOpTest<uint8_t>("Sub", TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                              TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                              -2.0f, 2.0f,  // Output range
                              17, ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
// Enable when this is fixed.
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Sub4D_Broadcast) {
  RunQDQBinaryOpTest<uint8_t>("Sub", TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                              TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f}),
                              -2.0f, 2.0f,  // Output range
                              17, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_SmallInputs) {
  RunQDQBinaryOpTest<uint8_t>("Div", TestInputDef<float>({1, 3, 8, 8}, false, -10.0f, 10.0f),
                              TestInputDef<float>({1, 3, 8, 8}, false, -10.0f, 10.0f),
                              -100.0f, 100.0f,  // Output range
                              17, ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
// Enable when this is fixed.
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Div4D_LargeInputs) {
  RunQDQBinaryOpTest<uint8_t>("Div", TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                              TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                              -100.0f, 100.0f,  // Output range
                              17, ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
// Enable when this is fixed.
// Fails accuracy when input0 has dims [1,3,768,768]
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Div4D_Broadcast) {
  RunQDQBinaryOpTest<uint8_t>("Div", TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                              TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f}),
                              -100.0f, 100.0f,  // Output range
                              17, ExpectedEPNodeAssignment::All);
}

// Test QDQ Mul
TEST_F(QnnHTPBackendTests, BinaryOp_Mul4D) {
  RunQDQBinaryOpTest<uint8_t>("Mul", TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                              TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                              -100.0f, 100.0f,  // Output range
                              17, ExpectedEPNodeAssignment::All);
}
// Test QDQ And
TEST_F(QnnHTPBackendTests, BinaryOp_And4D) {
  RunBinaryOpTest<bool>("And", TestInputDef<bool>({1, 2, 2, 2}, false, {false, false, true, true}),
                        TestInputDef<bool>({1, 2, 2, 2}, false, {false, true, false, true}),
                        17, ExpectedEPNodeAssignment::All);
}

// Test that Or is not yet supported on HTP backend.
TEST_F(QnnHTPBackendTests, BinaryOp_HTP_Or_Unsupported) {
  RunBinaryOpTest<bool>("Or", TestInputDef<bool>({1, 2, 2, 2}, false, {false, false, true, true}),
                        TestInputDef<bool>({1, 2, 2, 2}, false, {false, true, false, true}),
                        17, ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
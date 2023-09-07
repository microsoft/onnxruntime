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

template <typename InputType = float>
static GetTestModelFn BuildUnaryOpTestCase(const std::string& op_type, const TestInputDef<InputType>& input0_def,
                                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                           const std::string& domain = kOnnxDomain) {
  return [op_type, input0_def, attrs, domain](ModelTestBuilder& builder) {
    NodeArg* input0 = MakeTestInput(builder, input0_def);

    auto* output = builder.MakeOutput();
    auto& op_node = builder.AddNode(op_type, {input0}, {output}, domain);
    for (const auto& attr : attrs) {
      op_node.AddAttributeProto(attr);
    }
  };
}

// Creates the graph:
//                       _______________________
//                      |                       |
//    input_u8 -> DQ -> |       SimpleOp        | -> Q -> output_u8
//                      |_______________________|
//
// Currently used to test QNN EP.
template <typename InputQType>
GetTestQDQModelFn<InputQType> BuildQDQUnaryOpTestCase(const TestInputDef<float>& input_def,
                                                      const std::string& op_type,
                                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                      const std::string& domain = kOnnxDomain) {
  return [input_def, op_type, attrs, domain](ModelTestBuilder& builder,
                                             std::vector<QuantParams<InputQType>>& output_qparams) {
    auto* input = MakeTestInput(builder, input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams(input_def);
    auto* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    auto* op_output = builder.MakeIntermediate();
    auto& op_node = builder.AddNode(op_type, {input_qdq}, {op_output}, domain);

    for (const auto& attr : attrs) {
      op_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, op_output, output_qparams[0].scale, output_qparams[0].zero_point);
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
static void RunQDQUnaryOpTest(const TestInputDef<float>& input_def, const std::string& op_type,
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
  TestQDQModelAccuracy(BuildUnaryOpTestCase<float>(op_type, input_def, attrs, domain),
                       BuildQDQUnaryOpTestCase<InputQType>(input_def, op_type, attrs, domain),
                       provider_options,
                       opset_version,
                       expected_ep_assignment,
                       1e-5f);
}

// TODO: share with other op tests
// Creates the graph with two inputs and attributes
template <typename InputType>
static GetTestModelFn BuildOpTestCase(const std::string& op_type,
                                      const TestInputDef<InputType>& input0_def,
                                      const TestInputDef<InputType>& input1_def,
                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [op_type, input0_def, input1_def, attrs](ModelTestBuilder& builder) {
    NodeArg* input0 = MakeTestInput(builder, input0_def);
    NodeArg* input1 = MakeTestInput(builder, input1_def);

    auto* output = builder.MakeOutput();
    Node& onnx_node = builder.AddNode(op_type, {input0, input1}, {output});

    for (const auto& attr : attrs) {
      onnx_node.AddAttributeProto(attr);
    }
  };
}

// Creates the graph with two inputs and attributes
//                       _______________________
//                      |                       |
//   input0_u8 -> DQ -> |       SimpleOp        | -> Q -> output_u8
//   input1_u8 -> DQ -> |_______________________|
//
// Currently used to test QNN EP.
template <typename InputQType>
static GetTestQDQModelFn<InputQType> BuildQDQOpTestCase(const std::string& op_type,
                                                        const TestInputDef<float>& input0_def,
                                                        const TestInputDef<float>& input1_def,
                                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [op_type, input0_def, input1_def, attrs](ModelTestBuilder& builder,
                                                  std::vector<QuantParams<InputQType>>& output_qparams) {
    NodeArg* input0 = MakeTestInput(builder, input0_def);
    NodeArg* input1 = MakeTestInput(builder, input1_def);

    // input -> Q -> DQ -> Op
    QuantParams<InputQType> input0_qparams = GetTestInputQuantParams(input0_def);
    auto* qdq0_output = AddQDQNodePair<InputQType>(builder, input0, input0_qparams.scale, input0_qparams.zero_point);

    QuantParams<InputQType> input1_qparams = GetTestInputQuantParams(input1_def);
    auto* qdq1_output = AddQDQNodePair<InputQType>(builder, input1, input1_qparams.scale, input1_qparams.zero_point);

    // Op -> op_output
    auto* op_output = builder.MakeIntermediate();
    Node& onnx_node = builder.AddNode(op_type, {qdq0_output, qdq1_output}, {op_output});

    for (const auto& attr : attrs) {
      onnx_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, op_output, output_qparams[0].scale,
                                                      output_qparams[0].zero_point);
  };
}

template <typename InputQType = uint8_t>
static void RunQDQOpTest(const std::string& op_type,
                         const TestInputDef<float>& input0_def,
                         const TestInputDef<float>& input1_def,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         int opset_version,
                         ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, input0_def, input1_def, attrs),
                       BuildQDQOpTestCase<InputQType>(op_type, input0_def, input1_def, attrs),
                       provider_options,
                       opset_version,
                       expected_ep_assignment,
                       1e-5f);
}

template <typename InputType = float>
static void RunOpTest(const std::string& op_type,
                      const TestInputDef<InputType>& input0_def,
                      const TestInputDef<InputType>& input1_def,
                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                      int opset_version,
                      ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildOpTestCase<InputType>(op_type, input0_def, input1_def, attrs),
                  provider_options,
                  opset_version,
                  expected_ep_assignment);
}

// Check that QNN compiles DQ -> Gelu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Gelu) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),  // Input range [-10.0, 10.0f]
                    "Gelu",
                    {},
                    11,
                    ExpectedEPNodeAssignment::All,
                    kMSDomain);  // GeLu is a contrib op.
}

// Check that QNN compiles DQ -> Elu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Elu) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),  // Input range [-10.0, 10.0f]
                    "Elu",
                    {},
                    11,
                    ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> HardSwish -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSwish) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),  // Input range [-10.0, 10.0f]
                    "HardSwish",
                    {},
                    14,
                    ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Atan -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Atan) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),  // Input range [-10.0, 10.0f]
                    "Atan",
                    {},
                    14,
                    ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Asin -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Asin) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -0.5f, 0.5f),  // input range -0.5 to 0.5
                    "Asin", {},
                    13, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Sign -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Sign) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),
                    "Sign", {},
                    13, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Sin -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Sin) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -3.14159f, 3.14159f),
                    "Sin", {},
                    11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Cos -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Cos) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, {-3.14159f, -1.5f, -0.5f, 0.0f, 1.5, 3.14159f}),
                    "Cos", {},
                    11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Cos -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Cos_Inaccurate) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, {-3.14159f, -1.88436f, -0.542863f, 0.0f, 1.05622f, 3.14159f}),
                    "Cos", {},
                    11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Log -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Log) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, {3.14159f, 100.88436f, 10.542863f, 9.1f, 1.05622f, 3.14159f}),
                    "Log", {},
                    11, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that the default axis (-1) for SoftMax opset 13 works.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_DefaultAxis) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f),
                    "Softmax",
                    {},  // Uses default axis of -1 for opset 13
                    13, ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that an axis != -1 is not supported.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_UnsupportedAxis) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f),
                    "Softmax",
                    {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                    13, ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that the default axis (1) for SoftMax opset < 13 does not work.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax11_DefaultAxisFails) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f),
                    "Softmax",
                    {},  // Uses default axis of 1 for opset < 13.
                    11, ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that setting an axis value of -1 works for Softmax opset < 13.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax11_SetValidAxis) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f),
                    "Softmax",
                    {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                    11, ExpectedEPNodeAssignment::All);
}

// Test QDQ Abs op.
TEST_F(QnnHTPBackendTests, UnaryOp_Abs) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -10.0f, 10.0f),
                    "Abs",
                    {},
                    13, ExpectedEPNodeAssignment::All);
}

// Test QDQ Ceil op.
TEST_F(QnnHTPBackendTests, UnaryOp_Ceil) {
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 3}, false, -100.0f, 100.0f),
                    "Ceil",
                    {},
                    13, ExpectedEPNodeAssignment::All);
}

// Test QDQ DepthToSpace.
TEST_F(QnnHTPBackendTests, DepthToSpaceOp_CRD) {
  const std::vector<float> X = {0., 1., 2.,
                                3., 4., 5.,
                                9., 10., 11.,
                                12., 13., 14.,
                                18., 19., 20.,
                                21., 22., 23.,
                                27., 28., 29.,
                                30., 31., 32.};
  RunQDQUnaryOpTest(TestInputDef<float>({1, 4, 2, 3}, false, X),
                    "DepthToSpace",
                    {utils::MakeAttribute("blocksize", static_cast<int64_t>(2)),
                     utils::MakeAttribute("mode", "CRD")},
                    11, ExpectedEPNodeAssignment::All);
}

// Test QDQ DepthToSpace.
TEST_F(QnnHTPBackendTests, DepthToSpaceOp_DCR) {
  const std::vector<float> X = {0., 1., 2.,
                                3., 4., 5.,
                                9., 10., 11.,
                                12., 13., 14.,
                                18., 19., 20.,
                                21., 22., 23.,
                                27., 28., 29.,
                                30., 31., 32.};
  RunQDQUnaryOpTest(TestInputDef<float>({1, 4, 2, 3}, false, X),
                    "DepthToSpace",
                    {utils::MakeAttribute("blocksize", static_cast<int64_t>(2)),
                     utils::MakeAttribute("mode", "DCR")},
                    11, ExpectedEPNodeAssignment::All);
}

// Test QDQ SpaceToDepth.
TEST_F(QnnHTPBackendTests, SpaceToDepthOp) {
  const std::vector<float> X = {0.0f, 0.1f, 0.2f, 0.3f,
                                1.0f, 1.1f, 1.2f, 1.3f,

                                2.0f, 2.1f, 2.2f, 2.3f,
                                3.0f, 3.1f, 3.2f, 3.3f};
  RunQDQUnaryOpTest(TestInputDef<float>({1, 2, 2, 4}, false, X),
                    "SpaceToDepth",
                    {utils::MakeAttribute("blocksize", static_cast<int64_t>(2))},
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

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  TestQDQModelAccuracy(BuildUnaryOpTestCase<float>(op_type, input_def, {}),
                       BuildQDQUnaryOpTestCase<uint8_t>(input_def, op_type, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       1e-5f);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  // 2nd run will load and run from Qnn context cache binary file
  TestQDQModelAccuracy(BuildUnaryOpTestCase<float>(op_type, input_def, {}),
                       BuildQDQUnaryOpTestCase<uint8_t>(input_def, op_type, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       1e-5f);
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
  RunQDQOpTest<uint8_t>("Add",
                        TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                        TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Sub
TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D) {
  RunQDQOpTest<uint8_t>("Sub",
                        TestInputDef<float>({1, 3, 8, 8}, false, -10.0f, 10.0f),
                        TestInputDef<float>({1, 3, 8, 8}, false, -10.0f, 10.0f),
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D_LargeInputs) {
  RunQDQOpTest<uint8_t>("Sub",
                        TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                        TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D_Broadcast) {
  RunQDQOpTest<uint8_t>("Sub",
                        TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                        TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f}),
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_SmallInputs) {
  RunQDQOpTest<uint8_t>("Div",
                        TestInputDef<float>({1, 2, 2, 2}, false, {-10.0f, -8.0f, -1.0f, 0.0f, 1.0f, 2.1f, 8.0f, 10.0f}),
                        TestInputDef<float>({1, 2, 2, 2}, false, {5.0f, 4.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 5.0f}),
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// TODO: Enable when this is fixed.
// QNN v2.13: Inaccuracy detected for output 'output', element 2551923.
// Output quant params: scale=4100.92626953125, zero_point=126.
// Expected val: -277957.3125
// QNN QDQ val: 0 (err 277957.3125)
// CPU QDQ val: -516716.71875 (err 238759.40625)
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Div4D_LargeInputs) {
  RunQDQOpTest<uint8_t>("Div",
                        TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                        TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_Broadcast) {
  RunQDQOpTest<uint8_t>("Div",
                        TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                        TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f}),
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Mul
TEST_F(QnnHTPBackendTests, BinaryOp_Mul4D) {
  RunQDQOpTest<uint8_t>("Mul",
                        TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                        TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test And
TEST_F(QnnCPUBackendTests, BinaryOp_And4D) {
  RunOpTest<bool>("And",
                  TestInputDef<bool>({1, 4}, false, {false, false, true, true}),
                  TestInputDef<bool>({1, 4}, false, {false, true, false, true}),
                  {},
                  17,
                  ExpectedEPNodeAssignment::All);
}

// Test that Or is not yet supported on CPU backend.
TEST_F(QnnCPUBackendTests, BinaryOp_HTP_Or_Unsupported) {
  RunOpTest<bool>("Or",
                  TestInputDef<bool>({1, 4}, false, {false, false, true, true}),
                  TestInputDef<bool>({1, 4}, false, {false, true, false, true}),
                  {},
                  17,
                  ExpectedEPNodeAssignment::None);
}

// Test QDQ GridSample with bilinear
TEST_F(QnnHTPBackendTests, GridSample_Bilinear) {
  RunQDQOpTest<uint8_t>("GridSample",
                        TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                        TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f),
                        {utils::MakeAttribute("align_corners", static_cast<int64_t>(0)),
                         utils::MakeAttribute("mode", "bilinear"),
                         utils::MakeAttribute("padding_mode", "zeros")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ GridSample with align corners
TEST_F(QnnHTPBackendTests, GridSample_AlignCorners) {
  RunQDQOpTest<uint8_t>("GridSample",
                        TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                        TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f),
                        {utils::MakeAttribute("align_corners", static_cast<int64_t>(1)),
                         utils::MakeAttribute("mode", "bilinear"),
                         utils::MakeAttribute("padding_mode", "zeros")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ GridSample with padding mode: border
// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.046370312571525574, zero_point=129.
// Expected val: 3.3620510101318359
// QNN QDQ val: 3.2922921180725098 (err 0.069758892059326172)
// CPU QDQ val: 3.3850328922271729 (err 0.022981882095336914)
TEST_F(QnnHTPBackendTests, DISABLED_GridSample_BorderPadding) {
  RunQDQOpTest<uint8_t>("GridSample",
                        TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                        TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f),
                        {utils::MakeAttribute("mode", "bilinear"),
                         utils::MakeAttribute("padding_mode", "border")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ GridSample with nearest mode
TEST_F(QnnHTPBackendTests, GridSample_Nearest) {
  RunQDQOpTest<uint8_t>("GridSample",
                        TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                        TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f),
                        {utils::MakeAttribute("mode", "nearest")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ GridSample with reflection padding mode
// Inaccuracy detected for output 'output', element 2.
// Output quant params: scale=0.024269860237836838, zero_point=0.
// Expected val: 3.212885856628418
// QNN QDQ val: 3.1308119297027588 (err 0.08207392692565918)
// CPU QDQ val: 3.2036216259002686 (err 0.0092642307281494141)
TEST_F(QnnHTPBackendTests, DISABLED_GridSample_ReflectionPaddingMode) {
  RunQDQOpTest<uint8_t>("GridSample",
                        TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                        TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f),
                        {utils::MakeAttribute("padding_mode", "reflection")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
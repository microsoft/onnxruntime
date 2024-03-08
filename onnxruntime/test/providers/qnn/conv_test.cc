// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Creates a graph with a single float32 Conv operator. Used for testing CPU backend.
static GetTestModelFn BuildF32ConvTestCase(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                                           const TestInputDef<float>& weights_def,
                                           const TestInputDef<float>& bias_def,
                                           const std::vector<int64_t>& strides,
                                           const std::vector<int64_t>& pads,
                                           const std::vector<int64_t>& dilations,
                                           const std::string& auto_pad = "NOTSET") {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, auto_pad](ModelTestBuilder& builder) {
    std::vector<NodeArg*> conv_inputs = {
        MakeTestInput(builder, input_def),
        MakeTestInput(builder, weights_def)};

    if (!bias_def.GetShape().empty()) {
      conv_inputs.push_back(MakeTestInput(builder, bias_def));
    }

    auto* output = builder.MakeOutput();

    Node& convNode = builder.AddNode(conv_op_type, conv_inputs, {output});
    convNode.AddAttribute("auto_pad", auto_pad);

    if (!pads.empty() && auto_pad == "NOTSET") {
      convNode.AddAttribute("pads", pads);
    }

    if (!strides.empty()) {
      convNode.AddAttribute("strides", strides);
    }

    if (!dilations.empty()) {
      convNode.AddAttribute("dilations", dilations);
    }
  };
}

// Runs a Conv model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunCPUConvOpTest(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                             const TestInputDef<float>& weights_def,
                             const TestInputDef<float>& bias_def,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& pads,
                             const std::vector<int64_t>& dilations,
                             const std::string& auto_pad,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 13,
                             float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildF32ConvTestCase(conv_op_type, input_def, weights_def, bias_def, strides, pads, dilations, auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Creates a graph with a single Q/DQ Conv operator. Used for testing HTP backend.
template <typename ActivationQType, typename WeightQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQConvTestCase(const std::string& conv_op_type,
                                                               const TestInputDef<float>& input_def,
                                                               const TestInputDef<float>& weights_def,
                                                               const TestInputDef<float>& bias_def,
                                                               const std::vector<int64_t>& strides,
                                                               const std::vector<int64_t>& pads,
                                                               const std::vector<int64_t>& dilations,
                                                               const std::string& auto_pad = "NOTSET",
                                                               bool use_contrib_qdq = false) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, auto_pad, use_contrib_qdq](ModelTestBuilder& builder,
                                                std::vector<QuantParams<ActivationQType>>& output_qparams) {
    std::vector<NodeArg*> conv_inputs;

    // input -> Q/DQ ->
    auto* input = MakeTestInput(builder, input_def);
    QuantParams<ActivationQType> input_qparams = GetTestInputQuantParams<ActivationQType>(input_def);
    auto* input_qdq = AddQDQNodePair<ActivationQType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                      use_contrib_qdq);
    conv_inputs.push_back(input_qdq);

    // weights -> Q/DQ ->
    auto* weights = MakeTestInput(builder, weights_def);
    QuantParams<WeightQType> weights_qparams = GetTestInputQuantParams<WeightQType>(weights_def);
    auto* weights_qdq = AddQDQNodePair<WeightQType>(builder, weights, weights_qparams.scale,
                                                    weights_qparams.zero_point, use_contrib_qdq);
    conv_inputs.push_back(weights_qdq);

    // bias ->
    if (!bias_def.GetShape().empty()) {
      // Bias requirement taken from python quantization tool: onnx_quantizer.py::quantize_bias_static()
      const float bias_scale = input_qparams.scale * weights_qparams.scale;

      conv_inputs.push_back(MakeTestQDQBiasInput(builder, bias_def, bias_scale, use_contrib_qdq));
    }

    auto* conv_output = builder.MakeIntermediate();
    Node& conv_node = builder.AddNode(conv_op_type, conv_inputs, {conv_output});

    conv_node.AddAttribute("auto_pad", auto_pad);

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_node.AddAttribute("pads", pads);
    }
    if (!strides.empty()) {
      conv_node.AddAttribute("strides", strides);
    }
    if (!dilations.empty()) {
      conv_node.AddAttribute("dilations", dilations);
    }

    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(builder, conv_output, output_qparams[0].scale,
                                                           output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Runs a Conv model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename ActivationQType, typename WeightQType>
static void RunHTPConvOpTest(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                             const TestInputDef<float>& weights_def,
                             const TestInputDef<float>& bias_def,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& pads,
                             const std::vector<int64_t>& dilations,
                             const std::string& auto_pad,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             bool use_contrib_qdq = false,
                             int opset = 13,
                             QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildF32ConvTestCase(conv_op_type, input_def, weights_def, bias_def, strides, pads, dilations,
                                            auto_pad),
                       BuildQDQConvTestCase<ActivationQType, WeightQType>(conv_op_type, input_def, weights_def,
                                                                          bias_def, strides, pads, dilations,
                                                                          auto_pad, use_contrib_qdq),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as a dynamic input.
// TODO: Segfaults when calling graphFinalize(). v2.13
TEST_F(QnnCPUBackendTests, DISABLED_Convf32_dynamic_bias) {
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                   TestInputDef<float>({2, 1, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                   TestInputDef<float>({2}, false, -1.0f, 1.0f),           // Random dynamic bias
                   {1, 1},                                                 // default strides
                   {0, 0, 0, 0},                                           // default pads
                   {1, 1},                                                 // default dilations
                   "NOTSET",                                               // No auto-padding
                   ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnCPUBackendTests, Convf32_bias_initializer) {
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                   TestInputDef<float>({2, 1, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                   {1, 1},                                                 // default strides
                   {0, 0, 0, 0},                                           // default pads
                   {1, 1},                                                 // default dilations
                   "NOTSET",                                               // No auto-padding
                   ExpectedEPNodeAssignment::All);
}

// Tests Conv's auto_pad value "SAME_UPPER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, Convf32_AutoPadUpper) {
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 1, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({2, 1, 2, 2}, true, -1.0f, 1.0f),   // Random static weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                   {1, 1},                                                 // strides
                   {},                                                     // pads
                   {1, 1},                                                 // dilations
                   "SAME_UPPER",                                           // auto_pad
                   ExpectedEPNodeAssignment::All);
}

// Tests ConvTranspose's auto_pad value "SAME_UPPER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, ConvTransposef32_AutoPadUpper) {
  RunCPUConvOpTest("ConvTranspose",
                   TestInputDef<float>({1, 1, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({1, 2, 2, 2}, true, -1.0f, 1.0f),   // Random static weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                   {1, 1},                                                 // strides
                   {},                                                     // pads
                   {1, 1},                                                 // dilations
                   "SAME_UPPER",                                           // auto_pad
                   ExpectedEPNodeAssignment::All);
}

// Tests Conv's auto_pad value "SAME_LOWER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, Convf32_AutoPadLower) {
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 1, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({2, 1, 2, 2}, false, -1.0f, 1.0f),  // Random dynamic weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                   {1, 1},                                                 // strides
                   {},                                                     // pads
                   {1, 1},                                                 // dilations
                   "SAME_LOWER",                                           // auto_pad
                   ExpectedEPNodeAssignment::All);
}

// Tests ConvTranspose's auto_pad value "SAME_LOWER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, ConvTransposef32_AutoPadLower) {
  RunCPUConvOpTest("ConvTranspose",
                   TestInputDef<float>({1, 1, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({1, 2, 2, 2}, false, -1.0f, 1.0f),  // Random dynamic weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                   {1, 1},                                                 // strides
                   {},                                                     // pads
                   {1, 1},                                                 // dilations
                   "SAME_LOWER",                                           // auto_pad
                   ExpectedEPNodeAssignment::All);
}

// large input,output, pads
TEST_F(QnnCPUBackendTests, Convf32_large_input1_pad_bias_initializer) {
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 3, 60, 452}, false, 0.0f, 10.0f),  // Random dynamic input
                   TestInputDef<float>({16, 3, 3, 3}, true, 0.0f, 1.0f),      // Random dynamic weights
                   TestInputDef<float>({16}, true, -1.0f, 1.0f),              // Random static bias
                   {1, 1},
                   {1, 1, 1, 1},
                   {1, 1},
                   "NOTSET",
                   ExpectedEPNodeAssignment::All,
                   13,
                   1e-4f);
}

TEST_F(QnnCPUBackendTests, Convf32_large_input2_nopad_bias_initializer) {
#if defined(_WIN32)
  // Tolerance needs to be > 1.52588e-05 on Windows x64
  // TODO: Investigate why
  float fp32_abs_err = 1e-4f;
#else
  float fp32_abs_err = 1e-5f;  // default value
#endif

  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 32, 16, 113}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({16, 32, 1, 1}, false, -1.0f, 1.0f),    // Random dynamic weights
                   TestInputDef<float>({16}, true, -1.0f, 1.0f),               // Random static bias
                   {1, 1},
                   {0, 0, 0, 0},
                   {1, 1},
                   "NOTSET",
                   ExpectedEPNodeAssignment::All,
                   13,  // opset
                   fp32_abs_err);
}

// Test 1D Conv with static weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, Conv1Df32_StaticWeights_DefaultBias) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 2, 4}, false, input_data),               // Dynamic input
                   TestInputDef<float>({1, 2, 2}, true, {1.0f, 2.0f, 3.0f, 4.0f}),  // Static weights
                   TestInputDef<float>({1}, true, {1.0f}),                          // Initializer Bias
                   {1},                                                             // Strides
                   {0, 0},                                                          // Pads
                   {1},                                                             // Dilations
                   "NOTSET",
                   ExpectedEPNodeAssignment::All);
}

// Test 1D Conv with dynamic weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, Conv1Df32_DynamicWeights_DefaultBias) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 2, 4}, false, input_data),                // Dynamic input
                   TestInputDef<float>({1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),  // Dynamic weights
                   TestInputDef<float>(),                                            // Default bias
                   {1},                                                              // Strides
                   {0, 0},                                                           // Pads
                   {1},                                                              // Dilations
                   "NOTSET",
                   ExpectedEPNodeAssignment::All);
}

// Test 1D ConvTranspose with static weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, ConvTranspose1Df32_StaticWeights_DefaultBias) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunCPUConvOpTest("ConvTranspose",
                   TestInputDef<float>({1, 2, 4}, false, input_data),               // Dynamic input
                   TestInputDef<float>({2, 1, 2}, true, {1.0f, 2.0f, 3.0f, 4.0f}),  // Static weights
                   TestInputDef<float>({1}, true, {0.0f}),                          // Zero bias
                   {1},                                                             // Strides
                   {0, 0},                                                          // Pads
                   {1},                                                             // Dilations
                   "NOTSET",
                   ExpectedEPNodeAssignment::All);
}

// Test 1D ConvTranspose with dynamic weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, ConvTranspose1Df32_DynamicWeights_DefaultBias) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunCPUConvOpTest("ConvTranspose",
                   TestInputDef<float>({1, 2, 4}, false, input_data),                // Dynamic input
                   TestInputDef<float>({2, 1, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),  // Dynamic weights
                   TestInputDef<float>({1}, true, {0.0f}),                           // Zero bias
                   {1},                                                              // Strides
                   {0, 0},                                                           // Pads
                   {1},                                                              // Dilations
                   "NOTSET",
                   ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// The bug is from a QDQ model, and Conv node gets processed before it's producer Mul node
// A Transpose node gets inserted between Mul and the dynamic weight tensor shape on Conv
// to make Conv weight with shape HWNC
// However it changes Mul output shape to HWNC and cause issue
// It has to be QDQ model, because the DQ node with initializer on Conv gets processed first
// and DQ node requires its node unit to be processed
// So, Conv gets processed before Mul node
TEST_F(QnnHTPBackendTests, Test_QDQConvWithDynamicWeightsFromMul) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto BuildConvMulGraph = [](ModelTestBuilder& builder) {
    // DQ node for Conv input
    auto* dq_i_output = builder.MakeIntermediate();
    auto* conv_dq_input = builder.MakeInitializer<uint8_t>({1, 32, 16, 113}, static_cast<uint8_t>(0), static_cast<uint8_t>(127));

    // DQ node for Conv bias
    auto* dq_bias_output = builder.MakeIntermediate();
    auto* bias = builder.MakeInitializer<int32_t>({16}, static_cast<int32_t>(0), static_cast<int32_t>(127));

    // Mul node
    // DQ nodes for Mul
    auto* mul_dq1_output = builder.MakeIntermediate();
    auto* mul_input1 = builder.MakeInput<uint8_t>({16, 32, 1, 1}, static_cast<uint8_t>(0), static_cast<uint8_t>(127));

    auto* mul_dq2_output = builder.MakeIntermediate();
    auto* mul_input2 = builder.MakeInitializer<uint8_t>({16, 1, 1, 1}, static_cast<uint8_t>(0), static_cast<uint8_t>(127));
    builder.AddDequantizeLinearNode<uint8_t>(mul_input1, .03f, 0, mul_dq1_output);
    builder.AddDequantizeLinearNode<uint8_t>(mul_input2, .03f, 0, mul_dq2_output);

    auto* mul_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {mul_dq1_output, mul_dq2_output}, {mul_output});

    auto* mul_dq_output = AddQDQNodePair<uint8_t>(builder, mul_output, .03f, 0);

    builder.AddDequantizeLinearNode<uint8_t>(conv_dq_input, .04f, 0, dq_i_output);
    builder.AddDequantizeLinearNode<int32_t>(bias, .0012f, 0, dq_bias_output);
    // Conv node
    auto* conv_output = builder.MakeIntermediate();

    Node& conv_node = builder.AddNode("Conv", {dq_i_output, mul_dq_output, dq_bias_output}, {conv_output});
    conv_node.AddAttribute("auto_pad", "NOTSET");
    conv_node.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
    conv_node.AddAttribute("strides", std::vector<int64_t>{1, 1});
    conv_node.AddAttribute("dilations", std::vector<int64_t>{1, 1});

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(conv_output, .039f, 0, q_output);

    auto* dq_output = builder.MakeOutput();
    builder.AddDequantizeLinearNode<uint8_t>(q_output, .039f, 0, dq_output);
  };

  RunQnnModelTest(BuildConvMulGraph,
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::All,
                  4e-4f);  // Accuracy decreased slightly in QNN SDK 2.17.
                           // Expected: 9.94500065, Actual: 9.94537735
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as a dynamic input.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_bias_dynamic_input) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                                     TestInputDef<float>({1, 1, 3, 3}, true, -10.0f, 10.0f),  // Random static input
                                     TestInputDef<float>({1}, false, {2.0f}),                 // Dynamic bias
                                     {1, 1},                                                  // Strides
                                     {0, 0, 0, 0},                                            // Pads
                                     {1, 1},                                                  // Dilations
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13,     // opset
                                     // Need tolerance of 0.413% of output range after QNN SDK 2.17
                                     QDQTolerance(0.00413f));
}

// Tests 16-bit QDQ Conv with dynamic weights and bias (uses QNN's Conv2d)
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.0040235077030956745, zero_point=0.
// Expected val: 87.354057312011719
// QNN QDQ val: 0 (err 87.354057312011719)
// CPU QDQ val: 87.3583984375 (err 0.00434112548828125)
TEST_F(QnnHTPBackendTests, DISABLED_ConvU16S16S32_DynamicBias) {
  TestInputDef<float> input_def({1, 2, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 50));
  TestInputDef<float> weight_def({1, 2, 3, 3}, false, GetFloatDataInRange(-1.0f, 5.0f, 18));
  RunHTPConvOpTest<uint16_t, int16_t>("Conv",
                                      input_def,                                   // Input
                                      weight_def.OverrideValueRange(-5.0f, 5.0f),  // Weights (symmetric quant range)
                                      TestInputDef<float>({1}, false, {2.0f}),     // Bias
                                      {1, 1},                                      // Strides
                                      {0, 0, 0, 0},                                // Pads
                                      {1, 1},                                      // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit QDQ Conv with dynamic weights and bias (uses QNN's DepthwiseConv2d)
// TODO(adrianlizarraga): FAIL: Failed to finalize QNN graph. Error code 1002
TEST_F(QnnHTPBackendTests, DISABLED_DepthwiseConvU16S16S32_DynamicBias) {
  TestInputDef<float> input_def({1, 1, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 25));
  TestInputDef<float> weight_def({1, 1, 3, 3}, false, GetFloatDataInRange(-1.0f, 5.0f, 9));
  RunHTPConvOpTest<uint16_t, int16_t>("Conv",
                                      input_def,                                   // Input
                                      weight_def.OverrideValueRange(-5.0f, 5.0f),  // Weights (symmetric quant range)
                                      TestInputDef<float>({1}, false, {2.0f}),     // Bias
                                      {1, 1},                                      // Strides
                                      {0, 0, 0, 0},                                // Pads
                                      {1, 1},                                      // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit QDQ Conv with dynamic weights and no bias.
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.0039929896593093872, zero_point=0.
// Expected val: 85.354057312011719
// QNN QDQ val: 0 (err 85.354057312011719)
// CPU QDQ val: 85.358139038085938 (err 0.00408172607421875)
TEST_F(QnnHTPBackendTests, DISABLED_ConvU16S16S32_NoBias) {
  TestInputDef<float> input_def({1, 2, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 50));
  TestInputDef<float> weight_def({1, 2, 3, 3}, false, GetFloatDataInRange(-1.0f, 5.0f, 18));
  RunHTPConvOpTest<uint16_t, int16_t>("Conv",
                                      input_def,                                   // Input
                                      weight_def.OverrideValueRange(-5.0f, 5.0f),  // Weights (symmetric quant range)
                                      TestInputDef<float>(),                       // Bias
                                      {1, 1},                                      // Strides
                                      {0, 0, 0, 0},                                // Pads
                                      {1, 1},                                      // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit QDQ Conv with dynamic weights and no bias (uses QNN's DepthWiseConv2d)
// TODO(adrianlizarraga): FAIL: Failed to finalize QNN graph. Error code 1002
TEST_F(QnnHTPBackendTests, DISABLED_DepthwiseConvU16S16S32_NoBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 25);
  std::vector<float> weight_data = GetFloatDataInRange(-10.0f, 10.0f, 9);
  RunHTPConvOpTest<uint16_t, int16_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5}, false, input_data),   // Input
                                      TestInputDef<float>({1, 1, 3, 3}, false, weight_data),  // Weights
                                      TestInputDef<float>(),                                  // Bias
                                      {1, 1},                                                 // Strides
                                      {0, 0, 0, 0},                                           // Pads
                                      {1, 1},                                                 // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with static bias.
// Uses QNN's DepthwiseConv2d operator.
// TODO: Inaccuracy detected for output 'output', element 8.
// Output quant params: scale=0.0027466239407658577, zero_point=10194.
// Expected val: 152
// QNN QDQ val: 151.8004150390625 (err 0.1995849609375)
// CPU QDQ val: 151.9981689453125 (err 0.0018310546875)
TEST_F(QnnHTPBackendTests, DepthwiseConvU16U8S32_StaticBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 25);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 9);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 1, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>({1}, true, {2.0f}),                // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with static bias.
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.0040235077030956745, zero_point=0.
// Expected val: 87.354057312011719
// QNN QDQ val: 87.559577941894531 (err 0.2055206298828125)
// CPU QDQ val: 87.398635864257812 (err 0.04457855224609375)
TEST_F(QnnHTPBackendTests, ConvU16U8S32_StaticBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 50);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 18);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 2, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>({1}, true, {2.0f}),                // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with dynamic bias.
// Uses QNN's DepthwiseConv2d operator.
// TODO: Inaccuracy detected for output 'output', element 1.
// Output quant params: scale=0.0027466239407658577, zero_point=10194.
// Expected val: -13.000001907348633
// QNN QDQ val: -13.095903396606445 (err 0.0959014892578125)
// CPU QDQ val: -12.999771118164062 (err 0.0002307891845703125)
TEST_F(QnnHTPBackendTests, DepthwiseConvU16U8S32_DynamicBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 25);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 9);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 1, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>({1}, false, {2.0f}),               // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with dynamic bias.
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.0040235077030956745, zero_point=0.
// Expected val: 87.354057312011719
// QNN QDQ val: 87.559577941894531 (err 0.2055206298828125)
// CPU QDQ val: 87.398635864257812 (err 0.04457855224609375)
TEST_F(QnnHTPBackendTests, ConvU16U8S32_DynamicBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 50);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 18);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 2, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>({1}, false, {2.0f}),               // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with no bias
// TODO: Inaccuracy detected for output 'output', element 7.
// Output quant params: scale=0.0039929896593093872, zero_point=0.
// Expected val: 246.98667907714844
// QNN QDQ val: 247.82090759277344 (err 0.834228515625)
// CPU QDQ val: 247.24192810058594 (err 0.2552490234375)
TEST_F(QnnHTPBackendTests, ConvU16U8S32_NoBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 50);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 18);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 2, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>(),                                 // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with no bias
// Uses QNN's DepthwiseConv2d operator.
// TODO: Inaccuracy detected for output 'output', element 8.
// Output quant params: scale=0.0027466239407658577, zero_point=10923.
// Expected val: 150
// QNN QDQ val: 149.80087280273438 (err 0.199127197265625)
// CPU QDQ val: 149.99862670898438 (err 0.001373291015625)
TEST_F(QnnHTPBackendTests, DepthwiseConvU16U8S32_NoBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 25);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 9);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 1, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>(),                                 // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Test that dynamic weights with default bias works for Conv. This was previously not working
// on older versions of QNN sdk.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_DynamicWeight_NoBias) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 3, 32, 32}, false, -10.0f, 10.0f),  // Input
                                     TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),    // Weights
                                     TestInputDef<float>(),                                      // Bias
                                     {1, 1},                                                     // Strides
                                     {0, 0, 0, 0},                                               // Pads
                                     {1, 1},                                                     // Dilations
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Test that dynamic weights with default bias works for ConvTranspose. This was previously not working
// on older versions of QNN sdk.
TEST_F(QnnHTPBackendTests, ConvTransposeU8U8S32_DynamicWeight_NoBias) {
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 3, 32, 32}, false, -10.0f, 10.0f),  // Input
                                     TestInputDef<float>({3, 1, 4, 4}, false, -10.0f, 10.0f),    // Weights
                                     TestInputDef<float>(),                                      // Bias
                                     {1, 1},                                                     // Strides
                                     {0, 0, 0, 0},                                               // Pads
                                     {1, 1},                                                     // Dilations
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                                     TestInputDef<float>({1, 1, 3, 3}, true, -10.0f, 10.0f),  // Random static weight
                                     TestInputDef<float>({1}, true, {2.0f}),                  // Initializer bias
                                     {1, 1},                                                  // Strides
                                     {0, 0, 0, 0},                                            // Pads
                                     {1, 1},                                                  // Dilations
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13,     // opset
                                     // Need tolerance of 0.413% of output range after QNN SDK 2.17
                                     QDQTolerance(0.00413f));
}

// Tests 1D Conv with bias as an initializer.
TEST_F(QnnHTPBackendTests, Conv1DU8U8S32_bias_initializer) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0, 0},                                                      // pads
                                     {1},                                                         // dilations
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Tests 1D ConvTranspose with bias as an initializer.
TEST_F(QnnHTPBackendTests, ConvTranspose1DU8U8S32_bias_initializer) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0, 0},                                                      // pads
                                     {1},                                                         // dilations
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Tests auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvU8U8S32_AutoPadUpper) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {1, 1},                                               // strides
                                     {},                                                   // pads
                                     {1, 1},                                               // dilations
                                     "SAME_UPPER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests Conv1d auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, Conv1DU8U8S32_AutoPadUpper) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     "SAME_UPPER",                                                // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests TransposeConv1d auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvTranspose1DU8U8S32_AutoPadUpper) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     "SAME_UPPER",                                                // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests Conv's auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {1, 1},                                               // strides
                                     {},                                                   // pads
                                     {1, 1},                                               // dilations
                                     "SAME_LOWER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests ConvTranspose's auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvTransposeU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {1, 1},                                               // strides
                                     {},                                                   // pads
                                     {1, 1},                                               // dilations
                                     "SAME_LOWER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests Conv1d auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, Conv1DU8U8S32_AutoPadLower) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     "SAME_LOWER",                                                // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests ConvTranspose 1d auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvTranspose1DU8U8S32_AutoPadLower) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     "SAME_LOWER",                                                // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

TEST_F(QnnHTPBackendTests, ConvU8U8S32_large_input1_padding_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 3, 60, 452}, false, 0.f, 10.f),        // Dynamic input
                                     TestInputDef<float>({16, 3, 3, 3}, true, -1.f, 1.f),           // Static weights
                                     TestInputDef<float>({16}, true, std::vector<float>(16, 1.f)),  // Initializer bias
                                     {1, 1},
                                     {1, 1, 1, 1},
                                     {1, 1},
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13,     // opset
                                     // Need tolerance of 0.73% of output range after QNN SDK 2.17
                                     QDQTolerance(0.00730f));
}

TEST_F(QnnHTPBackendTests, ConvU8U8S32_large_input2_bias_initializer) {
#ifdef __linux__
  // On Linux QNN SDK 2.17: Need a tolerance of 0.785% of output range to pass.
  QDQTolerance tolerance = QDQTolerance(0.00785f);
#else
  QDQTolerance tolerance = QDQTolerance();
#endif
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 128, 8, 56}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({32, 128, 1, 1}, true, -1.f, 1.f),   // Random static weights
                                     TestInputDef<float>({32}, true, -1.f, 1.f),              // Random initializer bias
                                     {1, 1},
                                     {0, 0, 0, 0},
                                     {1, 1},
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,
                                     13,
                                     tolerance);
}

TEST_F(QnnHTPBackendTests, ConvU8U8S32_LargeInput_Dilations_Pads) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 3, 768, 1152}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({64, 3, 7, 7}, true, -1.f, 1.f),       // Static weights
                                     TestInputDef<float>({64}, true, -1.f, 1.f),                // Initializer bias
                                     {2, 2},                                                    // strides
                                     {3, 3, 3, 3},                                              // pads
                                     {1, 1},                                                    // dilations
                                     "NOTSET",                                                  // auto_pad
                                     ExpectedEPNodeAssignment::All);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif

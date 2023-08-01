// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

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
                  ExpectedEPNodeAssignment::All);
}

// Creates a graph with a single float32 Conv operator. Used for testing CPU backend.
static GetTestModelFn BuildF32ConvTestCase(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                                           const TestInputDef<float>& weights_def,
                                           const TestInputDef<float>& bias_def,
                                           const std::vector<int64_t>& strides,
                                           const std::vector<int64_t>& pads,
                                           const std::vector<int64_t>& dilations,
                                           const std::string& auto_pad = "NOTSET") {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads, dilations, auto_pad](ModelTestBuilder& builder) {
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
template <typename InputQType>
static GetTestModelFn BuildQDQConvTestCase(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                                           const TestInputDef<float>& weights_def,
                                           const TestInputDef<float>& bias_def,
                                           const std::vector<int64_t>& strides,
                                           const std::vector<int64_t>& pads,
                                           const std::vector<int64_t>& dilations,
                                           const std::string& auto_pad = "NOTSET") {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads, dilations, auto_pad](ModelTestBuilder& builder) {
    auto* output = builder.MakeOutput();

    using InputQLimits = std::numeric_limits<InputQType>;

    const float input_scale = 0.004f;
    const float weight_scale = 0.004f;
    const InputQType io_zp = (InputQLimits::min() + InputQLimits::max()) / 2 + 1;

    std::vector<NodeArg*> conv_inputs;

    // input -> Q/DQ ->
    auto* input = MakeTestInput(builder, input_def);
    auto* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_scale, io_zp);
    conv_inputs.push_back(input_qdq);

    // weights -> Q/DQ ->
    auto* weights = MakeTestInput(builder, weights_def);
    auto* weights_qdq = AddQDQNodePair<InputQType>(builder, weights, weight_scale, io_zp);
    conv_inputs.push_back(weights_qdq);

    // bias ->
    if (!bias_def.GetShape().empty()) {
      NodeArg* bias_int32 = nullptr;
      const float bias_scale = input_scale * weight_scale;  // Taken from python quantization tool: onnx_quantizer.py::quantize_bias_static()

      // Bias must be int32 to be detected as a QDQ node unit.
      // We must quantize the data.
      if (bias_def.IsRandomData()) {
        // Create random initializer def that is quantized to int32
        const auto& rand_info = bias_def.GetRandomDataInfo();
        TestInputDef<int32_t> bias_int32_def(bias_def.GetShape(), bias_def.IsInitializer(), static_cast<int32_t>(rand_info.min / bias_scale),
                                             static_cast<int32_t>(rand_info.max / bias_scale));
        bias_int32 = MakeTestInput(builder, bias_int32_def);
      } else {
        assert(bias_def.IsRawData());
        // Create raw data initializer def that is quantized to int32
        const auto& bias_f32_raw = bias_def.GetRawData();
        const size_t num_elems = bias_f32_raw.size();

        std::vector<int32_t> bias_int32_raw(num_elems);
        for (size_t i = 0; i < num_elems; i++) {
          bias_int32_raw[i] = static_cast<int32_t>(bias_f32_raw[i] / bias_scale);
        }

        TestInputDef<int32_t> bias_int32_def(bias_def.GetShape(), bias_def.IsInitializer(), bias_int32_raw);
        bias_int32 = MakeTestInput(builder, bias_int32_def);
      }

      auto* bias = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int32_t>(bias_int32, bias_scale, 0, bias);
      conv_inputs.push_back(bias);
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

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InputQType>(conv_output, input_scale, io_zp, q_output);
    builder.AddDequantizeLinearNode<InputQType>(q_output, input_scale, io_zp, output);
  };
}

// Runs a Conv model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename InputQType>
static void RunHTPConvOpTest(const std::string& conv_op_type, const TestInputDef<float>& input_def,
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
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildQDQConvTestCase<InputQType>(conv_op_type, input_def, weights_def, bias_def,
                                                   strides, pads, dilations, auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as a dynamic input.
// TODO: Segfaults when calling graphFinalize().
TEST_F(QnnCPUBackendTests, DISABLED_TestCPUConvf32_dynamic_bias) {
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
TEST_F(QnnCPUBackendTests, TestCPUConvf32_bias_initializer) {
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
TEST_F(QnnCPUBackendTests, TestCPUConvf32_AutoPadUpper) {
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
TEST_F(QnnCPUBackendTests, TestCPUConvTransposef32_AutoPadUpper) {
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
TEST_F(QnnCPUBackendTests, TestCPUConvf32_AutoPadLower) {
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
TEST_F(QnnCPUBackendTests, TestCPUConvTransposef32_AutoPadLower) {
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
TEST_F(QnnCPUBackendTests, TestCPUConvf32_large_input1_pad_bias_initializer) {
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

TEST_F(QnnCPUBackendTests, TestCPUConvf32_large_input2_nopad_bias_initializer) {
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
TEST_F(QnnCPUBackendTests, TestCPUConv1Df32_StaticWeights_DefaultBias) {
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 2, 4}, false, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}),  // Dynamic input
                   TestInputDef<float>({1, 2, 2}, true, {1.0f, 2.0f, 3.0f, 4.0f}),                           // Static weights
                   TestInputDef<float>({1}, true, {1.0f}),                                                   // Bias of 1.f
                   {1},                                                                                      // Strides
                   {0, 0},                                                                                   // Pads
                   {1},                                                                                      // Dilations
                   "NOTSET",
                   ExpectedEPNodeAssignment::All);
}

// Test 1D Conv with dynamic weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, TestCPUConv1Df32_DynamicWeights_DefaultBias) {
  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 2, 4}, false, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}),  // Dynamic input
                   TestInputDef<float>({1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),                          // Dynamic weights
                   TestInputDef<float>(),                                                                    // Default bias
                   {1},                                                                                      // Strides
                   {0, 0},                                                                                   // Pads
                   {1},                                                                                      // Dilations
                   "NOTSET",
                   ExpectedEPNodeAssignment::All);
}

// Test 1D ConvTranspose with static weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, TestCPUConvTranspose1Df32_StaticWeights_DefaultBias) {
  RunCPUConvOpTest("ConvTranspose",
                   TestInputDef<float>({1, 2, 4}, false, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}),  // Dynamic input
                   TestInputDef<float>({2, 1, 2}, true, {1.0f, 2.0f, 3.0f, 4.0f}),                           // Static weights
                   TestInputDef<float>({1}, true, {0.0f}),                                                   // Zero bias
                   {1},                                                                                      // Strides
                   {0, 0},                                                                                   // Pads
                   {1},                                                                                      // Dilations
                   "NOTSET",
                   ExpectedEPNodeAssignment::All);
}

// Test 1D ConvTranspose with dynamic weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, TestCPUConvTranspose1Df32_DynamicWeights_DefaultBias) {
  RunCPUConvOpTest("ConvTranspose",
                   TestInputDef<float>({1, 2, 4}, false, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}),  // Dynamic input
                   TestInputDef<float>({2, 1, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),                          // Dynamic weights
                   TestInputDef<float>({1}, true, {0.0f}),                                                   // Zero bias
                   {1},                                                                                      // Strides
                   {0, 0},                                                                                   // Pads
                   {1},                                                                                      // Dilations
                   "NOTSET",
                   ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as a dynamic input.
TEST_F(QnnHTPBackendTests, TestQDQConvU8S32_bias_dynamic_input) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 1, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                            TestInputDef<float>({1, 1, 3, 3}, true, -10.0f, 10.0f),  // Random static input
                            TestInputDef<float>({1}, false, {2.0f}),                 // Dynamic bias = 2.0f
                            {1, 1},                                                  // Strides
                            {0, 0, 0, 0},                                            // Pads
                            {1, 1},                                                  // Dilations
                            "NOTSET",
                            ExpectedEPNodeAssignment::All);
}

// Test that dynamic weights with default bias works for Conv. This was previously not working
// on older versions of QNN sdk.
TEST_F(QnnHTPBackendTests, TestQDQConvU8S32_DynamicWeight_NoBias) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 3, 32, 32}, false, 0.0f, 10.0f),  // Random dynamic input
                            TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random dynamic weights
                            TestInputDef<float>(),                                    // Default bias
                            {1, 1},                                                   // Strides
                            {0, 0, 0, 0},                                             // Pads
                            {1, 1},                                                   // Dilations
                            "NOTSET",
                            ExpectedEPNodeAssignment::All);
}

// Test that dynamic weights with default bias works for ConvTranspose. This was previously not working
// on older versions of QNN sdk.
TEST_F(QnnHTPBackendTests, TestQDQConvTransposeU8S32_DynamicWeight_NoBias) {
  RunHTPConvOpTest<uint8_t>("ConvTranspose",
                            TestInputDef<float>({1, 3, 32, 32}, false, 0.0f, 100.0f),  // Random dynamic input
                            TestInputDef<float>({3, 1, 4, 4}, false, -10.0f, 10.0f),   // Random dynamic weights
                            TestInputDef<float>(),                                     // Default bias
                            {1, 1},                                                    // Strides
                            {0, 0, 0, 0},                                              // Pads
                            {1, 1},                                                    // Dilations
                            "NOTSET",
                            ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8S32_bias_initializer) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 1, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                            TestInputDef<float>({1, 1, 3, 3}, true, -10.0f, 10.0f),  // Random static weight
                            TestInputDef<float>({1}, true, {2.0f}),                  // Initializer bias = 2.0f
                            {1, 1},                                                  // Strides
                            {0, 0, 0, 0},                                            // Pads
                            {1, 1},                                                  // Dilations
                            "NOTSET",
                            ExpectedEPNodeAssignment::All);
}

// Tests 1D Conv with bias as an initializer.
TEST_F(QnnHTPBackendTests, TestQDQConv1DU8S32_bias_initializer) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 2, 4}, false, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}),  // Dynamic input
                            TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),                       // Static weight
                            TestInputDef<float>({1}, true, {1.0f}),                                           // Initializer bias = 1.0f
                            {1},                                                                              // strides
                            {0, 0},                                                                           // pads
                            {1},                                                                              // dilations
                            "NOTSET",
                            ExpectedEPNodeAssignment::All);
}

// Tests 1D ConvTranspose with bias as an initializer.
TEST_F(QnnHTPBackendTests, TestQDQConvTranspose1DU8S32_bias_initializer) {
  RunHTPConvOpTest<uint8_t>("ConvTranspose",
                            TestInputDef<float>({1, 2, 4}, false, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}),  // Dynamic input
                            TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),                       // Static weight
                            TestInputDef<float>({1}, true, {1.0f}),                                           // Initializer bias = 1.0f
                            {1},                                                                              // strides
                            {0, 0},                                                                           // pads
                            {1},                                                                              // dilations
                            "NOTSET",
                            ExpectedEPNodeAssignment::All);
}

// Tests auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestQDQConvU8S32_AutoPadUpper) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                            TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                            TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias = 1.0f
                            {1, 1},                                               // strides
                            {},                                                   // pads
                            {1, 1},                                               // dilations
                            "SAME_UPPER",                                         // auto_pad
                            ExpectedEPNodeAssignment::All,
                            13,
                            1e-4f);
}

// Tests Conv1d auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestQDQConv1DU8U8S32_AutoPadUpper) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 2, 4}, false, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}),  // Dynamic input
                            TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),                       // Static weight
                            TestInputDef<float>({1}, true, {1.0f}),                                           // Initializer bias = 1.0f
                            {1},                                                                              // strides
                            {0},                                                                              // pads
                            {1},                                                                              // dilations
                            "SAME_UPPER",                                                                     // auto_pad
                            ExpectedEPNodeAssignment::All,
                            13,
                            1e-4f);
}

// Tests TransposeConv1d auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestQDQConvTranspose1DU8U8S32_AutoPadUpper) {
  RunHTPConvOpTest<uint8_t>("ConvTranspose",
                            TestInputDef<float>({1, 2, 4}, false, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}),  // Dynamic input
                            TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),                       // Static weight
                            TestInputDef<float>({1}, true, {1.0f}),                                           // Initializer bias = 1.0f
                            {1},                                                                              // strides
                            {0},                                                                              // pads
                            {1},                                                                              // dilations
                            "SAME_UPPER",                                                                     // auto_pad
                            ExpectedEPNodeAssignment::All,
                            13,
                            1e-4f);
}

// Tests Conv's auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                            TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                            TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias = 1.0f
                            {1, 1},                                               // strides
                            {},                                                   // pads
                            {1, 1},                                               // dilations
                            "SAME_LOWER",                                         // auto_pad
                            ExpectedEPNodeAssignment::All,
                            13,
                            1e-4f);
}

// Tests ConvTranspose's auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestQDQConvTransposeU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t>("ConvTranspose",
                            TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                            TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                            TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias = 1.0f
                            {1, 1},                                               // strides
                            {},                                                   // pads
                            {1, 1},                                               // dilations
                            "SAME_LOWER",                                         // auto_pad
                            ExpectedEPNodeAssignment::All,
                            13,
                            1e-4f);
}

// Tests Conv1d auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestQDQConv1DU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 2, 4}, false, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}),  // Dynamic input
                            TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),                       // Static weight
                            TestInputDef<float>({1}, true, {1.0f}),                                           // Initializer bias = 1.0f
                            {1},                                                                              // strides
                            {0},                                                                              // pads
                            {1},                                                                              // dilations
                            "SAME_LOWER",                                                                     // auto_pad
                            ExpectedEPNodeAssignment::All,
                            13,
                            1e-4f);
}

// Tests ConvTranspose 1d auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestQDQConvTranspose1DU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t>("ConvTranspose",
                            TestInputDef<float>({1, 2, 4}, false, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}),  // Dynamic input
                            TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),                       // Static weight
                            TestInputDef<float>({1}, true, {1.0f}),                                           // Initializer bias = 1.0f
                            {1},                                                                              // strides
                            {0},                                                                              // pads
                            {1},                                                                              // dilations
                            "SAME_LOWER",                                                                     // auto_pad
                            ExpectedEPNodeAssignment::All,
                            13,
                            1e-4f);
}

// TODO: re-enable tests once HTP issues are resolved
TEST_F(QnnHTPBackendTests, DISABLED_TestQDQConvU8U8S32_large_input1_padding_bias_initializer) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 3, 60, 452}, false, 0.f, 10.f),        // Dynamic input
                            TestInputDef<float>({16, 3, 3, 3}, true, -1.f, 1.f),           // Static weights
                            TestInputDef<float>({16}, true, std::vector<float>(16, 1.f)),  // Initializer bias = 1.f, 1.f, ...
                            {1, 1},
                            {1, 1, 1, 1},
                            {1, 1},
                            "NOTSET",
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, DISABLED_TestQDQConvU8S32_large_input2_bias_initializer) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 128, 8, 56}, false, 0.f, 10.f),  // Dynamic input
                            TestInputDef<float>({32, 128, 1, 1}, true, -1.f, 1.f),   // Random static weights
                            TestInputDef<float>({32}, true, -1.f, 1.f),              // Random initializer bias
                            {1, 1},
                            {0, 0, 0, 0},
                            {1, 1},
                            "NOTSET",
                            ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnHTPBackendTests, DISABLED_TestQDQConvU8U8S32_LargeInput_Dilations_Pads) {
  RunHTPConvOpTest<uint8_t>("Conv",
                            TestInputDef<float>({1, 3, 768, 1152}, false, 0.f, 10.f),  // Dynamic input
                            TestInputDef<float>({64, 3, 7, 7}, true, -1.f, 1.f),       // Random static weights
                            TestInputDef<float>({64}, true, -1.f, 1.f),                // Random initializer bias
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

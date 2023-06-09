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
TEST_F(QnnCPUBackendTests, Test_QDQConvWithDynamicWeightsFromMul) {
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

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildConvMulGraph,
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::All,
                  expected_nodes_in_partition,
                  "Test_ConvWithDynamicWeightsFromMul");
}

// Creates a graph with a single Conv operator. Used for testing CPU backend.
static GetTestModelFn BuildConvTestCase(const std::vector<int64_t>& input_shape,
                                        const std::vector<int64_t>& weights_shape,
                                        bool is_bias_initializer,
                                        const std::vector<int64_t>& strides,
                                        const std::vector<int64_t>& pads,
                                        const std::vector<int64_t>& dilations,
                                        const std::string& auto_pad = "NOTSET") {
  return [input_shape, weights_shape, is_bias_initializer, strides, pads, dilations, auto_pad](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(input_shape, 0.0f, 10.0f);
    auto* output = builder.MakeOutput();
    auto* weights = builder.MakeInitializer<float>(weights_shape, 0.0f, 1.0f);

    onnxruntime::NodeArg* bias = nullptr;

    if (is_bias_initializer) {
      bias = builder.MakeInitializer<float>({weights_shape[0]}, -1.0f, 1.0f);
    } else {
      bias = builder.MakeInput<float>({weights_shape[0]}, -1.0f, 1.0f);
    }

    Node& convNode = builder.AddNode("Conv", {input, weights, bias}, {output});
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
static void RunCPUConvOpTest(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& weights_shape,
                             bool is_bias_initializer,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& pads,
                             const std::vector<int64_t>& dilations,
                             const std::string& auto_pad,
                             ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                             int opset = 13,
                             float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildConvTestCase(input_shape, weights_shape, is_bias_initializer, strides, pads, dilations, auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description,
                  fp32_abs_err);
}

// Creates a graph with a single Q/DQ Conv operator. Used for testing HTP backend.
template <typename InputType, typename WeightType, typename BiasType, typename OutputType>
GetTestModelFn BuildQDQConvTestCase(const std::vector<int64_t>& input_shape,
                                    const std::vector<int64_t>& weights_shape,
                                    bool is_bias_initializer,
                                    const std::vector<int64_t>& strides,
                                    const std::vector<int64_t>& pads,
                                    const std::vector<int64_t>& dilations,
                                    const std::string& auto_pad = "NOTSET") {
  return [input_shape, weights_shape, is_bias_initializer, strides, pads, dilations, auto_pad](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    using InputLimits = std::numeric_limits<InputType>;
    using WeightLimits = std::numeric_limits<WeightType>;
    using OutputLimits = std::numeric_limits<OutputType>;

    InputType input_min_value = InputLimits::min();
    InputType input_max_value = InputLimits::max();

    WeightType weight_min_value = WeightLimits::min();
    WeightType weight_max_value = WeightLimits::max();

    auto* dq_w_output = builder.MakeIntermediate();
    auto* weight = builder.MakeInitializer<WeightType>(weights_shape, weight_min_value, weight_max_value);
    builder.AddDequantizeLinearNode<WeightType>(weight, .03f,
                                                (weight_min_value + weight_max_value) / 2 + 1,
                                                dq_w_output);

    auto* dq_bias_output = builder.MakeIntermediate();
    onnxruntime::NodeArg* bias = nullptr;

    if (is_bias_initializer) {
      bias = builder.MakeInitializer<BiasType>({weights_shape[0]}, static_cast<BiasType>(0),
                                               static_cast<BiasType>(127));
    } else {
      bias = builder.MakeInput<BiasType>({weights_shape[0]}, static_cast<BiasType>(0),
                                         static_cast<BiasType>(127));
    }

    builder.AddDequantizeLinearNode<BiasType>(bias, .0012f,
                                              0,
                                              dq_bias_output);

    auto* conv_output = builder.MakeIntermediate();
    auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, .04f,
                                                (input_min_value + input_max_value) / 2 + 1);
    Node& convNode = builder.AddNode("Conv", {dq_output, dq_w_output, dq_bias_output}, {conv_output});

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(conv_output, 1e-4f,
                                              (OutputLimits::min() + OutputLimits::max()) / 2 + 1,
                                              q_output);

    builder.AddDequantizeLinearNode<OutputType>(q_output, 1e-4f,
                                                (OutputLimits::min() + OutputLimits::max()) / 2 + 1,
                                                output_arg);
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

// Runs a Conv model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename InputType, typename WeightType, typename BiasType, typename OutputType>
static void RunHTPConvOpTest(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& weights_shape,
                             bool is_bias_initializer,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& pads,
                             const std::vector<int64_t>& dilations,
                             const std::string& auto_pad,
                             ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                             int opset = 13,
                             float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildQDQConvTestCase<InputType, WeightType, BiasType, OutputType>(input_shape, weights_shape,
                                                                                    is_bias_initializer,
                                                                                    strides, pads, dilations, auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description,
                  fp32_abs_err);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an input.
//
// TODO: Enable this test when QNN CPU backend (QNN sdk 2.10.0) fixes bug that causes graph finalization to
// throw a segfault when the bias is a non-static input.
TEST_F(QnnCPUBackendTests, DISABLED_TestCPUConvf32_bias_input) {
  RunCPUConvOpTest({1, 1, 3, 3}, {2, 1, 2, 2}, false, {1, 1}, {0, 0, 0, 0}, {1, 1}, "NOTSET", ExpectedEPNodeAssignment::All, "TestCPUConvf32_bias_input");
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnCPUBackendTests, TestCPUConvf32_bias_initializer) {
  RunCPUConvOpTest({1, 1, 3, 3}, {2, 1, 2, 2}, true, {1, 1}, {0, 0, 0, 0}, {1, 1}, "NOTSET", ExpectedEPNodeAssignment::All, "TestCPUConvf32_bias_initializer");
}

// Tests auto_pad value "SAME_UPPER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, TestCPUConvf32_AutoPadUpper) {
  RunCPUConvOpTest({1, 1, 3, 3},  // Input 0 shape
                   {2, 1, 2, 2},  // Input 1 (weights) shape
                   true,          // is_bias_initializer
                   {1, 1},        // strides
                   {},            // pads
                   {1, 1},        // dilations
                   "SAME_UPPER",  // auto_pad
                   ExpectedEPNodeAssignment::All,
                   "TestCPUConvf32_AutoPadUpper");
}

// Tests auto_pad value "SAME_LOWER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, TestCPUConvf32_AutoPadLower) {
  RunCPUConvOpTest({1, 1, 3, 3},  // Input 0 shape
                   {2, 1, 2, 2},  // Input 1 (weights) shape
                   true,          // is_bias_initializer
                   {1, 1},        // strides
                   {},            // pads
                   {1, 1},        // dilations
                   "SAME_LOWER",  // auto_pad
                   ExpectedEPNodeAssignment::All,
                   "TestCPUConvf32_AutoPadLower");
}

// large input,output, pads
// TODO: re-enable tests once Padding issues are resolved
TEST_F(QnnCPUBackendTests, DISABLED_TestCPUConvf32_large_input1_pad_bias_initializer) {
  RunCPUConvOpTest({1, 3, 60, 452}, {16, 3, 3, 3}, true, {1, 1}, {1, 1, 1, 1}, {1, 1}, "NOTSET", ExpectedEPNodeAssignment::All, "TestCPUConvf32_large_input1_pad_bias_initializer");
}

TEST_F(QnnCPUBackendTests, TestCPUConvf32_large_input2_nopad_bias_initializer) {
#if defined(_WIN32)
  // Tolerance needs to be > 1.52588e-05 on Windows x64
  // TODO: Investigate why
  float fp32_abs_err = 1e-4f;
#else
  float fp32_abs_err = 1e-5f;  // default value
#endif

  RunCPUConvOpTest({1, 32, 16, 113},
                   {16, 32, 1, 1},
                   true,
                   {1, 1},
                   {0, 0, 0, 0},
                   {1, 1},
                   "NOTSET",
                   ExpectedEPNodeAssignment::All,
                   "TestCPUConvf32_large_input2_nopad_bias_initializer",
                   13,  // opset
                   fp32_abs_err);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an input.
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8S32_bias_input) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>({1, 1, 5, 5}, {1, 1, 3, 3}, false, {1, 1}, {0, 0, 0, 0}, {1, 1}, "NOTSET", ExpectedEPNodeAssignment::All,
                                                       "TestQDQConvU8U8S32_bias_input");
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8S32_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>({1, 1, 5, 5}, {1, 1, 3, 3}, true, {1, 1}, {0, 0, 0, 0}, {1, 1}, "NOTSET", ExpectedEPNodeAssignment::All,
                                                       "TestQDQConvU8U8S32_bias_initializer");
}

// Tests auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestConvU8U8S32_AutoPadUpper) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>(
      {1, 1, 5, 5},  // input_shape
      {1, 1, 4, 4},  // weights_shape
      true,          // is_bias_initializer
      {1, 1},        // strides
      {},            // pads
      {1, 1},        // dilations
      "SAME_UPPER",  // auto_pad
      ExpectedEPNodeAssignment::All,
      "TestConvU8U8S32_AutoPadUpper",
      13,
      1e-4f);
}

// Tests auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, TestConvU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>(
      {1, 1, 5, 5},  // input_shape
      {1, 1, 4, 4},  // weights_shape
      true,          // is_bias_initializer
      {1, 1},        // strides
      {},            // pads
      {1, 1},        // dilations
      "SAME_LOWER",  // auto_pad
      ExpectedEPNodeAssignment::All,
      "TestConvU8U8S32_AutoPadLower",
      13,
      1e-4f);
}

// TODO: re-enable tests once HTP issues are resolved
TEST_F(QnnHTPBackendTests, DISABLED_TestQDQConvU8U8S32_large_input1_padding_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>({1, 3, 60, 452}, {16, 3, 3, 3}, true, {1, 1}, {1, 1, 1, 1}, {1, 1}, "NOTSET", ExpectedEPNodeAssignment::All,
                                                       "TestQDQConvU8U8S32_large_input1_padding_bias_initializer");
}

TEST_F(QnnHTPBackendTests, DISABLED_TestQDQConvU8U8S32_large_input2_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>({1, 128, 8, 56}, {32, 128, 1, 1}, true, {1, 1}, {0, 0, 0, 0}, {1, 1}, "NOTSET", ExpectedEPNodeAssignment::All,
                                                       "TestQDQConvU8U8S32_large_input2_bias_initializer");
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnHTPBackendTests, DISABLED_TestQDQConvU8U8S32_LargeInput_Dilations_Pads) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>(
      {1, 3, 768, 1152},  // input_shape
      {64, 3, 7, 7},      // weights_shape
      true,               // is_bias_initializer
      {2, 2},             // strides
      {3, 3, 3, 3},       // pads
      {1, 1},             // dilations
      "NOTSET",           // auto_pad
      ExpectedEPNodeAssignment::All,
      "TestQDQConvU8U8S32_large_input2_bias_initializer");
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <optional>
#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Information for activation node placed between the Conv and Q.
struct OutputActivationInfo {
  std::string op_type;  // Relu or Clip
  std::vector<float> const_inputs;
};

// Creates a graph with a single float32 Conv operator. Used for testing CPU backend.
static GetTestModelFn BuildF32ConvTestCase(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                                           const TestInputDef<float>& weights_def,
                                           const TestInputDef<float>& bias_def,
                                           const std::vector<int64_t>& strides,
                                           const std::vector<int64_t>& pads,
                                           const std::vector<int64_t>& dilations,
                                           std::optional<int64_t> group,
                                           const std::string& auto_pad = "NOTSET",
                                           std::optional<OutputActivationInfo> output_activation = std::nullopt) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, group, auto_pad, output_activation](ModelTestBuilder& builder) {
    std::vector<NodeArg*> conv_inputs = {
        MakeTestInput(builder, input_def),
        MakeTestInput(builder, weights_def)};

    if (!bias_def.GetShape().empty()) {
      conv_inputs.push_back(MakeTestInput(builder, bias_def));
    }

    auto* conv_output = output_activation.has_value() ? builder.MakeIntermediate() : builder.MakeOutput();

    Node& conv_node = builder.AddNode(conv_op_type, conv_inputs, {conv_output});
    conv_node.AddAttribute("auto_pad", auto_pad);

    if (group.has_value()) {
      conv_node.AddAttribute("group", group.value());
    }

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_node.AddAttribute("pads", pads);
    }

    if (!strides.empty()) {
      conv_node.AddAttribute("strides", strides);
    }

    if (!dilations.empty()) {
      conv_node.AddAttribute("dilations", dilations);
    }

    if (output_activation.has_value()) {
      NodeArg* output = builder.MakeOutput();
      std::vector<NodeArg*> activation_inputs = {conv_output};
      for (auto val : output_activation->const_inputs) {
        activation_inputs.push_back(builder.MakeScalarInitializer(val));
      }
      builder.AddNode(output_activation->op_type, activation_inputs, {output});
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
                             std::optional<int64_t> group,
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
  auto build_fn = BuildF32ConvTestCase(conv_op_type, input_def, weights_def, bias_def, strides, pads,
                                       dilations, group, auto_pad);
  RunQnnModelTest(build_fn,
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Creates a graph with a single Q/DQ Conv operator. Used for testing HTP backend.
template <typename ActivationQType, typename WeightQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQConvTestCase(
    const std::string& conv_op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    std::optional<int64_t> group,
    const std::string& auto_pad = "NOTSET",
    bool use_contrib_qdq = false,
    std::optional<OutputActivationInfo> output_activation = std::nullopt) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, group, auto_pad,
          use_contrib_qdq, output_activation](ModelTestBuilder& builder,
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

    if (group.has_value()) {
      conv_node.AddAttribute("group", group.value());
    }

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_node.AddAttribute("pads", pads);
    }
    if (!strides.empty()) {
      conv_node.AddAttribute("strides", strides);
    }
    if (!dilations.empty()) {
      conv_node.AddAttribute("dilations", dilations);
    }

    NodeArg* q_input = conv_output;
    if (output_activation.has_value()) {
      q_input = builder.MakeIntermediate();
      std::vector<NodeArg*> activation_inputs = {conv_output};
      for (auto val : output_activation->const_inputs) {
        activation_inputs.push_back(builder.MakeScalarInitializer(val));
      }
      builder.AddNode(output_activation->op_type, activation_inputs, {q_input});
    }

    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(builder, q_input, output_qparams[0].scale,
                                                           output_qparams[0].zero_point, use_contrib_qdq);
  };
}

template <typename ActivationQType, typename WeightQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQPerChannelConvTestCase(
    const std::string& conv_op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    int64_t weight_quant_axis,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    std::optional<int64_t> group,
    const std::string& auto_pad = "NOTSET",
    bool use_contrib_qdq = false,
    std::optional<OutputActivationInfo> output_activation = std::nullopt) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, group, auto_pad, use_contrib_qdq,
          weight_quant_axis, output_activation](ModelTestBuilder& builder,
                                                std::vector<QuantParams<ActivationQType>>& output_qparams) {
    std::vector<NodeArg*> conv_inputs;

    // input -> Q/DQ ->
    auto* input = MakeTestInput(builder, input_def);
    QuantParams<ActivationQType> input_qparams = GetTestInputQuantParams<ActivationQType>(input_def);
    auto* input_qdq = AddQDQNodePair<ActivationQType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                      use_contrib_qdq);
    conv_inputs.push_back(input_qdq);

    // Quantized(weights) -> DQ ->
    ORT_ENFORCE(weights_def.IsInitializer() && weights_def.IsRawData());
    std::vector<float> weight_scales;
    std::vector<WeightQType> weight_zero_points;
    TensorShape weights_shape = weights_def.GetTensorShape();
    int64_t pos_weight_quant_axis = weight_quant_axis;
    if (pos_weight_quant_axis < 0) {
      pos_weight_quant_axis += static_cast<int64_t>(weights_shape.NumDimensions());
    }
    GetTestInputQuantParamsPerChannel<WeightQType>(weights_def, weight_scales, weight_zero_points,
                                                   static_cast<size_t>(pos_weight_quant_axis), true);

    std::vector<WeightQType> quantized_weights;
    size_t num_weight_storage_elems = weights_shape.Size();
    if constexpr (std::is_same_v<WeightQType, Int4x2> || std::is_same_v<WeightQType, UInt4x2>) {
      num_weight_storage_elems = Int4x2::CalcNumInt4Pairs(weights_shape.Size());
    }
    quantized_weights.resize(num_weight_storage_elems);
    QuantizeValues<float, WeightQType>(weights_def.GetRawData(), quantized_weights, weights_shape,
                                       weight_scales, weight_zero_points, pos_weight_quant_axis);

    NodeArg* weights_initializer = builder.MakeInitializer<WeightQType>(weights_def.GetShape(), quantized_weights);
    NodeArg* weights_dq = builder.MakeIntermediate();
    Node& weights_dq_node = builder.AddDequantizeLinearNode<WeightQType>(weights_initializer, weight_scales,
                                                                         weight_zero_points, weights_dq,
                                                                         nullptr, use_contrib_qdq);
    weights_dq_node.AddAttribute("axis", weight_quant_axis);
    conv_inputs.push_back(weights_dq);

    // Quantized(bias) -> DQ ->
    if (!bias_def.GetShape().empty()) {
      // Bias requirement taken from python quantization tool: onnx_quantizer.py::quantize_bias_static()
      // bias_scale = input_scale * weight_scale
      // bias_zero_point = 0
      ORT_ENFORCE(bias_def.IsInitializer() && bias_def.IsRawData());
      std::vector<float> bias_scales = weight_scales;
      std::vector<int32_t> bias_zero_points(weight_scales.size(), 0);
      for (size_t i = 0; i < bias_scales.size(); i++) {
        bias_scales[i] *= input_qparams.scale;
      }

      TensorShape bias_shape = bias_def.GetTensorShape();
      std::vector<int32_t> quantized_biases(bias_shape.Size());
      QuantizeValues<float, int32_t>(bias_def.GetRawData(), quantized_biases, bias_shape, bias_scales,
                                     bias_zero_points, 0);

      NodeArg* bias_initializer = builder.MakeInitializer<int32_t>(bias_def.GetShape(), quantized_biases);
      NodeArg* bias_dq = builder.MakeIntermediate();
      Node& bias_dq_node = builder.AddDequantizeLinearNode<int32_t>(bias_initializer, bias_scales, bias_zero_points,
                                                                    bias_dq, nullptr, use_contrib_qdq);

      bias_dq_node.AddAttribute("axis", static_cast<int64_t>(0));
      conv_inputs.push_back(bias_dq);
    }

    auto* conv_output = builder.MakeIntermediate();
    Node& conv_node = builder.AddNode(conv_op_type, conv_inputs, {conv_output});

    conv_node.AddAttribute("auto_pad", auto_pad);

    if (group.has_value()) {
      conv_node.AddAttribute("group", group.value());
    }

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_node.AddAttribute("pads", pads);
    }
    if (!strides.empty()) {
      conv_node.AddAttribute("strides", strides);
    }
    if (!dilations.empty()) {
      conv_node.AddAttribute("dilations", dilations);
    }

    NodeArg* q_input = conv_output;
    if (output_activation.has_value()) {
      q_input = builder.MakeIntermediate();
      std::vector<NodeArg*> activation_inputs = {conv_output};
      for (auto val : output_activation->const_inputs) {
        activation_inputs.push_back(builder.MakeScalarInitializer(val));
      }
      builder.AddNode(output_activation->op_type, activation_inputs, {q_input});
    }

    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(builder, q_input, output_qparams[0].scale,
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
                             std::optional<int64_t> group,
                             const std::string& auto_pad,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             bool use_contrib_qdq = false,
                             int opset = 13,
                             QDQTolerance tolerance = QDQTolerance(),
                             std::optional<OutputActivationInfo> output_activation = std::nullopt) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildF32ConvTestCase(conv_op_type, input_def, weights_def, bias_def, strides, pads, dilations,
                                            group, auto_pad, output_activation),
                       BuildQDQConvTestCase<ActivationQType, WeightQType>(conv_op_type, input_def, weights_def,
                                                                          bias_def, strides, pads, dilations,
                                                                          group, auto_pad, use_contrib_qdq,
                                                                          output_activation),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

// Runs a QDQ Conv model (per-axis quantization on weight/bias) on the QNN HTP backend.
// Checks the graph node assignment, and that inference outputs for QNN EP and CPU EP match.
template <typename ActivationQType, typename WeightQType>
static void RunHTPConvOpPerChannelTest(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                                       const TestInputDef<float>& weights_def,
                                       const TestInputDef<float>& bias_def,
                                       int64_t weight_quant_axis,
                                       const std::vector<int64_t>& strides,
                                       const std::vector<int64_t>& pads,
                                       const std::vector<int64_t>& dilations,
                                       std::optional<int64_t> group,
                                       const std::string& auto_pad,
                                       ExpectedEPNodeAssignment expected_ep_assignment,
                                       bool use_contrib_qdq = false,
                                       int opset = 13,
                                       QDQTolerance tolerance = QDQTolerance(),
                                       std::optional<OutputActivationInfo> output_activation = std::nullopt) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto f32_fn = BuildF32ConvTestCase(conv_op_type, input_def, weights_def, bias_def, strides, pads, dilations,
                                     group, auto_pad, output_activation);
  auto qdq_fn = BuildQDQPerChannelConvTestCase<ActivationQType, WeightQType>(conv_op_type, input_def, weights_def,
                                                                             bias_def, weight_quant_axis, strides,
                                                                             pads, dilations, group, auto_pad,
                                                                             use_contrib_qdq, output_activation);
  TestQDQModelAccuracy(f32_fn, qdq_fn, provider_options, opset, expected_ep_assignment, tolerance);
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
                   1,                                                      // default group
                   "NOTSET",                                               // No auto-padding
                   ExpectedEPNodeAssignment::All);

  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 1, 3, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                   TestInputDef<float>({2, 1, 2, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                   TestInputDef<float>({2}, false, -1.0f, 1.0f),              // Random dynamic bias
                   {1, 1, 1},                                                 // default strides
                   {0, 0, 0, 0, 0, 0},                                        // default pads
                   {1, 1, 1},                                                 // default dilations
                   1,                                                         // default group
                   "NOTSET",                                                  // No auto-padding
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
                   1,                                                      // default group
                   "NOTSET",                                               // No auto-padding
                   ExpectedEPNodeAssignment::All);

  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 1, 3, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                   TestInputDef<float>({2, 1, 2, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                   {1, 1, 1},                                                 // default strides
                   {0, 0, 0, 0, 0, 0},                                        // default pads
                   {1, 1, 1},                                                 // default dilations
                   1,                                                         // default group
                   "NOTSET",                                                  // No auto-padding
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
                   1,                                                      // default group
                   "SAME_UPPER",                                           // auto_pad
                   ExpectedEPNodeAssignment::All);

  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 1, 3, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({2, 1, 2, 2, 2}, true, -1.0f, 1.0f),   // Random static weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                   {1, 1, 1},                                                 // strides
                   {},                                                        // pads
                   {1, 1, 1},                                                 // dilations
                   1,                                                         // default group
                   "SAME_UPPER",                                              // auto_pad
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
                   1,                                                      // default group
                   "SAME_UPPER",                                           // auto_pad
                   ExpectedEPNodeAssignment::All);

  RunCPUConvOpTest("ConvTranspose",
                   TestInputDef<float>({1, 1, 3, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({1, 2, 2, 2, 2}, true, -1.0f, 1.0f),   // Random static weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                   {1, 1, 1},                                                 // strides
                   {},                                                        // pads
                   {1, 1, 1},                                                 // dilations
                   1,                                                         // default group
                   "SAME_UPPER",                                              // auto_pad
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
                   1,                                                      // default group
                   "SAME_LOWER",                                           // auto_pad
                   ExpectedEPNodeAssignment::All);

  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 1, 3, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({2, 1, 2, 2, 2}, false, -1.0f, 1.0f),  // Random dynamic weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                   {1, 1, 1},                                                 // strides
                   {},                                                        // pads
                   {1, 1, 1},                                                 // dilations
                   1,                                                         // default group
                   "SAME_LOWER",                                              // auto_pad
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
                   1,                                                      // default group
                   "SAME_LOWER",                                           // auto_pad
                   ExpectedEPNodeAssignment::All);
}

// Tests ConvTranspose's auto_pad value "SAME_LOWER" (compares to CPU EP).
// Exception from graphFinalize
// Exception thrown at 0x00007FFFB7651630 (QnnCpu.dll) in onnxruntime_test_all.exe:
// 0xC0000005: Access violation reading location 0x0000000000000000.
TEST_F(QnnCPUBackendTests, DISABLED_ConvTranspose3D_f32_AutoPadLower) {
  RunCPUConvOpTest("ConvTranspose",
                   TestInputDef<float>({1, 1, 3, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({1, 2, 2, 2, 2}, false, -1.0f, 1.0f),  // Random dynamic weights
                   TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                   {1, 1, 1},                                                 // strides
                   {},                                                        // pads
                   {1, 1, 1},                                                 // dilations
                   1,                                                         // default group
                   "SAME_LOWER",                                              // auto_pad
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
                   1,  // default group
                   "NOTSET",
                   ExpectedEPNodeAssignment::All,
                   13,
                   1e-4f);

  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 3, 60, 452, 20}, false, 0.0f, 10.0f),  // Random dynamic input
                   TestInputDef<float>({16, 3, 3, 3, 3}, true, 0.0f, 1.0f),       // Random dynamic weights
                   TestInputDef<float>({16}, true, -1.0f, 1.0f),                  // Random static bias
                   {1, 1, 1},
                   {1, 1, 1, 1, 1, 1},
                   {1, 1, 1},
                   1,  // default group
                   "NOTSET",
                   ExpectedEPNodeAssignment::All,
                   13,
                   2e-4f);
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
                   1,  // default group
                   "NOTSET",
                   ExpectedEPNodeAssignment::All,
                   13,  // opset
                   fp32_abs_err);

  RunCPUConvOpTest("Conv",
                   TestInputDef<float>({1, 32, 16, 113, 12}, false, -3.0f, 3.0f),  // Random dynamic input
                   TestInputDef<float>({16, 32, 1, 1, 1}, false, -1.0f, 1.0f),     // Random dynamic weights
                   TestInputDef<float>({16}, true, -1.0f, 1.0f),                   // Random static bias
                   {1, 1, 1},
                   {0, 0, 0, 0, 0, 0},
                   {1, 1, 1},
                   1,  // default group
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
                   1,                                                               // default group
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
                   1,                                                                // default group
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
                   1,                                                               // default group
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
                   1,                                                                // default group
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
    auto* conv_dq_input = builder.MakeInitializer<uint8_t>({1, 32, 16, 113}, static_cast<uint8_t>(0),
                                                           static_cast<uint8_t>(127));

    // DQ node for Conv bias
    auto* dq_bias_output = builder.MakeIntermediate();
    auto* bias = builder.MakeInitializer<int32_t>({16}, static_cast<int32_t>(0), static_cast<int32_t>(127));

    // Mul node
    // DQ nodes for Mul
    auto* mul_dq1_output = builder.MakeIntermediate();
    auto* mul_input1 = builder.MakeInput<uint8_t>({16, 32, 1, 1}, static_cast<uint8_t>(0), static_cast<uint8_t>(127));

    auto* mul_dq2_output = builder.MakeIntermediate();
    auto* mul_input2 = builder.MakeInitializer<uint8_t>({16, 1, 1, 1}, static_cast<uint8_t>(0),
                                                        static_cast<uint8_t>(127));
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
                                     1,                                                       // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13,     // opset
                                     // Need tolerance of 0.413% of output range after QNN SDK 2.17
                                     QDQTolerance(0.00413f));

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                                     TestInputDef<float>({1, 1, 3, 3, 3}, true, -10.0f, 10.0f),  // Random static input
                                     TestInputDef<float>({1}, false, {2.0f}),                    // Dynamic bias
                                     {1, 1, 1},                                                  // Strides
                                     {0, 0, 0, 0, 0, 0},                                         // Pads
                                     {1, 1, 1},                                                  // Dilations
                                     1,                                                          // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13,     // opset
                                     // Need tolerance of 0.413% of output range after QNN SDK 2.17
                                     QDQTolerance(0.00413f));
}

// Test per-channel QDQ Conv. in0: u8, in1 (weight): s8, in2 (bias): s32, out: u8
TEST_F(QnnHTPBackendTests, ConvU8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// Test per-channel QDQ Conv with INT4 weights. in0: u16, in1 (weight): s4, in2 (bias): s32, out: u8
TEST_F(QnnHTPBackendTests, ConvU16S4S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint16_t, Int4x2>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               false,  // use_qdq_contrib_ops
                                               21);    // opset
}

// Test per-channel QDQ Conv with INT4 weights and no bias.
// in0: u16, in1 (weight): s4, out: u8
// Tests bug in QNN SDK 2.25 when validating Conv without a bias (QNN EP adds a dummy bias).
TEST_F(QnnHTPBackendTests, ConvU16S4_PerChannel_NoBias) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));

  RunHTPConvOpPerChannelTest<uint16_t, Int4x2>("Conv",
                                               input_def,
                                               weight_def,
                                               TestInputDef<float>(),
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               false,  // use_qdq_contrib_ops
                                               21);    // opset
}

// Test per-channel QDQ Conv with uint16 input[0], uint8 weights, and no bias.
// in0: u16, in1 (weight): s4, out: u8
// Tests bug in QNN SDK 2.25 when validating Conv without a bias (QNN EP adds a dummy bias).
TEST_F(QnnHTPBackendTests, ConvU16U8_PerTensor_NoBias) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));

  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      input_def,
                                      weight_def,
                                      TestInputDef<float>(),
                                      {1, 1},        // Strides
                                      {0, 0, 0, 0},  // Pads
                                      {1, 1},        // Dilations
                                      1,             // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      false,  // use_qdq_contrib_ops
                                      21);    // opset
}

TEST_F(QnnHTPBackendTests, ConvU16S4_PerChannel_NoBias_LargeINT4Weight) {
  std::vector<int64_t> input_shape = {1, 3072, 1, 512};
  std::vector<int64_t> weight_shape = {9216, 3072, 1, 1};
  std::vector<float> input_data(TensorShape(input_shape).Size(), 1.1f);
  input_data[0] = 2.2f;
  std::vector<float> weight_data(TensorShape(weight_shape).Size(), -0.1f);
  for (size_t c = 0; c < static_cast<size_t>(weight_shape[0]); c++) {
    size_t i = c * 3072;
    weight_data[i] = 0.1f;
  }

  TestInputDef<float> input_def(input_shape, false, input_data);
  TestInputDef<float> weight_def(weight_shape, true, weight_data);

#if 0
  RunHTPConvOpPerChannelTest<uint16_t, Int4x2>("Conv",
                                               input_def,
                                               weight_def,
                                               TestInputDef<float>(),
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               false,  // use_qdq_contrib_ops
                                               21);    // opset
#else
  // This test code is here to more directly test ONLY session initialization time with large INT4 weights.
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto qdq_model_fn = BuildQDQPerChannelConvTestCase<uint16_t, Int4x2>("Conv",
                                                                       input_def,
                                                                       weight_def,
                                                                       TestInputDef<float>(),
                                                                       0,             // weight quant axis
                                                                       {1, 1},        // Strides
                                                                       {0, 0, 0, 0},  // Pads
                                                                       {1, 1},        // Dilations
                                                                       1,             // default group
                                                                       "NOTSET",
                                                                       false);  // use_qdq_contrib_ops

  // Create QDQ model and serialize it to a string.
  auto& logging_manager = DefaultLoggingManager();
  const std::unordered_map<std::string, int> domain_to_version = {{"", 21}, {kMSDomain, 1}};
  onnxruntime::Model qdq_model("qdq_model", false, ModelMetaData(), PathString(),
                               IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                               logging_manager.DefaultLogger());

  std::vector<QuantParams<uint16_t>> output_qparams(1);
  output_qparams[0].scale = 0.0051529f;
  output_qparams[0].zero_point = 65535;
  ModelTestBuilder qdq_helper(qdq_model.MainGraph());
  std::string qdq_model_data;
  qdq_model_fn(qdq_helper, output_qparams);
  qdq_helper.SetGraphOutputs();
  ASSERT_STATUS_OK(qdq_model.MainGraph().Resolve());
  qdq_model.ToProto().SerializeToString(&qdq_model_data);

  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  auto qnn_ep = QnnExecutionProviderWithOptions(provider_options, &so);
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(qnn_ep)));
  ASSERT_STATUS_OK(session_object.Load(qdq_model_data.data(), static_cast<int>(qdq_model_data.size())));
  ASSERT_STATUS_OK(session_object.Initialize());
#endif
}

// Test fusion of DQs -> Conv -> Relu/Clip -> Q.
// User per-tensor quantization.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_ReluClipFusion) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  // DQs -> Conv (w/ bias) -> Relu -> Q
  OutputActivationInfo relu_info = {"Relu", {}};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     input_def,
                                     weight_def,
                                     bias_def,
                                     {1, 1},        // Strides
                                     {0, 0, 0, 0},  // Pads
                                     {1, 1},        // Dilations
                                     1,             // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     21,     // opset
                                     QDQTolerance(),
                                     relu_info);

  // DQs -> Conv (NO bias) -> Relu -> Q
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     input_def,
                                     weight_def,
                                     TestInputDef<float>(),
                                     {1, 1},        // Strides
                                     {0, 0, 0, 0},  // Pads
                                     {1, 1},        // Dilations
                                     1,             // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     21,     // opset
                                     QDQTolerance(),
                                     relu_info);

  // DQs -> Conv (w/ bias) -> Clip -> Q
  // Opset 6 Clip uses attributes for min/max
  OutputActivationInfo clip_info = {"Clip", {0.0f, 2.0f}};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     input_def,
                                     weight_def,
                                     bias_def,
                                     {1, 1},        // Strides
                                     {0, 0, 0, 0},  // Pads
                                     {1, 1},        // Dilations
                                     1,             // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     19,     // opset
                                     QDQTolerance(),
                                     clip_info);

  // DQs -> Conv (NO bias) -> Clip -> Q
  OutputActivationInfo clip_info_2 = {"Clip", {-6.0f, 6.0f}};
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      input_def,
                                      weight_def,
                                      TestInputDef<float>(),
                                      {1, 1},        // Strides
                                      {0, 0, 0, 0},  // Pads
                                      {1, 1},        // Dilations
                                      1,             // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      false,  // use_qdq_contrib_ops
                                      21,     // opset
                                      QDQTolerance(),
                                      clip_info_2);
}

// Test fusion of DQs -> Conv -> Relu/Clip -> Q.
// User per-channel quantization.
TEST_F(QnnHTPBackendTests, ConvS8S8S32_PerChannel_ReluClipFusion) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  // DQs -> Conv (w/ bias) -> Relu -> Q
  OutputActivationInfo relu_info = {"Relu", {}};
  RunHTPConvOpPerChannelTest<int8_t, int8_t>("Conv",
                                             input_def,
                                             weight_def,
                                             bias_def,
                                             0,             // weight quant axis
                                             {1, 1},        // Strides
                                             {0, 0, 0, 0},  // Pads
                                             {1, 1},        // Dilations
                                             1,             // default group
                                             "NOTSET",
                                             ExpectedEPNodeAssignment::All,
                                             false,  // use_qdq_contrib_ops
                                             21,     // opset
                                             QDQTolerance(),
                                             relu_info);

  // DQs -> Conv (w/ bias) -> Clip -> Q
  OutputActivationInfo clip_info = {"Clip", {0.0f, 6.0f}};
  RunHTPConvOpPerChannelTest<int8_t, int8_t>("Conv",
                                             input_def,
                                             weight_def,
                                             bias_def,
                                             0,             // weight quant axis
                                             {1, 1},        // Strides
                                             {0, 0, 0, 0},  // Pads
                                             {1, 1},        // Dilations
                                             1,             // default group
                                             "NOTSET",
                                             ExpectedEPNodeAssignment::All,
                                             false,  // use_qdq_contrib_ops
                                             21,     // opset
                                             QDQTolerance(),
                                             clip_info);
}

// Test per-channel QDQ Conv with INT4 weights and a negative weight quantization axis that still points to dimension 0.
TEST_F(QnnHTPBackendTests, ConvU16S4S32_PerChannel_NegativeWeightQuantAxis) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, Int4x2>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              -4,            // negative weight quant axis (same as 0)
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              21);    // opset
}

// Test per-channel QDQ Conv with INT4 weights. in0: u16, in1 (weight): s4, in2 (bias): s32, out: u8
// TODO(adrianlizarraga): Investigate inaccuracy for QNN EP.
//
// Output values for all EPs:
// CPU EP (f32 model): 25.143 21.554 17.964 10.785 7.195 3.605  -3.574  -7.164  -10.753
// CPU EP (qdq model): 24.670 21.103 17.536 10.254 6.689 2.972  -4.161  -7.728  -10.700
// QNN EP (qdq model): 27.186 27.186 27.186 21.541 6.685 -8.022 -10.548 -10.548 -10.548
TEST_F(QnnHTPBackendTests, ConvU16S4S32_PerChannel_AccuracyIssue) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  // Wrote out input data explicitly for easier reproduction.
  // std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size());
  std::vector<float> input_data = {-10.000f, -9.355f, -8.710f, -8.065f, -7.419f, -6.774f, -6.129f, -5.484f, -4.839f,
                                   -4.194f, -3.548f, -2.903f, -2.258f, -1.613f, -0.968f, -0.323f, 0.323f, 0.968f,
                                   1.613f, 2.258f, 2.903f, 3.548f, 4.194f, 4.839f, 5.484f, 6.129f, 6.774f,
                                   7.419f, 8.065f, 8.710f, 9.355f, 10.000f};

  // std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 1.0f, TensorShape(weight_shape).Size());
  std::vector<float> weight_data = {-1.000f, -0.913f, -0.826f, -0.739f, -0.652f, -0.565f, -0.478f, -0.391f, -0.304f,
                                    -0.217f, -0.130f, -0.043f, 0.043f, 0.130f, 0.217f, 0.304f, 0.391f, 0.478f,
                                    0.565f, 0.652f, 0.739f, 0.826f, 0.913f, 1.000f};

  // std::vector<float> bias_data = GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size());
  std::vector<float> bias_data = {-1.000f, 0.000f, 1.000f};

  TestInputDef<float> input_def(input_shape, false, input_data);
  TestInputDef<float> weight_def(weight_shape, true, weight_data);
  TestInputDef<float> bias_def(bias_shape, true, bias_data);

  RunHTPConvOpPerChannelTest<uint8_t, Int4x2>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              21,     // opset
                                              QDQTolerance(0.005f));
}

// Test per-channel QDQ Conv is rejected with weight axis != 0
TEST_F(QnnHTPBackendTests, Conv_PerChannel_UnsupportedAxis) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 3, 3};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              2,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::None,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// Test per-channel QDQ Conv. in0: u8, in1 (weight): s8, in2 (bias): s32, out: u8
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::QNN_Conv3d_w_scale
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x1a preparation failed with err:-1
// QnnDsp <E> "Conv" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// onnxruntime::qnn::QnnModel::FinalizeGraphs] Failed to finalize QNN graph.
TEST_F(QnnHTPBackendTests, DISABLED_Conv3D_U8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,                   // weight quant axis
                                              {1, 1, 1},           // Strides
                                              {0, 0, 0, 0, 0, 0},  // Pads
                                              {1, 1, 1},           // Dilations
                                              1,                   // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,
                                              13);
}

// Test per-channel QDQ Conv that maps to QNN's DepthwiseConv2d (input_chans == output_chans == group).
// in0: u8, in1 (weight): s8, in2 (bias): s32, out: u8
TEST_F(QnnHTPBackendTests, ConvDepthwiseU8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};   // (N, C, H, W)
  std::vector<int64_t> weight_shape = {2, 1, 2, 2};  // (C, M/group, kH, kW)
  std::vector<int64_t> bias_shape = {2};             // (M)

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              2,             // group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// Conv3D per-channel
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::QNN_Conv3d_w_scale
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x1a preparation failed with err:-1
// QnnDsp <E> "Conv" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// onnxruntime::qnn::QnnModel::FinalizeGraphs] Failed to finalize QNN graph.
TEST_F(QnnHTPBackendTests, DISABLED_Conv3D_U8S8S32_PerChannel2) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {2, 1, 2, 2, 2};
  std::vector<int64_t> bias_shape = {2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,                   // weight quant axis
                                              {1, 1, 1},           // Strides
                                              {0, 0, 0, 0, 0, 0},  // Pads
                                              {1, 1, 1},           // Dilations
                                              2,                   // group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,
                                              13);
}

// Test per-channel QDQ ConvTranspose. in0: u8, in1 (weight): s8, in2 (bias): s32, out: u8
TEST_F(QnnHTPBackendTests, ConvTransposeU8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("ConvTranspose",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              1,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// Test per-channel QDQ ConvTranspose is unsupported with weight axis != 1.
TEST_F(QnnHTPBackendTests, ConvTranspose_PerChannel_UnsupportedAxis) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 3, 3};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("ConvTranspose",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              2,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::None,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// ConvTranspose3D per-channel
// Disable it for 2.21 since it failed, re-enabled it for 2.22
TEST_F(QnnHTPBackendTests, DISABLED_ConvTranspose3D_U8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("ConvTranspose",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              1,                   // weight quant axis
                                              {1, 1, 1},           // Strides
                                              {0, 0, 0, 0, 0, 0},  // Pads
                                              {1, 1, 1},           // Dilations
                                              1,                   // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,
                                              13);
}

// Test per-channel QDQ Conv. in0: u16, in1 (weight): s8, in2 (bias): s32, out: u16
TEST_F(QnnHTPBackendTests, ConvU16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,  // use_qdq_contrib_ops
                                               13);   // opset
}

// Conv3D per-channel
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::QNN_Conv3d_w_scale
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x1a preparation failed with err:-1
// QnnDsp <E> "Conv" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// onnxruntime::qnn::QnnModel::FinalizeGraphs] Failed to finalize QNN graph.
TEST_F(QnnHTPBackendTests, DISABLED_Conv3D_U16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,                   // weight quant axis
                                               {1, 1, 1},           // Strides
                                               {0, 0, 0, 0, 0, 0},  // Pads
                                               {1, 1, 1},           // Dilations
                                               1,                   // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,
                                               13);
}

// Test per-channel QDQ ConvTranspose. in0: u16, in1 (weight): s8, in2 (bias): s32, out: u16
TEST_F(QnnHTPBackendTests, ConvTransposeU16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("ConvTranspose",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               1,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,  // use_qdq_contrib_ops
                                               13);   // opset
}

// Disable it for 2.21, re-enable it for 2.22
TEST_F(QnnHTPBackendTests, DISABLED_ConvTranspose3D_U16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("ConvTranspose",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               1,                   // weight quant axis
                                               {1, 1, 1},           // Strides
                                               {0, 0, 0, 0, 0, 0},  // Pads
                                               {1, 1, 1},           // Dilations
                                               1,                   // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,
                                               13);
}

// Test per-channel QDQ Conv that maps to QNN's DepthwiseConv2d (input_chans == output_chans == group).
// in0: u16, in1 (weight): s8, in2 (bias): s32, out: u16
TEST_F(QnnHTPBackendTests, ConvDepthwiseU16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};   // (N, C, H, W)
  std::vector<int64_t> weight_shape = {2, 1, 2, 2};  // (C, M/group, kH, kW)
  std::vector<int64_t> bias_shape = {2};             // (M)

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               2,             // group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,  // use_qdq_contrib_ops
                                               13);   // opset
}

// Test per-channel QDQ Conv3D
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::QNN_Conv3d_w_scale
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x1a preparation failed with err:-1
// QnnDsp <E> "Conv" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// onnxruntime::qnn::QnnModel::FinalizeGraphs] Failed to finalize QNN graph.
TEST_F(QnnHTPBackendTests, DISABLED_Conv3D_U16S8S32_PerChannel2) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {2, 1, 2, 2, 2};
  std::vector<int64_t> bias_shape = {2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, TensorShape(input_shape).Size()));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, TensorShape(weight_shape).Size()));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, TensorShape(bias_shape).Size()));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,                   // weight quant axis
                                               {1, 1, 1},           // Strides
                                               {0, 0, 0, 0, 0, 0},  // Pads
                                               {1, 1, 1},           // Dilations
                                               2,                   // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,
                                               13);
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
                                      1,                                           // default group
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
                                      1,                                           // default group
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
                                      1,                                           // default group
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
                                      1,                                                      // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with static bias.
// Uses QNN's DepthwiseConv2d operator.
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
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 125);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 27);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5, 5}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 1, 3, 3, 3}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>({1}, true, {2.0f}),                      // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with static bias.
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
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 150);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 36);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 2, 3, 3, 2}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>({1}, true, {2.0f}),                      // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with dynamic bias.
// Uses QNN's DepthwiseConv2d operator.
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
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 75);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 27);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 1, 3, 3, 3}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>({1}, false, {2.0f}),                     // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with dynamic bias.
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
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 150);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 36);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 2, 3, 3, 2}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>({1}, false, {2.0f}),                     // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with no bias
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
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 150);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 36);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 2, 3, 3, 2}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>(),                                       // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with no bias
// Uses QNN's DepthwiseConv2d operator.
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
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 75);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 18);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 1, 3, 3, 2}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>(),                                       // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
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
                                     1,                                                          // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 3, 32, 32, 32}, false, -10.0f, 10.0f),  // Input
                                     TestInputDef<float>({1, 3, 4, 4, 4}, false, -10.0f, 10.0f),     // Weights
                                     TestInputDef<float>(),                                          // Bias
                                     {1, 1, 1},                                                      // Strides
                                     {0, 0, 0, 0, 0, 0},                                             // Pads
                                     {1, 1, 1},                                                      // Dilations
                                     1,                                                              // default group
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
                                     1,                                                          // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// QNN op validation crash. Run correctly if by pass the QNN op validation
// Exception from backendValidateOpConfig:
// Exception thrown at 0x00007FFF9E0128B0 (QnnHtpPrepare.dll) in onnxruntime_test_all.exe:
// 0xC0000005: Access violation reading location 0x7079745F656C706D.
TEST_F(QnnHTPBackendTests, DISABLED_ConvTranspose3D_U8U8S32_DynamicWeight_NoBias) {
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 3, 32, 32, 32}, false, -10.0f, 10.0f),  // Input
                                     TestInputDef<float>({3, 1, 4, 4, 4}, false, -10.0f, 10.0f),     // Weights
                                     TestInputDef<float>(),                                          // Bias
                                     {1, 1, 1},                                                      // Strides
                                     {0, 0, 0, 0, 0, 0},                                             // Pads
                                     {1, 1, 1},                                                      // Dilations
                                     1,                                                              // default group
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
                                     1,                                                       // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13,     // opset
                                     // Need tolerance of 0.413% of output range after QNN SDK 2.17
                                     QDQTolerance(0.00413f));

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                                     TestInputDef<float>({1, 1, 3, 3, 3}, true, -10.0f, 10.0f),  // Random static weight
                                     TestInputDef<float>({1}, true, {2.0f}),                     // Initializer bias
                                     {1, 1, 1},                                                  // Strides
                                     {0, 0, 0, 0, 0, 0},                                         // Pads
                                     {1, 1, 1},                                                  // Dilations
                                     1,                                                          // default group
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
                                     1,                                                           // default group
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
                                     1,                                                           // default group
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
                                     1,                                                    // default group
                                     "SAME_UPPER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {1, 1, 1},                                               // strides
                                     {},                                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // default group
                                     "SAME_UPPER",                                            // auto_pad
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
                                     1,                                                           // default group
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
                                     1,                                                           // default group
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
                                     1,                                                    // default group
                                     "SAME_LOWER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {1, 1, 1},                                               // strides
                                     {},                                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // default group
                                     "SAME_LOWER",                                            // auto_pad
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
                                     1,                                                    // default group
                                     "SAME_LOWER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);

  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {1, 1, 1},                                               // strides
                                     {},                                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // default group
                                     "SAME_LOWER",                                            // auto_pad
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
                                     1,                                                           // default group
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
                                     1,                                                           // default group
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
                                     1,  // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13,     // opset
                                     // Need tolerance of 0.76% of output range after QNN SDK 2.19.2
                                     QDQTolerance(0.0076f));
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
                                     1,  // default group
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
                                     1,                                                         // default group
                                     "NOTSET",                                                  // auto_pad
                                     ExpectedEPNodeAssignment::All);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/providers/qnn/qnn_test_utils.h"

#include "core/graph/onnx_protobuf.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that creates a graph with MatMul operator.
static GetTestModelFn BuildMatMulOpTestCase(const TestInputDef<float>& input1_def,
                                            const TestInputDef<float>& input2_def) {
  return [input1_def, input2_def](ModelTestBuilder& builder) {
    NodeArg* input1 = MakeTestInput(builder, input1_def);
    NodeArg* input2 = MakeTestInput(builder, input2_def);
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("MatMul", {input1, input2}, {output});
  };
}

static void RunMatMulOpTest(bool is_htp_backend, const std::vector<int64_t>& shape_0,
                            const std::vector<int64_t>& shape_1, bool is_initializer_0, bool is_initializer_1,
                            ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                            int opset = 18, float f32_abs_err = 1e-4f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = is_htp_backend ? "htp" : "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildMatMulOpTestCase(
                      TestInputDef<float>(shape_0, is_initializer_0, GetSequentialFloatData(shape_0, 0.01f, 0.02f)),
                      TestInputDef<float>(shape_1, is_initializer_1, GetSequentialFloatData(shape_1, 0.02f, 0.02f))),
                  provider_options, opset, expected_ep_assignment, f32_abs_err);
}

// Returns a function that creates a graph with a QDQ MatMul operator.
template <typename Input0QType, typename Input1QType, typename OutputQType>
static GetTestQDQModelFn<OutputQType> BuildMatMulOpQDQTestCase(const TestInputDef<float>& input0_def,
                                                               const TestInputDef<float>& input1_def,
                                                               bool use_contrib_qdq) {
  return [input0_def, input1_def, use_contrib_qdq](ModelTestBuilder& builder,
                                                   std::vector<QuantParams<OutputQType>>& output_qparams) {
    // input1 -> Q -> DQ ->
    NodeArg* input0 = MakeTestInput(builder, input0_def);
    QuantParams<Input0QType> input0_qparams = GetTestInputQuantParams<Input0QType>(input0_def);
    auto* input0_qdq =
        AddQDQNodePair<Input0QType>(builder, input0, input0_qparams.scale, input0_qparams.zero_point, use_contrib_qdq);
    // input1 -> Q -> DQ ->
    NodeArg* input1 = MakeTestInput(builder, input1_def);
    QuantParams<Input1QType> input1_qparams = GetTestInputQuantParams<Input1QType>(input1_def);
    auto* input1_qdq =
        AddQDQNodePair<Input1QType>(builder, input1, input1_qparams.scale, input1_qparams.zero_point, use_contrib_qdq);

    // MatMul
    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("MatMul", {input0_qdq, input1_qdq}, {op_output});

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(builder, op_output, output_qparams[0].scale,
                                                       output_qparams[0].zero_point, use_contrib_qdq);
  };
}

template <typename Input0QType, typename WeightQType, typename OutputQType>
static GetTestQDQModelFn<OutputQType> BuildQDQPerChannelMatMulTestCase(const TestInputDef<float>& input_def,
                                                                       const TestInputDef<float>& weights_def,
                                                                       int64_t weight_quant_axis,
                                                                       bool use_contrib_qdq = false) {
  return [input_def, weights_def, weight_quant_axis, use_contrib_qdq](
             ModelTestBuilder& builder, std::vector<QuantParams<OutputQType>>& output_qparams) {
    std::vector<NodeArg*> matmul_inputs;

    // input -> Q/DQ ->
    auto* input = MakeTestInput(builder, input_def);
    QuantParams<Input0QType> input_qparams = GetTestInputQuantParams<Input0QType>(input_def);
    auto* input_qdq =
        AddQDQNodePair<Input0QType>(builder, input, input_qparams.scale, input_qparams.zero_point, use_contrib_qdq);
    matmul_inputs.push_back(input_qdq);

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
    QuantizeValues<float, WeightQType>(weights_def.GetRawData(), quantized_weights, weights_shape, weight_scales,
                                       weight_zero_points, pos_weight_quant_axis);

    NodeArg* weights_initializer = builder.MakeInitializer<WeightQType>(weights_def.GetShape(), quantized_weights);
    NodeArg* weights_dq = builder.MakeIntermediate();
    Node& weights_dq_node = builder.AddDequantizeLinearNode<WeightQType>(
        weights_initializer, weight_scales, weight_zero_points, weights_dq, nullptr, use_contrib_qdq);
    weights_dq_node.AddAttribute("axis", weight_quant_axis);
    matmul_inputs.push_back(weights_dq);

    auto* matmul_output = builder.MakeIntermediate();
    builder.AddNode("MatMul", matmul_inputs, {matmul_output});

    AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(builder, matmul_output, output_qparams[0].scale,
                                                       output_qparams[0].zero_point, use_contrib_qdq);
  };
}

template <typename Input0QType, typename Input1QType, typename OutputQType>
static void RunQDQMatMulOpTest(const std::vector<int64_t>& shape_0, const std::vector<int64_t>& shape_1,
                               bool is_initializer_0, bool is_initializer_1,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               int opset = 21, bool use_contrib_qdq = false,
                               QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestInputDef<float> input0_def(
      shape_0, is_initializer_0,
      GetFloatDataInRange(-0.1f, 0.1f,
                          static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                              std::multiplies<int64_t>()))));
  TestInputDef<float> input1_def(
      shape_1, is_initializer_1,
      GetFloatDataInRange(-0.1f, 0.1f,
                          static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                              std::multiplies<int64_t>()))));

  TestQDQModelAccuracy(
      BuildMatMulOpTestCase(input0_def, input1_def),
      BuildMatMulOpQDQTestCase<Input0QType, Input1QType, OutputQType>(input0_def, input1_def, use_contrib_qdq),
      provider_options, opset, expected_ep_assignment, tolerance);
}

template <typename InputQType, typename WeightQType, typename OutputQType>
static void RunQDQPerChannelMatMulOpTest(
    const std::vector<int64_t>& shape_input, const std::vector<int64_t>& shape_weight, int64_t weight_quant_axis,
    QDQTolerance tolerance = QDQTolerance(),
    ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All, int opset = 21,
    bool use_contrib_qdq = false, bool enable_fp16_precision = true) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  if (enable_fp16_precision) {
    provider_options["enable_htp_fp16_precision"] = "1";
  } else {
    provider_options["enable_htp_fp16_precision"] = "0";
  }

  TestInputDef<float> input_def(
      shape_input, false,
      GetFloatDataInRange(-0.1f, 0.1f,
                          static_cast<size_t>(std::accumulate(shape_input.begin(), shape_input.end(),
                                                              static_cast<int64_t>(1), std::multiplies<int64_t>()))));
  TestInputDef<float> weight_def(
      shape_weight, true,
      GetFloatDataInRange(-0.1f, 0.1f,
                          static_cast<size_t>(std::accumulate(shape_weight.begin(), shape_weight.end(),
                                                              static_cast<int64_t>(1), std::multiplies<int64_t>()))));

  TestQDQModelAccuracy(BuildMatMulOpTestCase(input_def, weight_def),
                       BuildQDQPerChannelMatMulTestCase<InputQType, WeightQType, OutputQType>(
                           input_def, weight_def, weight_quant_axis, use_contrib_qdq),
                       provider_options, opset, expected_ep_assignment, tolerance);
}

//
// CPU tests:
//
TEST_F(QnnCPUBackendTests, MatMulOp) {
  // RunMatMulOpTest(is_htp_backend, shape_0, shape_1, is_initializer_0, is_initializer_1)
  RunMatMulOpTest(false, {2, 3}, {3, 2}, false, false);
  RunMatMulOpTest(false, {2, 3}, {3, 2}, false, true);
  RunMatMulOpTest(false, {2, 3}, {3, 2}, true, false);
  RunMatMulOpTest(false, {2, 3}, {3, 2}, true, true);  // constant folding
  RunMatMulOpTest(false, {2, 3}, {2, 3, 2}, false, false);
  RunMatMulOpTest(false, {3, 3, 3}, {3, 2}, true, false);
  RunMatMulOpTest(false, {2, 3, 3, 3}, {3, 2}, false, true);
  RunMatMulOpTest(false, {2, 3, 3, 3}, {2, 3, 3, 2}, false, true);

#if defined(__linux__)
  // TODO: This fails on Linux (HTP emulation). Works on Windows ARM64.
  // Expected: contains 24 values, where each value and its corresponding value in 16-byte object <18-00 00-00 00-00 00-00 00-29 4E-53 A8-55 00-00> are an almost-equal pair
  // Actual: 16-byte object <18-00 00-00 00-00 00-00 80-28 3E-53 A8-55 00-00>, where the value pair (0.0285999943, 0) at index #12 don't match, which is -0.0286 from 0.0286
#else
  RunMatMulOpTest(false, {2, 1, 2, 3}, {3, 3, 2}, false, false);
#endif
  RunMatMulOpTest(false, {3}, {3}, false, false);
  RunMatMulOpTest(false, {3}, {3}, false, true);
  RunMatMulOpTest(false, {3}, {3}, true, false);
  RunMatMulOpTest(false, {3}, {3, 2}, false, false);
  RunMatMulOpTest(false, {3}, {3, 2}, false, true);
  RunMatMulOpTest(false, {3}, {3, 3, 2}, true, false);
  RunMatMulOpTest(false, {2, 3}, {3}, false, false);
  RunMatMulOpTest(false, {2, 3}, {3}, true, false);
  RunMatMulOpTest(false, {2, 3, 3, 3}, {3}, false, false);

  // Failed randomly on Linux
  // Expected: contains 36 values, where each value and its corresponding value in 16-byte object
  // <24-00 00-00 00-00 00-00 40-4A 47-42 4D-56 00-00> are an almost-equal pair
  // Actual: 16-byte object <24-00 00-00 00-00 00-00 80-39 2B-42 4D-56 00-00>, where the value pair (0.104199991, 0)
  // at index #18 don't match, which is -0.1042 from 0.1042
  // RunMatMulOpTest(false, {2, 3, 3, 3}, {3, 2}, true, false);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// HTP tests:
//
// Disable this for now as the QNN HTP backend is not stable on different versions and platforms so it failed randomly.
TEST_F(QnnHTPBackendTests, DISABLED_MatMulOp) {
  // RunMatMulOpTest(is_htp_backend, shape_0, shape_1, is_initializer_0, is_initializer_1, expected_ep_assignment,
  // opset, f32_abs_err)
  RunMatMulOpTest(true, {2, 3}, {3, 2}, false, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3}, {3, 2}, true, true);  // constant folding
  RunMatMulOpTest(true, {2, 3}, {2, 3, 2}, false, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3, 3, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3, 3, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3, 3, 3}, {2, 3, 3, 2}, false, true, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 1, 2, 3}, {3, 3, 2}, false, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {3}, {3}, false, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {3}, {3}, false, true, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {3}, {3}, true, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {3}, {3, 2}, false, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {3}, {3, 3, 2}, true, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3}, {3}, false, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3}, {3}, true, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
  RunMatMulOpTest(true, {2, 3, 3, 3}, {3}, false, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);

  // Failed randomly on Linux
  // Expected: contains 18 values, where each value and its corresponding value in 16-byte object
  // <12-00 00-00 00-00 00-00 40-3D CC-A5 5A-7A 00-00> are an almost-equal pair
  // Actual: 16-byte object <12-00 00-00 00-00 00-00 80-E8 CF-8F 5B-7A 00-00>, where the value pair
  // (0.0393999927, 98304.0078) at index #6 don't match, which is 98304 from 0.0394
  // RunMatMulOpTest(true, {3, 3, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, 18, 1e-2f);
}

TEST_F(QnnHTPBackendTests, MatMulOp_QDQ) {
  // UINT8
  // RunQDQMatMulOpTest(shape_0, shape_1, is_initializer_0, is_initializer_1, expected_ep_assignment, opset,
  // use_contrib_qdq)
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3, 2}, false, false);
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, 21,
                                                false, QDQTolerance(0.008f));
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 2, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, 18,
                                                true);
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 1, 3, 3}, {3, 3, 2}, false, true);
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({3}, {3}, false, false);
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3}, true, false);

  // UINT16, UINT8
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({2, 3}, {3, 2}, false, false);
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({2, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, 18, true);
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({2, 3, 3, 3}, {3, 2}, true, false);
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({3}, {3, 2}, false, true);
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({2, 3, 3, 3}, {3}, false, false);

  // UINT16, per-channel signed 4-bit weight
  // RunQDQPerChannelMatMulOpTest(shape_input, shape_weight, weight_quant_axis, tolerance, expected_ep_assignment,
  // opset, use_contrib_qdq, enable_fp16_precision)
  RunQDQPerChannelMatMulOpTest<uint16_t, Int4x2, uint16_t>({2, 3}, {3, 2}, 1);
  RunQDQPerChannelMatMulOpTest<uint16_t, Int4x2, uint16_t>({2, 3, 3, 3}, {3, 2}, -1, QDQTolerance(),
                                                           ExpectedEPNodeAssignment::All, 18, true);

  // UINT16, per-channel INT8 weight
  RunQDQPerChannelMatMulOpTest<uint16_t, int8_t, uint16_t>({2, 3}, {3, 2}, 1, QDQTolerance(),
                                                           ExpectedEPNodeAssignment::All, 21, false, false);
  RunQDQPerChannelMatMulOpTest<uint16_t, int8_t, uint16_t>({2, 3, 3}, {3}, -1, QDQTolerance(0.005f));
}

// Tests MatMul with two uint16 (quantized) inputs that are both dynamic.
// This exercises a workaround in QNN EP that inserts a QNN Convert op before input[1] (converts from uint16 to uint8).
// This workaround prevents a validation error for this specific MatMul configuration.
// Got specific shapes and input ranges (quant params) from customer model.
TEST_F(QnnHTPBackendTests, MatMulOp_QDQ_Regression_uint16_dynamic_inputs) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Test with rank 4 inputs
  {
    std::vector<int64_t> shape_0 = {1, 12, 512, 96};
    TestInputDef<float> input0_def(
        {1, 12, 512, 96}, false,
        GetFloatDataInRange(-5.087f, 4.992f,
                            static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));
    std::vector<int64_t> shape_1 = {1, 12, 96, 512};
    TestInputDef<float> input1_def(
        shape_1, false,
        GetFloatDataInRange(-6.772f, 7.258f,
                            static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));

    TestQDQModelAccuracy(
        BuildMatMulOpTestCase(input0_def, input1_def),
        BuildMatMulOpQDQTestCase<uint16_t, uint16_t, uint16_t>(input0_def, input1_def, false),
        provider_options, 21, ExpectedEPNodeAssignment::All, QDQTolerance());
  }

  // Test with input[1] as rank 1
  {
    std::vector<int64_t> shape_0 = {1, 12, 512, 96};
    TestInputDef<float> input0_def(
        {1, 12, 512, 96}, false,
        GetFloatDataInRange(-5.087f, 4.992f,
                            static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));
    std::vector<int64_t> shape_1 = {96};
    TestInputDef<float> input1_def(
        shape_1, false,
        GetFloatDataInRange(-6.772f, 7.258f,
                            static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));

    TestQDQModelAccuracy(
        BuildMatMulOpTestCase(input0_def, input1_def),
        BuildMatMulOpQDQTestCase<uint16_t, uint16_t, uint16_t>(input0_def, input1_def, false),
        provider_options, 21, ExpectedEPNodeAssignment::All, QDQTolerance());
  }
}

#ifndef __linux__
// Tests MatMul with two uint16 (quantized) inputs with weight as static.
// This exercises a workaround in QNN EP that inserts a QNN Convert op before input[1] (converts from uint16 to sint16).
// This workaround prevents a validation error for this specific MatMul configuration.
// Got specific shapes and input ranges (quant params) from customer model.
TEST_F(QnnHTPBackendTests, MatMulOp_QDQ_Regression_uint16_static_weight) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Test with rank 4 inputs
  {
    std::vector<int64_t> shape_0 = {1, 12, 512, 96};
    TestInputDef<float> input0_def(
        {1, 12, 512, 96}, false,
        GetFloatDataInRange(-5.087f, 4.992f,
                            static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));
    std::vector<int64_t> shape_1 = {1, 12, 96, 512};
    TestInputDef<float> input1_def(
        shape_1, true,
        GetFloatDataInRange(-6.772f, 7.258f,
                            static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));

    TestQDQModelAccuracy(
        BuildMatMulOpTestCase(input0_def, input1_def),
        BuildMatMulOpQDQTestCase<uint16_t, uint16_t, uint16_t>(input0_def, input1_def, false),
        provider_options, 21, ExpectedEPNodeAssignment::All, QDQTolerance());
  }

  // Test with input[1] as rank 1
  {
    std::vector<int64_t> shape_0 = {1, 12, 512, 96};
    TestInputDef<float> input0_def(
        {1, 12, 512, 96}, false,
        GetFloatDataInRange(-5.087f, 4.992f,
                            static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));
    std::vector<int64_t> shape_1 = {96};
    TestInputDef<float> input1_def(
        shape_1, true,
        GetFloatDataInRange(-6.772f, 7.258f,
                            static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));

    TestQDQModelAccuracy(
        BuildMatMulOpTestCase(input0_def, input1_def),
        BuildMatMulOpQDQTestCase<uint16_t, uint16_t, uint16_t>(input0_def, input1_def, false),
        provider_options, 21, ExpectedEPNodeAssignment::All, QDQTolerance());
  }
}
#endif

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"

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
    auto* input0_qdq = AddQDQNodePair<Input0QType>(builder, input0, input0_qparams.scale, input0_qparams.zero_point,
                                                   use_contrib_qdq);
    // input1 -> Q -> DQ ->
    NodeArg* input1 = MakeTestInput(builder, input1_def);
    QuantParams<Input1QType> input1_qparams = GetTestInputQuantParams<Input1QType>(input1_def);
    auto* input1_qdq = AddQDQNodePair<Input1QType>(builder, input1, input1_qparams.scale, input1_qparams.zero_point,
                                                   use_contrib_qdq);

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
  return [input_def, weights_def, weight_quant_axis,
          use_contrib_qdq](ModelTestBuilder& builder,
                           std::vector<QuantParams<OutputQType>>& output_qparams) {
    std::vector<NodeArg*> matmul_inputs;

    // input -> Q/DQ ->
    auto* input = MakeTestInput(builder, input_def);
    QuantParams<Input0QType> input_qparams = GetTestInputQuantParams<Input0QType>(input_def);
    auto* input_qdq = AddQDQNodePair<Input0QType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                  use_contrib_qdq);
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
    QuantizeValues<float, WeightQType>(weights_def.GetRawData(), quantized_weights, weights_shape,
                                       weight_scales, weight_zero_points, pos_weight_quant_axis);

    NodeArg* weights_initializer = builder.MakeInitializer<WeightQType>(weights_def.GetShape(), quantized_weights);
    NodeArg* weights_dq = builder.MakeIntermediate();
    Node& weights_dq_node = builder.AddDequantizeLinearNode<WeightQType>(weights_initializer, weight_scales,
                                                                         weight_zero_points, weights_dq,
                                                                         nullptr, use_contrib_qdq);
    weights_dq_node.AddAttribute("axis", weight_quant_axis);
    matmul_inputs.push_back(weights_dq);

    auto* matmul_output = builder.MakeIntermediate();
    builder.AddNode("MatMul", matmul_inputs, {matmul_output});

    AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(builder, matmul_output, output_qparams[0].scale,
                                                       output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Runs a QDQ per-channel MatMul model on the QNN HTP backend. Checks the graph node assignment, and that the
// QDQ model is accurate on QNN EP (compared to CPU EP).
template <typename Input0QType, typename WeightQType, typename OutputQType>
static void RunQDQPerChannelMatMulOpOpTest(const TestInputDef<float>& input_def,
                                           const TestInputDef<float>& weights_def,
                                           int64_t weight_quant_axis,
                                           ExpectedEPNodeAssignment expected_ep_assignment,
                                           int opset = 21,
                                           bool use_contrib_qdq = false,
                                           QDQTolerance tolerance = QDQTolerance(),
                                           bool enable_fp16_precision = true) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  if (enable_fp16_precision) {
    provider_options["enable_htp_fp16_precision"] = "1";
  } else {
    provider_options["enable_htp_fp16_precision"] = "0";
  }

  TestQDQModelAccuracy(BuildMatMulOpTestCase(input_def, weights_def),
                       BuildQDQPerChannelMatMulTestCase<Input0QType, WeightQType, OutputQType>(input_def,
                                                                                               weights_def,
                                                                                               weight_quant_axis,
                                                                                               use_contrib_qdq),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

// Runs an MatMul model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunMatMulOpOpTest(const TestInputDef<float>& input1_def,
                              const TestInputDef<float>& input2_def,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              int opset = 13,
                              float f32_abs_err = 1e-4f) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildMatMulOpTestCase(input1_def, input2_def),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  f32_abs_err);
}

// Runs a QDQ MatMul model on the QNN HTP backend. Checks the graph node assignment, and that the
// QDQ model is accurate on QNN EP (compared to CPU EP).
template <typename Input0QType, typename Input1QType, typename OutputQType>
static void RunQDQMatMulOpOpTest(const TestInputDef<float>& input1_def,
                                 const TestInputDef<float>& input2_def,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 18,
                                 bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildMatMulOpTestCase(input1_def, input2_def),
                       BuildMatMulOpQDQTestCase<Input0QType, Input1QType, OutputQType>(input1_def, input2_def,
                                                                                       use_contrib_qdq),
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

//
// CPU tests:
//

// TODO: Crashes during QNN CPU execution (QNN SDK 2.22)
TEST_F(QnnCPUBackendTests, DISABLED_MatMulOp) {
  RunMatMulOpOpTest(TestInputDef<float>({2, 3}, false, {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f}),
                    TestInputDef<float>({3, 2}, false, {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f}),
                    ExpectedEPNodeAssignment::All, 18);
}

// Test MatMul broadcasting
// Failed randomly on Linux
// Value of: expected_tensor.DataAsSpan<float>()
// Expected: contains 896 values, where each value and its corresponding value in 16-byte object
// <80-03 00-00 00-00 00-00 40-B8 53-08 CC-7F 00-00> are an almost-equal pair
// Actual: 16-byte object <80-03 00-00 00-00 00-00 C0-B7 43-08 CC-7F 00-00>, where the value pair
// (-5.19657087, 0) at index #29 don't match, which is 5.19657 from -5.19657
TEST_F(QnnCPUBackendTests, DISABLED_MatMulOp_Broadcast) {
  // Create two matrices with element values in the range [-10.0, 10.0].
  std::vector<float> input_a = GetFloatDataInRange(-10.0f, 10.0f, 28 * 64);
  std::vector<float> input_b = GetFloatDataInRange(-10.0f, 10.0f, 64 * 32);

  RunMatMulOpOpTest(TestInputDef<float>({28, 1, 64}, false, input_a),
                    TestInputDef<float>({64, 32}, false, input_b),
                    ExpectedEPNodeAssignment::All, 18, 0.0004f);
}

// TODO: Crashes during QNN CPU execution (QNN SDK 2.22)
TEST_F(QnnCPUBackendTests, DISABLED_MatMulOp_PaddingAndBroadcast_BLargerThanA) {
  std::vector<int64_t> input0_shape = {2, 3, 2};
  std::vector<int64_t> input1_shape = {3, 2, 2, 1};
  RunMatMulOpOpTest(TestInputDef<float>(input0_shape, false, GetSequentialFloatData(input0_shape)),
                    TestInputDef<float>(input1_shape, false, GetSequentialFloatData(input1_shape)),
                    ExpectedEPNodeAssignment::All, 7);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, MatMulOp_HTP_u8) {
  std::vector<float> input0_data = {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> input1_data = {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f};
  RunQDQMatMulOpOpTest<uint8_t, uint8_t, uint8_t>(TestInputDef<float>({2, 3}, false, input0_data),
                                                  TestInputDef<float>({3, 2}, false, input1_data),
                                                  ExpectedEPNodeAssignment::All, 18);
}

// Test QDQ MatMul with 16-bit act, 8-bit weights (static)
TEST_F(QnnHTPBackendTests, MatMulOp_HTP_A16_W8Static) {
  std::vector<float> input0_data = {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> input1_data = {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f};
  RunQDQMatMulOpOpTest<uint16_t, uint8_t, uint16_t>(TestInputDef<float>({2, 3}, false, input0_data),
                                                    TestInputDef<float>({3, 2}, true, input1_data),
                                                    ExpectedEPNodeAssignment::All,
                                                    18,
                                                    true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ per-channel MatMul with 16-bit act, signed 4-bit weights (static)
TEST_F(QnnHTPBackendTests, MatMulOp_PerChannel_A16_WeightInt4) {
  std::vector<float> input0_data = {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> input1_data = {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f};
  RunQDQPerChannelMatMulOpOpTest<uint16_t, Int4x2, uint16_t>(TestInputDef<float>({1, 1, 2, 3}, false, input0_data),
                                                             TestInputDef<float>({1, 1, 3, 2}, true, input1_data),
                                                             1,  // quantization axis
                                                             ExpectedEPNodeAssignment::All,
                                                             21,
                                                             false);
}

// Test QDQ per-channel MatMul with 16-bit act, unsigned 4-bit weights (static)
TEST_F(QnnHTPBackendTests, MatMulOp_PerChannel_A16_WeightUInt4) {
  std::vector<float> input0_data = {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> input1_data = {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f};
  RunQDQPerChannelMatMulOpOpTest<uint16_t, UInt4x2, uint16_t>(TestInputDef<float>({1, 1, 2, 3}, false, input0_data),
                                                              TestInputDef<float>({1, 1, 3, 2}, true, input1_data),
                                                              1,  // quantization axis
                                                              ExpectedEPNodeAssignment::All,
                                                              21,
                                                              false);
}

// Test QDQ per-channel MatMul with int8 act, int4 weights (static)
// QNN 2.27 regression. Also fails on QNN 2.28.2.
// Failed to finalize QNN graph. Error code: 1002
TEST_F(QnnHTPBackendTests, DISABLED_MatMulOp_PerChannel_AS8_WeightInt4) {
  std::vector<float> input0_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  std::vector<float> input1_data = {-2.0f, -1.0f, -0.5f, 0.0f, 1.0f, 2.0f};
  RunQDQPerChannelMatMulOpOpTest<int8_t, Int4x2, int8_t>(TestInputDef<float>({1, 1, 2, 3}, false, input0_data),
                                                         TestInputDef<float>({1, 1, 3, 2}, true, input1_data),
                                                         1,  // quantization axis
                                                         ExpectedEPNodeAssignment::All,
                                                         21,
                                                         false,
                                                         QDQTolerance(0.007f),
                                                         false);
}

// Test QDQ per-channel MatMul with 16-bit act, int8 weights (static)
TEST_F(QnnHTPBackendTests, MatMulOp_PerChannel_A16_WeightInt8) {
  std::vector<float> input0_data = {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> input1_data = {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f};
  RunQDQPerChannelMatMulOpOpTest<uint16_t, int8_t, uint16_t>(TestInputDef<float>({1, 1, 2, 3}, false, input0_data),
                                                             TestInputDef<float>({1, 1, 3, 2}, true, input1_data),
                                                             1,  // quantization axis
                                                             ExpectedEPNodeAssignment::All,
                                                             21,
                                                             false);
}

// Test QDQ MatMul with uint16 activation uint16 weights, both dynamic
// Inaccuracy detected for output 'output_0', element 1.
// Output quant params: scale=0.0015259021893143654, zero_point=0.
// Expected val: 40
// QNN QDQ val: 39.681087493896484 (err 0.31891250610351562)
// CPU QDQ val: 39.99847412109375 (err 0.00152587890625)
TEST_F(QnnHTPBackendTests, DISABLED_MatMulOp_HTP_A16_W16Dynamic) {
  std::vector<float> input0_data = {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> input1_data = {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f};
  RunQDQMatMulOpOpTest<uint16_t, uint16_t, uint16_t>(TestInputDef<float>({2, 3}, false, input0_data),
                                                     TestInputDef<float>({3, 2}, false, input1_data),
                                                     ExpectedEPNodeAssignment::All,
                                                     18,
                                                     true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ MatMul with uint16 activation uint16 weights, both dynamic
// Inaccuracy detected for output 'output_0', element 1.
// Output quant params: scale=0.71908456087112427, zero_point=1.
// Expected val: 46848.41015625
// QNN QDQ val: 46844.04296875 (err 4.3671875)
// CPU QDQ val: 46848.359375 (err 0.05078125)
TEST_F(QnnHTPBackendTests, DISABLED_MatMulOp_HTP_A16_W16DynamicLarge) {
  std::vector<float> input0_data = GetFloatDataInRange(-10.0f, 10.0f, 12 * 96 * 512);
  std::vector<float> input1_data = GetFloatDataInRange(-10.0f, 10.0f, 12 * 96 * 512);
  RunQDQMatMulOpOpTest<uint16_t, uint16_t, uint16_t>(TestInputDef<float>({1, 12, 96, 512}, false, input0_data),
                                                     TestInputDef<float>({1, 12, 512, 96}, false, input1_data),
                                                     ExpectedEPNodeAssignment::All,
                                                     18,
                                                     true);  // Use com.microsoft Q/DQ ops
}

// Test 16-bit QDQ MatMul with static weights
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.0015259021893143654, zero_point=0.
// Expected val: 98
// QNN QDQ val: 0.65461206436157227 (err 97.345390319824219)
// CPU QDQ val: 98.002593994140625 (err 0.002593994140625)
TEST_F(QnnHTPBackendTests, DISABLED_MatMulOp_HTP_A16_W16) {
  std::vector<float> input0_data = {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> input1_data = {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f};
  RunQDQMatMulOpOpTest<uint16_t, int16_t, uint16_t>(TestInputDef<float>({2, 3}, false, input0_data),
                                                    TestInputDef<float>({3, 2}, true, input1_data),
                                                    ExpectedEPNodeAssignment::All,
                                                    18,
                                                    true);  // Use com.microsoft Q/DQ ops
}

// Test 8-bit QDQ MatMul broadcasting
TEST_F(QnnHTPBackendTests, MatMulOp_Broadcast) {
  RunQDQMatMulOpOpTest<uint8_t, uint8_t, uint8_t>(TestInputDef<float>({28, 1, 64}, false, -10.0f, 10.0f),
                                                  TestInputDef<float>({64, 32}, false, -10.0f, 10.0f),
                                                  ExpectedEPNodeAssignment::All, 18);
}

// Test 16-bit QDQ MatMul broadcasting
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.0028538699261844158, zero_point=6050.
// Expected val: 169.76341247558594
// QNN QDQ val: -16.675161361694336 (err 186.43856811523438)
// CPU QDQ val: 169.762451171875 (err 0.0009613037109375)
TEST_F(QnnHTPBackendTests, DISABLED_MatMulOp_Broadcast_A16_W16) {
  std::vector<float> input_a = GetFloatDataInRange(-10.0f, 10.0f, 28 * 64);
  std::vector<float> input_b = GetFloatDataInRange(-10.0f, 10.0f, 64 * 32);

  RunQDQMatMulOpOpTest<uint16_t, int16_t, uint16_t>(TestInputDef<float>({28, 1, 64}, false, input_a),
                                                    TestInputDef<float>({64, 32}, true, input_b),
                                                    ExpectedEPNodeAssignment::All,
                                                    18,
                                                    true);  // Use com.microsoft Q/DQ ops
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

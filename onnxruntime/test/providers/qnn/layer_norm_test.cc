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

// Runs an LayerNorm model on the QNN CPU backend. Checks the graph node assignment and that inference
// outputs for QNN and CPU match.
static void RunLayerNormCpuTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& scale_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<float>("LayerNormalization", {input_def, scale_def}, {}, attrs),
                  provider_options,
                  17,
                  expected_ep_assignment);
}

#ifdef __linux__
// This CPU test fails on Linux, QNN SDK 2.17
// the value pair (-1.75661933, 0) at index #1 don't match, which is 1.75662 from -1.75662
TEST_F(QnnCPUBackendTests, DISABLED_LayerNorm) {
#else
TEST_F(QnnCPUBackendTests, LayerNorm) {
#endif
  RunLayerNormCpuTest(TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LayerNorm1D_Axis0) {
  RunLayerNormCpuTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LayerNorm1D_AxisLast) {
  RunLayerNormCpuTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 10.0f, 3)),
                      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LayerNorm2D) {
  RunLayerNormCpuTest(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                      TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                      {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LayerNorm3D) {
  RunLayerNormCpuTest(TestInputDef<float>({1, 2, 3, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 72)),
                      TestInputDef<float>({1, 2, 3, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 72)),
                      {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

template <typename InputQType, typename ScaleQType>
GetTestQDQModelFn<InputQType> BuildQDQLayerNormTestCase(const TestInputDef<float>& input_def,
                                                        const TestInputDef<float>& scale_def,
                                                        const TestInputDef<float>& bias_def,
                                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                        bool use_contrib_qdq_ops) {
  return [input_def, scale_def, bias_def, attrs,
          use_contrib_qdq_ops](ModelTestBuilder& builder,
                               std::vector<QuantParams<InputQType>>& output_qparams) {
    std::vector<NodeArg*> layer_norm_inputs;

    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                    use_contrib_qdq_ops);
    layer_norm_inputs.push_back(input_qdq);

    NodeArg* scale_qdq = nullptr;
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);

    if (scale_def.IsInitializer() && scale_def.IsRawData()) {
      // Quantized(scale weights) -> DQ ->
      std::vector<float> scale_scales = {scale_qparams.scale};
      std::vector<ScaleQType> scale_zps = {scale_qparams.zero_point};
      TensorShape scale_shape = scale_def.GetTensorShape();
      std::vector<ScaleQType> quantized_scales(scale_shape.Size());
      QuantizeValues<float, ScaleQType>(scale_def.GetRawData(), quantized_scales, scale_shape,
                                        scale_scales, scale_zps, std::nullopt);

      NodeArg* scale_initzer = builder.MakeInitializer<ScaleQType>(scale_def.GetShape(), quantized_scales);
      scale_qdq = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<ScaleQType>(scale_initzer, scale_scales, scale_zps, scale_qdq,
                                                  nullptr, use_contrib_qdq_ops);
    } else {
      // scale input -> Q -> DQ ->
      NodeArg* scale = MakeTestInput(builder, scale_def);
      scale_qdq = AddQDQNodePair<ScaleQType>(builder, scale, scale_qparams.scale, scale_qparams.zero_point,
                                             use_contrib_qdq_ops);
    }
    layer_norm_inputs.push_back(scale_qdq);

    if (!bias_def.GetShape().empty()) {
      const float bias_scale = input_qparams.scale * scale_qparams.scale;
      layer_norm_inputs.push_back(MakeTestQDQBiasInput(builder, bias_def, bias_scale, use_contrib_qdq_ops));
    }

    // LayerNormalization
    NodeArg* layer_norm_output = builder.MakeIntermediate();
    Node& layer_norm_node = builder.AddNode("LayerNormalization", layer_norm_inputs, {layer_norm_output});

    for (const auto& attr : attrs) {
      layer_norm_node.AddAttributeProto(attr);
    }

    // layer_norm_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, layer_norm_output, output_qparams[0].scale,
                                                      output_qparams[0].zero_point, use_contrib_qdq_ops);
  };
}

// Runs a QDQ LayerNorm model on the QNN HTP backend. Checks the graph node assignment and that inference
// outputs for QNN are as accurate as CPU EP (compares against f32 model and QDQ model).
template <typename InputQType, typename ScaleQType>
static void RunLayerNormQDQTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& scale_def,
                                const TestInputDef<float>& bias_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                bool use_contrib_qdq_ops = false) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase<float>("LayerNormalization", {input_def, scale_def}, {}, attrs),
                       BuildQDQLayerNormTestCase<InputQType, ScaleQType>(input_def, scale_def, bias_def, attrs,
                                                                         use_contrib_qdq_ops),
                       provider_options,
                       17,  // opset
                       expected_ep_assignment);
}

// Test that QNN HTP only supports axis = -1 (i.e., last dimension).
TEST_F(QnnHTPBackendTests, LayerNorm1D_Axis0_Unsupported) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, 0.0f, 10.0f),
                                        TestInputDef<float>({1, 2, 3}, true, 0.0f, 10.0f),
                                        TestInputDef<float>(),
                                        {utils::MakeAttribute("axis", static_cast<int64_t>(0))},  // Unsupported axis
                                        ExpectedEPNodeAssignment::None);
}

// Test accuracy of 8-bit QDQ LayerNorm with a static scale input.
TEST_F(QnnHTPBackendTests, LayerNorm1D_LastAxis_StaticScale_AU8_WU8) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                        TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                        TestInputDef<float>(),  // Implicit bias input
                                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of 8-bit QDQ LayerNorm with a static scale input and an explicit bias input (static).
TEST_F(QnnHTPBackendTests, LayerNorm1D_LastAxis_StaticScale_StaticBias_AU8_WU8_BU8) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                        TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                        TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LayerNorm1D_QNN2_24_ImplicitBias_ValidationBug) {
  // QNN 2.24 to 2.27: LayerNorm fails validation (intermittent) if the bias input is not provided. QNN EP will provide
  // an explicit bias of all zeros to get around this bug.
  // QNN 2.28.0: Validation bug is fixed, but get accuracy errors.
  // QNN 2.28.2: All fixed.
  for (size_t i = 0; i < 15; i++) {  // Run it multiple times since this is an intermittent bug.
    RunLayerNormQDQTest<uint16_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 1.0f, 6)),
                                           TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                           TestInputDef<float>(),  // Implicit bias input
                                           {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                           ExpectedEPNodeAssignment::All,
                                           true);
  }
}

TEST_F(QnnHTPBackendTests, LayerNorm1D_LastAxis_StaticScale_AU16_WU8) {
  // QNN 2.28.0: Get accuracy errors.
  // QNN 2.28.2: All fixed.
  RunLayerNormQDQTest<uint16_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                         TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),  // Static
                                         TestInputDef<float>(),
                                         {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},  // Last axis
                                         ExpectedEPNodeAssignment::All,
                                         true);  // Use 'com.microsoft' Q/DQ ops
}

// Test accuracy of 8-bit QDQ LayerNorm with a dynamic scale input.
//
// TODO(adrianlizarraga): Fails to finalize with QNN SDK 2.22. Still fails on QNN SDK 2.28.2.
// Verbose logs:
// Starting stage: Graph Transformations and Optimizations
// C:\...\QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::flat_to_vtcm
// C:\...\QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x102800000013 preparation failed with err:-1
// Completed stage: Graph Transformations and Optimizations (6247 us)
// QnnDsp <E> "node_token_15" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// Failed to finalize QNN graph.
TEST_F(QnnHTPBackendTests, DISABLED_LayerNorm1D_LastAxis_DynamicScale) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                        TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),  // Dynamic
                                        TestInputDef<float>(),
                                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},  // Last axis
                                        ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif

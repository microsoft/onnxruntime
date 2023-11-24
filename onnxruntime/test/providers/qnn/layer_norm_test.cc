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

TEST_F(QnnCPUBackendTests, LayerNorm) {
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
                                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, scale_def, attrs](ModelTestBuilder& builder,
                                       std::vector<QuantParams<InputQType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    // scale input -> Q -> DQ ->
    NodeArg* scale = MakeTestInput(builder, scale_def);
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);
    NodeArg* scale_qdq = AddQDQNodePair<ScaleQType>(builder, scale, scale_qparams.scale, scale_qparams.zero_point);

    // LayerNormalization
    NodeArg* layer_norm_output = builder.MakeIntermediate();
    Node& layer_norm_node = builder.AddNode("LayerNormalization", {input_qdq, scale_qdq}, {layer_norm_output});

    for (const auto& attr : attrs) {
      layer_norm_node.AddAttributeProto(attr);
    }

    // layer_norm_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, layer_norm_output, output_qparams[0].scale,
                                                      output_qparams[0].zero_point);
  };
}

// Runs a QDQ LayerNorm model on the QNN HTP backend. Checks the graph node assignment and that inference
// outputs for QNN are as accurate as CPU EP (compares against f32 model and QDQ model).
template <typename InputQType, typename ScaleQType>
static void RunLayerNormQDQTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& scale_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase<float>("LayerNormalization", {input_def, scale_def}, {}, attrs),
                       BuildQDQLayerNormTestCase<InputQType, ScaleQType>(input_def, scale_def, attrs),
                       provider_options,
                       17,  // opset
                       expected_ep_assignment);
}

// Test that QNN HTP only supports axis = -1 (i.e., last dimension).
TEST_F(QnnHTPBackendTests, LayerNorm1D_Axis0_Unsupported) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, 0.0f, 10.0f),
                                        TestInputDef<float>({1, 2, 3}, true, 0.0f, 10.0f),
                                        {utils::MakeAttribute("axis", static_cast<int64_t>(0))},  // Unsupported axis
                                        ExpectedEPNodeAssignment::None);
}

// Test accuracy of 8-bit QDQ LayerNorm with a static scale input. This used to fail on QNN DK 2.13,
// but was fixed in QNN SDK 2.14.
TEST_F(QnnHTPBackendTests, LayerNorm1D_LastAxis_StaticScale) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                        TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),  // Static
                                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},            // Last axis
                                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of 8-bit QDQ LayerNorm with a dynamic scale input.
// TODO(adrianlizarraga): Investigate graph finalization error in QNN SDK 2.14.1
// Failed QNN FinalizeGraphs: QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// C:\qnn_src\QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:232:ERROR:could not create op: q::flat_from_vtcm
// C:\qnn_src\QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1021:ERROR:Op 0x103d00000002 preparation failed with err:-1
TEST_F(QnnHTPBackendTests, DISABLED_LayerNorm1D_LastAxis_DynamicScale) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                        TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),  // Dynamic
                                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},             // Last axis
                                        ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif

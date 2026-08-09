// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "gtest/gtest.h"

#include "test/unittest_util/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

static void RunRMSNormCpuTest(const TestInputDef<float>& input_def,
                              const TestInputDef<float>& scale_def,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("RMSNormalization", {input_def, scale_def}, {}, attrs),
                  provider_options,
                  23,
                  expected_ep_assignment);
}

TEST_F(QnnCPUBackendTests, RMSNorm) {
  RunRMSNormCpuTest(TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                    TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                    {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, RMSNorm1D_Axis0) {
  RunRMSNormCpuTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                    TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                    {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, RMSNorm2D) {
  RunRMSNormCpuTest(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                    TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                    {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, RMSNorm3D) {
  RunRMSNormCpuTest(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                    TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                    {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

template <typename InputQType, typename ScaleQType>
GetTestQDQModelFn<InputQType> BuildQDQRMSNormTestCase(const TestInputDef<float>& input_def,
                                                      const TestInputDef<float>& scale_def,
                                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                      bool use_contrib_qdq_ops) {
  return [input_def, scale_def, attrs,
          use_contrib_qdq_ops](ModelTestBuilder& builder,
                               std::vector<QuantParams<InputQType>>& output_qparams) {
    std::vector<NodeArg*> rms_norm_inputs;

    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                    use_contrib_qdq_ops);
    rms_norm_inputs.push_back(input_qdq);

    NodeArg* scale_qdq = nullptr;
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);

    if (scale_def.IsInitializer() && scale_def.IsRawData()) {
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
      NodeArg* scale = MakeTestInput(builder, scale_def);
      scale_qdq = AddQDQNodePair<ScaleQType>(builder, scale, scale_qparams.scale, scale_qparams.zero_point,
                                             use_contrib_qdq_ops);
    }
    rms_norm_inputs.push_back(scale_qdq);

    NodeArg* rms_norm_output = builder.MakeIntermediate();
    Node& rms_norm_node = builder.AddNode("RMSNormalization", rms_norm_inputs, {rms_norm_output});

    for (const auto& attr : attrs) {
      rms_norm_node.AddAttributeProto(attr);
    }

    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, rms_norm_output, output_qparams[0].scale,
                                                      output_qparams[0].zero_point, use_contrib_qdq_ops);
  };
}

template <typename InputQType, typename ScaleQType>
static void RunRMSNormQDQTest(const TestInputDef<float>& input_def,
                              const TestInputDef<float>& scale_def,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              bool use_contrib_qdq_ops = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto qdq_model_fn = BuildQDQRMSNormTestCase<InputQType, ScaleQType>(input_def, scale_def, attrs,
                                                                      use_contrib_qdq_ops);
  GetTestModelFn model_fn = [qdq_model_fn, input_def](ModelTestBuilder& builder) {
    std::pair<float, float> input_range = input_def.GetRange();
    QuantParams<InputQType> output_qparams = QuantParams<InputQType>::Compute(input_range.first, input_range.second);
    std::vector<QuantParams<InputQType>> output_qparams_vec = {output_qparams};

    qdq_model_fn(builder, output_qparams_vec);
  };

  RunQnnModelTestHTPNoVerify(model_fn, provider_options, 23, expected_ep_assignment);
}

TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, 0.0f, 10.0f),
                                      TestInputDef<float>({3}, true, 0.0f, 10.0f),
                                      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AU8_WU8) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AU16_WU8) {
  RunRMSNormQDQTest<uint16_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                       TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                       {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                       ExpectedEPNodeAssignment::All,
                                       true);
}

TEST_F(QnnHTPBackendTests, RMSNormU8U8_4D_LastAxis) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18)),
                                      TestInputDef<float>({3}, true, GetFloatDataInRange(-2.0f, 2.0f, 3)),
                                      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_DynamicScale) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "onnx/onnx_pb.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that builds a model with a TopK operator.
template <typename DataType>
inline GetTestModelFn BuildTopKTestCase(const TestInputDef<DataType>& input_def,
                                        const TestInputDef<int64_t>& k_def,
                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                        bool cast_output_indices = true) {
  return [input_def, k_def, attrs, cast_output_indices](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput<DataType>(builder, input_def);
    NodeArg* k_input = MakeTestInput<int64_t>(builder, k_def);

    NodeArg* values_output = builder.MakeOutput();
    NodeArg* indices_output = cast_output_indices ? builder.MakeIntermediate() : builder.MakeOutput();
    Node& topk_node = builder.AddNode("TopK", {input, k_input}, {values_output, indices_output});

    for (const auto& attr : attrs) {
      topk_node.AddAttributeProto(attr);
    }

    // Cast indices to uint32
    if (cast_output_indices) {
      auto* uint32_indices_output = builder.MakeOutput();
      Node& cast_node = builder.AddNode("Cast", {indices_output}, {uint32_indices_output});
      const auto dst_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32;
      cast_node.AddAttribute("to", static_cast<int64_t>(dst_type));
    }
  };
}

// Runs a model with a TopK operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunTopKTestOnCPU(const TestInputDef<DataType>& input_def,
                             const TestInputDef<int64_t>& k_def,
                             const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 19) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildTopKTestCase<DataType>(input_def, k_def, attrs, false /*cast_output_indices*/),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that TopK with a dynamic K input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, TopK_DynamicK_Unsupported) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, false /* is_initializer */, {2}),
                          {},                               // Attributes
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test that TopK with an axis attribute that is not the last dimension is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, TopK_NonLastAxis_Unsupported) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test that TopK that returns the top k minimum values is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, TopK_MinValues_Unsupported) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {utils::MakeAttribute("largest", static_cast<int64_t>(0))},
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test TopK on CPU backend: top 2 largest floats from last axis
TEST_F(QnnCPUBackendTests, TopK_LargestFloats_LastAxis) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {},  // Attributes
                          ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest int32s from last axis
TEST_F(QnnCPUBackendTests, TopK_LargestInt32s_LastAxis) {
  std::vector<int32_t> input_data = {-6, -5, -4, -3, -2, 0, 1, 2, 3, 4, 5, 6};
  RunTopKTestOnCPU<int32_t>(TestInputDef<int32_t>({1, 2, 2, 3}, false, input_data),
                            TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                            {},  // Attributes
                            ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that creates a graph with a QDQ TopK operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQTopKTestCase(const TestInputDef<float>& input_def,
                                                  const TestInputDef<int64_t>& k_def,
                                                  const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                  bool use_contrib_qdq = false) {
  return [input_def, k_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                                    std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                   use_contrib_qdq);

    // K input
    NodeArg* k_input = MakeTestInput(builder, k_def);

    // Reshape op
    NodeArg* values_output = builder.MakeIntermediate();
    NodeArg* indices_output = builder.MakeIntermediate();
    Node& topk_node = builder.AddNode("TopK", {input_qdq, k_input}, {values_output, indices_output});

    for (const auto& attr : attrs) {
      topk_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    // NOTE: Input and output quantization parameters must be equal for Reshape.
    output_qparams[0] = input_qparams;  // Overwrite!
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, values_output, input_qparams.scale,
                                                     input_qparams.zero_point, use_contrib_qdq);

    // Cast indices to uint32 (HTP backend does not support int64 graph outputs)
    auto* uint32_indices_output = builder.MakeOutput();
    Node& cast_node = builder.AddNode("Cast", {indices_output}, {uint32_indices_output});
    const auto dst_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32;
    cast_node.AddAttribute("to", static_cast<int64_t>(dst_type));
  };
}

// Runs a QDQ TopK model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQTopKTestOnHTP(const TestInputDef<float>& input_def,
                                const TestInputDef<int64_t>& k_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 19,
                                bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto f32_model_builder = BuildTopKTestCase<float>(input_def, k_def, attrs, true /*cast_output_indices*/);
  auto qdq_model_builder = BuildQDQTopKTestCase<QType>(input_def, k_def, attrs, use_contrib_qdq);
  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test 8-bit QDQ TopK on HTP backend: top 2 largest floats from last axis
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U8_LastAxis) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ TopK on HTP backend: top 2 largest floats from last axis
// TODO: Inaccuracy detected for output 'output_0', element 6.
// Output quant params: scale=0.00061036087572574615, zero_point=32768.
// Expected val: -7.2340402603149414
// QNN QDQ val: -17.446556091308594 (err 10.212515830993652)
// CPU QDQ val: -7.2339968681335449 (err 4.3392181396484375e-05)
TEST_F(QnnHTPBackendTests, DISABLED_TopK_LargestFloats_U16_LastAxis) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                19,     // opset
                                true);  // Use com.microsoft Q/DQ ops
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

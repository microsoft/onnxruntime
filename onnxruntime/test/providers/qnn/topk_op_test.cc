// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that builds a model with a TopK operator.
template <typename DataType>
inline GetTestModelFn BuildTopKTestCase(const TestInputDef<DataType>& input_def,
                                        const TestInputDef<int64_t>& k_def,
                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, k_def, attrs](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput<DataType>(builder, input_def);
    NodeArg* k_input = MakeTestInput<int64_t>(builder, k_def);

    NodeArg* values_output = builder.MakeOutput();
    NodeArg* indices_output = builder.MakeOutput();
    Node& topk_node = builder.AddNode("TopK", {input, k_input}, {values_output, indices_output});

    for (const auto& attr : attrs) {
      topk_node.AddAttributeProto(attr);
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

  provider_options["backend_type"] = "cpu";

  RunQnnModelTest(BuildTopKTestCase<DataType>(input_def, k_def, attrs),
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

    // TopK_values_output -> Q -> DQ -> output
    // NOTE: Create output QDQ nodes before the TopK node so that TopK's 'values' output is the graph's first output.
    NodeArg* values_output = builder.MakeIntermediate();
    output_qparams[0] = input_qparams;  // Input and output qparams must be equal.
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, values_output, input_qparams.scale,
                                                     input_qparams.zero_point, use_contrib_qdq);
    // TopK node
    NodeArg* indices_output = builder.MakeOutput();
    Node& topk_node = builder.AddNode("TopK", {input_qdq, k_input}, {values_output, indices_output});

    for (const auto& attr : attrs) {
      topk_node.AddAttributeProto(attr);
    }
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

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildTopKTestCase<float>(input_def, k_def, attrs);
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
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U16_LastAxis) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

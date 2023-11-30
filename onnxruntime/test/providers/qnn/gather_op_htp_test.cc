// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Returns a function that creates a graph with a QDQ Gather operator.
template <typename QuantType, typename IndicesType>
GetTestQDQModelFn<QuantType> BuildQDQGatherTestCase(const TestInputDef<float>& input_def,
                                                    const TestInputDef<IndicesType>& indices_def,
                                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                    bool use_contrib_qdq = false) {
  return [input_def, indices_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                                          std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                   use_contrib_qdq);

    // indices input
    NodeArg* indices_input = MakeTestInput(builder, indices_def);

    // Gather op
    NodeArg* gather_output = builder.MakeIntermediate();
    Node& gather_node = builder.AddNode("Gather", {input_qdq, indices_input}, {gather_output});

    for (const auto& attr : attrs) {
      gather_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    // NOTE: Input and output quantization parameters must be equal for Gather.
    output_qparams[0] = input_qparams;  // Overwrite!
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, gather_output, input_qparams.scale,
                                                     input_qparams.zero_point, use_contrib_qdq);
  };
}

// Test the accuracy of a QDQ Gather model on QNN EP. Checks if the QDQ model on QNN EP as accurate as the QDQ model on CPU EP
// (compared to float32 model).
template <typename QuantType, typename IndicesType>
static void RunQDQGatherOpTest(const TestInputDef<float>& input_def,
                               const TestInputDef<IndicesType>& indices_def,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                               int opset,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto f32_model_builder = BuildOpTestCase<float, IndicesType>("Gather", {input_def}, {indices_def}, attrs);
  auto qdq_model_builder = BuildQDQGatherTestCase<QuantType, IndicesType>(input_def, indices_def, attrs,
                                                                          use_contrib_qdq);

  TestQDQModelAccuracy<QuantType>(f32_model_builder,
                                  qdq_model_builder,
                                  provider_options,
                                  opset,
                                  expected_ep_assignment);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results are as accurate as CPU EP.
//
// Static int64 indices with default axis.
TEST_F(QnnHTPBackendTests, GatherOp_IndicesStaticInt64_Axis0) {
  RunQDQGatherOpTest<uint8_t, int64_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                       TestInputDef<int64_t>({2, 2}, true, {0, 1, 1, 2}),
                                       {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                                       13,
                                       ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Gather with static int64 indices with default axis.
TEST_F(QnnHTPBackendTests, GatherOp_U16_IndicesStaticInt64_Axis0) {
  RunQDQGatherOpTest<uint16_t, int64_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                        TestInputDef<int64_t>({2, 2}, true, {0, 1, 1, 2}),
                                        {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                                        13,
                                        ExpectedEPNodeAssignment::All,
                                        true);  // Use 'com.microsoft' Q/DQ ops
}

// Tests that dynamic int64 indices are not supported on HTP backend.
TEST_F(QnnHTPBackendTests, GatherOp_IndicesDynamicInt64_Axis0) {
  RunQDQGatherOpTest<uint8_t, int64_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                       TestInputDef<int64_t>({2, 2}, false, {0, 1, 1, 2}),
                                       {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                                       13,
                                       ExpectedEPNodeAssignment::None);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results are as accurate as CPU EP.
//
// Static int32 indices with default axis.
TEST_F(QnnHTPBackendTests, GatherOp_IndicesStaticInt32_Axis0) {
  RunQDQGatherOpTest<uint8_t, int32_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                       TestInputDef<int32_t>({2, 2}, true, {0, 1, 1, 2}),
                                       {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                                       13,
                                       ExpectedEPNodeAssignment::All);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results are as accurate as CPU EP.
//
// Dynamic int32 indices with default axis.
TEST_F(QnnHTPBackendTests, GatherOp_IndicesDynamicInt32_Axis0) {
  RunQDQGatherOpTest<uint8_t, int32_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                       TestInputDef<int32_t>({2, 2}, false, {0, 1, 1, 2}),
                                       {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                                       13,
                                       ExpectedEPNodeAssignment::All);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results are as accurate as CPU EP.
//
// Static int32 indices with axis = 1
TEST_F(QnnHTPBackendTests, GatherOp_IndicesStaticInt32_Axis1) {
  RunQDQGatherOpTest<uint8_t, int32_t>(TestInputDef<float>({3, 3}, false, {1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f}),
                                       TestInputDef<int32_t>({1, 2}, true, {0, 2}),
                                       {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                                       13,
                                       ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
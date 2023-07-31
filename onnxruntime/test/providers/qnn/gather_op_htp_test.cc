// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Function that builds a float model with a Gather op.
template <typename IndicesType = int32_t>
static GetTestModelFn BuildGatherOpTestCase(const TestInputDef<float>& input_def,
                                            const TestInputDef<IndicesType>& indices_def,
                                            int64_t axis = 0) {
  return [input_def, indices_def, axis](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput(builder, input_def);
    NodeArg* indices = MakeTestInput(builder, indices_def);
    NodeArg* output = builder.MakeOutput();

    Node& gather_node = builder.AddNode("Gather", {input, indices}, {output});
    gather_node.AddAttribute("axis", axis);
  };
}

// Function that builds a QDQ model with a Gather op.
template <typename QuantType = uint8_t, typename IndicesType = int32_t>
static GetTestQDQModelFn<QuantType> BuildQDQGatherOpTestCase(const TestInputDef<float>& input_def,
                                                             const TestInputDef<IndicesType>& indices_def,
                                                             int64_t axis = 0) {
  return [input_def, indices_def, axis](ModelTestBuilder& builder,
                                        std::vector<QuantParams<QuantType>>& output_qparams) {
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    NodeArg* indices = MakeTestInput(builder, indices_def);

    NodeArg* gather_output = builder.MakeIntermediate();
    Node& gather_node = builder.AddNode("Gather", {input_qdq, indices}, {gather_output});
    gather_node.AddAttribute("axis", axis);

    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, gather_output, output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// Test the accuracy of a QDQ Gather model on QNN EP. Checks if the QDQ model on QNN EP as accurate as the QDQ model on CPU EP
// (compared to float32 model).
template <typename QuantType, typename IndicesType>
static void RunQDQGatherOpTest(const TestInputDef<float>& input_def, const TestInputDef<IndicesType>& indices_def,
                               int64_t axis, int opset, ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy<QuantType>(BuildGatherOpTestCase<IndicesType>(input_def, indices_def, axis),
                                  BuildQDQGatherOpTestCase<QuantType, IndicesType>(input_def, indices_def, axis),
                                  provider_options,
                                  opset,
                                  expected_ep_assignment,
                                  1e-5f);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results are as accurate as CPU EP.
//
// Static int64 indices with default axis.
TEST_F(QnnHTPBackendTests, GatherOp_IndicesStaticInt64_Axis0) {
  RunQDQGatherOpTest<uint8_t, int64_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                       TestInputDef<int64_t>({2, 2}, true, {0, 1, 1, 2}),
                                       0,
                                       13,
                                       ExpectedEPNodeAssignment::All);
}

// Tests that dynamic int64 indices are not supported on HTP backend.
TEST_F(QnnHTPBackendTests, GatherOp_IndicesDynamicInt64_Axis0) {
  RunQDQGatherOpTest<uint8_t, int64_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                       TestInputDef<int64_t>({2, 2}, false, {0, 1, 1, 2}),
                                       0,
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
                                       0,
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
                                       0,
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
                                       1,
                                       13,
                                       ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
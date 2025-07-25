// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "core/graph/node_attr_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Builds a QDQ model with ArgMin/ArgMax. The quantization parameters are computed from the provided input definition.
template <typename QType = uint8_t>
static GetTestQDQModelFn<QType> BuildQDQArgMxxTestCase(const std::string& op_type, TestInputDef<float> input_def,
                                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [op_type, input_def, attrs](ModelTestBuilder& builder,
                                     std::vector<QuantParams<QType>>& output_qparams) {
    ORT_UNUSED_PARAMETER(output_qparams);
    QuantParams<QType> input_qparams = GetTestInputQuantParams<QType>(input_def);

    auto* input = MakeTestInput(builder, input_def);

    // input -> Q -> DQ ->
    auto* input_qdq = AddQDQNodePair<QType>(builder, input, input_qparams.scale, input_qparams.zero_point);
    auto* argm_output = builder.MakeOutput();
    Node& argm_node = builder.AddNode(op_type, {input_qdq}, {argm_output});
    for (const auto& attr : attrs) {
      argm_node.AddAttributeProto(attr);
    }
  };
}

// Runs an ArgMax/ArgMin model on the specified QNN backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunArgMxxOpTest(const std::string& op_type, TestInputDef<float> input_def,
                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            const std::string& backend_name = "cpu", int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;

  RunQnnModelTest(BuildOpTestCase<float>(op_type, {input_def}, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ ArgMax/ArgMin model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment, and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (when compared to the baseline float32 model).
template <typename QType = uint8_t>
static void RunQDQArgMxxOpTest(const std::string& op_type, TestInputDef<float> input_def,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, attrs),   // baseline float32 model
                       BuildQDQArgMxxTestCase<QType>(op_type, input_def, attrs),  // QDQ model
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

//
// CPU tests:
//

// Test that ArgMax/ArgMin with default attributes works on QNN CPU backend. Compares output with CPU EP.
TEST_F(QnnCPUBackendTests, ArgMaxMin_DefaultAttrs) {
  RunArgMxxOpTest("ArgMax",
                  TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                  {},                                                       // All default ONNX attributes.
                  ExpectedEPNodeAssignment::All, "cpu", 13);
  RunArgMxxOpTest("ArgMin",
                  TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                  {},                                                       // All default ONNX attributes.
                  ExpectedEPNodeAssignment::All, "cpu", 13);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test that Q/DQ(uint8) ArgMax/ArgMin with default attributes works on HTP backend.
// Compares output with CPU EP.
TEST_F(QnnHTPBackendTests, ArgMaxMinU8_DefaultAttrs) {
  RunQDQArgMxxOpTest<uint8_t>("ArgMax",
                              TestInputDef<float>({1, 3, 4}, false, -10.0f, 10.0f),  // Random input.
                              {},                                                    // All default ONNX attributes.
                              ExpectedEPNodeAssignment::All, 13);
  RunQDQArgMxxOpTest<uint8_t>("ArgMin",
                              TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                              {},                                                       // Default ONNX attributes.
                              ExpectedEPNodeAssignment::All, 13);
}

// Tests that Q/DQ(uint8) ArgMax/ArgMin with axis of -1 works on HTP backend.
// Compares output with CPU EP.
TEST_F(QnnHTPBackendTests, ArgMaxMinU8_AxisLast) {
  RunQDQArgMxxOpTest<uint8_t>("ArgMax",
                              TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),   // Random input.
                              {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},  // axis is -1
                              ExpectedEPNodeAssignment::All, 13);
  RunQDQArgMxxOpTest<uint8_t>("ArgMin",
                              TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),   // Random input.
                              {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},  // axis is -1
                              ExpectedEPNodeAssignment::All, 13);
}

// Tests that Q/DQ(uint8) ArgMax/ArgMin with axis of -1 and keepdims = false works on HTP backend.
// Compares output with CPU EP.
TEST_F(QnnHTPBackendTests, ArgMaxMinU8_NotKeepDims) {
  RunQDQArgMxxOpTest<uint8_t>("ArgMax",
                              TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                              {utils::MakeAttribute("axis", static_cast<int64_t>(-1)),
                               utils::MakeAttribute("keepdims", static_cast<int64_t>(0))},
                              ExpectedEPNodeAssignment::All, 13);
  RunQDQArgMxxOpTest<uint8_t>("ArgMin",
                              TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                              {utils::MakeAttribute("keepdims", static_cast<int64_t>(0))},
                              ExpectedEPNodeAssignment::All, 13);
}

// Tests that Q/DQ ArgMax/ArgMin with select_last_index = 1 is not supported.
TEST_F(QnnHTPBackendTests, ArgMaxMinU8_SelectLastIndex_NonZero_Unsupported) {
  RunQDQArgMxxOpTest<uint8_t>("ArgMax",
                              TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                              {utils::MakeAttribute("select_last_index", static_cast<int64_t>(1))},
                              ExpectedEPNodeAssignment::None, 13);
  RunQDQArgMxxOpTest<uint8_t>("ArgMin",
                              TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                              {utils::MakeAttribute("select_last_index", static_cast<int64_t>(1))},
                              ExpectedEPNodeAssignment::None, 13);
}

// Tests that Q/DQ ArgMax/ArgMin with input rank > 4 on HTP is not supported.
TEST_F(QnnHTPBackendTests, ArgMaxMinU8_RankGreaterThan4_Unsupported) {
  RunQDQArgMxxOpTest<uint8_t>("ArgMax",
                              TestInputDef<float>({1, 3, 4, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                              {},
                              ExpectedEPNodeAssignment::None, 13);
  RunQDQArgMxxOpTest<uint8_t>("ArgMin",
                              TestInputDef<float>({1, 3, 4, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                              {},
                              ExpectedEPNodeAssignment::None, 13);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// Test that ArgMax/ArgMin with default attributes works on QNN GPU backend. Compares output with CPU EP.
// Disable Reason : Onnx Op need Int64 output. Can enable after CastOp Int32 to Int64 is done.
// Can enable after CastOp int32 to int64 is implemented in QnnGpu.
TEST_F(QnnGPUBackendTests, DISABLED_ArgMaxMin_DefaultAttrs) {
  RunArgMxxOpTest("ArgMax",
                  TestInputDef<float>({3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                  {},                                                    // All default ONNX attributes.
                  ExpectedEPNodeAssignment::All, "gpu", 13);
  RunArgMxxOpTest("ArgMin",
                  TestInputDef<float>({3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                  {},                                                    // All default ONNX attributes.
                  ExpectedEPNodeAssignment::All, "gpu", 13);
}

// Test that ArgMax/ArgMin with axis attribute works on QNN GPU backend. Compares output with CPU EP.
// Disable Reason : Onnx Op need Int64 output. Can enable after CastOp Int32 to Int64 is done.
// Can enable after CastOp int32 to int64 is implemented in QnnGpu.
TEST_F(QnnGPUBackendTests, DISABLED_ArgMaxMin_AxisAttr) {
  RunArgMxxOpTest("ArgMax",
                  TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                  {utils::MakeAttribute("axis", static_cast<int64_t>(1))},  // axis is 1
                  ExpectedEPNodeAssignment::All, "gpu", 13);
  RunArgMxxOpTest("ArgMin",
                  TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                  {utils::MakeAttribute("axis", static_cast<int64_t>(1))},  // axis is 1
                  ExpectedEPNodeAssignment::All, "gpu", 13);
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

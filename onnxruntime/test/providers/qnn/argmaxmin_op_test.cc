// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "core/graph/node_attr_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Builds a float32 model with ArgMin/ArgMax.
static GetTestModelFn BuildArgMxxTestCase(const std::string& op_type, TestInputDef<float> input_def,
                                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [op_type, input_def, attrs](ModelTestBuilder& builder) {
    auto* input = MakeTestInput(builder, input_def);

    auto* argm_output = builder.MakeIntermediate();
    Node& argm_node = builder.AddNode(op_type, {input}, {argm_output});
    for (const auto& attr : attrs) {
      argm_node.AddAttributeProto(attr);
    }

    // Add cast to uint32
    auto* output = builder.MakeOutput();
    Node& cast_node = builder.AddNode("Cast", {argm_output}, {output});
    const auto dst_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32;
    cast_node.AddAttribute("to", static_cast<int64_t>(dst_type));
  };
}

// Builds a QDQ model with ArgMin/ArgMax and a Cast to uint32. The quantization parameters are computed from the provided
// input definition.
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
    auto* argm_output = builder.MakeIntermediate();
    Node& argm_node = builder.AddNode(op_type, {input_qdq}, {argm_output});
    for (const auto& attr : attrs) {
      argm_node.AddAttributeProto(attr);
    }

    // Cast to uint32 (HTP does not support int64 as graph output)
    auto* output = builder.MakeOutput();
    Node& cast_node = builder.AddNode("Cast", {argm_output}, {output});
    const auto dst_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32;
    cast_node.AddAttribute("to", static_cast<int64_t>(dst_type));
  };
}

// Runs an ArgMax/ArgMin model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunCPUArgMxxOpTest(const std::string& op_type, TestInputDef<float> input_def,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildArgMxxTestCase(op_type, input_def, attrs),
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

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildArgMxxTestCase(op_type, input_def, attrs),            // baseline float32 model
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
  RunCPUArgMxxOpTest("ArgMax",
                     TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                     {},                                                       // All default ONNX attributes.
                     ExpectedEPNodeAssignment::All, 13);
  RunCPUArgMxxOpTest("ArgMin",
                     TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),  // Random input.
                     {},                                                       // All default ONNX attributes.
                     ExpectedEPNodeAssignment::All, 13);
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

// Test that ArgMax/ArgMin are not supported if they generate a graph output.
TEST_F(QnnHTPBackendTests, ArgMaxMin_AsGraphOutputUnsupported) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Utility function that creates a QDQ model with ArgMax/ArgMin that produce a graph output.
  auto model_builder_func = [](const std::string& op_type, const TestInputDef<float>& input_def,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) -> GetTestModelFn {
    return [op_type, input_def, attrs](ModelTestBuilder& builder) {
      QuantParams<uint8_t> input_qparams = GetTestInputQuantParams<uint8_t>(input_def);

      auto* input = MakeTestInput(builder, input_def);
      auto* output = builder.MakeOutput();

      // input -> Q -> DQ ->
      auto* input_qdq = AddQDQNodePair<uint8_t>(builder, input, input_qparams.scale, input_qparams.zero_point);

      Node& argm_node = builder.AddNode(op_type, {input_qdq}, {output});
      for (const auto& attr : attrs) {
        argm_node.AddAttributeProto(attr);
      }
    };
  };

  const int expected_nodes_in_graph = -1;  // Don't care exactly how many nodes in graph assigned to CPU EP.
  RunQnnModelTest(model_builder_func("ArgMax", TestInputDef<float>({1, 3, 4}, false, -1.0f, 1.0f), {}),
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::None,  // No nodes should be assigned to QNN EP!
                  expected_nodes_in_graph);
  RunQnnModelTest(model_builder_func("ArgMin", TestInputDef<float>({1, 3, 4}, false, -1.0f, 1.0f), {}),
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::None,  // No nodes should be assigned to QNN EP!
                  expected_nodes_in_graph);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

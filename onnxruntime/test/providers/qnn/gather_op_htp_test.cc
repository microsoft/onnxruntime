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

template <typename QuantType, typename IndicesType>
GetTestQDQModelFn<QuantType> BuildQDQGatherNdTestCase(const TestInputDef<float>& input_def,
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
    Node& gather_node = builder.AddNode("GatherND", {input_qdq, indices_input}, {gather_output});

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
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

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

// Tests that dynamic int64 indices are supported on HTP backend if the indices are a graph input.
// QNN SDK 2.23 added support for Cast from int64 to int32.
TEST_F(QnnHTPBackendTests, GatherOp_IndicesDynamicInt64_Axis0) {
  RunQDQGatherOpTest<uint8_t, int64_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                       TestInputDef<int64_t>({2, 2}, false, {0, 1, 1, 2}),
                                       {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                                       13,
                                       ExpectedEPNodeAssignment::All);
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

// negative indices
TEST_F(QnnHTPBackendTests, GatherOp_IndicesStaticInt32_NegativeIndices) {
  RunQDQGatherOpTest<uint8_t, int32_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                                       TestInputDef<int32_t>({2, 2}, true, {-1, 1, 1, 2}),
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

// disabled for QNN 2.28.0.241029 failed for accuracy validation
// Also fails on QNN 2.28.2.
// qdq@QNN_EP val: 3.6094117164611816 (err: 1.3094117641448975, err/output_range: 22.19342041015625%)
// qdq@CPU_EP val: 2.2905881404876709 (err: 0.0094118118286132812, err/output_range: 0.15952222049236298%)
// abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = 22.033897399902344%
// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results are as accurate as CPU EP.
//
// Static int32 indices with axis = 1
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, GatherOp_IndicesStaticInt32_Axis1) {
  RunQDQGatherOpTest<uint8_t, int32_t>(TestInputDef<float>({3, 3}, false, {1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f}),
                                       TestInputDef<int32_t>({1, 2}, true, {0, 2}),
                                       {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                                       13,
                                       ExpectedEPNodeAssignment::All);
}

// Runs a non-QDQ model on HTP and compares output to CPU EP.
template <typename InputType1 = float, typename InputType2 = float>
static void RunOpTest(const std::string& op_type,
                      const TestInputDef<InputType1>& input_def_1,
                      const TestInputDef<InputType2>& input_defs_2,
                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                      int opset_version,
                      ExpectedEPNodeAssignment expected_ep_assignment,
                      const std::string& op_domain = kOnnxDomain,
                      float fp32_abs_err = 1e-3f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildOpTestCase<InputType1, InputType2>(op_type, {input_def_1}, {input_defs_2}, attrs, op_domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Non-QDQ model, Gather with static input and dynamic int64 indices
// Fails with QNN SDK 2.35.0:
// Failed to finalize QNN graph. Error code: 1002
TEST_F(QnnHTPBackendTests, DISABLED_GatherOp_IndicesStaticInt64) {
  RunOpTest<float, int64_t>("Gather",
                            TestInputDef<float>({3, 2}, true, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f}),
                            TestInputDef<int64_t>({2, 2}, false, {0, 1, 1, 2}),
                            {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                            13,
                            ExpectedEPNodeAssignment::All);
}

// Test the accuracy of a QDQ GatherND model on QNN EP. Checks if the QDQ model on QNN EP is as accurate as the QDQ model on CPU EP.
template <typename QuantType, typename IndicesType>
static void RunQDQGatherNDOpTest(const TestInputDef<float>& input_def,
                                 const TestInputDef<IndicesType>& indices_def,
                                 const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                 int opset,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildOpTestCase<float, IndicesType>("GatherND", {input_def}, {indices_def}, attrs);
  auto qdq_model_builder = BuildQDQGatherNdTestCase<QuantType, IndicesType>(input_def, indices_def, attrs,
                                                                            use_contrib_qdq);

  TestQDQModelAccuracy<QuantType>(f32_model_builder,
                                  qdq_model_builder,
                                  provider_options,
                                  opset,
                                  expected_ep_assignment);
}

// Non-QDQ model, GatherND with static input and dynamic int64 indices
TEST_F(QnnHTPBackendTests, GatherNDOp_IndicesDynamicInt64) {
  RunOpTest<float, int64_t>(
      "GatherND",
      TestInputDef<float>({2, 2, 2}, true,  // Static input
                          {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}),
      TestInputDef<int64_t>({2, 2}, false,
                            {0, 0, 1, 1}),
      {},  // No attributes for GatherND
      13,  // Opset version
      ExpectedEPNodeAssignment::All);
}

// Static negative int64 indices with negative values and batch_dims = 0
TEST_F(QnnHTPBackendTests, GatherNDOp_Negative_IndicesInt64_BatchDims0) {
  RunOpTest<float, int64_t>(
      "GatherND",
      TestInputDef<float>({2, 2, 2}, true,  // Static input
                          {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}),
      TestInputDef<int64_t>({2, 2}, true,  // Static -ve indices with negative values
                            {-1, -1, 0, 0}),
      {utils::MakeAttribute("batch_dims", static_cast<int64_t>(0))},  // Attribute for batch_dims
      13,                                                             // Opset version
      ExpectedEPNodeAssignment::All);
}

// Static int64 indices with batch_dims = 0
TEST_F(QnnHTPBackendTests, GatherNDOp_QDQ_IndicesStaticInt64_BatchDims0) {
  RunQDQGatherNDOpTest<uint8_t, int64_t>(
      TestInputDef<float>({2, 2, 2}, false, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}),
      TestInputDef<int64_t>({2, 2}, true, {0, 0, 1, 1}),
      {utils::MakeAttribute("batch_dims", static_cast<int64_t>(0))},
      13,
      ExpectedEPNodeAssignment::All);
}

// Dynamic int64 indices with batch_dims = 0
TEST_F(QnnHTPBackendTests, GatherNDOp_QDQ_IndicesDynamicInt64_BatchDims0) {
  RunQDQGatherNDOpTest<uint8_t, int64_t>(
      TestInputDef<float>({2, 2, 2}, false, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}),
      TestInputDef<int64_t>({2, 2}, false, {0, 0, 1, 1}),
      {utils::MakeAttribute("batch_dims", static_cast<int64_t>(0))},
      13,
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif

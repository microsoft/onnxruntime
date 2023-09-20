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

// Runs a model with a Flatten operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunFlattenTestOnCPU(const TestInputDef<DataType>& input_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType>("Flatten", {input_def}, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Flatten input (rank4) with axis == 0.
TEST_F(QnnCPUBackendTests, Flatten_Rank4_Axis0) {
  RunFlattenTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                      {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

// Test that Flatten input (rank4) with axis == -1.
TEST_F(QnnCPUBackendTests, Flatten_Rank4_AxisNeg1) {
  RunFlattenTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                      ExpectedEPNodeAssignment::All);
}

// Test that Flatten input (rank5) with axis == 2.
TEST_F(QnnCPUBackendTests, Flatten_Rank5_Axis2) {
  RunFlattenTestOnCPU(TestInputDef<float>({1, 2, 3, 4, 4}, false, -10.0f, 10.0f),
                      {utils::MakeAttribute("axis", static_cast<int64_t>(2))},
                      ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Runs a model with a non-QDQ Flatten operator on the QNN HTP backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunFlattenTestOnHTP(const TestInputDef<DataType>& input_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType>("Flatten", {input_def}, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ Flatten model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQFlattenTestOnHTP(const TestInputDef<float>& input_def,
                                   const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   int opset = 13,
                                   bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto f32_model_builder = BuildOpTestCase<float>("Flatten", {input_def}, {}, attrs);
  auto qdq_model_builder = BuildQDQOpTestCase<QType>("Flatten", {input_def}, {}, attrs, kOnnxDomain, use_contrib_qdq);
  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test 8-bit QDQ Flatten input (rank4) with axis == 0.
TEST_F(QnnHTPBackendTests, Flatten_Rank4_Axis0) {
  RunQDQFlattenTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                  {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                                  ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Flatten input (rank4) with axis == 0.
TEST_F(QnnHTPBackendTests, Flatten_Rank4_Axis0_U16) {
  RunQDQFlattenTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                   {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
                                   ExpectedEPNodeAssignment::All,
                                   13,     // opset
                                   true);  // Use com.microsoft Q/DQ ops
}

// Test 8-bit QDQ Flatten input (rank4) with axis == -1.
TEST_F(QnnHTPBackendTests, Flatten_Rank4_AxisNeg1) {
  RunQDQFlattenTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                  {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                  ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Flatten input (rank4) with axis == -1.
TEST_F(QnnHTPBackendTests, Flatten_Rank4_AxisNeg1_U16) {
  RunQDQFlattenTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                   {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                   ExpectedEPNodeAssignment::All,
                                   13,     // opset
                                   true);  // Use com.microsoft Q/DQ ops
}

// Test 8-bit QDQ Flatten with an input of rank5.
TEST_F(QnnHTPBackendTests, Flatten_QDQ8bit_Rank5) {
  // We can't use the usual model-building functions because they add standalone Quantize and Dequantize nodes
  // at the input and output. These Q/DQ ops get lowered to QNN's Quantize and Dequantize operators, which DO NOT
  // support rank 5 tensors. Therefore, we have to create a test model that only instantiates the DQ -> Flatten -> Q
  // QDQ node group, which gets lowered to a single QNN Reshape node.
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    // input (u8) -> DQ ->
    NodeArg* quant_input = builder.MakeInput<uint8_t>({1, 2, 3, 4, 5}, 0, 255);
    NodeArg* input_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(quant_input, 1.0f, 0, input_dq);  // scale = 1.0, zp = 0

    // Flatten ->
    NodeArg* flatten_output = builder.MakeIntermediate();
    Node& flatten_node = builder.AddNode("Flatten", {input_dq}, {flatten_output});
    flatten_node.AddAttribute("axis", static_cast<int64_t>(2));

    // Q -> output (u8)
    NodeArg* output = builder.MakeOutput();
    builder.AddQuantizeLinearNode<uint8_t>(flatten_output, 1.0f, 0, output);  // scale = 1.0, zp = 0
  };

  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test that int32 non-QDQ Flatten runs on HTP backend.
TEST_F(QnnHTPBackendTests, Flatten_Int32_Rank4_Axis2) {
  std::vector<int32_t> input_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  RunFlattenTestOnHTP<int32_t>(TestInputDef<int32_t>({1, 3, 2, 2}, false, input_data),
                               {utils::MakeAttribute("axis", static_cast<int64_t>(2))},
                               ExpectedEPNodeAssignment::All);
}

// Test that rank 5 int32 Flatten runs on HTP backend.
TEST_F(QnnHTPBackendTests, Flatten_Int32_Rank5_Axis2) {
  std::vector<int32_t> input_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  RunFlattenTestOnHTP<int32_t>(TestInputDef<int32_t>({1, 3, 2, 2, 2}, false, input_data),
                               {utils::MakeAttribute("axis", static_cast<int64_t>(2))},
                               ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

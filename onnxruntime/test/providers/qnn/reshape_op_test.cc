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

// Runs a model with a Reshape operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunReshapeTestOnCPU(const TestInputDef<DataType>& input_def,
                                const TestInputDef<int64_t>& shape_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 19) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType, int64_t>("Reshape", {input_def}, {shape_def}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Reshape with a dynamic shape input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Reshape_DynamicShape_Unsupported) {
  RunReshapeTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                      TestInputDef<int64_t>({2}, false /* is_initializer */, {1, 48}),
                      {},                              // Attributes
                      ExpectedEPNodeAssignment::None,  // Should not be assigned to QNN EP.
                      19);                             // Opset
}

// Test that Reshape with an enabled 'allowzero' attribute is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Reshape_AllowZeroAttr_Unsupported) {
  RunReshapeTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                      TestInputDef<int64_t>({2}, true, {1, 48}),
                      {utils::MakeAttribute("allowzero", static_cast<int64_t>(1))},
                      ExpectedEPNodeAssignment::None,  // Should not be assigned to QNN EP.
                      19);                             // Opset
}

// Test Reshape of rank 4 -> rank 2.
TEST_F(QnnCPUBackendTests, Reshape_4D_f32) {
  RunReshapeTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                      TestInputDef<int64_t>({2}, true, {1, 48}),
                      {},  // Attributes
                      ExpectedEPNodeAssignment::All,
                      19);  // Opset
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that creates a graph with a QDQ Reshape operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQReshapeTestCase(const TestInputDef<float>& input_def,
                                                     const TestInputDef<int64_t>& shape_def,
                                                     const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                     bool use_contrib_qdq = false) {
  return [input_def, shape_def, attrs,
          use_contrib_qdq](ModelTestBuilder& builder,
                           std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                   use_contrib_qdq);

    // shape input
    NodeArg* shape_input = MakeTestInput(builder, shape_def);

    // Reshape op
    NodeArg* reshape_output = builder.MakeIntermediate();
    Node& reshape_node = builder.AddNode("Reshape", {input_qdq, shape_input}, {reshape_output});

    for (const auto& attr : attrs) {
      reshape_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    // NOTE: Input and output quantization parameters must be equal for Reshape.
    output_qparams[0] = input_qparams;  // Overwrite!
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, reshape_output, input_qparams.scale,
                                                     input_qparams.zero_point, use_contrib_qdq);
  };
}

// Runs a model with a non-QDQ Reshape operator on the QNN HTP backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunReshapeTestOnHTP(const TestInputDef<DataType>& input_def,
                                const TestInputDef<int64_t>& shape_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 19) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType, int64_t>("Reshape", {input_def}, {shape_def}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ Reshape model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQReshapeTestOnHTP(const TestInputDef<float>& input_def,
                                   const TestInputDef<int64_t>& shape_def,
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

  auto f32_model_builder = BuildOpTestCase<float, int64_t>("Reshape", {input_def}, {shape_def}, attrs);
  auto qdq_model_builder = BuildQDQReshapeTestCase<QType>(input_def, shape_def, attrs, use_contrib_qdq);
  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test that QDQ Reshape with a dynamic shape input is not supported by QNN EP.
TEST_F(QnnHTPBackendTests, Reshape_DynamicShape_Unsupported) {
  RunQDQReshapeTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                  TestInputDef<int64_t>({2}, false /* is_initializer */, {1, 48}),
                                  {},                              // Attributes
                                  ExpectedEPNodeAssignment::None,  // Should not be assigned to QNN EP.
                                  19);                             // Opset
}

// Test that QDQ Reshape with an enabled 'allowzero' attribute is not supported by QNN EP.
TEST_F(QnnHTPBackendTests, Reshape_AllowZeroAttr_Unsupported) {
  RunQDQReshapeTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                  TestInputDef<int64_t>({2}, true, {1, 48}),
                                  {utils::MakeAttribute("allowzero", static_cast<int64_t>(1))},
                                  ExpectedEPNodeAssignment::None,  // Should not be assigned to QNN EP.
                                  19);                             // Opset
}

// Test 8-bit QDQ Reshape of rank 4 -> rank 2.
TEST_F(QnnHTPBackendTests, Reshape_4D_u8) {
  RunQDQReshapeTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                  TestInputDef<int64_t>({2}, true, {1, 48}),
                                  {},  // Attributes
                                  ExpectedEPNodeAssignment::All,
                                  19);  // Opset
}

// Test 16-bit QDQ Reshape of rank 4 -> rank 2.
TEST_F(QnnHTPBackendTests, Reshape_4D_u16) {
  RunQDQReshapeTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                   TestInputDef<int64_t>({2}, true, {1, 48}),
                                   {},  // Attributes
                                   ExpectedEPNodeAssignment::All,
                                   19,     // Opset
                                   true);  // Use com.microsoft Q/DQ ops
}

// Test that int32 Reshape runs on HTP backend.
TEST_F(QnnHTPBackendTests, Reshape_4D_int32) {
  std::vector<int32_t> input_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  RunReshapeTestOnHTP<int32_t>(TestInputDef<int32_t>({1, 3, 2, 2}, false, input_data),
                               TestInputDef<int64_t>({3}, true, {1, 1, 12}),
                               {},  // Attributes
                               ExpectedEPNodeAssignment::All,
                               19);  // Opset
}

// Test QDQ Reshape with a shape value of 0 (copy dimension from input)
TEST_F(QnnHTPBackendTests, Reshape_4D_0MeansCopy) {
  RunQDQReshapeTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                  TestInputDef<int64_t>({3}, true, {1, 0, 16}),  // zero means copy => '(1, 3, 16)'
                                  {},                                            // Attributes
                                  ExpectedEPNodeAssignment::All,
                                  19);  // Opset
}

// Test QDQ Reshape with a shape value of -1 (dimension is inferred from the expected number of elements)
TEST_F(QnnHTPBackendTests, Reshape_4D_Neg1MeansInfer) {
  RunQDQReshapeTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                  TestInputDef<int64_t>({3}, true, {1, 3, -1}),  // -1 means infer => '(1, 3, 16)'
                                  {},                                            // Attributes
                                  ExpectedEPNodeAssignment::All,
                                  19);  // Opset
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

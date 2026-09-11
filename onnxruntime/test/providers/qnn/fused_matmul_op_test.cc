// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/constants.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a FusedMatMul operator on the QNN backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunFusedMatMulTest(const TestInputDef<DataType>& input_a_def,
                               const TestInputDef<DataType>& input_b_def,
                               bool transA,
                               bool transB,
                               bool transBatchA = false,
                               bool transBatchB = false,
                               float alpha = 1.0f,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               const std::string& backend_name = "cpu") {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;

  if (backend_name == "htp") {
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  auto model_builder = [input_a_def, input_b_def, transA, transB, transBatchA, transBatchB, alpha](ModelTestBuilder& builder) {
    NodeArg* input_a = MakeTestInput<DataType>(builder, input_a_def);
    NodeArg* input_b = MakeTestInput<DataType>(builder, input_b_def);
    std::vector<NodeArg*> inputs = {input_a, input_b};

    auto* output = builder.MakeOutput();

    Node& node = builder.AddNode("FusedMatMul", inputs, {output}, kMSDomain);
    node.AddAttribute("transA", static_cast<int64_t>(transA));
    node.AddAttribute("transB", static_cast<int64_t>(transB));
    node.AddAttribute("transBatchA", static_cast<int64_t>(transBatchA));
    node.AddAttribute("transBatchB", static_cast<int64_t>(transBatchB));
    node.AddAttribute("alpha", alpha);
  };

  RunQnnModelTest(model_builder,
                  provider_options,
                  13,  // opset version for contrib ops
                  expected_ep_assignment,
                  5e-3f);
}

// Tests the accuracy of a QDQ FusedMatMul model on QNN EP by comparing to CPU EP.
template <typename QType>
static void RunQDQFusedMatMulTest(const TestInputDef<float>& input_a_def,
                                  const TestInputDef<float>& input_b_def,
                                  bool transA,
                                  bool transB,
                                  bool transBatchA = false,
                                  bool transBatchB = false,
                                  float alpha = 1.0f,
                                  ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                                  const std::string& backend_name = "htp",
                                  bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestModelFn model_builder_fn = [input_a_def, input_b_def, transA, transB, transBatchA, transBatchB, alpha](ModelTestBuilder& builder) {
    NodeArg* input_a = MakeTestInput<float>(builder, input_a_def);
    NodeArg* input_b = MakeTestInput<float>(builder, input_b_def);
    std::vector<NodeArg*> inputs = {input_a, input_b};

    auto* output = builder.MakeOutput();

    Node& node = builder.AddNode("FusedMatMul", inputs, {output}, kMSDomain);
    node.AddAttribute("transA", static_cast<int64_t>(transA));
    node.AddAttribute("transB", static_cast<int64_t>(transB));
    node.AddAttribute("transBatchA", static_cast<int64_t>(transBatchA));
    node.AddAttribute("transBatchB", static_cast<int64_t>(transBatchB));
    node.AddAttribute("alpha", alpha);
  };

  GetTestQDQModelFn<QType> qdq_model_builder_fn = [input_a_def, input_b_def, transA, transB, transBatchA, transBatchB, alpha, use_contrib_qdq](
                                                      ModelTestBuilder& builder, std::vector<QuantParams<QType>>& output_qparams) {
    // Process input A with QDQ
    NodeArg* input_a = MakeTestInput<float>(builder, input_a_def);
    QuantParams<QType> input_a_qparams = GetTestInputQuantParams<QType>(input_a_def);
    NodeArg* input_a_qdq = AddQDQNodePair<QType>(builder, input_a, input_a_qparams.scale,
                                                 input_a_qparams.zero_point, use_contrib_qdq);

    // Process input B with QDQ
    NodeArg* input_b = MakeTestInput<float>(builder, input_b_def);
    QuantParams<QType> input_b_qparams = GetTestInputQuantParams<QType>(input_b_def);
    NodeArg* input_b_qdq = AddQDQNodePair<QType>(builder, input_b, input_b_qparams.scale,
                                                 input_b_qparams.zero_point, use_contrib_qdq);

    std::vector<NodeArg*> inputs = {input_a_qdq, input_b_qdq};

    // FusedMatMul -> op_output
    auto* op_output = builder.MakeIntermediate();
    Node& node = builder.AddNode("FusedMatMul", inputs, {op_output}, kMSDomain);
    node.AddAttribute("transA", static_cast<int64_t>(transA));
    node.AddAttribute("transB", static_cast<int64_t>(transB));
    node.AddAttribute("transBatchA", static_cast<int64_t>(transBatchA));
    node.AddAttribute("transBatchB", static_cast<int64_t>(transBatchB));
    node.AddAttribute("alpha", alpha);

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QType>(builder, op_output, output_qparams[0].scale,
                                                 output_qparams[0].zero_point, use_contrib_qdq);
  };

  TestQDQModelAccuracy(model_builder_fn,
                       qdq_model_builder_fn,
                       provider_options,
                       13,  // opset version for contrib ops
                       expected_ep_assignment,
                       QDQTolerance(5e-3f));
}

//
// CPU tests:
//

// Test FusedMatMul with default attributes (no transpose, alpha=1.0, no activation)
TEST_F(QnnCPUBackendTests, FusedMatMul_Default) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input A
      TestInputDef<float>({3, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input B
      false,                                                                    // transA
      false,                                                                    // transB
      false,                                                                    // transBatchA
      false,                                                                    // transBatchB
      1.0f,                                                                     // alpha
      ExpectedEPNodeAssignment::All);
}

// Test FusedMatMul with transpose A
TEST_F(QnnCPUBackendTests, FusedMatMul_TransposeA) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({3, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),   // input A
      TestInputDef<float>({3, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 12)),  // input B
      true,                                                                      // transA
      false,                                                                     // transB
      false,                                                                     // transBatchA
      false,                                                                     // transBatchB
      1.0f,                                                                      // alpha
      ExpectedEPNodeAssignment::All);
}

// Test FusedMatMul with transpose B
TEST_F(QnnCPUBackendTests, FusedMatMul_TransposeB) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input A
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input B
      false,                                                                    // transA
      true,                                                                     // transB
      false,                                                                    // transBatchA
      false,                                                                    // transBatchB
      1.0f,                                                                     // alpha
      ExpectedEPNodeAssignment::All);
}

// Test FusedMatMul with custom alpha
TEST_F(QnnCPUBackendTests, FusedMatMul_CustomAlpha) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input A
      TestInputDef<float>({3, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input B
      false,                                                                    // transA
      false,                                                                    // transB
      false,                                                                    // transBatchA
      false,                                                                    // transBatchB
      0.5f,                                                                     // alpha
      ExpectedEPNodeAssignment::All);
}

// Test FusedMatMul with all features combined
TEST_F(QnnCPUBackendTests, DISABLED_FusedMatMul_Combined) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({2, 4, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 24)),  // input A
      TestInputDef<float>({3, 4, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 12)),  // input B - adjusted shape for compatibility
      true,                                                                         // transA
      true,                                                                         // transB
      true,                                                                         // transBatchA
      true,                                                                         // transBatchB
      0.5f,                                                                         // alpha
      ExpectedEPNodeAssignment::All);
}

// Test FusedMatMul with higher rank tensors
TEST_F(QnnCPUBackendTests, FusedMatMul_HigherRank) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({2, 3, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 24)),  // input A
      TestInputDef<float>({2, 4, 5}, false, GetFloatDataInRange(-1.0f, 1.0f, 40)),  // input B
      false,                                                                        // transA
      false,                                                                        // transB
      false,                                                                        // transBatchA
      false,                                                                        // transBatchB
      1.0f,                                                                         // alpha
      ExpectedEPNodeAssignment::All);
}

// Test FusedMatMul with batch dimension transposition
TEST_F(QnnCPUBackendTests, FusedMatMul_BatchTranspose) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({2, 2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 16)),  // input A
      TestInputDef<float>({2, 4, 5}, false, GetFloatDataInRange(-1.0f, 1.0f, 40)),  // input B
      false,                                                                        // transA
      false,                                                                        // transB
      true,                                                                         // transBatchA
      false,                                                                        // transBatchB
      1.0f,                                                                         // alpha
      ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test FusedMatMul with default attributes on HTP
TEST_F(QnnHTPBackendTests, FusedMatMul_Default) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input A
      TestInputDef<float>({3, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input B
      false,                                                                    // transA
      false,                                                                    // transB
      false,                                                                    // transBatchA
      false,                                                                    // transBatchB
      1.0f,                                                                     // alpha
      ExpectedEPNodeAssignment::All,
      "htp");
}

// Test FusedMatMul with float16 inputs and custom alpha on HTP
TEST_F(QnnHTPBackendTests, FusedMatMul_Float16_CustomAlpha) {
  RunFusedMatMulTest<MLFloat16>(
      ConvertToFP16InputDef(TestInputDef<float>({2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6))),  // input A
      ConvertToFP16InputDef(TestInputDef<float>({3, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 6))),  // input B
      false,                                                                                           // transA
      false,                                                                                           // transB
      false,                                                                                           // transBatchA
      false,                                                                                           // transBatchB
      0.5f,                                                                                            // alpha
      ExpectedEPNodeAssignment::All,
      "htp");
}

// Test FusedMatMul with float16 inputs, transpose, and custom alpha on HTP
TEST_F(QnnHTPBackendTests, FusedMatMul_Float16_TransposeA_CustomAlpha) {
  RunFusedMatMulTest<MLFloat16>(
      ConvertToFP16InputDef(TestInputDef<float>({3, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 6))),   // input A
      ConvertToFP16InputDef(TestInputDef<float>({3, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 12))),  // input B
      true,                                                                                             // transA
      false,                                                                                            // transB
      false,                                                                                            // transBatchA
      false,                                                                                            // transBatchB
      1.702f,                                                                                           // alpha
      ExpectedEPNodeAssignment::All,
      "htp");
}

// Test FusedMatMul with batch dimension transposition on HTP
TEST_F(QnnHTPBackendTests, FusedMatMul_BatchTranspose) {
  RunFusedMatMulTest<float>(
      TestInputDef<float>({2, 2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 16)),  // input A
      TestInputDef<float>({2, 4, 5}, false, GetFloatDataInRange(-1.0f, 1.0f, 40)),  // input B
      false,                                                                        // transA
      false,                                                                        // transB
      true,                                                                         // transBatchA
      false,                                                                        // transBatchB
      1.0f,                                                                         // alpha
      ExpectedEPNodeAssignment::All,
      "htp");
}

// Test 8-bit QDQ FusedMatMul with default attributes on HTP
TEST_F(QnnHTPBackendTests, FusedMatMul_QDQ_U8_Default) {
  RunQDQFusedMatMulTest<uint8_t>(
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input A
      TestInputDef<float>({3, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input B
      false,                                                                    // transA
      false,                                                                    // transB
      false,                                                                    // transBatchA
      false,                                                                    // transBatchB
      1.0f,                                                                     // alpha
      ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ FusedMatMul with batch dimension transposition on HTP
TEST_F(QnnHTPBackendTests, FusedMatMul_QDQ_U8_BatchTranspose) {
  RunQDQFusedMatMulTest<uint8_t>(
      TestInputDef<float>({2, 2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 16)),  // input A
      TestInputDef<float>({2, 4, 5}, false, GetFloatDataInRange(-1.0f, 1.0f, 40)),  // input B
      false,                                                                        // transA
      false,                                                                        // transB
      true,                                                                         // transBatchA
      false,                                                                        // transBatchB
      1.0f,                                                                         // alpha
      ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ FusedMatMul with default attributes on HTP
TEST_F(QnnHTPBackendTests, FusedMatMul_QDQ_U16_Default) {
  RunQDQFusedMatMulTest<uint16_t>(
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input A
      TestInputDef<float>({3, 2}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),  // input B
      false,                                                                    // transA
      false,                                                                    // transB
      false,                                                                    // transBatchA
      false,                                                                    // transBatchB
      1.0f,                                                                     // alpha
      ExpectedEPNodeAssignment::All,
      "htp",
      true);  // Use com.microsoft Q/DQ ops
}

// Test 16-bit QDQ FusedMatMul with batch dimension transposition on HTP
TEST_F(QnnHTPBackendTests, FusedMatMul_QDQ_U16_BatchTranspose) {
  RunQDQFusedMatMulTest<uint16_t>(
      TestInputDef<float>({2, 2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 16)),  // input A
      TestInputDef<float>({2, 4, 5}, false, GetFloatDataInRange(-1.0f, 1.0f, 40)),  // input B
      false,                                                                        // transA
      false,                                                                        // transB
      true,                                                                         // transBatchA
      false,                                                                        // transBatchB
      1.0f,                                                                         // alpha
      ExpectedEPNodeAssignment::All,
      "htp",
      true);  // Use com.microsoft Q/DQ ops
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

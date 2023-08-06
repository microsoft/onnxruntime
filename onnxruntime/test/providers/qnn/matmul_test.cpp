// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that creates a graph with MatMul operator.
static GetTestModelFn BuildMatMulOpTestCase(const TestInputDef<float>& input1_def,
                                            const TestInputDef<float>& input2_def) {
  return [input1_def, input2_def](ModelTestBuilder& builder) {
    NodeArg* input1 = MakeTestInput(builder, input1_def);
    NodeArg* input2 = MakeTestInput(builder, input2_def);
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("MatMul", {input1, input2}, {output});
  };
}

// Returns a function that creates a graph with a QDQ MatMul operator.
template <typename QuantType>
static GetTestQDQModelFn<QuantType> BuildMatMulOpQDQTestCase(const TestInputDef<float>& input1_def,
                                                             const TestInputDef<float>& input2_def) {
  return [input1_def, input2_def](ModelTestBuilder& builder,
                                  std::vector<QuantParams<QuantType>>& output_qparams) {
    // input1 -> Q -> DQ ->
    NodeArg* input1 = MakeTestInput(builder, input1_def);
    QuantParams<QuantType> input1_qparams = GetTestInputQuantParams(input1_def);
    auto* input1_qdq = AddQDQNodePair<QuantType>(builder, input1, input1_qparams.scale, input1_qparams.zero_point);

    // input2 -> Q -> DQ ->
    NodeArg* input2 = MakeTestInput(builder, input2_def);
    QuantParams<QuantType> input2_qparams = GetTestInputQuantParams(input2_def);
    auto* input2_qdq = AddQDQNodePair<QuantType>(builder, input2, input2_qparams.scale, input2_qparams.zero_point);

    // MatMul
    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("MatMul", {input1_qdq, input2_qdq}, {op_output});

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, op_output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point);
  };
}

// Runs an MatMul model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunMatMulOpOpTest(const TestInputDef<float>& input1_def,
                              const TestInputDef<float>& input2_def,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              int opset = 13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildMatMulOpTestCase(input1_def, input2_def),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  2e-4f);
}

// Runs a QDQ MatMul model on the QNN HTP backend. Checks the graph node assignment, and that the
// QDQ model is accurate on QNN EP (compared to CPU EP).
template <typename QuantType>
static void RunQDQMatMulOpOpTest(const TestInputDef<float>& input1_def,
                                 const TestInputDef<float>& input2_def,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildMatMulOpTestCase(input1_def, input2_def),
                       BuildMatMulOpQDQTestCase<QuantType>(input1_def, input2_def),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       1e-5f);
}

//
// CPU tests:
//

TEST_F(QnnCPUBackendTests, MatMulOp) {
  RunMatMulOpOpTest(TestInputDef<float>({2, 3}, false, {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f}),
                    TestInputDef<float>({3, 2}, false, {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f}),
                    ExpectedEPNodeAssignment::All, 18);
}

// Test MatMul broadcasting
TEST_F(QnnCPUBackendTests, MatMulOp_Broadcast) {
  RunMatMulOpOpTest(TestInputDef<float>({28, 1, 64}, false, -10.0f, 10.0f),
                    TestInputDef<float>({64, 32}, false, -10.0f, 10.0f),
                    ExpectedEPNodeAssignment::All, 18);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, MatMulOp_HTP_u8) {
  RunQDQMatMulOpOpTest<uint8_t>(TestInputDef<float>({2, 3}, false, {-10.0f, -4.0f, -2.0f, 0.0f, 5.0f, 10.0f}),
                                TestInputDef<float>({3, 2}, false, {-10.0f, -6.0f, -1.0f, 0.0f, 3.0f, 10.0f}),
                                ExpectedEPNodeAssignment::All, 18);
}

// Test MatMul broadcasting
TEST_F(QnnHTPBackendTests, MatMulOp_Broadcast) {
  RunQDQMatMulOpOpTest<uint8_t>(TestInputDef<float>({28, 1, 64}, false, -10.0f, 10.0f),
                                TestInputDef<float>({64, 32}, false, -10.0f, 10.0f),
                                ExpectedEPNodeAssignment::All, 18);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

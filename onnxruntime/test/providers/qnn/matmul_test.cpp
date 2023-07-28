// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that creates a graph with MatMul operator.
static GetTestModelFn BuildMatMulOpTestCase(const std::vector<int64_t>& input1_shape,
                                            const std::vector<int64_t>& input2_shape) {
  return [input1_shape, input2_shape](ModelTestBuilder& builder) {
    // Random input data
    auto input1 = builder.MakeInput<float>(input1_shape, 0.0f, 10.0f);
    auto input2 = builder.MakeInput<float>(input2_shape, 0.0f, 10.0f);

    auto* output = builder.MakeOutput();
    builder.AddNode("MatMul", {input1, input2}, {output});
  };
}

// Returns a function that creates a graph with a QDQ AveragePool operator.
template <typename QuantType>
GetQDQTestCaseFn BuildMatMulOpQDQTestCase(const std::vector<int64_t>& input1_shape,
                                          const std::vector<int64_t>& input2_shape) {
  return [input1_shape, input2_shape](ModelTestBuilder& builder) {
    float pool_output_scale = 0.0038f;
    float q_scale = 0.0038f;
    QuantType pool_output_zp = std::numeric_limits<QuantType>::max() / 2;
    QuantType q_zp = std::numeric_limits<QuantType>::max() / 2;

    auto* input_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    using InputLimits = std::numeric_limits<QuantType>;

    // add QDQ input
    auto* q1_output = builder.MakeIntermediate();
    auto* dq1_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(input_arg,
                                             pool_output_scale,
                                             pool_output_zp,
                                             q1_output);
    builder.AddDequantizeLinearNode<QuantType>(q1_output,
                                               q_scale,
                                               q_zp,
                                               dq1_output);

    // add input b initializer (NNAPI only supports case of MatMul A*B - B is an initializer)
    auto* dq_2_output = builder.MakeIntermediate();
    auto* input_b = builder.MakeInitializer<QuantType>(input2_shape, InputLimits::min(), InputLimits::max());
    builder.AddDequantizeLinearNode<QuantType>(input_b,
                                               q_scale,
                                               q_zp,
                                               dq_2_output);

    // add MatMul operator
    auto* matmul_op_output = builder.MakeIntermediate();
    builder.AddNode("MatMul", {dq1_output, dq_2_output}, {matmul_op_output});

    // add QDQ output
    auto* q3_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(matmul_op_output,
                                             pool_output_scale,
                                             pool_output_zp,
                                             q3_output);
    builder.AddDequantizeLinearNode<QuantType>(q3_output,
                                               q_scale,
                                               q_zp,
                                               output_arg);
  };
}

// Runs an AveragePool model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunMatMulOpOpTest(const std::vector<int64_t>& input1_shape,
                              const std::vector<int64_t>& input2_shape,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              int opset = 13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildMatMulOpTestCase(input1_shape, input2_shape),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ AveragePool model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
template <typename QuantType>
static void RunQDQMatMulOpOpTest(const std::vector<int64_t>& input1_shape,
                                 const std::vector<int64_t>& input2_shape,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 18, float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildMatMulOpQDQTestCase<QuantType>(input1_shape, input2_shape),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//

TEST_F(QnnCPUBackendTests, TestMatMulOp) {
  RunMatMulOpOpTest({2, 2} /* input_shape1 */,
                    {2, 2} /* input_shape2 */,
                    ExpectedEPNodeAssignment::All, 18);
}

// QNN broadcast issue
TEST_F(QnnCPUBackendTests, DISABLED_TestMatMulOp2) {
  RunMatMulOpOpTest({28, 1, 64} /* input_shape1 */,
                    {64, 32} /* input_shape2 */,
                    ExpectedEPNodeAssignment::All, 18);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, TestMatMulOp_HTP_u8) {
  RunQDQMatMulOpOpTest<uint8_t>({2, 2} /* input_shape1 */,
                                {2, 2} /* input_shape2 */,
                                ExpectedEPNodeAssignment::All,
                                18, 0.00381f);
}

// QNN broadcast issue
TEST_F(QnnHTPBackendTests, DISABLED_TestMatMulOp2_HTP_u8) {
  RunQDQMatMulOpOpTest<uint8_t>({28, 1, 64} /* input_shape1 */,
                                {64, 32} /* input_shape2 */,
                                ExpectedEPNodeAssignment::All,
                                18, 0.00381f);
}

// QNN broadcast issue
TEST_F(QnnHTPBackendTests, DISABLED_TestMatMulOp3_HTP_u8) {
  RunQDQMatMulOpOpTest<uint8_t>({28, 1, 32} /* input_shape1 */,
                                {32, 2} /* input_shape2 */,
                                ExpectedEPNodeAssignment::All,
                                18, 0.00381f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

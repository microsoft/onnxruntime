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

// Runs a model with a Gemm operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunGemmTestOnCPU(const std::vector<TestInputDef<DataType>>& input_defs,
                             const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase("Gemm", input_defs, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Gemm with non-default 'alpha' or 'beta' attributes is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Gemm_NonDefaultAlphaBeta_Unsupported) {
  // Check that alpha != 1.0f is not supported.
  RunGemmTestOnCPU<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                           TestInputDef<float>({2, 4}, false, -10.0f, 10.0f)},
                          {utils::MakeAttribute("alpha", 1.5f)},
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.

  // Check that beta != 1.0f is not supported.
  RunGemmTestOnCPU<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                           TestInputDef<float>({2, 4}, false, -10.0f, 10.0f),
                           TestInputDef<float>({1, 4}, false, -1.0f, 1.0f)},
                          {utils::MakeAttribute("beta", 1.2f)},
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test that Gemm with general 2D bias (M, N) is NOT supported (unless M == 1).
// QNN's FullyConnected operator only supports `outputVector = ( inputAsVector * weightsMatrix ) + biasesVector`
TEST_F(QnnCPUBackendTests, Gemm_2D_Bias_Unsupported) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 12);

  // 2D matrix mul with bias not supported.
  RunGemmTestOnCPU<float>({TestInputDef<float>({2, 3}, false, input_a_data),
                           TestInputDef<float>({3, 4}, false, input_b_data),
                           TestInputDef<float>({2, 4}, false, -1.0f, 1.0f)},
                          {},
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.

  // However, 2D matrix mul without a bias is supported. Input A's 0th dimension is interpreted as `batch_size`.
  RunGemmTestOnCPU<float>({TestInputDef<float>({2, 3}, false, input_a_data),
                           TestInputDef<float>({3, 4}, false, input_b_data)},
                          {},
                          ExpectedEPNodeAssignment::All);  // Assigned to QNN EP.
}

// Test Gemm with dynamic (i.e., not initializer) inputs (A, B, Bias).
TEST_F(QnnCPUBackendTests, Gemm_Dynamic_A_B_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTestOnCPU<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                           TestInputDef<float>({6, 4}, false, input_b_data),
                           TestInputDef<float>({1, 4}, false, input_c_data)},
                          {},
                          ExpectedEPNodeAssignment::All);
}

// Test Gemm with static B and Bias inputs.
TEST_F(QnnCPUBackendTests, Gemm_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTestOnCPU<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                           TestInputDef<float>({6, 4}, true, input_b_data),
                           TestInputDef<float>({1, 4}, true, input_c_data)},
                          {},
                          ExpectedEPNodeAssignment::All);
}

// Test Gemm with transposed A/B and static B and Bias inputs.
TEST_F(QnnCPUBackendTests, Gemm_TransAB_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTestOnCPU<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                           TestInputDef<float>({4, 6}, true, input_b_data),
                           TestInputDef<float>({1, 4}, true, input_c_data)},
                          {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                           utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                          ExpectedEPNodeAssignment::All);
}

// Test Gemm with transposed A/B and dynamic (i.e., not initializer) B and Bias inputs.
TEST_F(QnnCPUBackendTests, Gemm_TransAB_Dynamic_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTestOnCPU<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                           TestInputDef<float>({4, 6}, false, input_b_data),
                           TestInputDef<float>({1, 4}, false, input_c_data)},
                          {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                           utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                          ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

#if 0
// Runs a QDQ Gemm model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (when compared to the baseline float32 model).
template <typename QType>
static void RunQDQGemmTestOnHTP(const TestInputDef<float>& input0_def,
                                const TestInputDef<float>& input1_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy<QType>(BuildOpTestCase<float>("Gemm", {input0_def, input1_def}, attrs),     // baseline float32 model
                              BuildQDQOpTestCase<QType>("Gemm", {input0_def, input1_def}, attrs),  // QDQ model
                              provider_options,
                              opset,
                              expected_ep_assignment);
}
#endif

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

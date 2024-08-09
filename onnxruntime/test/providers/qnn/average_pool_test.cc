// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>
#include <vector>

#include "core/graph/node_attr_utils.h"
#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs an AveragePool model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunAveragePoolOpTest(const std::string& op_type,
                                 const std::vector<TestInputDef<float>>& input_defs,
                                 const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<float>(op_type, input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ AveragePool model on the QNN HTP backend. Checks the graph node assignment, and that accuracy
// on QNN EP is at least as good as on CPU EP.
template <typename QuantType>
static void RunQDQAveragePoolOpTest(const std::string& op_type,
                                    const std::vector<TestInputDef<float>>& input_defs,
                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    int opset = 18,
                                    QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, input_defs, {}, attrs),
                       BuildQDQOpTestCase<QuantType>(op_type, input_defs, {}, attrs),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

//
// CPU tests:
//

// AveragePool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnCPUBackendTests, AveragePool_AsGlobal) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                        utils::MakeAttribute("strides", std::vector<int64_t>{3, 3})},
                       ExpectedEPNodeAssignment::All);
}

// Test GlobalAveragePool on QNN CPU backend.
TEST_F(QnnCPUBackendTests, GlobalAveragePool) {
  RunAveragePoolOpTest("GlobalAveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {},
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that counts padding.
TEST_F(QnnCPUBackendTests, AveragePool_CountIncludePad) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                        utils::MakeAttribute("count_include_pad", static_cast<int64_t>(1))},
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnCPUBackendTests, AveragePool_AutopadSameUpper) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                        utils::MakeAttribute("count_include_pad", static_cast<int64_t>(1)),
                        utils::MakeAttribute("auto_pad", "SAME_UPPER")},
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnCPUBackendTests, AveragePool_AutopadSameLower) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                        utils::MakeAttribute("count_include_pad", static_cast<int64_t>(1)),
                        utils::MakeAttribute("auto_pad", "SAME_LOWER")},
                       ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// QDQ AveragePool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnHTPBackendTests, AveragePool_AsGlobal) {
  std::vector<float> input = {32.1289f, -59.981f, -17.2799f, 62.7263f, 33.6205f, -19.3515f, -54.0113f, 37.5648f, 61.5357f,
                              -52.5769f, 27.3637f, -9.01382f, -65.5612f, 19.9497f, -47.9228f, 26.9813f, 83.064f, 0.362503f};
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                    utils::MakeAttribute("strides", std::vector<int64_t>{3, 3})},
                                   ExpectedEPNodeAssignment::All);
}

// Test accuracy for 8-bit QDQ GlobalAveragePool with input of rank 4.
TEST_F(QnnHTPBackendTests, GlobalAveragePool) {
  std::vector<float> input = GetFloatDataInRange(-32.0f, 32.0f, 18);

  RunQDQAveragePoolOpTest<uint8_t>("GlobalAveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {},
                                   ExpectedEPNodeAssignment::All);
}

// QDQ AveragePool that counts padding.
TEST_F(QnnHTPBackendTests, AveragePool_CountIncludePad_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                    utils::MakeAttribute("count_include_pad", static_cast<int64_t>(1))},
                                   ExpectedEPNodeAssignment::All,
                                   18,
                                   // Need tolerance of 0.414% of output range after QNN SDK 2.17
                                   QDQTolerance(0.00414f));
}

// QDQ AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnHTPBackendTests, AveragePool_AutopadSameUpper_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                    utils::MakeAttribute("auto_pad", "SAME_UPPER")},
                                   ExpectedEPNodeAssignment::All,
                                   18,
                                   // Need to use tolerance of 0.414% of output range after QNN SDK 2.17
                                   QDQTolerance(0.00414f));
}

// QDQ AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnHTPBackendTests, AveragePool_AutopadSameLower_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                    utils::MakeAttribute("auto_pad", "SAME_LOWER")},
                                   ExpectedEPNodeAssignment::All,
                                   18,
                                   // Need to use tolerance of 0.414% of output range after QNN SDK 2.17
                                   QDQTolerance(0.00414f));
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
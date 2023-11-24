// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs an Max/Min model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunCPUMinOrMaxOpTest(const std::string& op_type,
                                 const std::vector<TestInputDef<float>>& input_defs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<float>(op_type, input_defs, {}, {}, kOnnxDomain),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ Max/Min model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment, and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (when compared to the baseline float32 model).
template <typename QType = uint8_t>
static void RunQDQMinOrMaxOpTest(const std::string& op_type,
                                 const std::vector<TestInputDef<float>>& input_defs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, input_defs, {}, {}, kOnnxDomain),     // baseline float32 model
                       BuildQDQOpTestCase<QType>(op_type, input_defs, {}, {}, kOnnxDomain),  // QDQ model
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Min with 1 input is *NOT* supported on CPU backend.
TEST_F(QnnCPUBackendTests, Min_1Input_NotSupported) {
  RunCPUMinOrMaxOpTest("Min",
                       {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
                       ExpectedEPNodeAssignment::None, 13);
}

// Test that Max with 1 input is *NOT* supported on CPU backend.
TEST_F(QnnCPUBackendTests, Max_1Input_NotSupported) {
  RunCPUMinOrMaxOpTest("Max",
                       {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
                       ExpectedEPNodeAssignment::None, 13);
}

// Test Min with 2 inputs on CPU backend.
TEST_F(QnnCPUBackendTests, Min_2Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunCPUMinOrMaxOpTest("Min",
                       {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                       ExpectedEPNodeAssignment::All, 13);
}

// Test Max with 2 inputs on CPU backend.
TEST_F(QnnCPUBackendTests, Max_2Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunCPUMinOrMaxOpTest("Max",
                       {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                        TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                       ExpectedEPNodeAssignment::All, 13);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test that Min with 1 input is *NOT* supported on HTP backend.
TEST_F(QnnHTPBackendTests, Min_1Input_NotSupported) {
  RunQDQMinOrMaxOpTest("Min",
                       {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
                       ExpectedEPNodeAssignment::None, 13);
}

// Test that Max with 1 input is *NOT* supported on HTP backend.
TEST_F(QnnHTPBackendTests, Max_1Input_NotSupported) {
  RunQDQMinOrMaxOpTest("Max",
                       {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
                       ExpectedEPNodeAssignment::None, 13);
}

// Test accuracy of 8-bit Q/DQ Min with 2 inputs on HTP backend.
TEST_F(QnnHTPBackendTests, Min_2Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQMinOrMaxOpTest<uint8_t>("Min",
                                {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                 TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                                ExpectedEPNodeAssignment::All, 13);
}

// Test accuracy of 8-bit Q/DQ Max with 2 inputs on HTP backend.
TEST_F(QnnHTPBackendTests, Max_2Inputs) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQMinOrMaxOpTest<uint8_t>("Max",
                                {TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                                 TestInputDef<float>({1, 3, 4, 4}, false, input_data)},
                                ExpectedEPNodeAssignment::All, 13);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

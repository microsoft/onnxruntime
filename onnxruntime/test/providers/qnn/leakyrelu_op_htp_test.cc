// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Checks the accuracy of a QDQ LeakyRelu model by comparing to ORT CPU EP.
template <typename QuantType>
static void RunLeakyReluOpQDQTest(const TestInputDef<float>& input_def,
                                  const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                  int opset,
                                  ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase<float>("LeakyRelu", {input_def}, {}, attrs),
                       BuildQDQOpTestCase<QuantType>("LeakyRelu", {input_def}, {}, attrs),
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
TEST_F(QnnHTPBackendTests, LeakyReluOpSet15) {
  RunLeakyReluOpQDQTest<uint8_t>(TestInputDef<float>({1, 2, 3}, false, {-40.0f, -20.0f, 0.0f, 10.0f, 30.0f, 40.0f}),
                                 {utils::MakeAttribute("alpha", 0.2f)},
                                 15,
                                 ExpectedEPNodeAssignment::All);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
TEST_F(QnnHTPBackendTests, LeakyReluOpSet16) {
  RunLeakyReluOpQDQTest<uint8_t>(TestInputDef<float>({1, 2, 3}, false, {-40.0f, -20.0f, 0.0f, 10.0f, 30.0f, 40.0f}),
                                 {utils::MakeAttribute("alpha", 0.2f)},
                                 16,
                                 ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
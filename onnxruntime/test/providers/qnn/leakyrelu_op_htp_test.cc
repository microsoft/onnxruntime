// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

/**
 * Runs a LeakyRelu op model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param op_type The LeakyRelu op type (e.g., ReduceSum).
 * \param opset The opset version.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None)
 */
template <typename QuantType>
static void RunLeakyReluOpQDQTest(int opset,
                                  ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildQDQLeakyReluOpTestCase<QuantType>({2, 3, 4}),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQLeakyReluOpSet15) {
  RunLeakyReluOpQDQTest<uint8_t>(15);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQLeakyReluOpSet16) {
  RunLeakyReluOpQDQTest<uint8_t>(16);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
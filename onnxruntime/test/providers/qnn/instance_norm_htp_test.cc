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
 * Runs an InstanceNormalization model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_shape The input's shape.
 * \param epsilon The epsilon attribute.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param num_modes_in_graph The number of expected nodes in the graph.
 */
static void RunInstanceNormQDQTest(const std::vector<int64_t>& input_shape, float epsilon,
                                   ExpectedEPNodeAssignment expected_ep_assignment, int num_nodes_in_graph) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQInstanceNormTestCase<uint8_t, uint8_t, int32_t>(input_shape, epsilon),
                  provider_options,
                  18,
                  expected_ep_assignment,
                  num_nodes_in_graph);
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8) {
  RunInstanceNormQDQTest({1, 2, 3, 3}, 1e-05f, ExpectedEPNodeAssignment::All, 1);
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8Rank3) {
  RunInstanceNormQDQTest({1, 2, 3}, 1e-05f, ExpectedEPNodeAssignment::All, 1);
}

// Check that QNN InstanceNorm operator does not handle inputs with rank > 4.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8Rank5) {
  // No nodes should be assigned to QNN EP, and graph should have 5 (non-fused) nodes.
  RunInstanceNormQDQTest({1, 2, 3, 3, 3}, 1e-05f, ExpectedEPNodeAssignment::None, 5);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
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
 * Runs a Gather op model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param opset The opset version.
 * \param scalar_indices whether the incidices input is scalar or not.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None)
 */
template <typename QuantType, typename IndicesType>
static void RunGatherOpQDQTest(int opset, bool scalar_indices = false,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  if (scalar_indices) {
    RunQnnModelTest(BuildQDQGatherOpScalarIndicesTestCase<QuantType, IndicesType>({2, 3, 4},  // input shape
                                                                                  1,          // indices
                                                                                  1),         // axis
                    provider_options,
                    opset,
                    expected_ep_assignment);
  } else {
    RunQnnModelTest(BuildQDQGatherOpTestCase<QuantType, IndicesType>({2, 3, 4},                    // input shape
                                                                     std::vector<IndicesType>{1},  // indices
                                                                     {1},                          // indices_shape
                                                                     1),                           // axis
                    provider_options,
                    opset,
                    expected_ep_assignment);
  }
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQGatherOpU8) {
  RunGatherOpQDQTest<uint8_t, int64_t>(11);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQGatherOpI8) {
  RunGatherOpQDQTest<int8_t, int32_t>(11);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQGatherOpScalarIndicesU8) {
  RunGatherOpQDQTest<uint8_t, int64_t>(11, true);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQGatherOpScalarIndicesI8) {
  RunGatherOpQDQTest<int8_t, int32_t>(11, true);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
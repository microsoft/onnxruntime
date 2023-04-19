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
 * Runs a ReduceOp model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param op_type The ReduceOp type (e.g., ReduceSum).
 * \param opset The opset version. Some opset versions have "axes" as an attribute or input.
 * \param test_description Description of the test for error reporting.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None)
 * \param keepdims Common attribute for all reduce operations.
 */
template <typename QuantType>
static void RunReduceOpQDQTest(const std::string& op_type, int opset, const char* test_description,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               bool keepdims = true) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildQDQReduceOpTestCase<QuantType>(op_type,
                                                      {2, 2},                                // Input shape
                                                      ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                      {0, 1},                                // Axes
                                                      keepdims,                              // keepdims
                                                      false),                                // noop_with_empty_axes
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description);
}

//
// ReduceSum
//

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceSum", 13, "TestQDQReduceSumU8Opset13");
}

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 11, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumU8Opset11) {
  RunReduceOpQDQTest<uint8_t>("ReduceSum", 11, "TestQDQReduceSumU8Opset11");
}

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumS8Opset13) {
  RunReduceOpQDQTest<int8_t>("ReduceSum", 13, "TestQDQReduceSumS8Opset13");
}

//
// ReduceMax
//

// ReduceMax on Linux's HTP emulator is always off by an amount equal to the final DQ.scale
// Works fine on windows arm64.
#if !defined(__linux__)
// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMaxU8Opset18) {
  RunReduceOpQDQTest<uint8_t>("ReduceMax", 18, "TestQDQReduceMaxU8Opset18");
}

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMaxU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMax", 13, "TestQDQReduceMaxU8Opset13");
}

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMaxS8Opset18) {
  RunReduceOpQDQTest<int8_t>("ReduceMax", 18, "TestQDQReduceMaxS8Opset18");
}
#endif  // !defined(__linux__)

//
// ReduceMin
//
// ReduceMin on Linux's HTP emulator is always off by an amount equal to the final DQ.scale
// Works fine on windows arm64.
#if !defined(__linux__)
// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMinU8Opset18) {
  RunReduceOpQDQTest<uint8_t>("ReduceMin", 18, "TestQDQReduceMinU8Opset18");
}

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMinU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMin", 13, "TestQDQReduceMinU8Opset13");
}

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// Uses int8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQReduceMinS8Opset18) {
  RunReduceOpQDQTest<int8_t>("ReduceMin", 18, "TestQDQReduceMinS8Opset18");
}
#endif  // !defined(__linux__)

//
// ReduceMean
//

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMeanU8Opset18) {
  RunReduceOpQDQTest<uint8_t>("ReduceMean", 18, "TestQDQReduceMeanU8Opset18");
}

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMeanU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMean", 13, "TestQDQReduceMeanU8Opset13");
}

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMeanS8Opset18) {
  RunReduceOpQDQTest<int8_t>("ReduceMean", 18, "TestQDQReduceMeanS8Opset18");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
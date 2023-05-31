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

// Creates a graph with a single logical operator (e.g., Equal). Used for testing CPU backend.
static GetTestModelFn BuildLogicalOpTestCase(const std::string& op_type, const std::vector<int64_t>& shape) {
  return [op_type, shape](ModelTestBuilder& builder) {
    auto* input0 = builder.MakeInput<float>(shape, 0.0f, 20.0f);
    auto* input1 = builder.MakeInput<float>(shape, 0.0f, 20.0f);
    auto* output = builder.MakeOutput();

    builder.AddNode(op_type, {input0, input1}, {output});
  };
}

// Creates a graph with a single Q/DQ logical operator. Used for testing HTP backend.
template <typename InputQType = uint8_t>
static GetTestModelFn BuildQDQLogicalOpTestCase(const std::string& op_type, const std::vector<int64_t>& shape) {
  return [op_type, shape](ModelTestBuilder& builder) {
    const InputQType zero_point = std::numeric_limits<InputQType>::max() / 2;
    constexpr float qdq_scale = 0.0038f;

    auto* input0 = builder.MakeInput<float>(shape, -1.0f, 1.0f);
    auto* input1 = builder.MakeInput<float>(shape, -1.0f, 1.0f);
    auto* output = builder.MakeOutput();

    // input -> Q -> DQ -> Op
    auto* qdq0_output = AddQDQNodePair<InputQType>(builder, input0, qdq_scale, zero_point);
    auto* qdq1_output = AddQDQNodePair<InputQType>(builder, input1, qdq_scale, zero_point);

    // Op -> output
    builder.AddNode(op_type, {qdq0_output, qdq1_output}, {output});
  };
}

// Runs a model with a logical operator on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunCPULogicalOpTest(const std::string& op_type, const std::vector<int64_t>& shape,
                                ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                                int opset = 17) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
  fp32_abs_err = 1.5e-5f;  // On linux we need slightly larger tolerance.
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildLogicalOpTestCase(op_type, shape),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description);
}

// Runs a model with a logical operator on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename QuantType>
static void RunQDQLogicalOpTest(const std::string& op_type, const std::vector<int64_t>& shape,
                                ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                                int opset = 17) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildQDQLogicalOpTestCase<QuantType>(op_type, shape),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description);
}

//
// CPU tests:
//

TEST_F(QnnCPUBackendTests, LogicalOpEqual4D) {
  RunCPULogicalOpTest("Equal", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpEqual4D");
}

TEST_F(QnnCPUBackendTests, LogicalOpGreater4D) {
  RunCPULogicalOpTest("Greater", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpGreater4D");
}

// TODO: Add support for GreaterOrEqual.
// According to the ONNX spec, logical operators that are implemented as
// functions (e.g., LessOrEqual or GreaterOrEqual) do not have output type/shape inference.
// We need to handle this uniquely.
TEST_F(QnnCPUBackendTests, DISABLED_LogicalOpGreaterOrEqual4D) {
  RunCPULogicalOpTest("GreaterOrEqual", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpGreaterOrEqual4D");
}

TEST_F(QnnCPUBackendTests, LogicalOpLess4D) {
  RunCPULogicalOpTest("Less", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpLess4D");
}

// TODO: Add support for LessOrEqual.
// According to the ONNX spec, logical operators that are implemented as
// functions (e.g., LessOrEqual or GreaterOrEqual) do not have output type/shape inference.
// We need to handle this uniquely.
TEST_F(QnnCPUBackendTests, DISABLED_LogicalOpLessOrEqual4D) {
  RunCPULogicalOpTest("LessOrEqual", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpLessOrEqual4D");
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, LogicalOpEqual4D) {
  RunQDQLogicalOpTest<uint8_t>("Equal", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpEqual4D");
}

TEST_F(QnnHTPBackendTests, LogicalOpGreater4D) {
  RunQDQLogicalOpTest<uint8_t>("Greater", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpGreater4D");
}

// TODO: Add support for GreaterOrEqual.
// According to the ONNX spec, logical operators that are implemented as
// functions (e.g., LessOrEqual or GreaterOrEqual) do not have output type/shape inference.
// We need to handle this uniquely.
TEST_F(QnnHTPBackendTests, DISABLED_LogicalOpGreaterOrEqual4D) {
  RunQDQLogicalOpTest<uint8_t>("GreaterOrEqual", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpGreaterOrEqual4D");
}

TEST_F(QnnHTPBackendTests, LogicalOpLess4D) {
  RunQDQLogicalOpTest<uint8_t>("Less", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpLess4D");
}

// TODO: Add support for LessOrEqual.
// According to the ONNX spec, logical operators that are implemented as
// functions (e.g., LessOrEqual or GreaterOrEqual) do not have output type/shape inference.
// We need to handle this uniquely.
TEST_F(QnnHTPBackendTests, DISABLED_LogicalOpLessOrEqual4D) {
  RunQDQLogicalOpTest<uint8_t>("LessOrEqual", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All, "LogicalOpLessOrEqual4D");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

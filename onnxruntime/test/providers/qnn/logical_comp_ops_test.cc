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
    NodeArg* output = nullptr;

    // Explicitly set output type/shape for logical comparison ops implemented as functions.
    if (op_type == "GreaterOrEqual" || op_type == "LessOrEqual") {
      output = builder.MakeOutput<bool>(shape);
    } else {
      output = builder.MakeOutput();
    }

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
    NodeArg* output = nullptr;

    // Explicitly set output type/shape for logical comparison ops implemented as functions.
    if (op_type == "GreaterOrEqual" || op_type == "LessOrEqual") {
      output = builder.MakeOutput<bool>(shape);
    } else {
      output = builder.MakeOutput();
    }

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
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 17) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildLogicalOpTestCase(op_type, shape),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a model with a logical operator on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename QuantType>
static void RunQDQLogicalOpTest(const std::string& op_type, const std::vector<int64_t>& shape,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 17) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildQDQLogicalOpTestCase<QuantType>(op_type, shape),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

TEST_F(QnnCPUBackendTests, LogicalOpEqual4D) {
  RunCPULogicalOpTest("Equal", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LogicalOpGreater4D) {
  RunCPULogicalOpTest("Greater", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LogicalOpGreaterOrEqual4D) {
  RunCPULogicalOpTest("GreaterOrEqual", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LogicalOpLess4D) {
  RunCPULogicalOpTest("Less", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LogicalOpLessOrEqual4D) {
  RunCPULogicalOpTest("LessOrEqual", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, LogicalOpEqual4D) {
  RunQDQLogicalOpTest<uint8_t>("Equal", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogicalOpGreater4D) {
  RunQDQLogicalOpTest<uint8_t>("Greater", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogicalOpGreaterOrEqual4D) {
  RunQDQLogicalOpTest<uint8_t>("GreaterOrEqual", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogicalOpLess4D) {
  RunQDQLogicalOpTest<uint8_t>("Less", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogicalOpLessOrEqual4D) {
  RunQDQLogicalOpTest<uint8_t>("LessOrEqual", {1, 3, 16, 16}, ExpectedEPNodeAssignment::All);
}

// Test for bug 44777546.
// Tests a QDQ graph with an Equal node followed by a Cast.
TEST_F(QnnHTPBackendTests, EqualToCast4D) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Model building function that creates a QDQ graph with an Equal node followed by
  // a Cast to float32.
  auto build_qdq_equal_to_cast = [](ModelTestBuilder& builder) {
    constexpr uint8_t zero_point = 0;
    constexpr float qdq_scale = 0.0038f;
    const std::vector<int64_t> input_shape = {1, 3, 8, 8};

    auto* input0 = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
    auto* input1 = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
    auto* output = builder.MakeOutput();

    // input -> Q -> DQ -> Op
    auto* qdq0_output = AddQDQNodePair<uint8_t>(builder, input0, qdq_scale, zero_point);
    auto* qdq1_output = AddQDQNodePair<uint8_t>(builder, input1, qdq_scale, zero_point);

    // Equal ->
    auto* equal_output = builder.MakeIntermediate();
    builder.AddNode("Equal", {qdq0_output, qdq1_output}, {equal_output});

    // -> Cast -> output
    Node& cast_node = builder.AddNode("Cast", {equal_output}, {output});
    cast_node.AddAttribute("to",
                           static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  };

  RunQnnModelTest(build_qdq_equal_to_cast,
                  provider_options,
                  17,  // opset
                  ExpectedEPNodeAssignment::All,
                  1);  // expected nodes in graph
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

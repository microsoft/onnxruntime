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

// Creates the graph:
//                       _______________________
//                      |                       |
//    input_u8 -> DQ -> |       SimpleOp        | -> Q -> output_u8
//                      |_______________________|
//
// Currently used to test QNN EP.
template <typename InputQType>
GetQDQTestCaseFn BuildQDQSingleInputOpTestCase(const std::vector<int64_t>& input_shape, const std::string& op_type) {
  return [input_shape, op_type](ModelTestBuilder& builder) {
    const InputQType quant_zero_point = 0;
    const float quant_scale = 1.0f;

    auto* input = builder.MakeInput<InputQType>(input_shape, std::numeric_limits<InputQType>::min(),
                                                std::numeric_limits<InputQType>::max());
    auto* dq_input = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputQType>(input, quant_scale, quant_zero_point, dq_input);

    auto* op_output = builder.MakeIntermediate();
    builder.AddNode(op_type, {dq_input}, {op_output}, kMSDomain);

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InputQType>(op_output, quant_scale, quant_zero_point, q_output);

    auto* final_output = builder.MakeOutput();
    builder.AddDequantizeLinearNode<InputQType>(q_output, quant_scale, quant_zero_point, final_output);
  };
}

/**
 * Runs an BatchNormalization model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_shape The input's shape.
 * \param test_description Description of the test for error reporting.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param num_modes_in_graph The number of expected nodes in the graph.
 */
static void RunQDQSingleInputOpTest(const std::vector<int64_t>& input_shape, const std::string& op_type,
                                    const char* test_description,
                                    ExpectedEPNodeAssignment expected_ep_assignment, int num_nodes_in_graph) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQSingleInputOpTestCase<uint8_t>(input_shape, op_type),
                  provider_options,
                  11,
                  expected_ep_assignment,
                  num_nodes_in_graph,
                  test_description);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQGeluTest) {
  RunQDQSingleInputOpTest({1, 2, 3}, "Gelu", "TestQDQGeluTest", ExpectedEPNodeAssignment::All, 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
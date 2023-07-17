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

// Creates the following graph if axes is an input (newer opsets):
//                                _______________________
//    input (f32) -> Q -> DQ ->  |                       | -> Q -> DQ -> output (f32)
// axes (int32, initializer) ->  |       Reduce___       |
//                               |_______________________|
//
// Creates the following graph if axes is an attribute (older opsets):
//                                _______________________
//    input (f32) -> Q -> DQ ->  |                       | -> Q -> DQ -> output (f32)
//                               |       Reduce___       |
//                               |_______________________|
//
template <typename QuantType>
GetTestModelFn BuildQDQReduceOpTestCase(const std::string& reduce_op_type, const std::vector<int64_t>& input_shape,
                                        bool axes_as_input, const std::vector<int64_t>& axes, bool keepdims,
                                        bool noop_with_empty_axes) {
  return [reduce_op_type, input_shape, axes_as_input, axes, keepdims,
          noop_with_empty_axes](ModelTestBuilder& builder) {
    using QuantTypeLimits = std::numeric_limits<QuantType>;
    QuantType input_quant_min_value = QuantTypeLimits::min();
    QuantType input_quant_max_value = QuantTypeLimits::max();

    auto* input_data = builder.MakeInput<float>(input_shape, -100.0f, 100.0f);
    auto* final_output = builder.MakeOutput();

    // input_data -> Q/DQ ->
    auto* input_qdq_output = AddQDQNodePair<QuantType>(builder, input_data, .04f,
                                                       (input_quant_min_value + input_quant_max_value) / 2 + 1);

    // -> ReduceOp (e.g., ReduceSum) ->
    std::vector<NodeArg*> reduce_op_inputs;
    reduce_op_inputs.push_back(input_qdq_output);

    if (axes_as_input) {
      reduce_op_inputs.push_back(builder.MakeInitializer({static_cast<int64_t>(axes.size())}, axes));
    }

    auto* reduce_sum_output = builder.MakeIntermediate();
    Node& reduce_sum_node = builder.AddNode(reduce_op_type, reduce_op_inputs, {reduce_sum_output});
    reduce_sum_node.AddAttribute("keepdims", static_cast<int64_t>(keepdims));

    if (axes_as_input) {
      reduce_sum_node.AddAttribute("noop_with_empty_axes", static_cast<int64_t>(noop_with_empty_axes));
    } else {
      reduce_sum_node.AddAttribute("axes", axes);
    }

    // -> Q/DQ -> final_output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(reduce_sum_output, .039f,
                                             (QuantTypeLimits::min() + QuantTypeLimits::max()) / 2 + 1,
                                             q_output);

    builder.AddDequantizeLinearNode<QuantType>(q_output, .039f,
                                               (QuantTypeLimits::min() + QuantTypeLimits::max()) / 2 + 1,
                                               final_output);
  };
}

/**
 * Runs a ReduceOp model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param op_type The ReduceOp type (e.g., ReduceSum).
 * \param opset The opset version. Some opset versions have "axes" as an attribute or input.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None)
 * \param keepdims Common attribute for all reduce operations.
 */
template <typename QuantType>
static void RunReduceOpQDQTest(const std::string& op_type, int opset, const std::vector<int64_t>& input_shape,
                               const std::vector<int64_t>& axes,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               bool keepdims = true) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // If QNN EP can support all ops, then we expect a single fused node in the graph.
  // Otherwise, we'll get a graph with 5 individual nodes handled by CPU EP.
  constexpr bool noop_with_empty_axes = false;
  RunQnnModelTest(BuildQDQReduceOpTestCase<QuantType>(op_type,
                                                      input_shape,
                                                      ReduceOpHasAxesInput(op_type, opset),  // New opset changed axes to input.
                                                      axes,
                                                      keepdims,
                                                      noop_with_empty_axes),
                  provider_options,
                  opset,
                  expected_ep_assignment);
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
  RunReduceOpQDQTest<uint8_t>("ReduceSum", 13, {2, 2}, {0, 1});
}

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 11, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumU8Opset11) {
  RunReduceOpQDQTest<uint8_t>("ReduceSum", 11, {1, 3, 4, 4}, {0, 1, 2, 3});
}

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumS8Opset13) {
  RunReduceOpQDQTest<int8_t>("ReduceSum", 13, {2, 2}, {0, 1});
}

// Tests that keepdims = false generates expected results.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumS8Opset13_NoKeepDims) {
  RunReduceOpQDQTest<int8_t>("ReduceSum", 13, {2, 2}, {1}, ExpectedEPNodeAssignment::All, false);
}

// Test that we don't support rank 5 Reduce ops.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumS8Opset13_Rank5Unsupported) {
  RunReduceOpQDQTest<int8_t>("ReduceSum", 13, {1, 3, 4, 4, 2}, {0, 1, 2, 3, 4}, ExpectedEPNodeAssignment::None);
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
  RunReduceOpQDQTest<uint8_t>("ReduceMax", 18, {2, 2}, {0, 1});
}

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMaxU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMax", 13, {2, 2}, {0, 1});
}

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMaxS8Opset18) {
  RunReduceOpQDQTest<int8_t>("ReduceMax", 18, {2, 2}, {0, 1});
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
  RunReduceOpQDQTest<uint8_t>("ReduceMin", 18, {2, 2}, {0, 1});
}

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMinU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMin", 13, {2, 2}, {0, 1});
}

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// Uses int8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQReduceMinS8Opset18) {
  RunReduceOpQDQTest<int8_t>("ReduceMin", 18, {2, 2}, {0, 1});
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
  RunReduceOpQDQTest<uint8_t>("ReduceMean", 18, {2, 2}, {0, 1});
}

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMeanU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMean", 13, {2, 2}, {0, 1});
}

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMeanS8Opset18) {
  RunReduceOpQDQTest<int8_t>("ReduceMean", 18, {2, 2}, {0, 1});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
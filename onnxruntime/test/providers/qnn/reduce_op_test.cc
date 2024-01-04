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

/**
 * Creates a graph with a single reduce operator (e.g., ReduceSum, ReduceMin, etc.). Reduce operators take the
 * axes of reduction as either a node attribute or an optional input (depending on opset).
 *
 * \param reduce_op_type The string denoting the reduce operator's type (e.g., "ReduceSum").
 * \param input_def The input definition (shape, data, etc.)
 * \param axes_as_input True if the "axes" are specified as a node input.
 * \param axes The axes of reduction.
 * \param keepdims True if the output's rank should match the input. This is a node attribute that defaults to true.
 * \param noop_with_empty_axes True if empty axes should force the node to act as a NoOp (no operation).
 *                             This is a node attribute that defaults to false.
 * \param domain The domain to assign to the graph node.
 *
 * \return A function that builds the graph with the provided builder.
 */
template <typename DataType>
static GetTestModelFn BuildReduceOpTestCase(const std::string& reduce_op_type,
                                            const TestInputDef<DataType>& input_def,
                                            bool axes_as_input, std::vector<int64_t> axes, bool keepdims,
                                            bool noop_with_empty_axes) {
  return [reduce_op_type, input_def, axes_as_input, axes, keepdims,
          noop_with_empty_axes](ModelTestBuilder& builder) {
    std::vector<NodeArg*> input_args;

    // Input data arg
    input_args.push_back(MakeTestInput(builder, input_def));

    // Axes input (initializer) for newer opsets.
    if (axes_as_input) {
      input_args.push_back(builder.MakeInitializer({static_cast<int64_t>(axes.size())}, axes));
    }

    auto* reduce_sum_output = builder.MakeOutput();
    Node& reduce_sum_node = builder.AddNode(reduce_op_type, input_args, {reduce_sum_output});
    reduce_sum_node.AddAttribute("keepdims", static_cast<int64_t>(keepdims));

    // Older opsets have "axes" as a node attribute.
    if (!axes_as_input) {
      reduce_sum_node.AddAttribute("axes", axes);
    } else {
      reduce_sum_node.AddAttribute("noop_with_empty_axes", static_cast<int64_t>(noop_with_empty_axes));
    }
  };
}

/**
 * Runs a ReduceOp model on the QNN CPU backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param op_type The ReduceOp type (e.g., ReduceSum).
 * \param input_def The input definition (shape, data, etc.)
 * \param axes The axes of reduction.
 * \param opset The opset version. Some opset versions have "axes" as an attribute or input.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None)
 * \param keepdims Common attribute for all reduce operations.
 */
template <typename DataType>
static void RunReduceOpCpuTest(const std::string& op_type,
                               const TestInputDef<DataType>& input_def,
                               const std::vector<int64_t>& axes,
                               bool keepdims,
                               int opset,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildReduceOpTestCase<DataType>(op_type,
                                                  input_def,  //{2, 2},  // input shape
                                                  ReduceOpHasAxesInput(op_type, opset),
                                                  axes,  //{0, 1},  // axes
                                                  keepdims,
                                                  false),  // noop_with_empty_axes
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// ReduceSum
//

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is int32.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, ReduceSumOpset13_Int32) {
  RunReduceOpCpuTest<int32_t>("ReduceSum",
                              TestInputDef<int32_t>({2, 2}, false, -10.0f, 10.0f),
                              std::vector<int64_t>{0, 1},
                              true,  // keepdims
                              13,
                              ExpectedEPNodeAssignment::All);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is int32.
// - Uses opset 11, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, ReduceSumOpset11_Int32) {
  RunReduceOpCpuTest<int32_t>("ReduceSum",
                              TestInputDef<int32_t>({2, 2}, false, -10.0f, 10.0f),
                              std::vector<int64_t>{0, 1},
                              true,  // keepdims
                              11,
                              ExpectedEPNodeAssignment::All);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, ReduceSumOpset13_Float) {
  RunReduceOpCpuTest<float>("ReduceSum",
                            TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            13,
                            ExpectedEPNodeAssignment::All);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 11, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, ReduceSumOpset11_Float) {
  RunReduceOpCpuTest<float>("ReduceSum",
                            TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            11,
                            ExpectedEPNodeAssignment::All);
}

//
// ReduceProd
//

// Test creates a graph with a ReduceProd node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, ReduceProdOpset18) {
  RunReduceOpCpuTest<float>("ReduceProd",
                            TestInputDef<float>({2, 2}, false, {-10.0f, -8.2f, 0.0f, 10.0f}),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            18,
                            ExpectedEPNodeAssignment::All);
}

// TODO: Investigate slight inaccuracy. x64 Windows/Linux require a slightly larger error tolerance greater than 1.5e-5f.
// LOG: ... the value pair (208.881729, 208.881744) at index #0 don't match, which is 1.52588e-05 from 208.882
TEST_F(QnnCPUBackendTests, ReduceProdOpset18_SlightlyInaccurate_WindowsLinuxX64) {
  RunReduceOpCpuTest<float>("ReduceProd",
                            TestInputDef<float>({2, 2}, false, {3.21289f, -5.9981f, -1.72799f, 6.27263f}),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            18,
                            ExpectedEPNodeAssignment::All,
                            2e-5f);  // x64 Linux & Windows require larger tolerance.
}

// Test creates a graph with a ReduceProd node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, ReduceProdOpset13) {
  RunReduceOpCpuTest<float>("ReduceProd",
                            TestInputDef<float>({2, 2}, false, {-10.0f, -8.2f, 0.0f, 10.0f}),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            13,
                            ExpectedEPNodeAssignment::All);
}

//
// ReduceMax
//

// Test creates a graph with a ReduceMax node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, ReduceMaxOpset18) {
  RunReduceOpCpuTest<float>("ReduceMax",
                            TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            18,
                            ExpectedEPNodeAssignment::All);
}

// Test creates a graph with a ReduceMax node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, ReduceMaxOpset13) {
  RunReduceOpCpuTest<float>("ReduceMax",
                            TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            13,
                            ExpectedEPNodeAssignment::All);
}

//
// ReduceMin
//

// Test creates a graph with a ReduceMin node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, ReduceMinOpset18) {
  RunReduceOpCpuTest<float>("ReduceMin",
                            TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            18,
                            ExpectedEPNodeAssignment::All);
}

// Test creates a graph with a ReduceMin node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, ReduceMinOpset13) {
  RunReduceOpCpuTest<float>("ReduceMin",
                            TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            13,
                            ExpectedEPNodeAssignment::All);
}

//
// ReduceMean
//

// Test creates a graph with a ReduceMean node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, ReduceMeanOpset18) {
  RunReduceOpCpuTest<float>("ReduceMean",
                            TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            18,
                            ExpectedEPNodeAssignment::All);
}

// Test creates a graph with a ReduceMean node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, ReduceMeanOpset13) {
  RunReduceOpCpuTest<float>("ReduceMean",
                            TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                            std::vector<int64_t>{0, 1},
                            true,  // keepdims
                            13,
                            ExpectedEPNodeAssignment::All);
}

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
GetTestQDQModelFn<QuantType> BuildQDQReduceOpTestCase(const std::string& reduce_op_type,
                                                      const TestInputDef<float>& input_def,
                                                      bool axes_as_input, const std::vector<int64_t>& axes, bool keepdims,
                                                      bool noop_with_empty_axes) {
  return [reduce_op_type, input_def, axes_as_input, axes, keepdims,
          noop_with_empty_axes](ModelTestBuilder& builder,
                                std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    auto* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    // -> ReduceOp (e.g., ReduceSum) ->
    std::vector<NodeArg*> reduce_op_inputs;
    reduce_op_inputs.push_back(input_qdq);

    if (axes_as_input) {
      reduce_op_inputs.push_back(builder.MakeInitializer({static_cast<int64_t>(axes.size())}, axes));
    }

    auto* op_output = builder.MakeIntermediate();
    Node& reduce_sum_node = builder.AddNode(reduce_op_type, reduce_op_inputs, {op_output});
    reduce_sum_node.AddAttribute("keepdims", static_cast<int64_t>(keepdims));

    if (axes_as_input) {
      reduce_sum_node.AddAttribute("noop_with_empty_axes", static_cast<int64_t>(noop_with_empty_axes));
    } else {
      reduce_sum_node.AddAttribute("axes", axes);
    }

    // -> Q -> DQ -> final output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, op_output, output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

/**
 * Runs a ReduceOp model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param op_type The ReduceOp type (e.g., ReduceSum).
 * \param input_def The input definition (shape, data, etc.).
 * \param axes The axes input (or attribute).
 * \param keepdims Common attribute for all reduce operations.
 * \param opset The opset version. Some opset versions have "axes" as an attribute or input.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None)
 * \param fp32_abs_err Error tolerance.
 */
template <typename QuantType>
static void RunReduceOpQDQTest(const std::string& op_type,
                               const TestInputDef<float>& input_def,
                               const std::vector<int64_t>& axes,
                               bool keepdims,
                               int opset,
                               ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  constexpr bool noop_with_empty_axes = false;
  const bool axes_as_input = ReduceOpHasAxesInput(op_type, opset);  // Later opsets have "axes" as an input.

  TestQDQModelAccuracy(BuildReduceOpTestCase<float>(op_type, input_def, axes_as_input, axes, keepdims,
                                                    noop_with_empty_axes),
                       BuildQDQReduceOpTestCase<QuantType>(op_type, input_def, axes_as_input, axes, keepdims,
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
TEST_F(QnnHTPBackendTests, ReduceSumU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceSum",
                              TestInputDef<float>({2, 2}, false, {-10.0f, 3.21289f, -5.9981f, 10.0f}),
                              {0, 1},  // axes
                              true,    // keepdims
                              13,      // opset
                              ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ ReduceSum of last axis.
TEST_F(QnnHTPBackendTests, ReduceSumU8Opset13_LastAxis) {
  const std::vector<float> input_data = {3.21289f, -5.9981f, -1.72799f, 6.27263f};
  RunReduceOpQDQTest<uint8_t>("ReduceSum",
                              TestInputDef<float>({2, 2}, false, input_data),
                              {1},   // axes
                              true,  // keepdims
                              13,    // opset
                              ExpectedEPNodeAssignment::All);
}
// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 11, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, ReduceSumU8Opset11) {
  RunReduceOpQDQTest<uint8_t>("ReduceSum",
                              TestInputDef<float>({2, 2}, false, {-10.0f, 3.21289f, -5.9981f, 10.0f}),
                              {0, 1},  // axes
                              true,    // keepdims
                              11,      // opset
                              ExpectedEPNodeAssignment::All);
}

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, ReduceSumS8Opset13) {
  // non-symmetrical input range so output sum is not trivially zero.
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 20.0f, 9);

  RunReduceOpQDQTest<int8_t>("ReduceSum",
                             TestInputDef<float>({3, 3}, false, input_data),
                             {0, 1},  // axes
                             true,    // keepdims
                             13,      // opset
                             ExpectedEPNodeAssignment::All);
}

// Tests that keepdims = false generates expected results.
TEST_F(QnnHTPBackendTests, ReduceSumS8Opset13_NoKeepDims) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 9);

  RunReduceOpQDQTest<int8_t>("ReduceSum",
                             TestInputDef<float>({3, 3}, false, input_data),
                             {1},    // axes
                             false,  // keepdims
                             13,     // opset
                             ExpectedEPNodeAssignment::All);
}

// Test rank 5 ReduceSum (s8 quant) with axes = [0, 1, 2, 3, 4], keep_dims = true
TEST_F(QnnHTPBackendTests, ReduceSumS8Opset13_Rank5) {
  RunReduceOpQDQTest<int8_t>("ReduceSum",
                             TestInputDef<float>({1, 3, 4, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 96)),
                             {0, 1, 2, 3, 4},  // axes
                             true,             // keepdims
                             13,               // opset
                             ExpectedEPNodeAssignment::All);
}

// Test that QNN validation APIs reject inputs of unsupported ranks.
TEST_F(QnnHTPBackendTests, ReduceSumS8Opset13_Rank6_Unsupported) {
  RunReduceOpQDQTest<int8_t>("ReduceSum",
                             TestInputDef<float>({1, 3, 4, 4, 2, 1}, false, GetFloatDataInRange(-10.0f, 10.0f, 96)),
                             {-1},                             // axes
                             false,                            // keepdims
                             13,                               // opset
                             ExpectedEPNodeAssignment::None);  // Not assigned to QNN EP
}

// Test rank 5 ReduceSum (u8 quant) with axes = [-1], keep_dims = false
TEST_F(QnnHTPBackendTests, ReduceSumU8Opset13_Rank5_LastAxis) {
  constexpr size_t num_elems = 2ULL * 12 * 124 * 2 * 4;
  std::vector<float> input_data = GetFloatDataInRange(-100.0f, 100.0f, num_elems);
  RunReduceOpQDQTest<uint8_t>("ReduceSum",
                              TestInputDef<float>({2, 12, 124, 2, 4}, false, input_data),
                              {-1},   // axes
                              false,  // keepdims
                              13,     // opset
                              ExpectedEPNodeAssignment::All);
}

//
// ReduceMax
//

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, ReduceMaxU8Opset18) {
  RunReduceOpQDQTest<uint8_t>("ReduceMax",
                              TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                              {0, 1},  // axes
                              true,    // keepdims
                              18,      // opset
                              ExpectedEPNodeAssignment::All);
}

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, ReduceMaxU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMax",
                              TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                              {0, 1},  // axes
                              true,    // keepdims
                              13,      // opset
                              ExpectedEPNodeAssignment::All);
}

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, ReduceMaxS8Opset18) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 9);

  RunReduceOpQDQTest<int8_t>("ReduceMax",
                             TestInputDef<float>({3, 3}, false, input_data),
                             {0, 1},  // axes
                             true,    // keepdims
                             18,      // opset
                             ExpectedEPNodeAssignment::All);
}

//
// ReduceMin
//

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, ReduceMinU8Opset18) {
  RunReduceOpQDQTest<uint8_t>("ReduceMin",
                              TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                              {0, 1},  // axes
                              true,    // keepdims
                              18,      // opset
                              ExpectedEPNodeAssignment::All);
}

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, ReduceMinU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMin",
                              TestInputDef<float>({2, 2}, false, -10.0f, 10.0f),
                              {0, 1},  // axes
                              true,    // keepdims
                              13,      // opset
                              ExpectedEPNodeAssignment::All);
}

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// Uses int8 as the quantization type.
TEST_F(QnnHTPBackendTests, ReduceMinS8Opset18) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 9);

  RunReduceOpQDQTest<int8_t>("ReduceMin",
                             TestInputDef<float>({3, 3}, false, input_data),
                             {0, 1},  // axes
                             true,    // keepdims
                             18,      // opset
                             ExpectedEPNodeAssignment::All);
}

//
// ReduceMean
//

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, ReduceMeanU8Opset18) {
  RunReduceOpQDQTest<uint8_t>("ReduceMean",
                              TestInputDef<float>({2, 2}, false, {-10.0f, 3.21289f, -5.9981f, 10.0f}),
                              {0, 1},  // axes
                              true,    // keepdims
                              18,      // opset
                              ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ ReduceMean of last axis
TEST_F(QnnHTPBackendTests, ReduceMeanU8Opset18_LastAxis) {
  const std::vector<float> input_data = {3.21289f, -5.9981f, -1.72799f, 6.27263f};
  RunReduceOpQDQTest<uint8_t>("ReduceMean",
                              TestInputDef<float>({2, 2}, false, input_data),
                              {1},   // axes
                              true,  // keepdims
                              18,    // opset
                              ExpectedEPNodeAssignment::All);
}

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, ReduceMeanU8Opset13) {
  RunReduceOpQDQTest<uint8_t>("ReduceMean",
                              TestInputDef<float>({2, 2}, false, {-10.0f, 3.21289f, -5.9981f, 10.0f}),
                              {0, 1},  // axes
                              true,    // keepdims
                              13,      // opset
                              ExpectedEPNodeAssignment::All);
}

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, ReduceMeanS8Opset18) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 20.0f, 48);

  RunReduceOpQDQTest<int8_t>("ReduceMean",
                             TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                             {0, 1, 2, 3},  // axes
                             true,          // keepdims
                             18,            // opset
                             ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif

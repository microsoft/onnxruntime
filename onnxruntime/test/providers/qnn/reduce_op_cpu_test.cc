// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

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
 * \param input_shape The shape of the input. Input data is randomly generated with this shape.
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
                                            const std::vector<int64_t>& input_shape,
                                            bool axes_as_input, std::vector<int64_t> axes, bool keepdims,
                                            bool noop_with_empty_axes) {
  return [reduce_op_type, input_shape, axes_as_input, axes, keepdims,
          noop_with_empty_axes](ModelTestBuilder& builder) {
    std::vector<NodeArg*> input_args;

    // Input data arg
    input_args.push_back(builder.MakeInput<DataType>(input_shape, static_cast<DataType>(0),
                                                     static_cast<DataType>(20)));

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
 * \param opset The opset version. Some opset versions have "axes" as an attribute or input.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None)
 * \param keepdims Common attribute for all reduce operations.
 */
template <typename DataType>
static void RunReduceOpCpuTest(const std::string& op_type, int opset,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               bool keepdims = true) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildReduceOpTestCase<DataType>(op_type,
                                                  {2, 2},  // input shape
                                                  ReduceOpHasAxesInput(op_type, opset),
                                                  {0, 1},  // axes
                                                  keepdims,
                                                  false),  // noop_with_empty_axes
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// ReduceSum
//

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is int32.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, TestInt32ReduceSumOpset13) {
  RunReduceOpCpuTest<int32_t>("ReduceSum", 13);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is int32.
// - Uses opset 11, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, TestInt32ReduceSumOpset11) {
  RunReduceOpCpuTest<int32_t>("ReduceSum", 11);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, TestFloatReduceSumOpset13) {
  RunReduceOpCpuTest<float>("ReduceSum", 13);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 11, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, TestFloatReduceSumOpset11) {
  RunReduceOpCpuTest<float>("ReduceSum", 11);
}

//
// ReduceProd
//

// Test creates a graph with a ReduceProd node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, TestReduceProdOpset18) {
  RunReduceOpCpuTest<float>("ReduceProd", 18);
}

// Test creates a graph with a ReduceProd node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, TestReduceProdOpset13) {
  RunReduceOpCpuTest<float>("ReduceProd", 13);
}

//
// ReduceMax
//

// Test creates a graph with a ReduceMax node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, TestReduceMaxOpset18) {
  RunReduceOpCpuTest<float>("ReduceMax", 18);
}

// Test creates a graph with a ReduceMax node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, TestReduceMaxOpset13) {
  RunReduceOpCpuTest<float>("ReduceMax", 13);
}

//
// ReduceMin
//

// Test creates a graph with a ReduceMin node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, TestReduceMinOpset18) {
  RunReduceOpCpuTest<float>("ReduceMin", 18);
}

// Test creates a graph with a ReduceMin node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, TestReduceMinOpset13) {
  RunReduceOpCpuTest<float>("ReduceMin", 13);
}

//
// ReduceMean
//

// Test creates a graph with a ReduceMean node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnCPUBackendTests, TestReduceMeanOpset18) {
  RunReduceOpCpuTest<float>("ReduceMean", 18);
}

// Test creates a graph with a ReduceMean node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnCPUBackendTests, TestReduceMeanOpset13) {
  RunReduceOpCpuTest<float>("ReduceMean", 13);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
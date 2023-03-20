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
static GetQDQTestCaseFn BuildReduceOpTestCase(const std::string& reduce_op_type,
                                              const std::vector<int64_t>& input_shape,
                                              bool axes_as_input, std::vector<int64_t> axes, bool keepdims,
                                              bool noop_with_empty_axes, const std::string& domain) {
  return [reduce_op_type, input_shape, axes_as_input, axes, keepdims,
          noop_with_empty_axes, domain](ModelTestBuilder& builder) {
    std::vector<NodeArg*> input_args;

    // Input data arg
    input_args.push_back(builder.MakeInput<DataType>(input_shape, static_cast<DataType>(0),
                                                     static_cast<DataType>(20)));

    // Axes input (initializer) for newer opsets.
    if (axes_as_input) {
      input_args.push_back(builder.MakeInitializer({static_cast<int64_t>(axes.size())}, axes));
    }

    auto* reduce_sum_output = builder.MakeOutput();
    Node& reduce_sum_node = builder.AddNode(reduce_op_type, input_args, {reduce_sum_output}, domain);
    reduce_sum_node.AddAttribute("keepdims", static_cast<int64_t>(keepdims));

    // Older opsets have "axes" as a node attribute.
    if (!axes_as_input) {
      reduce_sum_node.AddAttribute("axes", axes);
    } else {
      reduce_sum_node.AddAttribute("noop_with_empty_axes", static_cast<int64_t>(noop_with_empty_axes));
    }
  };
}

//
// ReduceSum
//

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is int32.
// - Uses opset 13, which has "axes" as an input.
TEST(QnnCPUBackendTests, TestInt32ReduceSumOpset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceSum";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<int32_t>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                              {0, 1}, true, false, domain),
               "qnn_int32_reduce_sum_opset13",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is int32.
// - Uses opset 11, which has "axes" as an attribute.
TEST(QnnCPUBackendTests, TestInt32ReduceSumOpset11) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 11;
  const std::string op_type = "ReduceSum";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<int32_t>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                              {0, 1}, true, false, domain),
               "qnn_int32_reduce_sum_opset11",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an input.
TEST(QnnCPUBackendTests, TestFloatReduceSumOpset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceSum";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_sum_opset13",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a graph with a ReduceSum node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 11, which has "axes" as an attribute.
TEST(QnnCPUBackendTests, TestFloatReduceSumOpset11) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 11;
  const std::string op_type = "ReduceSum";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_sum_opset11",
               provider_options,
               verification_params,
               domain_to_version);
}

//
// ReduceProd
//

// Test creates a graph with a ReduceProd node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST(QnnCPUBackendTests, TestReduceProdOpset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceProd";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_prod_opset18",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a graph with a ReduceProd node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST(QnnCPUBackendTests, TestReduceProdOpset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceProd";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_prod_opset13",
               provider_options,
               verification_params,
               domain_to_version);
}

//
// ReduceMax
//

// Test creates a graph with a ReduceMax node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST(QnnCPUBackendTests, TestReduceMaxOpset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMax";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_max_opset18",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a graph with a ReduceMax node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST(QnnCPUBackendTests, TestReduceMaxOpset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceMax";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_max_opset13",
               provider_options,
               verification_params,
               domain_to_version);
}

//
// ReduceMin
//

// Test creates a graph with a ReduceMin node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST(QnnCPUBackendTests, TestReduceMinOpset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMin";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_min_opset18",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a graph with a ReduceMin node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST(QnnCPUBackendTests, TestReduceMinOpset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceMin";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_min_opset13",
               provider_options,
               verification_params,
               domain_to_version);
}


//
// ReduceMean
//

// Test creates a graph with a ReduceMean node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 18, which has "axes" as an input.
TEST(QnnCPUBackendTests, TestReduceMeanOpset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMean";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_mean_opset18",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a graph with a ReduceMean node, and checks that all
// nodes are supported by the QNN EP (cpu backend), and that the inference results match the CPU EP results.
//
// - The input and output data type is float.
// - Uses opset 13, which has "axes" as an attribute.
TEST(QnnCPUBackendTests, TestReduceMeanOpset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceMean";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildReduceOpTestCase<float>(op_type, {2, 2}, ReduceOpHasAxesInput(op_type, opset),
                                            {0, 1}, true, false, domain),
               "qnn_float_reduce_mean_opset13",
               provider_options,
               verification_params,
               domain_to_version);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
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

//
// ReduceSum
//

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumU8Opset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceSum -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceSum";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<uint8_t>(op_type,
                                                 {2, 2},                                // Input shape
                                                 ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                 {0, 1},                                // Axes
                                                 true,                                  // keepdims
                                                 false,                                 // noop_with_empty_axes
                                                 domain),
               "qnn_qdq_reduce_sum_u8_htp",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 11, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumU8Opset11) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceSum -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 11;
  const std::string op_type = "ReduceSum";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<uint8_t>(op_type,
                                                 {2, 2},                                // Input shape
                                                 ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                 {0, 1},                                // Axes
                                                 true,                                  // keepdims
                                                 false,                                 // noop_with_empty_axes
                                                 domain),
               "qnn_qdq_reduce_sum_u8_htp",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a Q -> DQ -> ReduceSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses int8 as the quantization type.
// - Uses opset 13, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceSumS8Opset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceSum -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceSum";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<int8_t>(op_type,
                                                {2, 2},                                // Input shape
                                                ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                {0, 1},                                // Axes
                                                true,                                  // keepdims
                                                false,                                 // noop_with_empty_axes
                                                domain),
               "qnn_qdq_reduce_sum_s8_htp",
               provider_options,
               verification_params,
               domain_to_version);
}

//
// ReduceMax
//

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMaxU8Opset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMax -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMax";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<uint8_t>(op_type,
                                                 {2, 2},                                // Input shape
                                                 ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                 {0, 1},                                // Axes
                                                 true,                                  // keepdims
                                                 false,                                 // noop_with_empty_axes
                                                 domain),
               "qnn_qdq_reduce_max_u8_htp",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMaxU8Opset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMax -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceMax";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<uint8_t>(op_type,
                                                 {2, 2},                                // Input shape
                                                 ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                 {0, 1},                                // Axes
                                                 true,                                  // keepdims
                                                 false,                                 // noop_with_empty_axes
                                                 domain),
               "qnn_qdq_reduce_max_u8_htp",
               provider_options,
               verification_params,
               domain_to_version);
}

// Test creates a Q -> DQ -> ReduceMax -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// Uses int8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQReduceMaxS8Opset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMax -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMax";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<int8_t>(op_type,
                                                {2, 2},                                // Input shape
                                                ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                {0, 1},                                // Axes
                                                true,                                  // keepdims
                                                false,                                 // noop_with_empty_axes
                                                domain),
               "qnn_qdq_reduce_max_s8_htp",
               provider_options,
               verification_params,
               domain_to_version);
}

//
// ReduceMin
//

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMinU8Opset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMin -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMin";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<uint8_t>(op_type,
                                                 {2, 2},                                // Input shape
                                                 ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                 {0, 1},                                // Axes
                                                 true,                                  // keepdims
                                                 false,                                 // noop_with_empty_axes
                                                 domain),
               "qnn_qdq_reduce_min_u8_htp",
               provider_options,
               verification_params, domain_to_version);
}

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMinU8Opset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMin -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceMin";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<uint8_t>(op_type,
                                                 {2, 2},                                // Input shape
                                                 ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                 {0, 1},                                // Axes
                                                 true,                                  // keepdims
                                                 false,                                 // noop_with_empty_axes
                                                 domain),
               "qnn_qdq_reduce_min_u8_htp",
               provider_options,
               verification_params, domain_to_version);
}

// Test creates a Q -> DQ -> ReduceMin -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// Uses int8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQReduceMinS8Opset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMin -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMin";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<int8_t>(op_type,
                                                {2, 2},                                // Input shape
                                                ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                {0, 1},                                // Axes
                                                true,                                  // keepdims
                                                false,                                 // noop_with_empty_axes
                                                domain),
               "qnn_qdq_reduce_min_s8_htp",
               provider_options,
               verification_params, domain_to_version);
}

//
// ReduceMean
//

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 18, which has "axes" as an input.
TEST_F(QnnHTPBackendTests, TestQDQReduceMeanU8Opset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMean -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMean";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<uint8_t>(op_type,
                                                 {2, 2},                                // Input shape
                                                 ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                 {0},                                   // Axes
                                                 true,                                  // keepdims
                                                 false,                                 // noop_with_empty_axes
                                                 domain),
               "qnn_qdq_reduce_mean_u8_htp",
               provider_options,
               verification_params, domain_to_version);
}

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
// - Uses opset 13, which has "axes" as an attribute.
TEST_F(QnnHTPBackendTests, TestQDQReduceMeanU8Opset13) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMean -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 13;
  const std::string op_type = "ReduceMean";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<uint8_t>(op_type,
                                                 {2, 2},                                // Input shape
                                                 ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                 {0},                                   // Axes
                                                 true,                                  // keepdims
                                                 false,                                 // noop_with_empty_axes
                                                 domain),
               "qnn_qdq_reduce_mean_u8_htp",
               provider_options,
               verification_params, domain_to_version);
}

// Test creates a Q -> DQ -> ReduceMean -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// Uses int8 as the quantization type.
TEST_F(QnnHTPBackendTests, TestQDQReduceMeanS8Opset18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> ReduceMean -> Q node unit assigned to QNN EP";
  };

  const std::string domain = "";
  const int opset = 18;
  const std::string op_type = "ReduceMean";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;
  const std::unordered_map<std::string, int> domain_to_version = {{domain, opset}};

  RunModelTest(BuildQDQReduceOpTestCase<int8_t>(op_type,
                                                {2, 2},                                // Input shape
                                                ReduceOpHasAxesInput(op_type, opset),  // Axes is an input
                                                {1},                                   // Axes
                                                true,                                  // keepdims
                                                false,                                 // noop_with_empty_axes
                                                domain),
               "qnn_qdq_reduce_mean_s8_htp",
               provider_options,
               verification_params, domain_to_version);
}

}  // namespace test
}  // namespace onnxruntime

#endif
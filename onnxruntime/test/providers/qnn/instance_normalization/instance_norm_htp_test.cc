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

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> InstanceNormalization -> Q node unit assigned to QNN EP";
  };

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunModelTest(BuildQDQInstanceNormTestCase<uint8_t, uint8_t, int32_t>(
                   {1, 2, 3, 3} /* input_shape */,
                   1e-05f /* epsilon */),
               "qnn_qdq_test_graph_instance_norm_u8",
               provider_options,
               verification_params);
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8Rank3) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> InstanceNormalization -> Q node unit assigned to QNN EP";
  };

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunModelTest(BuildQDQInstanceNormTestCase<uint8_t, uint8_t, int32_t>(
                   {1, 2, 3} /* input_shape */,
                   1e-05f /* epsilon */),
               "qnn_qdq_test_graph_instance_norm_u8_rank3",
               provider_options,
               verification_params);
}

// Check that QNN InstanceNorm operator does not handle inputs with rank > 4.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8Rank5) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::None;  // No graph nodes should be assigned to QNN

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunModelTest(BuildQDQInstanceNormTestCase<uint8_t, uint8_t, int32_t>(
                   {1, 2, 3, 3, 3} /* input_shape */,
                   1e-05f /* epsilon */),
               "qnn_qdq_test_graph_instance_norm_u8_rank5",
               provider_options,
               verification_params);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
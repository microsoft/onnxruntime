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

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> Conv -> Q node unit assigned to QNN EP";
  };

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;

  RunModelTest(BuildQDQConvTestCase<uint8_t /* InputType */,
                                    uint8_t /* WeightType */,
                                    int32_t /* BiasType */,
                                    uint8_t /* OutputType */>(
                   {1, 1, 5, 5} /* input_shape */,
                   {1, 1, 3, 3} /* weights_shape */),
               "qnn_qdq_test_graph_conv_u8u8",
               provider_options,
               verification_params);  // two transpose nodes would be added before and after
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
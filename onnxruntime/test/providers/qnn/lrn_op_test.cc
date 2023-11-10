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

// Creates a graph with a single LRN operator. Used for testing CPU backend.
static GetTestModelFn BuildLRNTestCase(const std::vector<int64_t>& shape, int64_t size,
                                       float alpha = 0.0001f, float beta = 0.75f, float bias = 1.0f) {
  return [shape, size, alpha, beta, bias](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(shape, 0.0f, 20.0f);
    auto* output = builder.MakeOutput();

    Node& lrn_node = builder.AddNode("LRN", {input}, {output});
    lrn_node.AddAttribute("size", size);
    lrn_node.AddAttribute("alpha", alpha);
    lrn_node.AddAttribute("beta", beta);
    lrn_node.AddAttribute("bias", bias);
  };
}

// Q/DQ scaled used to build Q/DQ test model. This is a global constant
// because results from HTP backend are off by exactly this amount.
static constexpr float qdq_scale = 0.0038f;

// Creates a graph with a single Q/DQ LRN operator. Used for testing HTP backend.
template <typename InputQType = uint8_t>
static GetTestModelFn BuildQDQLRNTestCase(const std::vector<int64_t>& shape, int64_t size,
                                          float alpha = 0.0001f, float beta = 0.75f, float bias = 1.0f) {
  return [shape, size, alpha, beta, bias](ModelTestBuilder& builder) {
    const InputQType zero_point = std::numeric_limits<InputQType>::max() / 2;

    auto* input = builder.MakeInput<float>(shape, -1.0f, 1.0f);
    auto* output = builder.MakeOutput();

    // input -> Q -> DQ -> LRN
    auto* qdq_output = AddQDQNodePair<InputQType>(builder, input, qdq_scale, zero_point);
    auto* lrn_output = builder.MakeIntermediate();

    Node& lrn_node = builder.AddNode("LRN", {qdq_output}, {lrn_output});
    lrn_node.AddAttribute("size", size);
    lrn_node.AddAttribute("alpha", alpha);
    lrn_node.AddAttribute("beta", beta);
    lrn_node.AddAttribute("bias", bias);

    // -> Q -> DQ -> output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InputQType>(lrn_output, qdq_scale, zero_point, q_output);
    builder.AddDequantizeLinearNode<InputQType>(q_output, qdq_scale, zero_point, output);
  };
}

// Runs an LRN model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunCPULRNOpTest(const std::vector<int64_t>& shape, int64_t size,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            float alpha = 0.0001f, float beta = 0.75f, float bias = 1.0f, int opset = 13) {
  ProviderOptions provider_options;
  float fp32_abs_err = 1e-5f;  // default tolerance

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
  fp32_abs_err = 1.5e-5f;  // On linux we need slightly larger tolerance.
#endif

  RunQnnModelTest(BuildLRNTestCase(shape, size, alpha, beta, bias),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Runs an LRN model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename QuantType>
static void RunQDQLRNOpTest(const std::vector<int64_t>& shape, int64_t size,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            float alpha = 0.0001f, float beta = 0.75f, float bias = 1.0f,
                            int opset = 13, float fp32_abs_err = qdq_scale) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildQDQLRNTestCase<QuantType>(shape, size, alpha, beta, bias),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err + 0.0001f);
}

//
// CPU tests:
//

TEST_F(QnnCPUBackendTests, TestCPULRNSize3) {
  RunCPULRNOpTest({1, 128, 4, 5}, 3, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TestCPULRNSize5) {
  RunCPULRNOpTest({1, 128, 4, 5}, 5, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TestCPULRN_size_larger_than_channel) {
  RunCPULRNOpTest({1, 128, 4, 5}, 255, ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, TestHTPLRNSize3) {
  RunQDQLRNOpTest<uint8_t>({1, 128, 4, 5}, 3, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, TestHTPLRNSize5) {
  RunQDQLRNOpTest<uint8_t>({1, 128, 4, 5}, 5, ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, TestHTPLRN_size_larger_than_channel) {
  RunQDQLRNOpTest<uint8_t>({1, 128, 4, 5}, 255, ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

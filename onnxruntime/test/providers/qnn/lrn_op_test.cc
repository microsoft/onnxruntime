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
static GetTestModelFn BuildLRNTestCase(const TestInputDef<float>& input_def, int64_t size,
                                       float alpha = 0.0001f, float beta = 0.75f, float bias = 1.0f) {
  return [input_def, size, alpha, beta, bias](ModelTestBuilder& builder) {
    auto* input = MakeTestInput(builder, input_def);
    auto* output = builder.MakeOutput();

    Node& lrn_node = builder.AddNode("LRN", {input}, {output});
    lrn_node.AddAttribute("size", size);
    lrn_node.AddAttribute("alpha", alpha);
    lrn_node.AddAttribute("beta", beta);
    lrn_node.AddAttribute("bias", bias);
  };
}

// Creates a graph with a single Q/DQ LRN operator. Used for testing HTP backend.
template <typename InputQType = uint8_t>
static GetTestQDQModelFn<InputQType> BuildQDQLRNTestCase(const TestInputDef<float>& input_def, int64_t size,
                                                         float alpha = 0.0001f, float beta = 0.75f, float bias = 1.0f) {
  return [input_def, size, alpha, beta, bias](ModelTestBuilder& builder,
                                              std::vector<QuantParams<InputQType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    // LRN
    NodeArg* lrn_output = builder.MakeIntermediate();
    Node& lrn_node = builder.AddNode("LRN", {input_qdq}, {lrn_output});
    lrn_node.AddAttribute("size", size);
    lrn_node.AddAttribute("alpha", alpha);
    lrn_node.AddAttribute("beta", beta);
    lrn_node.AddAttribute("bias", bias);

    // LRN output -> Q -> DQ -> final output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, lrn_output, output_qparams[0].scale,
                                                      output_qparams[0].zero_point);
  };
}

// Runs an LRN model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunCPULRNOpTest(const TestInputDef<float>& input_def, int64_t size,
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

  RunQnnModelTest(BuildLRNTestCase(input_def, size, alpha, beta, bias),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Runs an LRN model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename QuantType>
static void RunQDQLRNOpTest(const TestInputDef<float>& input_def, int64_t size,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            float alpha = 0.0001f, float beta = 0.75f, float bias = 1.0f,
                            int opset = 13, QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildLRNTestCase(input_def, size, alpha, beta, bias),
                       BuildQDQLRNTestCase<QuantType>(input_def, size, alpha, beta, bias),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

//
// CPU tests:
//

TEST_F(QnnCPUBackendTests, LRNSize3) {
  RunCPULRNOpTest(TestInputDef<float>({1, 128, 4, 5}, false, -10.0f, 10.0f),
                  3,  // Size
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LRNSize5) {
  RunCPULRNOpTest(TestInputDef<float>({1, 128, 4, 5}, false, -10.0f, 10.0f),
                  5,  // Size
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LRN_size_larger_than_channel) {
  RunCPULRNOpTest(TestInputDef<float>({1, 128, 4, 5}, false, -10.0f, 10.0f),
                  255,  // Size
                  ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, LRNSize3) {
  RunQDQLRNOpTest<uint8_t>(TestInputDef<float>({1, 128, 4, 5}, false, -10.0f, 10.0f),
                           3,  // Size
                           ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LRNSize5) {
  RunQDQLRNOpTest<uint8_t>(TestInputDef<float>({1, 128, 4, 5}, false, -10.0f, 10.0f),
                           5,  // Size
                           ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LRN_size_larger_than_channel) {
  RunQDQLRNOpTest<uint8_t>(TestInputDef<float>({1, 128, 4, 5}, false, -10.0f, 10.0f),
                           255,  // Size
                           ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

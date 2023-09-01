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

// Creates a function that builds a model with a LeakyRelu operator.
static GetTestModelFn BuildLeakyReluOpTestCase(const TestInputDef<float>& input_def, float alpha) {
  return [input_def, alpha](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput(builder, input_def);
    NodeArg* output = builder.MakeOutput();
    Node& leakyrelu_node = builder.AddNode("LeakyRelu", {input}, {output});
    leakyrelu_node.AddAttribute("alpha", alpha);
  };
}

// Creates a function that builds a QDQ model with a LeakyRelu operator.
template <typename QuantType>
static GetTestQDQModelFn<QuantType> BuildQDQLeakyReluOpTestCase(const TestInputDef<float>& input_def,
                                                                float alpha) {
  return [input_def, alpha](ModelTestBuilder& builder,
                            std::vector<QuantParams<QuantType>>& output_qparams) {
    // input => Q => DQ =>
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    // LeakryRelu
    auto* leakyrelu_output = builder.MakeIntermediate();
    Node& leakyrelu_node = builder.AddNode("LeakyRelu", {input_qdq}, {leakyrelu_output});
    leakyrelu_node.AddAttribute("alpha", alpha);

    // => Q => DQ -> final output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, leakyrelu_output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point);
  };
}

// Checks the accuracy of a QDQ LeakyRelu model by comparing to ORT CPU EP.
template <typename QuantType>
static void RunLeakyReluOpQDQTest(const TestInputDef<float>& input_def,
                                  float alpha,
                                  int opset,
                                  ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildLeakyReluOpTestCase(input_def, alpha),
                       BuildQDQLeakyReluOpTestCase<QuantType>(input_def, alpha),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       1e-5f);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
TEST_F(QnnHTPBackendTests, LeakyReluOpSet15) {
  RunLeakyReluOpQDQTest<uint8_t>(TestInputDef<float>({1, 2, 3}, false, {-40.0f, -20.0f, 0.0f, 10.0f, 30.0f, 40.0f}),
                                 0.2f,
                                 15,
                                 ExpectedEPNodeAssignment::All);
}

// Test creates a DQ -> Gather -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results match the CPU EP results.
//
// - Uses uint8 as the quantization type.
TEST_F(QnnHTPBackendTests, LeakyReluOpSet16) {
  RunLeakyReluOpQDQTest<uint8_t>(TestInputDef<float>({1, 2, 3}, false, {-40.0f, -20.0f, 0.0f, 10.0f, 30.0f, 40.0f}),
                                 0.2f,
                                 16,
                                 ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
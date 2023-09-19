// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "onnx/onnx_pb.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Clip operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunClipTestOnCPU(const std::vector<TestInputDef<DataType>>& input_defs,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType>("Clip", input_defs, {}),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Clip with a dynamic min or max input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Clip_Dynamic_MinMax_Unsupported) {
  // Dynamic min input is not supported.
  RunClipTestOnCPU<float>({TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                           TestInputDef<float>({}, false /* is_initializer */, {-5.0f})},
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
  // Dynamic max input is not supported.
  RunClipTestOnCPU<float>({TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                           TestInputDef<float>({}, true, {-5.0f}),
                           TestInputDef<float>({}, false, {5.0f})},
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test Clip with default min/max.
TEST_F(QnnCPUBackendTests, Clip_4D_f32_DefaultMinMax) {
  RunClipTestOnCPU<float>({TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))},
                          ExpectedEPNodeAssignment::All);
}

// Test Clip with 5D input.
TEST_F(QnnCPUBackendTests, Clip_5D_f32) {
  RunClipTestOnCPU<float>({TestInputDef<float>({1, 1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                           TestInputDef<float>({}, true, {-5.0f}),
                           TestInputDef<float>({}, true, {5.0f})},
                          ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that builds a model with a QDQ Clip operator. Only the first input is quantized.
template <typename InputQType>
inline GetTestQDQModelFn<InputQType> BuildQDQClipTestCase(const std::vector<TestInputDef<float>>& input_defs) {
  return [input_defs](ModelTestBuilder& builder, std::vector<QuantParams<InputQType>>& output_qparams) {
    const size_t num_inputs = input_defs.size();
    std::vector<NodeArg*> op_inputs;
    op_inputs.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; i++) {
      const TestInputDef<float>& input_def = input_defs[i];
      NodeArg* input = MakeTestInput<float>(builder, input_def);

      if (i == 0) {  // Only input 0 is quantized.
        QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
        NodeArg* input_after_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale,
                                                              input_qparams.zero_point);
        op_inputs.push_back(input_after_qdq);
      } else {
        op_inputs.push_back(input);
      }
    }

    // Op -> op_output
    auto* clip_output = builder.MakeIntermediate();
    builder.AddNode("Clip", op_inputs, {clip_output});

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, clip_output, output_qparams[0].scale,
                                                      output_qparams[0].zero_point);
  };
}

// Runs a QDQ Clip model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (when compared to the baseline float32 model).
template <typename QType>
static void RunQDQClipTestOnHTP(const std::vector<TestInputDef<float>>& input_defs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase("Clip", input_defs, {}),  // baseline float32 model
                       BuildQDQClipTestCase<QType>(input_defs),  // QDQ model
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test QDQ Clip with default min/max.
// NOTE: The Clip operator is *optimized* away during L1 optimizations, so QNN EP does not get a graph with a Clip op.
// Instead, QNN EP will get a graph with a Q -> DQ.
// - Original sequence: Q1 -> DQ1 -> Clip -> Q2 -> DQ2
// - ClipQuantFusion: Fuses Clip -> QuantizeLinear resulting in Q1 -> DQ1 -> Q2' -> DQ2
// - DoubleQDQPairsRemover: Simplifies remaining Q1 -> DQ1 -> Q2' -> DQ2 sequence to Q1 -> DQ2.
TEST_F(QnnHTPBackendTests, Clip_U8_DefaultMinMax_Rank4) {
  RunQDQClipTestOnHTP<uint8_t>({TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))},
                               ExpectedEPNodeAssignment::All);
}

// Test QDQ Clip with non-default min and max inputs. QNN EP will get a graph with a Clip operator.
TEST_F(QnnHTPBackendTests, Clip_U8_Rank4) {
  RunQDQClipTestOnHTP<uint8_t>({TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                TestInputDef<float>({}, true, {-5.0f}),
                                TestInputDef<float>({}, true, {5.0f})},
                               ExpectedEPNodeAssignment::All);
}

// Test QDQ Clip of rank 5.
TEST_F(QnnHTPBackendTests, Clip_U8_Rank5) {
  // We can't use the usual model-building functions because they add standalone Quantize and Dequantize nodes
  // at the input and output. These Q/DQ ops get lowered to QNN's Quantize and Dequantize operators, which DO NOT
  // support rank 5 tensors. Therefore, we have to create a test model that only instantiates the DQ -> Clip -> Q
  // QDQ node group, which gets lowered to a single QNN Clip node.
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    // input (u8) -> DQ ->
    NodeArg* quant_input = builder.MakeInput<uint8_t>({1, 1, 2, 2, 2}, {0, 1, 6, 10, 20, 100, 128, 255});
    NodeArg* input_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(quant_input, 1.0f, 0, input_dq);  // scale = 1.0, zp = 0

    // Min/Max initializers
    NodeArg* min_input = builder.MakeScalarInitializer(5.0f);
    NodeArg* max_input = builder.MakeScalarInitializer(100.0f);

    // Clip ->
    NodeArg* clip_output = builder.MakeIntermediate();
    builder.AddNode("Clip", {input_dq, min_input, max_input}, {clip_output});

    // Q -> output (u8)
    NodeArg* output = builder.MakeOutput();
    builder.AddQuantizeLinearNode<uint8_t>(clip_output, 1.0f, 0, output);  // scale = 1.0, zp = 0
  };

  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

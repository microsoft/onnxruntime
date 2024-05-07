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
static void RunClipTest(const TestInputDef<DataType>& input_def,
                        const std::vector<TestInputDef<DataType>>& min_max_defs,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        bool on_cpu_backend = true,
                        int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = on_cpu_backend ? "QnnCpu.dll" : "QnnHtp.dll";
#else
  provider_options["backend_path"] = on_cpu_backend ? "libQnnCpu.so" : "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType, DataType>("Clip", {input_def}, min_max_defs, {}),
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
  RunClipTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                     {TestInputDef<float>({}, false /* is_initializer */, {-5.0f})},
                     ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
  // Dynamic max input is not supported.
  RunClipTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                     {TestInputDef<float>({}, true, {-5.0f}),
                      TestInputDef<float>({}, false, {5.0f})},
                     ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test Clip with default min/max.
TEST_F(QnnCPUBackendTests, Clip_4D_f32_DefaultMinMax) {
  RunClipTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                     {},  // Don't specify min/max inputs.
                     ExpectedEPNodeAssignment::All);
}

// Test Clip with 5D input.
TEST_F(QnnCPUBackendTests, Clip_5D_f32) {
  RunClipTest<float>(TestInputDef<float>({1, 1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                     {TestInputDef<float>({}, true, {-5.0f}),
                      TestInputDef<float>({}, true, {5.0f})},
                     ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test Clip with float32 on HTP
TEST_F(QnnHTPBackendTests, Clip_f32) {
  bool on_cpu_backend = false;
  RunClipTest<float>(TestInputDef<float>({1, 1, 3, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 12)),
                     {TestInputDef<float>({}, true, {-5.0f}),
                      TestInputDef<float>({}, true, {5.0f})},
                     ExpectedEPNodeAssignment::All,
                     on_cpu_backend);
}

// Test Clip with int32 on HTP
TEST_F(QnnHTPBackendTests, Clip_int32) {
  bool on_cpu_backend = false;
  RunClipTest<int32_t>(TestInputDef<int32_t>({1, 1, 3, 2}, false, {1, 2, -5, 3, -10, 25}),
                       {TestInputDef<int32_t>({}, true, {-5}),
                        TestInputDef<int32_t>({}, true, {5})},
                       ExpectedEPNodeAssignment::All,
                       on_cpu_backend);
}

// Runs a QDQ Clip model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQClipTestOnHTP(const TestInputDef<float>& input_def,
                                const std::vector<TestInputDef<float>>& min_max_defs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13,
                                bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto f32_model_builder = BuildOpTestCase<float, float>("Clip", {input_def}, {min_max_defs}, {});
  auto qdq_model_builder = BuildQDQOpTestCase<QType, float>("Clip", {input_def}, {min_max_defs}, {},
                                                            kOnnxDomain, use_contrib_qdq);

  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test 8-bit QDQ Clip with default min/max.
// NOTE: The Clip operator is *optimized* away during L1 optimizations, so QNN EP does not get a graph with a Clip op.
// Instead, QNN EP will get a graph with a Q -> DQ.
// - Original sequence: Q1 -> DQ1 -> Clip -> Q2 -> DQ2
// - ClipQuantFusion: Fuses Clip -> QuantizeLinear resulting in Q1 -> DQ1 -> Q2' -> DQ2
// - DoubleQDQPairsRemover: Simplifies remaining Q1 -> DQ1 -> Q2' -> DQ2 sequence to Q1 -> DQ2.
TEST_F(QnnHTPBackendTests, Clip_U8_DefaultMinMax_Rank4) {
  RunQDQClipTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               {},  // Don't specify min/max inputs.
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Clip with default min/max.
// NOTE: The Clip operator is *optimized* away during L1 optimizations, so QNN EP does not get a graph with a Clip op.
// Instead, QNN EP will get a graph with a Q -> DQ.
// - Original sequence: Q1 -> DQ1 -> Clip -> Q2 -> DQ2
// - ClipQuantFusion: Fuses Clip -> QuantizeLinear resulting in Q1 -> DQ1 -> Q2' -> DQ2
// - DoubleQDQPairsRemover: Simplifies remaining Q1 -> DQ1 -> Q2' -> DQ2 sequence to Q1 -> DQ2.
TEST_F(QnnHTPBackendTests, Clip_U16_DefaultMinMax_Rank4) {
  RunQDQClipTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                {},  // Don't specify min/max inputs.
                                ExpectedEPNodeAssignment::All,
                                13,     // opset
                                true);  // Use com.microsoft Q/DQ ops
}

// Test 8-bit QDQ Clip with non-default min and max inputs. QNN EP will get a graph with a Clip operator.
TEST_F(QnnHTPBackendTests, Clip_U8_Rank4) {
  RunQDQClipTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               {TestInputDef<float>({}, true, {-5.0f}),
                                TestInputDef<float>({}, true, {5.0f})},
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Clip with non-default min and max inputs. QNN EP will get a graph with a Clip operator.
TEST_F(QnnHTPBackendTests, Clip_U16_Rank4) {
  RunQDQClipTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                {TestInputDef<float>({}, true, {-5.0f}),
                                 TestInputDef<float>({}, true, {5.0f})},
                                ExpectedEPNodeAssignment::All,
                                13,     // opset
                                true);  // Use com.microsoft Q/DQ ops
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

// Test FP16 Clip with min (FP16)
TEST_F(QnnHTPBackendTests, Clip_FP16) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto f32_input = TestInputDef<float>({1, 3, 2, 2}, false,
                                       {-10.0f, -8.0f, -3.5f, 2.2f,
                                        1.3f, 1.5f, 3.2f, 5.8f,
                                        5.8f, 9.7f, 8.5f, 8.9f});
  std::vector<MLFloat16> f16_data;
  std::for_each(f32_input.GetRawData().begin(), f32_input.GetRawData().end(),
                [&f16_data](const float data) {
                  f16_data.push_back(static_cast<MLFloat16>(data));
                });
  auto f16_input = TestInputDef<MLFloat16>({1, 3, 2, 2}, false, f16_data);

  const float min_f32 = 1.2f;
  const MLFloat16 min_f16 = static_cast<MLFloat16>(min_f32);
  auto f32_min_input = TestInputDef<float>({}, true, {min_f32});
  auto f16_min_input = TestInputDef<MLFloat16>({}, true, {min_f16});

  auto f32_model_builder = BuildOpTestCase<float, float>("Clip", {f32_input}, {f32_min_input}, {});
  auto f16_model_builder = BuildOpTestCase<MLFloat16, MLFloat16>("Clip", {f16_input}, {f16_min_input}, {});
  int opset = 13;
  ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All;

  TestFp16ModelAccuracy(f32_model_builder,
                        f16_model_builder,
                        provider_options,
                        opset,
                        expected_ep_assignment);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

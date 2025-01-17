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

// Runs a model with a Tile operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunTileTestOnCPU(const TestInputDef<DataType>& input_def,
                             const TestInputDef<int64_t>& repeats_def,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType, int64_t>("Tile", {input_def}, {repeats_def}, {}),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Test that Tile with a dynamic repeats input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Tile_DynamicRepeats_Unsupported) {
  RunTileTestOnCPU(TestInputDef<float>({2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),
                   TestInputDef<int64_t>({2}, false /* is_initializer */, {1, 2}),
                   ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test that Tile with rank 4 float input.
TEST_F(QnnCPUBackendTests, Tile_F32_Rank4) {
  std::vector<float> input_data = {-4.0f, -3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  RunTileTestOnCPU(TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                   TestInputDef<int64_t>({4}, true /* is_initializer */, {1, 2, 1, 1}),
                   ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that creates a graph with a QDQ Tile operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQTileTestCase(const TestInputDef<float>& input_def,
                                                  const TestInputDef<int64_t>& repeats_def,
                                                  bool use_contrib_qdq = false) {
  return [input_def, repeats_def, use_contrib_qdq](ModelTestBuilder& builder,
                                                   std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                   use_contrib_qdq);

    // repeats input
    NodeArg* repeats_input = MakeTestInput(builder, repeats_def);

    // Tile op
    NodeArg* tile_output = builder.MakeIntermediate();
    builder.AddNode("Tile", {input_qdq, repeats_input}, {tile_output});

    // op_output -> Q -> DQ -> output
    // NOTE: Input and output quantization parameters must be equal for Tile.
    output_qparams[0] = input_qparams;  // Overwrite!
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, tile_output, input_qparams.scale,
                                                     input_qparams.zero_point, use_contrib_qdq);
  };
}

// Runs a QDQ Tile model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQTileTestOnHTP(const TestInputDef<float>& input_def,
                                const TestInputDef<int64_t>& repeats_def,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13,
                                bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto f32_model_builder = BuildOpTestCase<float, int64_t>("Tile", {input_def}, {repeats_def}, {});
  auto qdq_model_builder = BuildQDQTileTestCase<QType>(input_def, repeats_def, use_contrib_qdq);
  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test 8-bit QDQ Tile with rank 4 input.
TEST_F(QnnHTPBackendTests, Tile_U8_Rank4) {
  std::vector<float> input_data = {-4.0f, -3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  RunQDQTileTestOnHTP<uint8_t>(TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                               TestInputDef<int64_t>({4}, true /* is_initializer */, {1, 2, 1, 1}),
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Tile with rank 4 input.
TEST_F(QnnHTPBackendTests, Tile_U16_Rank4) {
  std::vector<float> input_data = {-4.0f, -3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  RunQDQTileTestOnHTP<uint16_t>(TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                                TestInputDef<int64_t>({4}, true /* is_initializer */, {1, 2, 1, 1}),
                                ExpectedEPNodeAssignment::All,
                                13,     // opset
                                true);  // Use com.microsoft Q/DQ ops
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

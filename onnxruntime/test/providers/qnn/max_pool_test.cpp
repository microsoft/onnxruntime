// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "core/graph/node_attr_utils.h"
#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that creates a graph with a single MaxPool operator.
static GetTestModelFn BuildMaxPoolTestCase(const TestInputDef<float>& input_def,
                                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, attrs](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput(builder, input_def);
    NodeArg* output = builder.MakeOutput();
    Node& pool_node = builder.AddNode("MaxPool", {input}, {output});

    for (const auto& attr : attrs) {
      pool_node.AddAttributeProto(attr);
    }
  };
}

// Returns a function that creates a graph with a QDQ MaxPool operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildMaxPoolQDQTestCase(const TestInputDef<float>& input_def,
                                                     const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, attrs](ModelTestBuilder& builder,
                            std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    // MaxPool
    NodeArg* pool_output = builder.MakeIntermediate();
    Node& pool_node = builder.AddNode("MaxPool", {input_qdq}, {pool_output});

    for (const auto& attr : attrs) {
      pool_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    // NOTE: Input and output quantization parameters must be equal for MaxPool.
    output_qparams[0] = input_qparams;  // Overwrite!
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, pool_output, input_qparams.scale,
                                                     input_qparams.zero_point);
  };
}

// Runs an MaxPool model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunMaxPoolOpTest(const TestInputDef<float>& input_def,
                             const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildMaxPoolTestCase(input_def, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ MaxPool model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
template <typename QuantType>
static void RunQDQMaxPoolOpTest(const TestInputDef<float>& input_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildMaxPoolTestCase(input_def, attrs),
                       BuildMaxPoolQDQTestCase<QuantType>(input_def, attrs),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       1e-5f);
}

//
// CPU tests:
//

// MaxPool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnCPUBackendTests, MaxPool_Global) {
  RunMaxPoolOpTest(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                   {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                    utils::MakeAttribute("strides", std::vector<int64_t>{3, 3}),
                    utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                    utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                    utils::MakeAttribute("ceil_mode", static_cast<int64_t>(0)),
                    utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                    utils::MakeAttribute("auto_pad", "NOTSET")},
                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, MaxPool_Large_Input) {
  RunMaxPoolOpTest(TestInputDef<float>({1, 125, 8, 56}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                   {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                    utils::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                    utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                    utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                    utils::MakeAttribute("ceil_mode", static_cast<int64_t>(0)),
                    utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                    utils::MakeAttribute("auto_pad", "NOTSET")},
                   ExpectedEPNodeAssignment::All);
}

// QNN v2.13, backendValidateOpConfig() failed for node `MaxPool` of type `PoolMax2d` with error code 4003
TEST_F(QnnCPUBackendTests, DISABLED_MaxPool_Ceil) {
  RunMaxPoolOpTest(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                   {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                    utils::MakeAttribute("strides", std::vector<int64_t>{3, 3}),
                    utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                    utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                    utils::MakeAttribute("ceil_mode", static_cast<int64_t>(1)),
                    utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                    utils::MakeAttribute("auto_pad", "NOTSET")},
                   ExpectedEPNodeAssignment::All);
}

// QNN v2.13, backendValidateOpConfig() failed for node `MaxPool` of type `PoolMax2d` with error code 4003
TEST_F(QnnCPUBackendTests, DISABLED_MaxPool_Large_Input2_Ceil) {
  RunMaxPoolOpTest(TestInputDef<float>({1, 128, 16, 113}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                   {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                    utils::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                    utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                    utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                    utils::MakeAttribute("ceil_mode", static_cast<int64_t>(1)),
                    utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                    utils::MakeAttribute("auto_pad", "NOTSET")},
                   ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//
// QDQ MaxPool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnHTPBackendTests, MaxPool_Global_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                               {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                utils::MakeAttribute("strides", std::vector<int64_t>{3, 3}),
                                utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                                utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                                utils::MakeAttribute("ceil_mode", static_cast<int64_t>(0)),
                                utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                                utils::MakeAttribute("auto_pad", "NOTSET")},
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, MaxPool_Large_Input_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>(TestInputDef<float>({1, 125, 8, 56}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                               {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                utils::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                                utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                                utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                                utils::MakeAttribute("ceil_mode", static_cast<int64_t>(0)),
                                utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                                utils::MakeAttribute("auto_pad", "NOTSET")},
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, MaxPool_Ceil_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                               {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                utils::MakeAttribute("strides", std::vector<int64_t>{3, 3}),
                                utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                                utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                                utils::MakeAttribute("ceil_mode", static_cast<int64_t>(1)),
                                utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                                utils::MakeAttribute("auto_pad", "NOTSET")},
                               ExpectedEPNodeAssignment::All);
}

// QNN v2.13: Inaccuracy detected for output 'output', element 58367.
// Output quant params: scale=0.078431375324726105, zero_point=127.
// Expected val: 5.6846914291381836
// QNN QDQ val: -5.3333334922790527 (err 11.018024444580078)
// CPU QDQ val: 5.6470589637756348 (err 0.037632465362548828)
TEST_F(QnnHTPBackendTests, DISABLED_MaxPool_Large_Input2_Ceil_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>(TestInputDef<float>({1, 128, 16, 113}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                               {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2}),
                                utils::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                                utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}),
                                utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                                utils::MakeAttribute("ceil_mode", static_cast<int64_t>(1)),
                                utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                                utils::MakeAttribute("auto_pad", "NOTSET")},
                               ExpectedEPNodeAssignment::All);
}

// QNN v2.13: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnHTPBackendTests, DISABLED_MaxPool_LargeInput_1Pads) {
  RunQDQMaxPoolOpTest<uint8_t>(TestInputDef<float>({1, 64, 384, 576}, false, -10.0f, 10.0f),  // Dynamic input with range [-10, 10]
                               {utils::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                utils::MakeAttribute("strides", std::vector<int64_t>{2, 2}),
                                utils::MakeAttribute("pads", std::vector<int64_t>{1, 1, 1, 1}),
                                utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}),
                                utils::MakeAttribute("ceil_mode", static_cast<int64_t>(0)),
                                utils::MakeAttribute("storage_order", static_cast<int64_t>(0)),
                                utils::MakeAttribute("auto_pad", "NOTSET")},
                               ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
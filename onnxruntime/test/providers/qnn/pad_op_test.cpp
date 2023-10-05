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

// Returns a function that creates a graph with a single Pad operator.
static GetTestModelFn BuildPadTestCase(const TestInputDef<float>& data_def,
                                       const TestInputDef<int64_t>& pads_def,
                                       const TestInputDef<float>& constant_value_def,
                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                       bool has_constant_value = true) {
  return [data_def, pads_def, constant_value_def, attrs, has_constant_value](ModelTestBuilder& builder) {
    NodeArg* data = MakeTestInput(builder, data_def);
    NodeArg* pads = MakeTestInput(builder, pads_def);
    std::vector<NodeArg*> inputs{data, pads};
    if (has_constant_value) {
      NodeArg* constant_value = MakeTestInput(builder, constant_value_def);
      inputs.push_back(constant_value);
    }
    NodeArg* output = builder.MakeOutput();
    Node& pad_node = builder.AddNode("Pad", inputs, {output});

    for (const auto& attr : attrs) {
      pad_node.AddAttributeProto(attr);
    }
  };
}

// Returns a function that creates a graph with a QDQ Pad operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildPadQDQTestCase(const TestInputDef<float>& data_def,
                                                 const TestInputDef<int64_t>& pads_def,
                                                 const TestInputDef<float>& constant_value_def,
                                                 const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                 bool has_constant_value,
                                                 bool constant_value_quantized) {
  return [data_def, pads_def, constant_value_def, attrs, has_constant_value, constant_value_quantized](ModelTestBuilder& builder,
                                                                                                       std::vector<QuantParams<QuantType>>& output_qparams) {
    std::vector<NodeArg*> inputs;
    // data -> Q -> DQ ->
    NodeArg* data = MakeTestInput(builder, data_def);
    QuantParams<QuantType> data_qparams = GetTestInputQuantParams<QuantType>(data_def);
    NodeArg* data_qdq = AddQDQNodePair<QuantType>(builder, data, data_qparams.scale, data_qparams.zero_point);
    inputs.push_back(data_qdq);

    // pads
    NodeArg* pads = MakeTestInput(builder, pads_def);
    inputs.push_back(pads);

    // constant_value -- QNN support both quantized and non-quantized
    if (has_constant_value) {
      if (constant_value_quantized) {
        // constant_value -> Q -> DQ ->
        NodeArg* constant_value = MakeTestInput(builder, constant_value_def);
        QuantParams<QuantType> constant_value_qparams = GetTestInputQuantParams<QuantType>(constant_value_def);
        NodeArg* constant_value_qdq = AddQDQNodePair<QuantType>(builder, constant_value,
                                                                constant_value_qparams.scale,
                                                                constant_value_qparams.zero_point);
        inputs.push_back(constant_value_qdq);
      } else {
        NodeArg* constant_value = MakeTestInput(builder, constant_value_def);
        inputs.push_back(constant_value);
      }
    }

    NodeArg* output = builder.MakeIntermediate();
    Node& pad_node = builder.AddNode("Pad", inputs, {output});

    for (const auto& attr : attrs) {
      pad_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point);
  };
}

// Runs an Pad model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunPadOpTest(const TestInputDef<float>& data_def,
                         const TestInputDef<int64_t>& pads_def,
                         const TestInputDef<float>& constant_value_def,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         bool has_constant_value = true,
                         int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildPadTestCase(data_def, pads_def, constant_value_def, attrs, has_constant_value),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ Pad model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
template <typename QuantType>
static void RunQDQPadOpTest(const TestInputDef<float>& data_def,
                            const TestInputDef<int64_t>& pads_def,
                            const TestInputDef<float>& constant_value_def,
                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            bool has_constant_value = true,
                            bool constant_value_quantized = true,
                            int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildPadTestCase(data_def, pads_def, constant_value_def, attrs),
                       BuildPadQDQTestCase<QuantType>(data_def, pads_def, constant_value_def, attrs,
                                                      has_constant_value, constant_value_quantized),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       1e-5f);
}

//
// CPU tests:
//

// Pad 2d
TEST_F(QnnCPUBackendTests, Pad2d) {
  RunPadOpTest(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
               TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
               TestInputDef<float>({1}, true, {0.0f}),
               {utils::MakeAttribute("mode", "constant")},
               ExpectedEPNodeAssignment::All);
}

// Pad 2d, pads input not initializer
TEST_F(QnnCPUBackendTests, Pad2dPadsNotIni) {
  RunPadOpTest(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
               TestInputDef<int64_t>({4}, false, {0, 2, 0, 0}),
               TestInputDef<float>({1}, true, {0.0f}),
               {utils::MakeAttribute("mode", "constant")},
               ExpectedEPNodeAssignment::None);
}

// Pad reflect mode
// Expected: contains 12 values, where each value and its corresponding value in 16-byte object <0C-00 00-00 00-00 00-00 40-01 23-05 EC-01 00-00> are an almost-equal pair
// Actual: 16-byte object <0C-00 00-00 00-00 00-00 40-01 12-05 EC-01 00-00>, where the value pair (1.2, 0) at index #1 don't match, which is -1.2 from 1.2
TEST_F(QnnCPUBackendTests, DISABLED_PadModeReflect) {
  bool has_constant_value = false;
  RunPadOpTest(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
               TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
               TestInputDef<float>({1}, true, {0.0f}),
               {utils::MakeAttribute("mode", "reflect")},
               ExpectedEPNodeAssignment::All,
               has_constant_value);
}

// Pad edge mode
TEST_F(QnnCPUBackendTests, PadModeEdge) {
  bool has_constant_value = false;
  RunPadOpTest(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
               TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
               TestInputDef<float>({1}, true, {0.0f}),
               {utils::MakeAttribute("mode", "edge")},
               ExpectedEPNodeAssignment::All,
               has_constant_value);
}

// Pad wrap mode not supported
TEST_F(QnnCPUBackendTests, PadModeWrap) {
  bool has_constant_value = false;
  RunPadOpTest(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
               TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
               TestInputDef<float>({1}, true, {0.0f}),
               {utils::MakeAttribute("mode", "wrap")},
               ExpectedEPNodeAssignment::None,  // not supported
               has_constant_value);
}

// Pad 4d
TEST_F(QnnCPUBackendTests, Pad4d) {
  RunPadOpTest(TestInputDef<float>({1, 2, 2, 2}, false,
                                   {1.0f, 1.0f,
                                    1.0f, 1.0f,
                                    1.0f, 1.0f,
                                    1.0f, 1.0f}),
               TestInputDef<int64_t>({8}, true, {0, 0, 0, 1, 0, 0, 0, 1}),
               TestInputDef<float>({1}, true, {0.0f}),
               {utils::MakeAttribute("mode", "constant")},
               ExpectedEPNodeAssignment::All);
}

// Pad 5d supported
TEST_F(QnnCPUBackendTests, Pad5d) {
  RunPadOpTest(TestInputDef<float>({1, 2, 2, 2, 2}, false, GetFloatDataInRange(1.0f, 10.0f, 16)),
               TestInputDef<int64_t>({10}, true, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0}),
               TestInputDef<float>({1}, true, {5.0f}),
               {utils::MakeAttribute("mode", "constant")},
               ExpectedEPNodeAssignment::All);
}

// Pad 6d supported
TEST_F(QnnCPUBackendTests, Pad6d) {
  RunPadOpTest(TestInputDef<float>({1, 2, 2, 2, 2, 2}, false, GetFloatDataInRange(1.0f, 10.0f, 32)),
               TestInputDef<int64_t>({12}, true, {0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}),
               TestInputDef<float>({1}, true, {0.0f}),
               {utils::MakeAttribute("mode", "constant")},
               ExpectedEPNodeAssignment::None);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//
// QDQ Pad
TEST_F(QnnHTPBackendTests, PadNoConstantValue) {
  bool has_constant_value_input = false;
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
                           TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
                           TestInputDef<float>({1}, true, {0.0f}),
                           {utils::MakeAttribute("mode", "constant")},
                           ExpectedEPNodeAssignment::All,
                           has_constant_value_input);
}

TEST_F(QnnHTPBackendTests, PadHasConstantValueNonQuantized) {
  bool has_constant_value_input = true;
  bool constant_value_quantized = false;
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
                           TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
                           TestInputDef<float>({1}, true, {0.0f}),
                           {utils::MakeAttribute("mode", "constant")},
                           ExpectedEPNodeAssignment::All,
                           has_constant_value_input,
                           constant_value_quantized);
}

TEST_F(QnnHTPBackendTests, PadHasConstantValueQuantized) {
  bool has_constant_value_input = true;
  bool constant_value_quantized = true;
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
                           TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
                           TestInputDef<float>({1}, true, {0.0f}),
                           {utils::MakeAttribute("mode", "constant")},
                           ExpectedEPNodeAssignment::All,
                           has_constant_value_input,
                           constant_value_quantized);
}

// QNN graph execute error. Error code: 6031
TEST_F(QnnHTPBackendTests, DISABLED_PadReflectMode) {
  bool has_constant_value_input = false;
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
                           TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
                           TestInputDef<float>({1}, true, {0.0f}),
                           {utils::MakeAttribute("mode", "reflect")},
                           ExpectedEPNodeAssignment::All,
                           has_constant_value_input);
}

TEST_F(QnnHTPBackendTests, PadEdgeMode) {
  bool has_constant_value_input = false;
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
                           TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
                           TestInputDef<float>({1}, true, {0.0f}),
                           {utils::MakeAttribute("mode", "edge")},
                           ExpectedEPNodeAssignment::All,
                           has_constant_value_input);
}

// wrap mode not supported
TEST_F(QnnHTPBackendTests, PadWrapMode) {
  bool has_constant_value_input = false;
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({3, 2}, false, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f}),
                           TestInputDef<int64_t>({4}, true, {0, 2, 0, 0}),
                           TestInputDef<float>({1}, true, {0.0f}),
                           {utils::MakeAttribute("mode", "wrap")},
                           ExpectedEPNodeAssignment::None,
                           has_constant_value_input);
}

TEST_F(QnnHTPBackendTests, Pad4d) {
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({1, 2, 2, 2}, false,
                                               {1.0f, 2.0f,
                                                3.0f, 4.0f,
                                                5.0f, 6.0f,
                                                7.0f, 8.0f}),
                           TestInputDef<int64_t>({8}, true, {0, 0, 0, 1, 0, 0, 0, 1}),
                           TestInputDef<float>({1}, true, {5.0f}),
                           {utils::MakeAttribute("mode", "constant")},
                           ExpectedEPNodeAssignment::All);
}

// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.035294119268655777, zero_point=0.
// Expected val: 9
// QNN QDQ val: 8.0117654800415039 (err 0.98823451995849609)
// CPU QDQ val: 9 (err 0)
// QNN limitation? pad_constant_value has to be within the range of input[0].
// Here pad_constant_value = 9.0 > max(input[0]) = 8.0
TEST_F(QnnHTPBackendTests, DISABLED_Pad4dOutOfRangePadConstantValue) {
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({1, 2, 2, 2}, false,
                                               {1.0f, 2.0f,
                                                3.0f, 4.0f,
                                                5.0f, 6.0f,
                                                7.0f, 8.0f}),
                           TestInputDef<int64_t>({8}, true, {0, 0, 0, 1, 0, 0, 0, 1}),
                           TestInputDef<float>({1}, true, {9.0f}),  // pad_constant_value out of input[0] range
                           {utils::MakeAttribute("mode", "constant")},
                           ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Pad5d) {
  RunQDQPadOpTest<uint8_t>(TestInputDef<float>({1, 2, 2, 2, 2}, false, GetFloatDataInRange(1.0f, 10.0f, 16)),
                           TestInputDef<int64_t>({10}, true, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0}),
                           TestInputDef<float>({1}, true, {2.0f}),
                           {utils::MakeAttribute("mode", "constant")},
                           ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
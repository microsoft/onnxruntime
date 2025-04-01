// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Upsample operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunUpsampleTestOnCPU(const std::string& op_type,
                                 const TestInputDef<DataType>& input_def,
                                 const TestInputDef<float>& scales_def,
                                 std::vector<ONNX_NAMESPACE::AttributeProto> attrs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 9) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  if (opset <= 7) {
    const std::vector<float>& scales = scales_def.GetRawData();
    attrs.push_back(utils::MakeAttribute("scales", scales));

    RunQnnModelTest(BuildOpTestCase<DataType>(op_type, {input_def}, {}, attrs),
                    provider_options,
                    opset,
                    expected_ep_assignment);
  } else {
    RunQnnModelTest(BuildOpTestCase<DataType, float>(op_type, {input_def}, {scales_def}, attrs),
                    provider_options,
                    opset,
                    expected_ep_assignment);
  }
}

//
// CPU tests:
//

// Test that Upsample with a dynamic scales input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Upsample_DynamicScales_Unsupported) {
  RunUpsampleTestOnCPU("Upsample",
                       TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, false /* is_initializer */, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "nearest")},  // Attributes
                       ExpectedEPNodeAssignment::None,             // Should not be assigned to QNN EP.
                       9);                                         // Opset
}

// Test Upsample with opset-9, mode `nearest`
TEST_F(QnnCPUBackendTests, Upsample_4D_Nearest_opset9) {
  RunUpsampleTestOnCPU("Upsample",
                       TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "nearest")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       9);  // Opset
}

// Test Upsample with opset-9, mode `linear`
TEST_F(QnnCPUBackendTests, Upsample_4D_Linear_opset9) {
  RunUpsampleTestOnCPU("Upsample",
                       TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "linear")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       9);  // Opset
}

// Test Upsample with opset-7, mode `nearest`
TEST_F(QnnCPUBackendTests, Upsample_4D_Nearest_opset7) {
  RunUpsampleTestOnCPU("Upsample",
                       TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "nearest")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       7);  // Opset
}

// Test Upsample with opset-7, mode `linear`
TEST_F(QnnCPUBackendTests, Upsample_4D_Linear_opset7) {
  RunUpsampleTestOnCPU("Upsample",
                       TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "linear")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       7);  // Opset
}

// Test Upsample 5D
TEST_F(QnnCPUBackendTests, Upsample_5D) {
  RunUpsampleTestOnCPU("Upsample",
                       TestInputDef<float>({1, 3, 4, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({5}, true, {1.0f, 1.0f, 1.5f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "nearest")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       9);  // Opset
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that creates a graph with a QDQ Upsample operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQUpsampleTestCase(const std::string& op_type,
                                                      const TestInputDef<float>& input_def,
                                                      const TestInputDef<int64_t>& scales_def,
                                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                      int opset = 10,
                                                      bool use_contrib_qdq = false) {
  return [input_def, scales_def, attrs,
          use_contrib_qdq, op_type](ModelTestBuilder& builder,
                                    std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                   use_contrib_qdq);

    if (opset <= 7) {
      // Upsample op
      NodeArg* upsample_output = builder.MakeIntermediate();
      Node& upsample_node = builder.AddNode(op_type, {input_qdq}, {upsample_output});
    } else {
      // scales input
      NodeArg* scales_input = MakeTestInput(builder, scales_def);

      // Upsample op
      NodeArg* upsample_output = builder.MakeIntermediate();
      Node& upsample_node = builder.AddNode(op_type, {input_qdq, scales_input}, {upsample_output});
    }

    for (auto& attr : attrs) {
      upsample_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    // Input and output quantization parameters are equal for Upsample.
    output_qparams[0] = input_qparams;
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, upsample_output, input_qparams.scale,
                                                     input_qparams.zero_point, use_contrib_qdq);
  };
}

// Run a QDQ Upsample model on the QNN HTP EP and the ORT CPU EP. Check the graph node assignment and the QDQ model
// inference on QNN HTP EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQUpsampleTestOnHTP(const std::string& op_type,
                                    const TestInputDef<float>& input_def,
                                    const TestInputDef<float>& scales_def,
                                    std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    int opset = 10,
                                    bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  if (opset <= 7) {
    const std::vector<float>& scales = scales_def.GetRawData();
    attrs.push_back(utils::MakeAttribute("scales", scales));

    auto f32_model_builder = BuildOpTestCase<float>(op_type, {input_def}, {}, attrs);
  } else {
    auto f32_model_builder = BuildOpTestCase<float, float>(op_type, {input_def}, {scales_def}, attrs);
  }

  auto qdq_model_builder = BuildQDQUpsampleTestCase<QType>(op_type, input_def, scales_def, attrs,
                                                           opset, use_contrib_qdq);
  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test that QDQ Upsample with a dynamic scales input is not supported by QNN EP.
TEST_F(QnnHTPBackendTests, Upsample_DynamicScales_Unsupported) {
  RunQDQUpsampleTestOnHTP<uint8_t>("Upsample",
                                   TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                   TestInputDef<float>({4}, false /* is_initializer */, {1.0f, 1.0f, 1.5f, 1.5f}),
                                   {utils::MakeAttribute("mode", "nearest")},  // Attributes
                                   ExpectedEPNodeAssignment::None,             // Should not be assigned to QNN EP.
                                   10);                                        // Opset
}

// Test QDQ Upsample with opset-9, mode `nearest`
TEST_F(QnnHTPBackendTests, Upsample_4D_Nearest_opset9) {
  RunQDQUpsampleTestOnHTP<uint8_t>("Upsample",
                                   TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                   TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                                   {utils::MakeAttribute("mode", "nearest")},  // Attributes
                                   ExpectedEPNodeAssignment::All,
                                   9);  // Opset
}

// Test QDQ Upsample with opset-9, mode `linear`
TEST_F(QnnHTPBackendTests, Upsample_4D_Linear_opset9) {
  RunQDQUpsampleTestOnHTP<uint8_t>("Upsample",
                                   TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                   TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                                   {utils::MakeAttribute("mode", "linear")},  // Attributes
                                   ExpectedEPNodeAssignment::All,
                                   9);  // Opset
}

// Test QDQ Upsample with opset-7, mode `nearest`
TEST_F(QnnHTPBackendTests, Upsample_4D_Nearest_opset7) {
  RunQDQUpsampleTestOnHTP<uint8_t>("Upsample",
                                   TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                   TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                                   {utils::MakeAttribute("mode", "nearest")},  // Attributes
                                   ExpectedEPNodeAssignment::All,
                                   7);  // Opset
}

// Test QDQ Upsample with opset-7, mode `linear`
TEST_F(QnnHTPBackendTests, Upsample_4D_Linear_opset7) {
  RunQDQUpsampleTestOnHTP<uint8_t>("Upsample",
                                   TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                   TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                                   {utils::MakeAttribute("mode", "linear")},  // Attributes
                                   ExpectedEPNodeAssignment::All,
                                   7);  // Opset
}

// Test QDQ Upsample 5D
TEST_F(QnnHTPBackendTests, Upsample_5D) {
  RunQDQUpsampleTestOnHTP<uint8_t>("Upsample",
                                   TestInputDef<float>({1, 3, 4, 4, 4}, false, -10.0f, 10.0f),
                                   TestInputDef<float>({5}, true, {1.0f, 1.0f, 1.5f, 1.5f, 1.5f}),
                                   {utils::MakeAttribute("mode", "nearest")},  // Attributes
                                   ExpectedEPNodeAssignment::All,
                                   9);  // Opset
}

// Test QDQ Upsample 6D not supported for HTP backend
TEST_F(QnnHTPBackendTests, Upsample_6D) {
  RunQDQUpsampleTestOnHTP<uint8_t>("Upsample",
                                   TestInputDef<float>({1, 3, 4, 4, 4, 4}, false, -10.0f, 10.0f),
                                   TestInputDef<float>({6}, true, {1.0f, 1.0f, 1.5f, 1.5f, 1.5f, 1.5f}),
                                   {utils::MakeAttribute("mode", "nearest")},  // Attributes
                                   ExpectedEPNodeAssignment::None,
                                   9);  // Opset
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

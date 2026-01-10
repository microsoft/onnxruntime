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
static void RunUpsampleTestOnCPU(const TestInputDef<DataType>& input_def,
                                 const TestInputDef<float>& scales_def,
                                 std::vector<ONNX_NAMESPACE::AttributeProto>&& attrs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 9) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  if (opset <= 7) {
    const std::vector<float>& scales = scales_def.GetRawData();
    attrs.push_back(utils::MakeAttribute("scales", scales));

    RunQnnModelTest(BuildOpTestCase<DataType>("Upsample", {input_def}, {}, attrs),
                    provider_options,
                    opset,
                    expected_ep_assignment);
  } else {
    RunQnnModelTest(BuildOpTestCase<DataType, float>("Upsample", {input_def}, {scales_def}, attrs),
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
  RunUpsampleTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, false /* is_initializer */, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "nearest")},  // Attributes
                       ExpectedEPNodeAssignment::None,             // Should not be assigned to QNN EP.
                       9);                                         // Opset
}

// Test Upsample with opset-9, mode `nearest`
TEST_F(QnnCPUBackendTests, Upsample_4D_Nearest_opset9) {
  RunUpsampleTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "nearest")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       9);  // Opset
}

// Test Upsample with opset-9, mode `linear`
TEST_F(QnnCPUBackendTests, Upsample_4D_Linear_opset9) {
  RunUpsampleTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "linear")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       9);  // Opset
}

// Test Upsample with opset-7, mode `nearest`
TEST_F(QnnCPUBackendTests, Upsample_4D_Nearest_opset7) {
  RunUpsampleTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "nearest")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       7);  // Opset
}

// Test Upsample with opset-7, mode `linear`
TEST_F(QnnCPUBackendTests, Upsample_4D_Linear_opset7) {
  RunUpsampleTestOnCPU(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({4}, true, {1.0f, 1.0f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "linear")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       7);  // Opset
}

// Test Upsample 5D
TEST_F(QnnCPUBackendTests, Upsample_5D) {
  RunUpsampleTestOnCPU(TestInputDef<float>({1, 3, 4, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({5}, true, {1.0f, 1.0f, 1.5f, 1.5f, 1.5f}),
                       {utils::MakeAttribute("mode", "nearest")},  // Attributes
                       ExpectedEPNodeAssignment::All,
                       9);  // Opset
}

/*
QNN HTP backend tests for the QDQ Upsample model is bypassed and can not be enabled.

ONNX Upsample is deprecated in domain version 10. However, ONNX QuantizeLinear and DequantizeLinear are enabled in
domain version 10. Their conditions are mutually exclusive, so it is not possible for these ops to coexist in the
same domain version.
*/

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

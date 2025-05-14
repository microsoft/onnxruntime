// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a non-QDQ model on HTP and compares output to CPU EP.
template <typename InputType1 = float, typename InputType2 = float>
static void RunOpTest(const std::string& op_type,
                      const TestInputDef<InputType1>& input_def_1,
                      const TestInputDef<InputType2>& input_defs_2,
                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                      int opset_version,
                      ExpectedEPNodeAssignment expected_ep_assignment,
                      const std::string& op_domain = kOnnxDomain,
                      float fp32_abs_err = 1e-3f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildOpTestCase<InputType1, InputType2>(op_type, {input_def_1}, {input_defs_2}, attrs, op_domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Non-QDQ model, CumSum with float input and axis input as initializer
TEST_F(QnnHTPBackendTests, CumSum_float_int32_e0_r0) {
  RunOpTest<float, int32_t>("CumSum",
                            TestInputDef<float>({3, 2}, false, {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f}),
                            TestInputDef<int32_t>({}, true, {0}),
                            {utils::MakeAttribute("exclusive", static_cast<int64_t>(0)),
                             utils::MakeAttribute("reverse", static_cast<int64_t>(0))},
                            17,
                            ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, CumSum_float_int32_e0_r1) {
  RunOpTest<float, int32_t>("CumSum",
                            TestInputDef<float>({3, 2}, false, {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f}),
                            TestInputDef<int32_t>({}, true, {0}),
                            {utils::MakeAttribute("exclusive", static_cast<int64_t>(0)),
                             utils::MakeAttribute("reverse", static_cast<int64_t>(1))},
                            17,
                            ExpectedEPNodeAssignment::All);
}
TEST_F(QnnHTPBackendTests, CumSum_float_int32_e1_r0) {
  RunOpTest<float, int32_t>("CumSum",
                            TestInputDef<float>({3, 2}, false, {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f}),
                            TestInputDef<int32_t>({}, true, {0}),
                            {utils::MakeAttribute("exclusive", static_cast<int64_t>(1)),
                             utils::MakeAttribute("reverse", static_cast<int64_t>(0))},
                            17,
                            ExpectedEPNodeAssignment::All);
}
TEST_F(QnnHTPBackendTests, CumSum_float_int32_e1_r1) {
  RunOpTest<float, int32_t>("CumSum",
                            TestInputDef<float>({3, 2}, false, {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f}),
                            TestInputDef<int32_t>({}, true, {0}),
                            {utils::MakeAttribute("exclusive", static_cast<int64_t>(1)),
                             utils::MakeAttribute("reverse", static_cast<int64_t>(1))},
                            17,
                            ExpectedEPNodeAssignment::All);
}

}  // namespace test
}  // namespace onnxruntime

#endif

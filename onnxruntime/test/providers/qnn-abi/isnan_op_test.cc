// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <string>

#include "test/providers/qnn-abi/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a IsNaN operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunIsNanTest(const std::vector<TestInputDef<DataType>>& input_defs,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         float fp32_abs_err = 1e-5,
                         const std::string& backend_name = "cpu",
                         int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTestABI(BuildOpTestCase<DataType>("IsNaN", input_defs, {}, {}, kOnnxDomain),
                     provider_options,
                     opset,
                     expected_ep_assignment,
                     fp32_abs_err);
}

//
// CPU tests:
//

TEST_F(QnnABICPUBackendTests, IsNaN_Scalar) {
  const std::vector<int64_t> input_shape{};  // scalar
  const std::vector<float> input_data{std::numeric_limits<float>::quiet_NaN()};

  RunIsNanTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      ExpectedEPNodeAssignment::All,
                      0.0f);
}

TEST_F(QnnABICPUBackendTests, IsNaN_Mix_2d) {
  const std::vector<int64_t> input_shape{2, 4};
  const std::vector<float> input_data{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f,
                                      3.0f, 4.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};

  RunIsNanTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      ExpectedEPNodeAssignment::All,
                      0.0f);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//
// Skip qdq test since IsNaN only support fp.
TEST_F(QnnABIHTPBackendTests, IsNaN_Scalar) {
  const std::vector<int64_t> input_shape{};
  const std::vector<float> input_data{std::numeric_limits<float>::quiet_NaN()};

  RunIsNanTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      ExpectedEPNodeAssignment::All,
                      0.0f,
                      "htp");
}

TEST_F(QnnABIHTPBackendTests, IsNaN_Mix_2d) {
  const std::vector<int64_t> input_shape{2, 4};
  const std::vector<float> input_data{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f,
                                      3.0f, 4.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};

  RunIsNanTest<float>({TestInputDef<float>(input_shape, false, input_data)},
                      ExpectedEPNodeAssignment::All,
                      0.0f,
                      "htp");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

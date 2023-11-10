// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that creates a graph with a single AveragePool operator.
static GetTestModelFn BuildAveragePoolTestCase(const std::vector<int64_t>& shape,
                                               const std::vector<int64_t>& kernel_shape,
                                               const std::vector<int64_t>& strides,
                                               const std::vector<int64_t>& pads,
                                               int64_t count_include_pad,
                                               const std::string& auto_pad = "NOTSET") {
  return [shape, kernel_shape, strides, pads,
          count_include_pad, auto_pad](ModelTestBuilder& builder) {
    // Random input data
    auto input = builder.MakeInput<float>(shape, 0.0f, 10.0f);

    auto* output = builder.MakeOutput();
    Node& pool_node = builder.AddNode("AveragePool", {input}, {output});

    pool_node.AddAttribute("kernel_shape", kernel_shape);

    if (!strides.empty()) {
      pool_node.AddAttribute("strides", strides);
    }

    pool_node.AddAttribute("auto_pad", auto_pad);

    if (!pads.empty() && auto_pad == "NOTSET") {
      pool_node.AddAttribute("pads", pads);
    }

    if (count_include_pad > 0) {
      pool_node.AddAttribute("count_include_pad", count_include_pad);
    }
  };
}

// Returns a function that creates a graph with a QDQ AveragePool operator.
template <typename QuantType>
GetQDQTestCaseFn BuildAveragePoolQDQTestCase(const std::vector<int64_t>& shape,
                                             const std::vector<int64_t>& kernel_shape,
                                             const std::vector<int64_t>& strides,
                                             const std::vector<int64_t>& pads,
                                             int64_t count_include_pad,
                                             const std::string& auto_pad = "NOTSET") {
  return [shape, kernel_shape, strides, pads,
          count_include_pad, auto_pad](ModelTestBuilder& builder) {
    float dq_scale = 0.0038f;
    float pool_output_scale = 0.0038f;
    float q_scale = 0.0038f;
    QuantType dq_zp = std::numeric_limits<QuantType>::max() / 2;
    QuantType pool_output_zp = std::numeric_limits<QuantType>::max() / 2;
    QuantType q_zp = std::numeric_limits<QuantType>::max() / 2;

    auto* input_arg = builder.MakeInput<float>(shape, -1.0f, 1.0f);
    auto* output_arg = builder.MakeOutput();

    // add QDQ + AveragePool
    auto* dq_output = AddQDQNodePair<QuantType>(builder, input_arg, dq_scale, dq_zp);
    auto* averagepool_output = builder.MakeIntermediate();
    Node& pool_node = builder.AddNode("AveragePool", {dq_output}, {averagepool_output});

    pool_node.AddAttribute("kernel_shape", kernel_shape);

    if (!strides.empty()) {
      pool_node.AddAttribute("strides", strides);
    }

    pool_node.AddAttribute("auto_pad", auto_pad);

    if (!pads.empty() && auto_pad == "NOTSET") {
      pool_node.AddAttribute("pads", pads);
    }

    if (count_include_pad > 0) {
      pool_node.AddAttribute("count_include_pad", count_include_pad);
    }

    // add QDQ output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(averagepool_output,
                                             pool_output_scale,
                                             pool_output_zp,
                                             q_output);
    builder.AddDequantizeLinearNode<QuantType>(q_output,
                                               q_scale,
                                               q_zp,
                                               output_arg);
  };
}

// Runs an AveragePool model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunAveragePoolOpTest(const std::vector<int64_t>& shape,
                                 const std::vector<int64_t>& kernel_shape,
                                 const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& pads,
                                 int64_t count_include_pad,
                                 const std::string& auto_pad,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildAveragePoolTestCase(shape, kernel_shape, strides, pads, count_include_pad, auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ AveragePool model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
template <typename QuantType>
static void RunQDQAveragePoolOpTest(const std::vector<int64_t>& shape,
                                    const std::vector<int64_t>& kernel_shape,
                                    const std::vector<int64_t>& strides,
                                    const std::vector<int64_t>& pads,
                                    int64_t count_include_pad,
                                    const std::string& auto_pad,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    int opset = 18, float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildAveragePoolQDQTestCase<QuantType>(shape, kernel_shape, strides, pads, count_include_pad,
                                                         auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//

// AveragePool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnCPUBackendTests, TestAveragePool_Global) {
  RunAveragePoolOpTest({1, 2, 3, 3},  // shape
                       {3, 3},        // kernel_shape
                       {3, 3},        // strides
                       {0, 0, 0, 0},  // pads
                       0,             // count_include_pad
                       "NOTSET",
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that counts padding.
TEST_F(QnnCPUBackendTests, TestAveragePool_CountIncludePad) {
  RunAveragePoolOpTest({1, 2, 3, 3},  // shape
                       {1, 1},        // kernel_shape
                       {1, 1},        // strides
                       {0, 0, 0, 0},  // pads
                       1,             // count_include_pad
                       "NOTSET",
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnCPUBackendTests, TestAveragePool_AutopadSameUpper) {
  RunAveragePoolOpTest({1, 2, 3, 3},  // shape
                       {1, 1},        // kernel_shape
                       {1, 1},        // strides
                       {},            // pads
                       1,             // count_include_pad
                       "SAME_UPPER",
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnCPUBackendTests, TestAveragePool_AutopadSameLower) {
  RunAveragePoolOpTest({1, 2, 3, 3},  // shape
                       {1, 1},        // kernel_shape
                       {1, 1},        // strides
                       {},            // pads
                       1,             // count_include_pad
                       "SAME_LOWER",
                       ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// QDQ AveragePool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnHTPBackendTests, TestAveragePool_Global_HTP_u8) {
  RunQDQAveragePoolOpTest<uint8_t>({1, 2, 3, 3},  // shape
                                   {3, 3},        // kernel_shape
                                   {3, 3},        // strides
                                   {0, 0, 0, 0},  // pads
                                   0,             // count_include_pad
                                   "NOTSET",
                                   ExpectedEPNodeAssignment::All);
}

// QDQ AveragePool that counts padding.
TEST_F(QnnHTPBackendTests, TestAveragePool_CountIncludePad_HTP_u8) {
  RunQDQAveragePoolOpTest<uint8_t>({1, 2, 3, 3},  // shape
                                   {1, 1},        // kernel_shape
                                   {1, 1},        // strides
                                   {0, 0, 0, 0},  // pads
                                   1,             // count_include_pad
                                   "NOTSET",
                                   ExpectedEPNodeAssignment::All,
                                   18, 0.00381f);
}

// QDQ AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnHTPBackendTests, TestAveragePool_AutopadSameUpper_HTP_u8) {
  RunQDQAveragePoolOpTest<uint8_t>({1, 2, 3, 3},  // shape
                                   {1, 1},        // kernel_shape
                                   {1, 1},        // strides
                                   {},            // pads
                                   0,             // count_include_pad
                                   "SAME_UPPER",
                                   ExpectedEPNodeAssignment::All,
                                   18, 0.00381f);
}

// QDQ AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnHTPBackendTests, TestAveragePool_AutopadSameLower_HTP_u8) {
  RunQDQAveragePoolOpTest<uint8_t>({1, 2, 3, 3},  // shape
                                   {1, 1},        // kernel_shape
                                   {1, 1},        // strides
                                   {},            // pads
                                   0,             // count_include_pad
                                   "SAME_LOWER",
                                   ExpectedEPNodeAssignment::All,
                                   18, 0.00381f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
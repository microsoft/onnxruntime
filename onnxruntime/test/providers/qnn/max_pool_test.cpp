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

// Returns a function that creates a graph with a single MaxPool operator.
static GetTestModelFn BuildMaxPoolTestCase(const std::vector<int64_t>& shape,
                                           const std::vector<int64_t>& kernel_shape,
                                           const std::vector<int64_t>& strides,
                                           const std::vector<int64_t>& pads,
                                           const std::vector<int64_t>& dilations,
                                           int64_t ceil_mode,
                                           int64_t storage_order,
                                           const std::string& auto_pad = "NOTSET") {
  return [shape, kernel_shape, strides, pads, dilations,
          ceil_mode, storage_order, auto_pad](ModelTestBuilder& builder) {
    // Random input data
    auto input = builder.MakeInput<float>(shape, 0.0f, 10.0f);

    auto* output = builder.MakeOutput();
    Node& pool_node = builder.AddNode("MaxPool", {input}, {output});

    pool_node.AddAttribute("kernel_shape", kernel_shape);

    if (!strides.empty()) {
      pool_node.AddAttribute("strides", strides);
    }

    if (!dilations.empty()) {
      pool_node.AddAttribute("dilations", dilations);
    }

    pool_node.AddAttribute("auto_pad", auto_pad);

    if (!pads.empty() && auto_pad == "NOTSET") {
      pool_node.AddAttribute("pads", pads);
    }

    if (ceil_mode > 0) {
      pool_node.AddAttribute("ceil_mode", ceil_mode);
    }

    if (storage_order > 0) {
      pool_node.AddAttribute("storage_order", storage_order);
    }
  };
}

// Returns a function that creates a graph with a QDQ MaxPool operator.
template <typename QuantType>
GetQDQTestCaseFn BuildMaxPoolQDQTestCase(const std::vector<int64_t>& shape,
                                         const std::vector<int64_t>& kernel_shape,
                                         const std::vector<int64_t>& strides,
                                         const std::vector<int64_t>& pads,
                                         const std::vector<int64_t>& dilations,
                                         int64_t ceil_mode,
                                         int64_t storage_order,
                                         const std::string& auto_pad = "NOTSET") {
  return [shape, kernel_shape, strides, pads, dilations,
          ceil_mode, storage_order, auto_pad](ModelTestBuilder& builder) {
    float dq_scale = 0.0038f;
    float pool_output_scale = 0.0038f;
    float q_scale = 0.0038f;
    QuantType dq_zp = std::numeric_limits<QuantType>::max() / 2;
    QuantType pool_output_zp = std::numeric_limits<QuantType>::max() / 2;
    QuantType q_zp = std::numeric_limits<QuantType>::max() / 2;

    auto* input_arg = builder.MakeInput<float>(shape, -1.0f, 1.0f);
    auto* output_arg = builder.MakeOutput();

    // add QDQ + MaxPool
    auto* dq_output = AddQDQNodePair<QuantType>(builder, input_arg, dq_scale, dq_zp);
    auto* MaxPool_output = builder.MakeIntermediate();
    Node& pool_node = builder.AddNode("MaxPool", {dq_output}, {MaxPool_output});

    pool_node.AddAttribute("kernel_shape", kernel_shape);

    if (!strides.empty()) {
      pool_node.AddAttribute("strides", strides);
    }

    if (!dilations.empty()) {
      pool_node.AddAttribute("dilations", dilations);
    }

    pool_node.AddAttribute("auto_pad", auto_pad);

    if (!pads.empty() && auto_pad == "NOTSET") {
      pool_node.AddAttribute("pads", pads);
    }

    if (ceil_mode > 0) {
      pool_node.AddAttribute("ceil_mode", ceil_mode);
    }

    if (storage_order > 0) {
      pool_node.AddAttribute("storage_order", storage_order);
    }

    // add QDQ output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(MaxPool_output,
                                             pool_output_scale,
                                             pool_output_zp,
                                             q_output);
    builder.AddDequantizeLinearNode<QuantType>(q_output,
                                               q_scale,
                                               q_zp,
                                               output_arg);
  };
}

// Runs an MaxPool model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunMaxPoolOpTest(const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& kernel_shape,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& pads,
                             const std::vector<int64_t>& dilations,
                             int64_t ceil_mode,
                             int64_t storage_order,
                             const std::string& auto_pad,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildMaxPoolTestCase(shape, kernel_shape, strides, pads, dilations, ceil_mode, storage_order, auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ MaxPool model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
template <typename QuantType>
static void RunQDQMaxPoolOpTest(const std::vector<int64_t>& shape,
                                const std::vector<int64_t>& kernel_shape,
                                const std::vector<int64_t>& strides,
                                const std::vector<int64_t>& pads,
                                const std::vector<int64_t>& dilations,
                                int64_t ceil_mode,
                                int64_t storage_order,
                                const std::string& auto_pad,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 18, float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildMaxPoolQDQTestCase<QuantType>(shape, kernel_shape, strides, pads, dilations, ceil_mode, storage_order, auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//

// MaxPool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnCPUBackendTests, TestMaxPool_Global) {
  RunMaxPoolOpTest({1, 2, 3, 3},  // shape
                   {3, 3},        // kernel_shape
                   {3, 3},        // strides
                   {0, 0, 0, 0},  // pads
                   {1, 1},        // dialations
                   0,             // ceil_mode
                   0,             // storage_order
                   "NOTSET",      // auto_pad
                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TestMaxPool_Large_Input) {
  RunMaxPoolOpTest({1, 125, 8, 56},  // shape
                   {2, 2},           // kernel_shape
                   {2, 2},           // strides
                   {0, 0, 0, 0},     // pads
                   {1, 1},           // dialations
                   0,                // ceil_mode
                   0,                // storage_order
                   "NOTSET",         // auto_pad
                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TestMaxPool_Large_Input2) {
  RunMaxPoolOpTest({1, 128, 16, 113},  // shape
                   {2, 2},             // kernel_shape
                   {2, 2},             // strides
                   {0, 0, 0, 0},       // pads
                   {1, 1},             // dialations
                   0,                  // ceil_mode
                   0,                  // storage_order
                   "NOTSET",           // auto_pad
                   ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnCPUBackendTests, DISABLED_TestMaxPool_Ceil) {
  RunMaxPoolOpTest({1, 2, 3, 3},  // shape
                   {3, 3},        // kernel_shape
                   {3, 3},        // strides
                   {0, 0, 0, 0},  // pads
                   {1, 1},        // dialations
                   1,             // ceil_mode
                   0,             // storage_order
                   "NOTSET",      // auto_pad
                   ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnCPUBackendTests, DISABLED_TestMaxPool_Large_Input2_Ceil) {
  RunMaxPoolOpTest({1, 128, 16, 113},  // shape
                   {2, 2},             // kernel_shape
                   {2, 2},             // strides
                   {0, 0, 0, 0},       // pads
                   {1, 1},             // dialations
                   1,                  // ceil_mode
                   0,                  // storage_order
                   "NOTSET",           // auto_pad
                   ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//
// QDQ MaxPool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnHTPBackendTests, TestMaxPool_Global_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>({1, 2, 3, 3},  // shape
                               {3, 3},        // kernel_shape
                               {3, 3},        // strides
                               {0, 0, 0, 0},  // pads
                               {1, 1},        // dialations
                               0,             // ceil_mode
                               0,             // storage_order
                               "NOTSET",      // auto_pad
                               ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnHTPBackendTests, DISABLED_TestMaxPool_Large_Input_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>({1, 125, 8, 56},  // shape
                               {2, 2},           // kernel_shape
                               {2, 2},           // strides
                               {0, 0, 0, 0},     // pads
                               {1, 1},           // dialations
                               0,                // ceil_mode
                               0,                // storage_order
                               "NOTSET",         // auto_pad
                               ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnHTPBackendTests, DISABLED_TestMaxPool_Large_Input2_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>({1, 128, 16, 113},  // shape
                               {2, 2},             // kernel_shape
                               {2, 2},             // strides
                               {0, 0, 0, 0},       // pads
                               {1, 1},             // dialations
                               0,                  // ceil_mode
                               0,                  // storage_order
                               "NOTSET",           // auto_pad
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, TestMaxPool_Ceil_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>({1, 2, 3, 3},  // shape
                               {3, 3},        // kernel_shape
                               {3, 3},        // strides
                               {0, 0, 0, 0},  // pads
                               {1, 1},        // dialations
                               1,             // ceil_mode
                               0,             // storage_order
                               "NOTSET",      // auto_pad
                               ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnHTPBackendTests, DISABLED_TestMaxPool_Large_Input2_Ceil_HTP_u8) {
  RunQDQMaxPoolOpTest<uint8_t>({1, 128, 16, 113},  // shape
                               {2, 2},             // kernel_shape
                               {2, 2},             // strides
                               {0, 0, 0, 0},       // pads
                               {1, 1},             // dialations
                               1,                  // ceil_mode
                               0,                  // storage_order
                               "NOTSET",           // auto_pad
                               ExpectedEPNodeAssignment::All);
}

// TODO: Certain large input sizes cause the QNN graph to fail to finalize with error 1002 (QNN_COMMON_ERROR_MEM_ALLOC).
TEST_F(QnnHTPBackendTests, DISABLED_TestMaxPool_LargeInput_1Pads) {
  RunQDQMaxPoolOpTest<uint8_t>({1, 64, 384, 576},  // shape
                               {3, 3},             // kernel_shape
                               {2, 2},             // strides
                               {1, 1, 1, 1},       // pads
                               {1, 1},             // dialations
                               0,                  // ceil_mode
                               0,                  // storage_order
                               "NOTSET",           // auto_pad
                               ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
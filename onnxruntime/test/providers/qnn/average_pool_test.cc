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
static GetTestModelFn BuildAveragePoolTestCase(const TestInputDef<float>& input_def,
                                               const std::vector<int64_t>& kernel_shape,
                                               const std::vector<int64_t>& strides,
                                               const std::vector<int64_t>& pads,
                                               int64_t count_include_pad,
                                               const std::string& auto_pad = "NOTSET") {
  return [input_def, kernel_shape, strides, pads,
          count_include_pad, auto_pad](ModelTestBuilder& builder) {
    auto* input = MakeTestInput(builder, input_def);

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
GetTestQDQModelFn<QuantType> BuildAveragePoolQDQTestCase(const TestInputDef<float>& input_def,
                                                         const std::vector<int64_t>& kernel_shape,
                                                         const std::vector<int64_t>& strides,
                                                         const std::vector<int64_t>& pads,
                                                         int64_t count_include_pad,
                                                         const std::string& auto_pad = "NOTSET") {
  return [input_def, kernel_shape, strides, pads,
          count_include_pad, auto_pad](ModelTestBuilder& builder,
                                       std::vector<QuantParams<QuantType>>& output_qparams) {
    auto* input_arg = MakeTestInput(builder, input_def);

    // add QDQ + AveragePool
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams(input_def);
    auto* dq_output = AddQDQNodePair<QuantType>(builder, input_arg, input_qparams.scale, input_qparams.zero_point);
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

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, averagepool_output,
                                                     output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// Runs an AveragePool model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunAveragePoolOpTest(const TestInputDef<float>& input_def,
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

  RunQnnModelTest(BuildAveragePoolTestCase(input_def, kernel_shape, strides, pads, count_include_pad, auto_pad),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ AveragePool model on the QNN HTP backend. Checks the graph node assignment, and that accuracy
// on QNN EP is at least as good as on CPU EP.
template <typename QuantType>
static void RunQDQAveragePoolOpTest(const TestInputDef<float>& input_def,
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

  TestQDQModelAccuracy(BuildAveragePoolTestCase(input_def, kernel_shape, strides, pads, count_include_pad, auto_pad),
                       BuildAveragePoolQDQTestCase<QuantType>(input_def, kernel_shape, strides, pads, count_include_pad,
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
TEST_F(QnnCPUBackendTests, AveragePool_Global) {
  RunAveragePoolOpTest(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),  // random input
                       {3, 3},                                                   // kernel_shape
                       {3, 3},                                                   // strides
                       {0, 0, 0, 0},                                             // pads
                       0,                                                        // count_include_pad
                       "NOTSET",
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that counts padding.
TEST_F(QnnCPUBackendTests, AveragePool_CountIncludePad) {
  RunAveragePoolOpTest(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),  // random input
                       {1, 1},                                                   // kernel_shape
                       {1, 1},                                                   // strides
                       {0, 0, 0, 0},                                             // pads
                       1,                                                        // count_include_pad
                       "NOTSET",
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnCPUBackendTests, AveragePool_AutopadSameUpper) {
  RunAveragePoolOpTest(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),  // random input
                       {1, 1},                                                   // kernel_shape
                       {1, 1},                                                   // strides
                       {},                                                       // pads
                       1,                                                        // count_include_pad
                       "SAME_UPPER",
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnCPUBackendTests, AveragePool_AutopadSameLower) {
  RunAveragePoolOpTest(TestInputDef<float>({1, 2, 3, 3}, false, -10.0f, 10.0f),  // random input
                       {1, 1},                                                   // kernel_shape
                       {1, 1},                                                   // strides
                       {},                                                       // pads
                       1,                                                        // count_include_pad
                       "SAME_LOWER",
                       ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// QDQ AveragePool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnHTPBackendTests, AveragePool_Global_HTP) {
  std::vector<float> input = {32.1289f, -59.981f, -17.2799f, 62.7263f, 33.6205f, -19.3515f, -54.0113f, 37.5648f, 61.5357f,
                              -52.5769f, 27.3637f, -9.01382f, -65.5612f, 19.9497f, -47.9228f, 26.9813f, 83.064f, 0.362503f};
  RunQDQAveragePoolOpTest<uint8_t>(TestInputDef<float>({1, 2, 3, 3}, false, input),
                                   {3, 3},        // kernel_shape
                                   {3, 3},        // strides
                                   {0, 0, 0, 0},  // pads
                                   0,             // count_include_pad
                                   "NOTSET",
                                   ExpectedEPNodeAssignment::All);
}

// QDQ AveragePool that counts padding.
TEST_F(QnnHTPBackendTests, AveragePool_CountIncludePad_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>(TestInputDef<float>({1, 2, 3, 3}, false, input),
                                   {1, 1},        // kernel_shape
                                   {1, 1},        // strides
                                   {0, 0, 0, 0},  // pads
                                   1,             // count_include_pad
                                   "NOTSET",
                                   ExpectedEPNodeAssignment::All,
                                   18);
}

// QDQ AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnHTPBackendTests, AveragePool_AutopadSameUpper_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>(TestInputDef<float>({1, 2, 3, 3}, false, input),
                                   {1, 1},  // kernel_shape
                                   {1, 1},  // strides
                                   {},      // pads
                                   0,       // count_include_pad
                                   "SAME_UPPER",
                                   ExpectedEPNodeAssignment::All,
                                   18);
}

// QDQ AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnHTPBackendTests, AveragePool_AutopadSameLower_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>(TestInputDef<float>({1, 2, 3, 3}, false, input),
                                   {1, 1},  // kernel_shape
                                   {1, 1},  // strides
                                   {},      // pads
                                   0,       // count_include_pad
                                   "SAME_LOWER",
                                   ExpectedEPNodeAssignment::All,
                                   18);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
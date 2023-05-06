// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Creates a graph with a single Conv operator. Used for testing CPU backend.
static GetTestModelFn BuildConvTestCase(const std::vector<int64_t>& input_shape,
                                        const std::vector<int64_t>& weights_shape,
                                        bool is_bias_initializer) {
  return [input_shape, weights_shape, is_bias_initializer](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(input_shape, 0.0f, 10.0f);
    auto* output = builder.MakeOutput();
    auto* weights = builder.MakeInitializer<float>(weights_shape, 0.0f, 1.0f);

    onnxruntime::NodeArg* bias = nullptr;

    if (is_bias_initializer) {
      bias = builder.MakeInitializer<float>({weights_shape[0]}, -1.0f, 1.0f);
    } else {
      bias = builder.MakeInput<float>({weights_shape[0]}, -1.0f, 1.0f);
    }

    builder.AddNode("Conv", {input, weights, bias}, {output});
  };
}

// Runs a Conv model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunCPUConvOpTest(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& weights_shape,
                             bool is_bias_initializer,
                             ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                             int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildConvTestCase(input_shape, weights_shape, is_bias_initializer),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description);
}

// Creates a graph with a single Q/DQ Conv operator. Used for testing HTP backend.
template <typename InputType, typename WeightType, typename BiasType, typename OutputType>
GetTestModelFn BuildQDQConvTestCase(const std::vector<int64_t>& input_shape,
                                    const std::vector<int64_t>& weights_shape,
                                    bool is_bias_initializer = true) {
  return [input_shape, weights_shape, is_bias_initializer](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    using InputLimits = std::numeric_limits<InputType>;
    using WeightLimits = std::numeric_limits<WeightType>;
    using OutputLimits = std::numeric_limits<OutputType>;

    InputType input_min_value = InputLimits::min();
    InputType input_max_value = InputLimits::max();

    WeightType weight_min_value = WeightLimits::min();
    WeightType weight_max_value = WeightLimits::max();

    auto* dq_w_output = builder.MakeIntermediate();
    auto* weight = builder.MakeInitializer<WeightType>(weights_shape, weight_min_value, weight_max_value);
    builder.AddDequantizeLinearNode<WeightType>(weight, .03f,
                                                (weight_min_value + weight_max_value) / 2 + 1,
                                                dq_w_output);

    auto* dq_bias_output = builder.MakeIntermediate();
    onnxruntime::NodeArg* bias = nullptr;

    if (is_bias_initializer) {
      bias = builder.MakeInitializer<BiasType>({weights_shape[0]}, static_cast<BiasType>(0),
                                               static_cast<BiasType>(127));
    } else {
      bias = builder.MakeInput<BiasType>({weights_shape[0]}, static_cast<BiasType>(0),
                                         static_cast<BiasType>(127));
    }

    builder.AddDequantizeLinearNode<BiasType>(bias, .0012f,
                                              0,
                                              dq_bias_output);

    auto* conv_output = builder.MakeIntermediate();
    auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, .04f,
                                                (input_min_value + input_max_value) / 2 + 1);
    builder.AddNode("Conv", {dq_output, dq_w_output, dq_bias_output}, {conv_output});

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(conv_output, .039f,
                                              (OutputLimits::min() + OutputLimits::max()) / 2 + 1,
                                              q_output);

    builder.AddDequantizeLinearNode<OutputType>(q_output, .039f,
                                                (OutputLimits::min() + OutputLimits::max()) / 2 + 1,
                                                output_arg);
  };
}

// Runs a Conv model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename InputType, typename WeightType, typename BiasType, typename OutputType>
static void RunHTPConvOpTest(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& weights_shape,
                             bool is_bias_initializer,
                             ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                             int opset = 13) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildQDQConvTestCase<InputType, WeightType, BiasType, OutputType>(input_shape, weights_shape,
                                                                                    is_bias_initializer),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an input.
//
// TODO: Enable this test when QNN CPU backend (QNN sdk 2.10.0) fixes bug that causes graph finalization to
// throw a segfault when the bias is a non-static input.
TEST_F(QnnCPUBackendTests, DISABLED_TestCPUConvf32_bias_input) {
  RunCPUConvOpTest({1, 1, 3, 3}, {2, 1, 2, 2}, false, ExpectedEPNodeAssignment::All, "TestCPUConvf32_bias_input");
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnCPUBackendTests, TestCPUConvf32_bias_initializer) {
  RunCPUConvOpTest({1, 1, 3, 3}, {2, 1, 2, 2}, true, ExpectedEPNodeAssignment::All, "TestCPUConvf32_bias_initializer");
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an input.
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8S32_bias_input) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>({1, 1, 5, 5}, {1, 1, 3, 3}, false, ExpectedEPNodeAssignment::All,
                                                       "TestQDQConvU8U8S32_bias_input");
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8S32_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t, int32_t, uint8_t>({1, 1, 5, 5}, {1, 1, 3, 3}, true, ExpectedEPNodeAssignment::All,
                                                       "TestQDQConvU8U8S32_bias_initializer");
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
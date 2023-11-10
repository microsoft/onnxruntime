// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

static void RunLayerNormCpuTest(const std::vector<int64_t>& shape) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  auto BuildLayerNormTestCase = [](const std::vector<int64_t>& shape) -> GetTestModelFn {
    return [shape](ModelTestBuilder& builder) {
      // Random input data
      auto input = builder.MakeInput<float>(shape, 0.0f, 10.0f);
      auto scale = builder.MakeInput<float>(shape, 0.0f, 10.0f);

      auto* output = builder.MakeOutput();
      Node& layer_norm_node = builder.AddNode("LayerNormalization", {input, scale}, {output});

      layer_norm_node.AddAttribute("axis", static_cast<int64_t>(0));
    };
  };

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildLayerNormTestCase(shape),
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::All,
                  expected_nodes_in_partition);
}

TEST_F(QnnCPUBackendTests, TestLayerNorm) {
  RunLayerNormCpuTest({2, 3});
}

TEST_F(QnnCPUBackendTests, TestLayerNorm1D) {
  RunLayerNormCpuTest({1, 2, 3});
}

TEST_F(QnnCPUBackendTests, TestLayerNorm2D) {
  RunLayerNormCpuTest({1, 2, 3, 3});
}

TEST_F(QnnCPUBackendTests, TestLayerNorm3D) {
  RunLayerNormCpuTest({1, 2, 3, 3, 4});
}

template <typename InputQType, typename ScaleQType>
GetQDQTestCaseFn BuildQDQLayerNormTestCase(const std::vector<int64_t>& input_shape,
                                           const std::vector<int64_t>& scale_shape,
                                           int64_t axis_value = 0) {
  return [input_shape, scale_shape, axis_value](ModelTestBuilder& builder) {
    const InputQType quant_zero_point = 0;
    // const float quant_scale = 1.0f;

    auto* input = builder.MakeInput<InputQType>(input_shape, std::numeric_limits<InputQType>::min(),
                                                std::numeric_limits<InputQType>::max());
    auto* dq_input = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputQType>(input, 0.0039f, quant_zero_point, dq_input);

    auto* dq_scale_output = builder.MakeIntermediate();
    auto* scale = builder.MakeInitializer<ScaleQType>(scale_shape, static_cast<ScaleQType>(1), static_cast<ScaleQType>(127));
    builder.AddDequantizeLinearNode<ScaleQType>(scale, 0.0028f, quant_zero_point, dq_scale_output);

    auto* layernorm_output = builder.MakeIntermediate();
    Node& layer_norm_node = builder.AddNode("LayerNormalization", {dq_input, dq_scale_output}, {layernorm_output});
    layer_norm_node.AddAttribute("axis", axis_value);

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InputQType>(layernorm_output, 0.00377f, quant_zero_point, q_output);

    auto* final_output = builder.MakeOutput();
    builder.AddDequantizeLinearNode<InputQType>(q_output, 0.00377f,
                                                quant_zero_point,
                                                final_output);
  };
}

/**
 * Runs an LayerNormalization model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_shape The input's shape.
 * \param scale_shape The scale's shape.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param num_modes_in_graph The number of expected nodes in the graph.
 * \param axis_value The axis value.
 */
static void RunLayerNormQDQTest(const std::vector<int64_t>& input_shape,
                                const std::vector<int64_t>& scale_shape,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int64_t axis_value = 0) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildQDQLayerNormTestCase<uint8_t, uint8_t>(input_shape, scale_shape, axis_value),
                  provider_options,
                  11,
                  expected_ep_assignment);
}

// Check that QNN compiles DQ -> LayerNormalization -> Q as a single unit.
// Use an input of rank 3.
// Failed QNN op validation: QnnDsp <E> Param[0] has incorrect Value 3
TEST_F(QnnHTPBackendTests, DISABLED_TestQDQLayerNorm1DAxis0) {
  RunLayerNormQDQTest({1, 2, 3}, {1, 2, 3}, ExpectedEPNodeAssignment::All);
}

// Failed QNN FinalizeGraphs: QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
TEST_F(QnnHTPBackendTests, DISABLED_TestQDQLayerNorm1DAxis2) {
  RunLayerNormQDQTest({1, 2, 3}, {3}, ExpectedEPNodeAssignment::All, -1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
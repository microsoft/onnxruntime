// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "graph_transform_test_builder.h"

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/session/inference_session.h"

#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {

using GetQDQTestCaseFn = std::function<void(ModelTestBuilder& builder)>;

template <typename T>
typename std::enable_if<IsTypeQuantLinearCompatible<T>::value, NodeArg*>::type
AddQDQNodePair(ModelTestBuilder& builder, NodeArg* q_input, float scale, T zp = T()) {
  auto* q_output = builder.MakeIntermediate();
  auto* dq_output = builder.MakeIntermediate();
  builder.AddQuantizeLinearNode<T>(q_input, scale, zp, q_output);
  builder.AddDequantizeLinearNode<T>(q_output, scale, zp, dq_output);
  return dq_output;
}

template <typename T>
typename std::enable_if<IsTypeQuantLinearCompatible<T>::value, NodeArg*>::type
AddQDQNodePairWithOutputAsGraphOutput(ModelTestBuilder& builder, NodeArg* q_input, float scale, T zp = T()) {
  auto* q_output = builder.MakeIntermediate();
  auto* dq_output = builder.MakeOutput();
  builder.AddQuantizeLinearNode<T>(q_input, scale, zp, q_output);
  builder.AddDequantizeLinearNode<T>(q_output, scale, zp, dq_output);
  return dq_output;
}

template <typename InputType, typename WeightType, typename BiasType, typename OutputType>
GetQDQTestCaseFn BuildQDQConvTestCase(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
  return [input_shape, weights_shape](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    using InputLimits = std::numeric_limits<InputType>;
    using WeightLimits = std::numeric_limits<WeightType>;
    using OutputLimits = std::numeric_limits<OutputType>;

    InputType input_min_value = InputLimits::min();
    InputType input_max_value = InputLimits::max();

    WeightType weight_min_value = WeightLimits::min();
    WeightType weight_max_value = WeightLimits::max();

    // the reason that we reduce weight range by half for int8 weight type comes from the case when
    // running on cpu, MLAS kernel will overflow for uint8 activation and int8 weight with avx2 and avx512 extension
    // reduced weight range can prevent the overflow.
    if constexpr (std::is_same<WeightType, int8_t>::value) {
      weight_min_value /= 2;
      weight_max_value /= 2;
    }

    auto* dq_w_output = builder.MakeIntermediate();
    auto* weight = builder.MakeInitializer<WeightType>(weights_shape, weight_min_value, weight_max_value);
    builder.AddDequantizeLinearNode<WeightType>(weight, .03f,
                                                (weight_min_value + weight_max_value) / 2 + 1,
                                                dq_w_output);

    auto* dq_bias_output = builder.MakeIntermediate();
    auto* bias = builder.MakeInitializer<BiasType>({weights_shape[0]}, static_cast<BiasType>(0), static_cast<BiasType>(127));
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

template <typename InputType, typename OutputType>
GetQDQTestCaseFn BuildQDQAveragePoolTestCase(const std::vector<int64_t>& input_shape) {
  return [input_shape](ModelTestBuilder& builder) {

#ifdef USE_NNAPI  // NNAPI require consistent scales/ZPs for DQ -> Pool -> Q
    float dq_scale = 0.0038f;
    float pool_output_scale = 0.0038f;
    float q_scale = 0.0038f;
    InputType dq_zp = std::numeric_limits<OutputType>::max() / 2;
    InputType pool_output_zp = std::numeric_limits<OutputType>::max() / 2;
    InputType q_zp = std::numeric_limits<OutputType>::max() / 2;
#else
    float dq_scale = 0.0035f;
    float pool_output_scale = 0.0038f;
    float q_scale = 0.0039f;
    InputType dq_zp = 7;
    InputType pool_output_zp = std::numeric_limits<OutputType>::max() / 2;
    InputType q_zp = std::numeric_limits<OutputType>::max() / 2;
#endif

    auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();
    // add QDQ + AveragePool
    auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, dq_scale, dq_zp);
    auto* averagepool_output = builder.MakeIntermediate();
    Node& pool_node = builder.AddNode("AveragePool", {dq_output}, {averagepool_output});
    std::vector<int64_t> pads((input_shape.size() - 2) * 2, 1);
    pool_node.AddAttribute("pads", pads);
    std::vector<int64_t> kernel_shape(input_shape.size() - 2, 3);
    pool_node.AddAttribute("kernel_shape", kernel_shape);

    // add QDQ output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(averagepool_output,
                                              pool_output_scale,
                                              pool_output_zp,
                                              q_output);
    builder.AddDequantizeLinearNode<OutputType>(q_output,
                                                q_scale,
                                                q_zp,
                                                output_arg);
  };
}

template <typename InputType, typename OutputType>
GetQDQTestCaseFn BuildQDQGlobalAveragePoolTestCase(const std::vector<int64_t>& input_shape) {
  return [input_shape](ModelTestBuilder& builder) {
    float dq_scale = 0.0035f;
    float pool_output_scale = 0.0038f;
    float q_scale = 0.0039f;
    InputType dq_zp = 7;
    InputType pool_output_zp = std::numeric_limits<OutputType>::max() / 2;
    InputType q_zp = std::numeric_limits<OutputType>::max() / 2;

    auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();
    // add QDQ + GlobalAveragePool
    auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, dq_scale, dq_zp);
    auto* globalaveragepool_output = builder.MakeIntermediate();
    builder.AddNode("GlobalAveragePool", {dq_output}, {globalaveragepool_output});

    // add QDQ output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(globalaveragepool_output,
                                              pool_output_scale,
                                              pool_output_zp,
                                              q_output);
    builder.AddDequantizeLinearNode<OutputType>(q_output,
                                                q_scale,
                                                q_zp,
                                                output_arg);
  };
}

GetQDQTestCaseFn BuildQDQResizeTestCase(const std::vector<int64_t>& input_shape,
                                        const std::vector<int64_t>& sizes_data,
                                        const std::string& mode = "nearest",
                                        const std::string& coordinate_transformation_mode = "half_pixel");

template <typename Input1Type, typename Input2Type, typename OutputType>
GetQDQTestCaseFn BuildBinaryOpTestCase(const std::vector<int64_t>& input_shape,
                                       const std::string& op_type) {
  return [input_shape, op_type](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* input2_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

#ifdef USE_NNAPI  // NNAPI require consistent scales for DQ -> bin_op_input and bin_op_output-> Q
    float q_scale = 0.008f;
    float op_input_scale = 0.008f;
    float op_output_scale = 0.0076f;
    float dq_scale = 0.0076f;
#else
    float q_scale = 0.008f;
    float op_input_scale = 0.0079f;
    float op_output_scale = 0.0076f;
    float dq_scale = 0.0078f;
#endif

    // add QDQ 1
    auto* q1_output = builder.MakeIntermediate();
    auto* dq1_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<Input1Type>(input1_arg,
                                              q_scale,
                                              std::numeric_limits<Input1Type>::max() / 2,
                                              q1_output);
    builder.AddDequantizeLinearNode<Input1Type>(q1_output,
                                                op_input_scale,
                                                std::numeric_limits<Input1Type>::max() / 2,
                                                dq1_output);

    // add QDQ 2
    auto* q2_output = builder.MakeIntermediate();
    auto* dq2_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<Input2Type>(input2_arg,
                                              q_scale,
                                              std::numeric_limits<Input2Type>::max() / 2,
                                              q2_output);
    builder.AddDequantizeLinearNode<Input2Type>(q2_output,
                                                op_input_scale,
                                                std::numeric_limits<Input2Type>::max() / 2,
                                                dq2_output);

    // add binary operator
    auto* binary_op_output = builder.MakeIntermediate();
    builder.AddNode(op_type, {dq1_output, dq2_output}, {binary_op_output});

    // add QDQ output
    auto* q3_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(binary_op_output,
                                              op_output_scale,
                                              std::numeric_limits<OutputType>::max() / 2,
                                              q3_output);
    builder.AddDequantizeLinearNode<OutputType>(q3_output,
                                                dq_scale,
                                                std::numeric_limits<OutputType>::max() / 2,
                                                output_arg);
  };
}

template <typename InputType, typename OutputType>
GetQDQTestCaseFn BuildQDQTransposeTestCase(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& perms) {
  return [input_shape, perms](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<InputType>(input_shape,
                                                   std::numeric_limits<InputType>::min(),
                                                   std::numeric_limits<InputType>::max());
    auto* output_arg = builder.MakeOutput();

    InputType dq_zp = std::numeric_limits<InputType>::max() / 2;
    OutputType q_zp = std::numeric_limits<OutputType>::max() / 2;

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputType>(input_arg, .003f, dq_zp, dq_output);

    // add Transpose
    auto* transpose_output = builder.MakeIntermediate();
    Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
    transpose_node.AddAttribute("perm", perms);

    // add Q
    builder.AddQuantizeLinearNode<OutputType>(transpose_output, .003f, q_zp, output_arg);
  };
}

template <typename InputType, typename OutputType>
GetQDQTestCaseFn BuildQDQSoftMaxTestCase(const std::vector<int64_t>& input_shape, const int64_t& axis,
                                         float output_scales, OutputType output_zero_point) {
  return [input_shape, axis, output_scales, output_zero_point](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<InputType>(input_shape,
                                                   std::numeric_limits<InputType>::min(),
                                                   std::numeric_limits<InputType>::max());

    auto* output_arg = builder.MakeOutput();

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputType>(input_arg, .003f, std::numeric_limits<InputType>::max() / 2, dq_output);

    // add SoftMax
    auto* softmax_output = builder.MakeIntermediate();
    Node& softmax_node = builder.AddNode("Softmax", {dq_output}, {softmax_output});

    softmax_node.AddAttribute("axis", axis);

    // add Q
    builder.AddQuantizeLinearNode<OutputType>(softmax_output, output_scales, output_zero_point, output_arg);
  };
}

GetQDQTestCaseFn BuildQDQReshapeTestCase(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& reshape_shape);

GetQDQTestCaseFn BuildQDQConcatTestCase(const std::vector<std::vector<int64_t>>& input_shapes,
                                        int64_t axis,
                                        bool has_input_float = false,
                                        bool has_input_int8 = false,
                                        bool has_output_int8 = false);

GetQDQTestCaseFn BuildQDQConcatTestCaseUnsupportedInputScaleZp();

}  // namespace test
}  // namespace onnxruntime

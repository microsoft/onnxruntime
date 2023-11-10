// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>

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
GetQDQTestCaseFn BuildQDQConvTransposeTestCase(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
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
    builder.AddNode("ConvTranspose", {dq_output, dq_w_output, dq_bias_output}, {conv_output});

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(conv_output, .039f,
                                              (OutputLimits::min() + OutputLimits::max()) / 2 + 1,
                                              q_output);

    builder.AddDequantizeLinearNode<OutputType>(q_output, .039f,
                                                (OutputLimits::min() + OutputLimits::max()) / 2 + 1,
                                                output_arg);
  };
}

// Creates the following graph:
//                                _______________________
//    input (f32) -> Q -> DQ ->  |                       | -> Q -> DQ -> output (f32)
// axes (int32, initializer) ->  |         Gather        |
//                               |_______________________|
//
template <typename QuantType, typename IndicesType>
GetQDQTestCaseFn BuildQDQGatherOpTestCase(const std::vector<int64_t>& input_shape,
                                          const std::vector<IndicesType> indices,
                                          const std::vector<int64_t>& indices_shape,
                                          int64_t axis) {
  return [input_shape, indices, indices_shape, axis](ModelTestBuilder& builder) {
    auto* input_data = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
    auto* final_output = builder.MakeOutput();

    // input_data -> Q/DQ ->
    auto* input_qdq_output = AddQDQNodePair<QuantType>(builder, input_data, .003f, 1);

    auto* indices_input = builder.MakeInitializer<IndicesType>(indices_shape, indices);

    auto* gather_output = builder.MakeIntermediate();
    Node& gather_node = builder.AddNode("Gather", {input_qdq_output, indices_input}, {gather_output});
    gather_node.AddAttribute("axis", axis);

    // -> Q/DQ -> final_output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(gather_output, .003f, 1,
                                             q_output);

    builder.AddDequantizeLinearNode<QuantType>(q_output, .003f, 1,
                                               final_output);
  };
}

// Creates the following graph:
//                                _______________________
//    input (f32) -> Q -> DQ ->  |                       | -> Q -> DQ -> output (f32)
// axes (int32, initializer) ->  |         Gather        |
//                               |_______________________|
//
template <typename QuantType, typename IndicesType>
GetQDQTestCaseFn BuildQDQGatherOpScalarIndicesTestCase(const std::vector<int64_t>& input_shape,
                                                       const IndicesType indices,
                                                       int64_t axis) {
  return [input_shape, indices, axis](ModelTestBuilder& builder) {
    auto* input_data = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
    auto* final_output = builder.MakeOutput();

    // input_data -> Q/DQ ->
    auto* input_qdq_output = AddQDQNodePair<QuantType>(builder, input_data, .003f, 1);

    auto* indices_input = builder.MakeScalarInitializer<IndicesType>(indices);

    auto* gather_output = builder.MakeIntermediate();
    Node& gather_node = builder.AddNode("Gather", {input_qdq_output, indices_input}, {gather_output});
    gather_node.AddAttribute("axis", axis);

    // -> Q/DQ -> final_output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(gather_output, .003f, 1,
                                             q_output);

    builder.AddDequantizeLinearNode<QuantType>(q_output, .003f, 1,
                                               final_output);
  };
}

// Creates the following graph:
//                                _______________________
//                               |                       |
//    input (f32) -> Q -> DQ ->  |       LeakyRelu       | -> Q -> DQ -> output (f32)
//                               |_______________________|
//
template <typename QuantType>
GetQDQTestCaseFn BuildQDQLeakyReluOpTestCase(const std::vector<int64_t>& input_shape) {
  return [input_shape](ModelTestBuilder& builder) {
    auto* input_data = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
    auto* final_output = builder.MakeOutput();

    // input_data -> Q/DQ ->
    auto* input_qdq_output = AddQDQNodePair<QuantType>(builder, input_data, 0.0473f, 137);

    auto* leakyrelu_output = builder.MakeIntermediate();
    Node& leakyrelu_node = builder.AddNode("LeakyRelu", {input_qdq_output}, {leakyrelu_output});
    leakyrelu_node.AddAttribute("alpha", 0.2f);

    // -> Q/DQ -> final_output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(leakyrelu_output, 0.02696f, 48,
                                             q_output);

    builder.AddDequantizeLinearNode<QuantType>(q_output, 0.02696f, 48,
                                               final_output);
  };
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
GetQDQTestCaseFn BuildQDQAveragePoolTestCase(const std::vector<int64_t>& input_shape,
                                             int64_t count_include_pad = 0) {
  return [input_shape, count_include_pad](ModelTestBuilder& builder) {

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
    if (count_include_pad > 0) {
      pool_node.AddAttribute("count_include_pad", count_include_pad);
    }

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
GetQDQTestCaseFn BuildQDQMaxPoolTestCase(const std::vector<int64_t>& input_shape, bool q_eq_dq = false) {
  return [input_shape, q_eq_dq](ModelTestBuilder& builder) {
    float dq_scale = 0.0035f;
    float pool_output_scale = q_eq_dq ? dq_scale : 0.0038f;
    float q_scale = 0.0039f;
    InputType dq_zp = 7;
    InputType pool_output_zp = q_eq_dq ? dq_zp : std::numeric_limits<OutputType>::max() / 2;
    InputType q_zp = std::numeric_limits<OutputType>::max() / 2;

    auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();
    // add QDQ + MaxPool
    auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, dq_scale, dq_zp);
    auto* maxpool_output = builder.MakeIntermediate();
    Node& pool_node = builder.AddNode("MaxPool", {dq_output}, {maxpool_output});
    std::vector<int64_t> pads((input_shape.size() - 2) * 2, 1);
    pool_node.AddAttribute("pads", pads);
    std::vector<int64_t> kernel_shape(input_shape.size() - 2, 3);
    pool_node.AddAttribute("kernel_shape", kernel_shape);
    pool_node.AddAttribute("ceil_mode", static_cast<int64_t>(0));

    // add QDQ output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(maxpool_output,
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

template <typename InputType = uint8_t>
GetQDQTestCaseFn BuildQDQResizeTestCase(const std::vector<int64_t>& input_shape,
                                        const std::vector<int64_t>& sizes_data,
                                        const std::string& mode = "nearest",
                                        const std::string& coordinate_transformation_mode = "half_pixel",
                                        const std::string& nearest_mode = "round_prefer_floor",
                                        bool add_dq_output_float = false) {
  static_assert(std::is_same_v<InputType, int8_t> || std::is_same_v<InputType, uint8_t>);
  return [input_shape, sizes_data, mode, coordinate_transformation_mode,
          nearest_mode, add_dq_output_float](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<InputType>(input_shape,
                                                    std::numeric_limits<InputType>::min(),
                                                    std::numeric_limits<InputType>::max());
    auto* roi = builder.MakeInitializer<float>({0}, {});
    auto* scales = builder.MakeInitializer<float>({0}, {});
    auto* sizes = builder.Make1DInitializer<int64_t>(sizes_data);
    NodeArg* output_arg = 0;

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputType>(input1_arg, .003f, 1, dq_output);

    // add Resize
    auto* resize_output = builder.MakeIntermediate();
    Node& resize_node = builder.AddNode("Resize", {dq_output, roi, scales, sizes}, {resize_output});

    resize_node.AddAttribute("mode", mode);
    resize_node.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);

    if (mode == "nearest") {
      resize_node.AddAttribute("nearest_mode", nearest_mode);
    }

    if (add_dq_output_float) {
      // add Q
      output_arg = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<InputType>(resize_output, .003f, 1, output_arg);
      auto* f_dq_output = builder.MakeOutput();
      builder.AddDequantizeLinearNode<InputType>(output_arg, .003f, 1, f_dq_output);
    } else {
      output_arg = builder.MakeOutput();
      // add Q
      builder.AddQuantizeLinearNode<InputType>(resize_output, .003f, 1, output_arg);
    }
  };
}

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
GetQDQTestCaseFn BuildConsolidationTestCase(
    const std::vector<int64_t>& input_shape,
    const int64_t& axis) {
  return [input_shape, axis](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    InputType dq_zp = std::numeric_limits<InputType>::max() / 2;
    OutputType q_zp = std::numeric_limits<OutputType>::max() / 2;
    auto* upper_dq_output = builder.MakeIntermediate();
    auto* upper_q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InputType>(input_arg, .003f, q_zp, upper_q_output);
    builder.AddDequantizeLinearNode<InputType>(upper_q_output, .003f, dq_zp, upper_dq_output);

    // add Split

    auto* split_output_1 = builder.MakeIntermediate();
    auto* split_output_2 = builder.MakeIntermediate();
    auto* split_output_3 = builder.MakeIntermediate();
    Node& split_node = builder.AddNode("Split", {upper_dq_output}, {split_output_1, split_output_2, split_output_3});
    split_node.AddAttribute("axis", axis);
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      split_node.AddAttribute("num_outputs", static_cast<int64_t>(3));
    }

    // add Q
    auto* lower_q_output_1 = builder.MakeIntermediate();
    auto* lower_q_output_2 = builder.MakeIntermediate();
    auto* lower_q_output_3 = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(split_output_1, .003f, q_zp, lower_q_output_1);
    builder.AddQuantizeLinearNode<OutputType>(split_output_2, .003f, q_zp, lower_q_output_2);
    builder.AddQuantizeLinearNode<OutputType>(split_output_3, .003f, q_zp, lower_q_output_3);
    auto* q_split_output_1 = builder.MakeOutput();
    auto* q_split_output_2 = builder.MakeOutput();
    auto* q_split_output_3 = builder.MakeOutput();
    builder.AddDequantizeLinearNode<OutputType>(lower_q_output_1, .003f, dq_zp, q_split_output_1);
    builder.AddDequantizeLinearNode<OutputType>(lower_q_output_2, .003f, dq_zp, q_split_output_2);
    builder.AddDequantizeLinearNode<OutputType>(lower_q_output_3, .003f, dq_zp, q_split_output_3);
  };
}

template <typename Type1, typename Type2, typename Type3, typename Type4>
GetQDQTestCaseFn BuildDoubleQDQTestCases(Type1 zp_1, Type2 zp_2, Type3 zp_3, Type4 zp_4,
                                         float scale_1, float scale_2, float scale_3, float scale_4) {
  return [=](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(
        {11, 22, 33, 44},
        std::numeric_limits<Type1>::min() * (scale_1 + scale_3) / 2,
        std::numeric_limits<Type1>::max() * (scale_1 + scale_3) / 2);
    NodeArg* q1_output = builder.MakeIntermediate();
    NodeArg* dq1_output = builder.MakeIntermediate();
    NodeArg* q2_output = builder.MakeIntermediate();
    NodeArg* dq2_output = builder.MakeOutput();
    builder.AddQuantizeLinearNode<Type1>(input_arg, scale_1, zp_1, q1_output);
    builder.AddDequantizeLinearNode<Type2>(q1_output, scale_2, zp_2, dq1_output);
    builder.AddQuantizeLinearNode<Type3>(dq1_output, scale_3, zp_3, q2_output);
    builder.AddDequantizeLinearNode<Type4>(q2_output, scale_4, zp_4, dq2_output);
  };
}

template <typename T>
GetQDQTestCaseFn BuildDoubleQDQWithoutLastOutput(int output_index) {
  return [=](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({2, 3, 4}, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    T zp = (std::numeric_limits<T>::max() - std::numeric_limits<T>::min()) / 2;
    float scale = 0.003f;
    std::vector<NodeArg*> outputs(4);
    for (auto i = 0; i < 4; i++) {
      if (output_index == i) {
        outputs[i] = builder.MakeOutput();
      } else {
        outputs[i] = builder.MakeIntermediate();
      }
    }
    builder.AddQuantizeLinearNode<T>(input_arg, scale, zp, outputs[0]);
    builder.AddDequantizeLinearNode<T>(outputs[0], scale, zp, outputs[1]);
    builder.AddQuantizeLinearNode<T>(outputs[1], scale, zp, outputs[2]);
    builder.AddDequantizeLinearNode<T>(outputs[2], scale, zp, outputs[3]);
  };
}

template <typename InputType, typename OutputType>
GetQDQTestCaseFn BuildQDQSplitTestCase(
    const std::vector<int64_t>& input_shape,
    const int64_t& axis) {
  return [input_shape, axis](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<InputType>(input_shape,
                                                   std::numeric_limits<InputType>::min(),
                                                   std::numeric_limits<InputType>::max());

    InputType dq_zp = std::numeric_limits<InputType>::max() / 2;
    OutputType q_zp = std::numeric_limits<OutputType>::max() / 2;
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputType>(input_arg, .003f, dq_zp, dq_output);

    // add Split

    auto* split_output_1 = builder.MakeIntermediate();
    auto* split_output_2 = builder.MakeIntermediate();
    auto* split_output_3 = builder.MakeIntermediate();
    Node& split_node = builder.AddNode("Split", {dq_output}, {split_output_1, split_output_2, split_output_3});
    split_node.AddAttribute("axis", axis);
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      split_node.AddAttribute("num_outputs", static_cast<int64_t>(3));
    }

    // add Q
    auto* q_split_output_1 = builder.MakeOutput();
    auto* q_split_output_2 = builder.MakeOutput();
    auto* q_split_output_3 = builder.MakeOutput();
    builder.AddQuantizeLinearNode<OutputType>(split_output_1, .003f, q_zp, q_split_output_1);  // Model input (node_token_1)
    builder.AddQuantizeLinearNode<OutputType>(split_output_2, .003f, q_zp, q_split_output_2);  // Model input (node_token_2)
    builder.AddQuantizeLinearNode<OutputType>(split_output_3, .003f, q_zp, q_split_output_3);
  };
}
template <typename InputType>
GetQDQTestCaseFn BuildQDQWhereTestCase(
    const std::vector<int64_t>& cond_shape,
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape) {
  return [cond_shape, x_shape, y_shape](ModelTestBuilder& builder) {
    auto* input_cond_arg = builder.MakeInputBool(cond_shape);
    auto* input_x_arg = builder.MakeInput<InputType>(x_shape,
                                                     std::numeric_limits<InputType>::min(),
                                                     std::numeric_limits<InputType>::max());
    auto* input_y_arg = builder.MakeInput<InputType>(y_shape,
                                                     std::numeric_limits<InputType>::min(),
                                                     std::numeric_limits<InputType>::max());
    InputType zp = std::numeric_limits<InputType>::max() / 2;
    constexpr float scale = 0.003f;
    auto* dq_x_output = builder.MakeIntermediate();
    auto* dq_y_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InputType>(input_x_arg, scale, zp, dq_x_output);
    builder.AddDequantizeLinearNode<InputType>(input_y_arg, scale, zp, dq_y_output);
    // add Where

    auto* where_output = builder.MakeIntermediate();
    builder.AddNode("Where", {input_cond_arg, dq_x_output, dq_y_output}, {where_output});

    // add Q
    auto* q_where_output = builder.MakeOutput();
    builder.AddQuantizeLinearNode<InputType>(where_output, scale, zp, q_where_output);  // Model input (node_token_1)
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

GetQDQTestCaseFn BuildQDQMatMulTestCase(const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape);

template <typename Input1Type, typename Input2Type, typename OutputType, typename BiasType = int32_t>
GetQDQTestCaseFn BuildQDQGemmTestCase(const std::vector<int64_t>& input1_shape,
                                      const std::vector<int64_t>& input2_shape,
                                      bool has_bias,
                                      const int64_t& transB) {
  return [input1_shape, input2_shape, has_bias, transB](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    typedef std::numeric_limits<Input1Type> Input1Limits;
    typedef std::numeric_limits<Input2Type> Input2Limits;
    typedef std::numeric_limits<OutputType> OutputTypeLimits;

    std::vector<NodeArg*> input_args;

    // add QDQ A
    auto* q1_output = builder.MakeIntermediate();
    auto* dq1_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<Input1Type>(input1_arg,
                                              .039f,
                                              (Input1Limits::max() + Input1Limits::min()) / 2 + 1,
                                              q1_output);
    builder.AddDequantizeLinearNode<Input1Type>(q1_output,
                                                .039f,
                                                (Input1Limits::max() + Input1Limits::min()) / 2 + 1,
                                                dq1_output);

    input_args.push_back(dq1_output);

    // add QDQ B
    auto* dq2_output = builder.MakeIntermediate();
    auto* input_b = builder.MakeInitializer<Input2Type>(input2_shape, Input2Limits::min(), Input2Limits::max());
    builder.AddDequantizeLinearNode<Input2Type>(input_b, 0.04f,
                                                (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                dq2_output);
    input_args.push_back(dq2_output);

    if (has_bias) {
      auto* dq_bias_output = builder.MakeIntermediate();
      auto* bias = builder.MakeInitializer<BiasType>({input2_shape[0]}, static_cast<BiasType>(0), static_cast<BiasType>(127));
      builder.AddDequantizeLinearNode<BiasType>(bias, 0.00156f,
                                                0,
                                                dq_bias_output);
      input_args.push_back(dq_bias_output);
    }

    Node* gemm_node = nullptr;

    auto* gemm_op_output = builder.MakeIntermediate();
    gemm_node = &builder.AddNode("Gemm", input_args, {gemm_op_output});

    // add QDQ output
    auto* q3_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<OutputType>(gemm_op_output,
                                              .039f,
                                              (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                              q3_output);
    builder.AddDequantizeLinearNode<OutputType>(q3_output,
                                                .039f,
                                                (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                output_arg);

    gemm_node->AddAttribute("transB", transB);
  };
}

std::vector<std::string> GetNodeOpTypesInTopologicalOrder(const Graph& graph);

}  // namespace test
}  // namespace onnxruntime

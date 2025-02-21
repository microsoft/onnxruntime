// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "onnx/defs/data_type_utils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "cmath"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class AttentionOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
};

std::vector<int64_t> generate_indices(int64_t batch_size, int64_t kv_num_heads, int64_t sequence_length) {
  std::vector<int64_t> indices;
  for (int64_t i = 0; i < sequence_length; ++i) {
    for (int64_t j = 0; j < batch_size * kv_num_heads; ++j) {
      indices.push_back(j / kv_num_heads);
      indices.push_back(j % kv_num_heads);
    }
  }
  return indices;
}

std::vector<int64_t> repeat_sequence(int64_t sequence_length, int64_t kv_num_heads, int64_t batch_size) {
  std::vector<int64_t> repeated;
  for (int64_t i = 0; i < sequence_length; ++i) {
    for (int64_t j = 0; j < batch_size * kv_num_heads; ++j) {
      repeated.push_back(i);
    }
  }
  return repeated;
}

/** GroupQueryAttention SubGraph.
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size, P is past_sequence_length
               N is number of attention heads, H is head size, and W=N*H, h=Sqrt(H)
               B and S could be symbolic. ? means it is optional.
    GQA inputs: query, key value, past_key, past_value, seqlens_k, total_sequence_length
    Notes: If the datatype of the inputs (qkv and past kv) is float16, we cast them to float32 to ensure data precision.

          query             key       value
            |                |          |
         Reshape          Reshape     Reshape (B,S,H,N)
            |      past_key  /          |
            |         |     /           |
          q_Transpose |    /            |
           (0,2,1,3)  |   /             |                    seqlens_k
             \        |  /              |                     /     |
              \       | /               |                    /      |
present_key<---\----ScatterND <---------|-----(scatter_indices*)    |
               |      |                 |        /                  |
               |  k_Transpose           |       /                   |
               \  (0,1,3,2)             |      /                    |
                \     |                 |     /                     |
                 \  Expand(G)           |    /                      |
                  \   |     past_value  |   /                       |
                qk_MatMul          \    |  /                        |
                      |            ScatterND-----> present_value    |
                  qk_Div              /                             |
                      |              /                              |
                     Add <----------/--------(attention_bias, one/finfo_min mask*)
                      |            /
                    Softmax     Expand(G)
                       \         /
                        \       /
                      qkv_MatMul
                             |
                          Transpose (perm=0,2,1,3)
                             |
                          Reshape---(shape=B,S,W)
                             |
                           output
*/

Status AttentionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  if (op_type != "GroupQueryAttention") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported normalization op: ", op_type);
  }

  const auto& input_defs = node.InputDefs();

  emscripten::val query_input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val key_input = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val value_input = model_builder.GetOperand(input_defs[2]->Name());
  emscripten::val past_key_input = model_builder.GetOperand(input_defs[3]->Name());
  emscripten::val past_value_input = model_builder.GetOperand(input_defs[4]->Name());
  emscripten::val seqlens_k_input = model_builder.GetOperand(input_defs[5]->Name());

  std::vector<int64_t> input_q_shape, input_k_shape, input_v_shape, input_past_k_shape, input_past_v_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_q_shape, logger), "Cannot get query shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_k_shape, logger), "Cannot get key shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[2], input_v_shape, logger), "Cannot get value shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[3], input_past_k_shape, logger), "Cannot get past_key shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[4], input_past_v_shape, logger), "Cannot get past_value shape");
  const auto q_rank = input_q_shape.size();
  const auto k_rank = input_k_shape.size();
  const auto v_rank = input_v_shape.size();
  const auto past_k_rank = input_past_k_shape.size();
  const auto past_v_rank = input_past_v_shape.size();
  ORT_RETURN_IF_NOT(q_rank == 3 && k_rank == 3 && v_rank == 3,
                    "The qkv shape should be batch_size * sequence_length * hidden_size.");
  ORT_RETURN_IF_NOT(past_k_rank == 4 && past_v_rank == 4, "The qkv shape should be BNSH.");

  NodeAttrHelper helper(node);
  uint32_t kv_num_heads = helper.Get("kv_num_heads", 32);
  uint32_t num_heads = helper.Get("num_heads", 32);

  uint32_t qkv_batch_size = SafeInt<uint32_t>(input_q_shape[0]);
  uint32_t qkv_sequence_length = SafeInt<uint32_t>(input_q_shape[1]);
  uint32_t qkv_hidden_size = SafeInt<uint32_t>(input_q_shape[2]);
  uint32_t head_size = SafeInt<uint32_t>(qkv_hidden_size / num_heads);
  uint32_t past_sequence_length = SafeInt<uint32_t>(input_past_k_shape[2]);
  uint32_t group_size = SafeInt<uint32_t>(num_heads / kv_num_heads);

  std::vector<uint32_t> reshape_output_shape = {qkv_batch_size, qkv_sequence_length, qkv_hidden_size};
  std::vector<uint32_t> scatter_indices_shape = {qkv_batch_size, qkv_sequence_length, kv_num_heads, 3};
  std::vector<uint32_t> reshape_tensor_shape = {qkv_batch_size, qkv_sequence_length, num_heads, head_size};
  std::vector<uint32_t> group_broadcast_tensor_shape_1 = {qkv_batch_size, kv_num_heads, 1, past_sequence_length,
                                                          head_size};
  std::vector<uint32_t> group_broadcast_tensor_shape_2 = {qkv_batch_size, kv_num_heads, group_size,
                                                          past_sequence_length, head_size};
  std::vector<uint32_t> group_broadcast_tensor_shape_3 = {qkv_batch_size, num_heads, past_sequence_length, head_size};

  emscripten::val common_options = emscripten::val::object();
  if (input_defs[0]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/GQA/preprocess/cast/query_input");
    query_input = model_builder.GetBuilder().call<emscripten::val>("cast", query_input, emscripten::val("float32"),
                                                                   common_options);
  }
  if (input_defs[1]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/GQA/preprocess/cast/key_input");
    key_input =
        model_builder.GetBuilder().call<emscripten::val>("cast", key_input, emscripten::val("float32"), common_options);
  }
  if (input_defs[2]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/GQA/preprocess/cast/value_input");
    value_input = model_builder.GetBuilder().call<emscripten::val>("cast", value_input, emscripten::val("float32"),
                                                                   common_options);
  }
  if (input_defs[3]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/GQA/preprocess/cast/past_key_input");
    past_key_input = model_builder.GetBuilder().call<emscripten::val>("cast", past_key_input,
                                                                      emscripten::val("float32"), common_options);
  }
  if (input_defs[4]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/GQA/preprocess/cast/past_value_input");
    past_value_input = model_builder.GetBuilder().call<emscripten::val>("cast", past_value_input,
                                                                        emscripten::val("float32"), common_options);
  }

  auto left = generate_indices(qkv_batch_size, kv_num_heads, qkv_sequence_length);
  auto right = repeat_sequence(qkv_sequence_length, kv_num_heads, qkv_batch_size);

  emscripten::val desc_left = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_left, ONNX_NAMESPACE::TensorProto_DataType_INT64), "Unsupported data type");
  emscripten::val dims_left =
      emscripten::val::array(std::vector<uint32_t>({qkv_batch_size * qkv_sequence_length * kv_num_heads, 2}));
  desc_left.set("dimensions", dims_left);
  desc_left.set("shape", dims_left);
  emscripten::val left_buffer = emscripten::val::global("BigInt64Array").new_(emscripten::val::array(left));
  emscripten::val left_constant = model_builder.GetBuilder().call<emscripten::val>("constant", desc_left, left_buffer);

  emscripten::val desc_right = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_right, ONNX_NAMESPACE::TensorProto_DataType_INT64), "Unsupported data type");
  emscripten::val dims_right =
      emscripten::val::array(std::vector<uint32_t>({qkv_batch_size * qkv_sequence_length * kv_num_heads, 1}));
  desc_right.set("dimensions", dims_right);
  desc_right.set("shape", dims_right);
  emscripten::val right_buffer = emscripten::val::global("BigInt64Array").new_(emscripten::val::array(right));
  emscripten::val right_constant =
      model_builder.GetBuilder().call<emscripten::val>("constant", desc_right, right_buffer);

  // query_input -> reshape(B,S,N,H) -> transpose(B,N,S,H) -> new_query
  common_options.set("label", node.Name() + "/GQA/query/reshape");
  emscripten::val reshaped_query = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", query_input, emscripten::val::array(reshape_tensor_shape), common_options);

  emscripten::val options = emscripten::val::object();
  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  options.set("label", node.Name() + "/GQA/query/transpose");
  emscripten::val new_query = model_builder.GetBuilder().call<emscripten::val>("transpose", reshaped_query, options);

  std::vector<uint32_t> reshape_kv_shape = {qkv_batch_size, qkv_sequence_length, kv_num_heads, head_size};
  common_options.set("label", node.Name() + "/GQA/key/reshape_1");
  emscripten::val key_for_scatter = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", key_input, emscripten::val::array(reshape_kv_shape), common_options);

  common_options.set("label", node.Name() + "/GQA/value/reshape_1");
  emscripten::val value_for_scatter = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", value_input, emscripten::val::array(reshape_kv_shape), common_options);

  common_options.set("label", node.Name() + "seqlens_k_casted");
  emscripten::val seqlens_k_casted = model_builder.GetBuilder().call<emscripten::val>(
      "cast", seqlens_k_input, emscripten::val("int64"), common_options);

  // The prefill and decode stages require different index construction for ScatterND operations.
  // Similar to other EPs like CPU and DirectML, when qkv_sequence_length > 1, the key and value are scattered to the
  // beginning of kv cache.
  std::vector<uint8_t> first_condition({(qkv_sequence_length > 1)});
  emscripten::val desc = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc, ONNX_NAMESPACE::TensorProto_DataType_UINT8), "Unsupported data type");
  emscripten::val dims_condition = emscripten::val::array(std::vector<uint32_t>({1}));
  desc.set("dimensions", dims_condition);
  desc.set("shape", dims_condition);
  emscripten::val first_condition_buffer =
      emscripten::val::global("Uint8Array").new_(emscripten::val::array(first_condition));
  emscripten::val condition_constant =
      model_builder.GetBuilder().call<emscripten::val>("constant", desc, first_condition_buffer);

  std::vector<int64_t> value_zero({0});
  emscripten::val desc_value_zero = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_value_zero, ONNX_NAMESPACE::TensorProto_DataType_INT64),
                    "Unsupported data type");
  emscripten::val dims_zero = emscripten::val::array(std::vector<uint32_t>({1}));
  desc_value_zero.set("dimensions", dims_zero);
  desc_value_zero.set("shape", dims_zero);
  emscripten::val value_zero_buffer = emscripten::val::global("BigInt64Array").new_(emscripten::val::array(value_zero));
  emscripten::val value_zero_constant =
      model_builder.GetBuilder().call<emscripten::val>("constant", desc_value_zero, value_zero_buffer);

  common_options.set("label", node.Name() + "/GQA/scatter/where");
  emscripten::val scatter_pos = model_builder.GetBuilder().call<emscripten::val>(
      "where", condition_constant, value_zero_constant, seqlens_k_casted, common_options);

  common_options.set("label", node.Name() + "/GQA/reshaped_qkv_sequence_length_shape/add");
  emscripten::val reshaped_qkv_sequence_length_shape_plus =
      model_builder.GetBuilder().call<emscripten::val>("add", right_constant, scatter_pos, common_options);

  common_options.set("label", node.Name() + "/GQA/scatter");
  std::vector<emscripten::val> inputs({left_constant, reshaped_qkv_sequence_length_shape_plus});
  uint32_t axis = 1;
  emscripten::val pre_scatter_indices =
      model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(inputs), axis, common_options);

  common_options.set("label", node.Name() + "/GQA/pre_scatter_indices/reshape");
  emscripten::val scatter_indices = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", pre_scatter_indices, emscripten::val::array(scatter_indices_shape), common_options);

  emscripten::val scatter_indices_casted =
      model_builder.GetBuilder().call<emscripten::val>("cast", scatter_indices, emscripten::val("int32"));

  common_options.set("label", node.Name() + "/GQA/present_key/ScatterND");
  emscripten::val present_key = model_builder.GetBuilder().call<emscripten::val>(
      "scatterND", past_key_input, scatter_indices_casted, key_for_scatter, common_options);

  common_options.set("label", node.Name() + "/GQA/present_value/ScatterND");
  emscripten::val present_value = model_builder.GetBuilder().call<emscripten::val>(
      "scatterND", past_value_input, scatter_indices_casted, value_for_scatter, common_options);

  emscripten::val true_present_key;
  emscripten::val true_present_value;
  if (group_size != 1) {
    // present_key/value(B,kv_N,P,H) -> reshape(B,kv_N,1,P,H) -> expand(B,kv_N,N/kv_N,P,H) -> reshape(B,N,P,H) ->
    // true_present_key/value
    common_options.set("label", node.Name() + "/GQA/true_present_key/reshape_1");
    true_present_key = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", present_key, emscripten::val::array(group_broadcast_tensor_shape_1), common_options);
    common_options.set("label", node.Name() + "/GQA/true_present_key/expand");
    true_present_key = model_builder.GetBuilder().call<emscripten::val>(
        "expand", true_present_key, emscripten::val::array(group_broadcast_tensor_shape_2), common_options);
    common_options.set("label", node.Name() + "/GQA/true_present_key/reshape_2");
    true_present_key = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", true_present_key, emscripten::val::array(group_broadcast_tensor_shape_3), common_options);

    common_options.set("label", node.Name() + "/GQA/true_present_value/reshape_1");
    true_present_value = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", present_value, emscripten::val::array(group_broadcast_tensor_shape_1), common_options);
    common_options.set("label", node.Name() + "/GQA/true_present_value/expand");
    true_present_value = model_builder.GetBuilder().call<emscripten::val>(
        "expand", true_present_value, emscripten::val::array(group_broadcast_tensor_shape_2), common_options);
    common_options.set("label", node.Name() + "/GQA/true_present_value/reshape_2");
    true_present_value = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", true_present_value, emscripten::val::array(group_broadcast_tensor_shape_3), common_options);
  } else {  // no need for broadcast
    true_present_key = present_key;
    true_present_value = present_value;
  }

  // true_present_key(B,N,P,H) -> transpose(B,N,H,P)
  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 1, 3, 2})));
  options.set("label", node.Name() + "/GQA/present_key/transpose");
  true_present_key = model_builder.GetBuilder().call<emscripten::val>("transpose", true_present_key, options);

  common_options.set("label", node.Name() + "/GQA/qkv/matmul_1");
  emscripten::val matmul_output =
      model_builder.GetBuilder().call<emscripten::val>("matmul", new_query, true_present_key, common_options);

  std::vector<float> scale({static_cast<float>(sqrt(head_size))});
  emscripten::val desc_scale = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_scale, ONNX_NAMESPACE::TensorProto_DataType_FLOAT), "Unsupported data type");
  emscripten::val dims_scale = emscripten::val::array(std::vector<uint32_t>({1}));
  desc_scale.set("dimensions", dims_scale);
  desc_scale.set("shape", dims_scale);
  emscripten::val scale_buffer = emscripten::val::global("Float32Array").new_(emscripten::val::array(scale));
  emscripten::val scale_constant =
      model_builder.GetBuilder().call<emscripten::val>("constant", desc_scale, scale_buffer);

  common_options.set("label", node.Name() + "/GQA/qkv/div");
  emscripten::val div_output =
      model_builder.GetBuilder().call<emscripten::val>("div", matmul_output, scale_constant, common_options);

  // static_cast<int64_t>(qkv_batch_size), static_cast<int64_t>(num_heads), static_cast<int64_t>(qkv_sequence_length),
  // static_cast<int64_t>(past_sequence_length)
  std::vector<int64_t> mask_shape_ones_shape(qkv_batch_size * num_heads * qkv_sequence_length * past_sequence_length,
                                             1);
  emscripten::val desc_mask_shape_ones_shape = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_mask_shape_ones_shape, ONNX_NAMESPACE::TensorProto_DataType_INT64),
                    "Unsupported data type");
  emscripten::val dims_mask_shape = emscripten::val::array(
      std::vector<uint32_t>({qkv_batch_size, num_heads, qkv_sequence_length, past_sequence_length}));
  desc_mask_shape_ones_shape.set("dimensions", dims_mask_shape);
  desc_mask_shape_ones_shape.set("shape", dims_mask_shape);
  emscripten::val mask_shape_ones_shape_buffer =
      emscripten::val::global("BigInt64Array").new_(emscripten::val::array(mask_shape_ones_shape));
  emscripten::val mask_shape_ones_shape_constant = model_builder.GetBuilder().call<emscripten::val>(
      "constant", desc_mask_shape_ones_shape, mask_shape_ones_shape_buffer);

  options.set("label", node.Name() + "range_of_mask_shape");
  options.set("exclusive", true);
  options.set("reversed", false);
  emscripten::val neq_left = model_builder.GetBuilder().call<emscripten::val>(
      "cumulativeSum", mask_shape_ones_shape_constant, gsl::narrow<uint32_t>(3), options);

  std::vector<uint32_t> reshape_pre_neq_right = {past_sequence_length, qkv_sequence_length};
  std::vector<int64_t> pre_neq_right_data_range(qkv_sequence_length);
  std::iota(pre_neq_right_data_range.begin(), pre_neq_right_data_range.end(), 1);

  emscripten::val desc_pre_neq_right_data_range = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_pre_neq_right_data_range, ONNX_NAMESPACE::TensorProto_DataType_INT64),
                    "Unsupported data type");
  emscripten::val dims_pre_neq_right_data_range = emscripten::val::array(std::vector<uint32_t>({qkv_sequence_length}));
  desc_pre_neq_right_data_range.set("dimensions", dims_pre_neq_right_data_range);
  desc_pre_neq_right_data_range.set("shape", dims_pre_neq_right_data_range);
  emscripten::val pre_neq_right_data_range_buffer =
      emscripten::val::global("BigInt64Array").new_(emscripten::val::array(pre_neq_right_data_range));
  emscripten::val pre_neq_right_data_range_constant = model_builder.GetBuilder().call<emscripten::val>(
      "constant", desc_pre_neq_right_data_range, pre_neq_right_data_range_buffer);

  common_options.set("label", node.Name() + "/GQA/attn_mask/add");
  emscripten::val pre_neq_right = model_builder.GetBuilder().call<emscripten::val>(
      "add", pre_neq_right_data_range_constant, scatter_pos, common_options);

  common_options.set("label", node.Name() + "/GQA/expand_neq_right");
  emscripten::val expanded_neq_right = model_builder.GetBuilder().call<emscripten::val>(
      "expand", pre_neq_right, emscripten::val::array(reshape_pre_neq_right), common_options);

  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({1, 0})));
  options.set("label", node.Name() + "/GQA/neq_right/transpose");
  emscripten::val neq_right =
      model_builder.GetBuilder().call<emscripten::val>("transpose", expanded_neq_right, options);

  common_options.set("label", node.Name() + "/GQA/attn_mask/condition");
  emscripten::val condition =
      model_builder.GetBuilder().call<emscripten::val>("lesser", neq_left, neq_right, common_options);

  std::vector<float_t> value_one({1});
  emscripten::val desc_value_one = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_value_one, ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
                    "Unsupported data type");
  emscripten::val dims_one = emscripten::val::array(std::vector<uint32_t>({1}));
  desc_value_one.set("dimensions", dims_one);
  desc_value_one.set("shape", dims_one);
  emscripten::val value_one_buffer = emscripten::val::global("Float32Array").new_(emscripten::val::array(value_one));
  emscripten::val value_one_constant =
      model_builder.GetBuilder().call<emscripten::val>("constant", desc_value_one, value_one_buffer);

  std::vector<float_t> finfo_min({-3.4028234663852886e+38});
  emscripten::val desc_finfo_min = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_finfo_min, ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
                    "Unsupported data type");
  emscripten::val dims_finfo_min = emscripten::val::array(std::vector<uint32_t>({1}));
  desc_finfo_min.set("dimensions", dims_finfo_min);
  desc_finfo_min.set("shape", dims_finfo_min);
  emscripten::val finfo_min_buffer = emscripten::val::global("Float32Array").new_(emscripten::val::array(finfo_min));
  emscripten::val finfo_min_constant =
      model_builder.GetBuilder().call<emscripten::val>("constant", desc_finfo_min, finfo_min_buffer);

  common_options.set("label", node.Name() + "/GQA/attn_mask/where");
  emscripten::val attn_mask = model_builder.GetBuilder().call<emscripten::val>("where", condition, value_one_constant,
                                                                               finfo_min_constant, common_options);

  common_options.set("label", node.Name() + "/GQA/attn_mask/softmax_input");
  emscripten::val softmax_input =
      model_builder.GetBuilder().call<emscripten::val>("add", div_output, attn_mask, common_options);

  common_options.set("label", node.Name() + "/GQA/attn_mask/softmax_input");
  int32_t softmax_axis = 3;
  emscripten::val softmax_output =
      model_builder.GetBuilder().call<emscripten::val>("softmax", softmax_input, softmax_axis, common_options);

  common_options.set("label", node.Name() + "/GQA/qkv/matmul_2");
  emscripten::val attn_output =
      model_builder.GetBuilder().call<emscripten::val>("matmul", softmax_output, true_present_value, common_options);

  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  options.set("label", node.Name() + "/GQA/qkv/transpose");
  emscripten::val transposed_attn_output =
      model_builder.GetBuilder().call<emscripten::val>("transpose", attn_output, options);

  common_options.set("label", node.Name() + "/GQA/qkv/reshape");
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", transposed_attn_output, emscripten::val::array(reshape_output_shape), common_options);

  if (node.OutputDefs()[0]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/GQA/postprocess/cast/output");
    output =
        model_builder.GetBuilder().call<emscripten::val>("cast", output, emscripten::val("float16"), common_options);
  }
  if (node.OutputDefs()[1]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/GQA/postprocess/cast/present_key");
    present_key = model_builder.GetBuilder().call<emscripten::val>("cast", present_key, emscripten::val("float16"),
                                                                   common_options);
  }
  if (node.OutputDefs()[2]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/GQA/postprocess/cast/present_value");
    present_value = model_builder.GetBuilder().call<emscripten::val>("cast", present_value, emscripten::val("float16"),
                                                                     common_options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  model_builder.AddOperand(node.OutputDefs()[1]->Name(), std::move(present_key));
  model_builder.AddOperand(node.OutputDefs()[2]->Name(), std::move(present_value));

  return Status::OK();
}

// Operator support related.

bool AttentionOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                           const WebnnDeviceType /* device_type */,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  NodeAttrHelper helper(node);

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get input shape.";
    return false;
  }

  const auto& output_defs = node.OutputDefs();
  if (output_defs.size() != 3) {
    LOGS(logger, VERBOSE) << op_type << " output count must be Three.";
    return false;
  }

  return true;
}

bool AttentionOpBuilder::HasSupportedInputsImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                                const emscripten::val& wnn_limits,
                                                const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();

  if (input_defs.size() < 7) {
    LOGS(logger, VERBOSE) << op_type << " requires at least seven inputs.";
    return false;
  }

  return true;
}

void CreateAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0) return;

  constexpr static std::string_view op_types[] = {
      "GroupQueryAttention",
  };

  op_registrations.builders.push_back(std::make_unique<AttentionOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime

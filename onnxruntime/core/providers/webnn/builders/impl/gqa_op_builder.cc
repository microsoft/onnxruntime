// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include <cmath>

#include "base_op_builder.h"
#include "attention_helper.h"

namespace onnxruntime {
namespace webnn {

class GroupQueryAttentionOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node, const WebnnDeviceType /* device_type */,
                         const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

void GroupQueryAttentionOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // We check the value of input total_sequence_length in function IsOpSupportedImpl,
  // and it should be an initializer and does not participate in Op calculation.
  const auto input_name = node.InputDefs()[6]->Name();
  model_builder.AddInitializerToSkip(input_name);
  model_builder.AddInputToSkip(input_name);
}

std::vector<int32_t> generate_indices(int32_t batch_size, int32_t kv_num_heads, int32_t sequence_length) {
  std::vector<int32_t> indices;
  for (int32_t i = 0; i < sequence_length; ++i) {
    for (int32_t j = 0; j < batch_size * kv_num_heads; ++j) {
      indices.push_back(j / kv_num_heads);
      indices.push_back(j % kv_num_heads);
    }
  }
  return indices;
}

std::vector<int32_t> repeat_sequence(int32_t sequence_length, int32_t kv_num_heads, int32_t batch_size) {
  std::vector<int32_t> repeated;
  for (int32_t i = 0; i < sequence_length; ++i) {
    for (int32_t j = 0; j < batch_size * kv_num_heads; ++j) {
      repeated.push_back(i);
    }
  }
  return repeated;
}

/** GroupQueryAttention SubGraph.
 Abbreviations: B is batch_size, S is sequence_length, W is hidden_size, P is past_sequence_length
                N is number of attention heads, kv_N is number of attention heads for kv, H is head size
                G is group size, and G=N/kv_N, W=N*H, h=Sqrt(H).
    GQA inputs: query, key(optional), value(optional), past_key(optional), past_value(optional),
                seqlens_k, total_sequence_length, cos_cache(optional), sin_cache(optional), position_ids(optional)
    Notes:
      - key, value, past_key, past_value can be empty (optional inputs).
      - When key/value are empty, query contains packed QKV.
      - When past_key/past_value are empty, this is the first token (prefill mode).
      - When do_rotary is true, cos_cache and sin_cache must be provided.

          query      key               value
            |         |                  |
      (RotaryEmb)  (RotaryEmb)           |
            |         |                  |
         Reshape   Reshape            Reshape (B,S,H,N)     seqlens_k
            |         |                  |                  /       |
            |         |       past_value |   (scatter_indices*)     |
        q_Transpose   |              \   |   /                      |
        (0,2,1,3)     | past_key    ScatterND-----------------------|------> present_value
             \        |  /              |                           |
present_key<--\----ScatterND         Expand(G)      (attention_bias, one/finfo_min mask*)
               \      |                 |              /
               |   Expand(G)            |             /
               |      |                 |            /
               |  k_Transpose           |           /
               |   (0,1,3,2)            |          /
               |      |                 |         /
            +---------------------------------------+
            |        ScaledDotProductAttention      |
            +---------------------------------------+
                             |
                           output
*/

Status GroupQueryAttentionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  NodeAttrHelper helper(node);
  const int32_t local_window_size = helper.Get("local_window_size", -1);
  const uint32_t kv_num_heads = helper.Get("kv_num_heads", 0);
  const uint32_t num_heads = helper.Get("num_heads", 0);
  const bool do_rotary = static_cast<bool>(helper.Get("do_rotary", 0));
  const bool rotary_interleaved = static_cast<bool>(helper.Get("rotary_interleaved", 0));

  // Check if optional inputs exist
  const bool has_key = TensorExists(input_defs, 1);
  const bool has_value = TensorExists(input_defs, 2);
  const bool has_past_key = TensorExists(input_defs, 3);
  const bool has_past_value = TensorExists(input_defs, 4);
  const bool has_cos_cache = TensorExists(input_defs, 7);
  const bool has_sin_cache = TensorExists(input_defs, 8);
  const bool has_position_ids = TensorExists(input_defs, 9);

  emscripten::val query_input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val key_input = has_key ? model_builder.GetOperand(input_defs[1]->Name()) : emscripten::val::undefined();
  emscripten::val value_input = has_value ? model_builder.GetOperand(input_defs[2]->Name()) : emscripten::val::undefined();
  emscripten::val past_key_input = has_past_key ? model_builder.GetOperand(input_defs[3]->Name()) : emscripten::val::undefined();
  emscripten::val past_value_input = has_past_value ? model_builder.GetOperand(input_defs[4]->Name()) : emscripten::val::undefined();
  emscripten::val seqlens_k_input = model_builder.GetOperand(input_defs[5]->Name());
  emscripten::val cos_cache = has_cos_cache ? model_builder.GetOperand(input_defs[7]->Name()) : emscripten::val::undefined();
  emscripten::val sin_cache = has_sin_cache ? model_builder.GetOperand(input_defs[8]->Name()) : emscripten::val::undefined();

  std::vector<int64_t> input_q_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_q_shape, logger), "Cannot get query shape");

  // Calculate hidden_size and head_size based on whether key/value are provided
  uint32_t qkv_hidden_size;
  uint32_t head_size;
  if (has_key) {
    // query shape is (batch_size, sequence_length, num_heads * head_size)
    qkv_hidden_size = SafeInt<uint32_t>(input_q_shape[2]);
    head_size = SafeInt<uint32_t>(qkv_hidden_size / num_heads);
  } else {
    // query contains packed QKV: (batch_size, sequence_length, num_heads * head_size + 2 * kv_num_heads * head_size)
    // hidden_size = num_heads * head_size, so we derive: head_size = d / (num_heads + 2 * kv_num_heads)
    uint32_t d = SafeInt<uint32_t>(input_q_shape[2]);
    head_size = d / (num_heads + 2 * kv_num_heads);
    qkv_hidden_size = num_heads * head_size;
  }

  // Get past_sequence_length from past_key if available, otherwise it's 0 (first token)
  uint32_t past_sequence_length = 0;
  if (has_past_key) {
    std::vector<int64_t> input_past_k_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[3], input_past_k_shape, logger), "Cannot get past_key shape");
    past_sequence_length = SafeInt<uint32_t>(input_past_k_shape[2]);
  }

  const uint32_t batch_size = SafeInt<uint32_t>(input_q_shape[0]);
  const uint32_t qkv_sequence_length = SafeInt<uint32_t>(input_q_shape[1]);

  emscripten::val position_ids = emscripten::val::undefined();
  bool use_position_ids_as_offset = false;
  if (has_position_ids) {
    position_ids = model_builder.GetOperand(input_defs[9]->Name());
  } else {
    // If position_ids is not provided, we need to derive it from the context.
    // We distinguish prefill vs decode by qkv_sequence_length (not has_past_key), because
    // with pre-allocated KV cache (freeDimensionOverrides), has_past_key is always true.
    //
    // - Prefill (qkv_sequence_length > 1): positions start from 0
    // - Decode (qkv_sequence_length == 1): position = seqlens_k (the actual sequence position)
    //
    // Note: We cannot use past_sequence_length from the static shape because it represents the
    // pre-allocated cache size (total_sequence_length), not the actual number of valid tokens.
    if (qkv_sequence_length == 1) {
      // During decode, use seqlens_k as the position offset for rotary embedding.
      // seqlens_k has shape [batch_size], but we need [batch_size, 1] to properly broadcast
      // with position_ids_range which has shape [1, sequence_length] in ApplyRotaryEmbedding.
      emscripten::val reshape_options = emscripten::val::object();
      reshape_options.set("label", node.Name() + "_/GQA/seqlens_k_reshape_for_position");

      emscripten::val reshaped_seqlens_k = model_builder.GetBuilder().call<emscripten::val>(
          "reshape", seqlens_k_input, emscripten::val::array(std::vector<uint32_t>({batch_size, 1})), reshape_options);

      // seqlens_k is INT32, but position_ids_range in ApplyRotaryEmbedding may be INT64
      // if int64 is supported. We need to cast to match the expected type.
      if (model_builder.IsInt64Supported()) {
        emscripten::val cast_options = emscripten::val::object();
        cast_options.set("label", node.Name() + "_/GQA/seqlens_k_cast_to_int64");
        position_ids = model_builder.GetBuilder().call<emscripten::val>(
            "cast", reshaped_seqlens_k, emscripten::val("int64"), cast_options);
      } else {
        position_ids = reshaped_seqlens_k;
      }
    } else {
      // During prefill, use 0 as the offset (positions will be 0, 1, 2, ..., sequence_length-1)
      if (model_builder.IsInt64Supported()) {
        position_ids = model_builder.CreateOrGetConstant<int64_t>(
            ONNX_NAMESPACE::TensorProto_DataType_INT64, static_cast<int64_t>(0), {1});
      } else {
        position_ids = model_builder.CreateOrGetConstant<int32_t>(
            ONNX_NAMESPACE::TensorProto_DataType_INT32, static_cast<int32_t>(0), {1});
      }
    }
    use_position_ids_as_offset = true;
  }

  const uint32_t group_size = SafeInt<uint32_t>(num_heads / kv_num_heads);

  const float scale_value = helper.Get("scale", 1 / sqrt(static_cast<float>(head_size)));

  const std::vector<uint32_t> reshape_output_shape = {batch_size, qkv_sequence_length, qkv_hidden_size};
  const std::vector<uint32_t> scatter_indices_shape = {batch_size, qkv_sequence_length, kv_num_heads, 3};
  const std::vector<uint32_t> reshape_tensor_shape = {batch_size, qkv_sequence_length, num_heads, head_size};

  emscripten::val common_options = emscripten::val::object();
  emscripten::val common_desc = emscripten::val::object();

  int32_t q_type = 0;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], q_type, logger), "Could not get input data type.");

  // Split packed QKV if key and value are not provided separately
  if (!has_key) {
    // query contains packed QKV: (batch_size, sequence_length, num_heads * head_size + 2 * kv_num_heads * head_size)
    const uint32_t kv_hidden_size = kv_num_heads * head_size;
    const std::vector<uint32_t> splits{qkv_hidden_size, kv_hidden_size, kv_hidden_size};
    emscripten::val split_options = emscripten::val::object();
    split_options.set("label", node.Name() + "_/GQA/split_packed_qkv");
    split_options.set("axis", 2);
    emscripten::val split_result = model_builder.GetBuilder().call<emscripten::val>(
        "split", query_input, emscripten::val::array(splits), split_options);
    query_input = split_result[0];
    key_input = split_result[1];
    value_input = split_result[2];
  }

  // Apply rotary embedding if do_rotary is true
  if (do_rotary && has_cos_cache && has_sin_cache) {
    // Determine rotary_embedding_dim from cos_cache shape
    std::vector<int64_t> cos_cache_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[7], cos_cache_shape, logger), "Cannot get cos_cache shape");
    const uint32_t rotary_embedding_dim = static_cast<uint32_t>(cos_cache_shape[1] * 2);

    // Reshape query to (batch_size, sequence_length, num_heads, head_size) for rotary embedding
    const std::vector<uint32_t> query_reshape_for_rotary = {batch_size, qkv_sequence_length, num_heads, head_size};
    common_options.set("label", node.Name() + "_/GQA/query/reshape_for_rotary");
    emscripten::val reshaped_query_for_rotary = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", query_input, emscripten::val::array(query_reshape_for_rotary), common_options);

    // Apply rotary embedding to query
    emscripten::val rotary_query_output;
    ORT_RETURN_IF_ERROR(ApplyRotaryEmbedding(
        model_builder,
        node.Name() + "_query",
        reshaped_query_for_rotary,
        cos_cache,
        sin_cache,
        position_ids,
        q_type,
        batch_size,
        qkv_sequence_length,
        num_heads,
        head_size,
        rotary_embedding_dim,
        rotary_interleaved,
        true,
        use_position_ids_as_offset,  // position_ids_is_offset
        rotary_query_output));

    // Reshape back to (batch_size, sequence_length, hidden_size)
    common_options.set("label", node.Name() + "_/GQA/query/reshape_after_rotary");
    query_input = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", rotary_query_output, emscripten::val::array(std::vector<uint32_t>({batch_size, qkv_sequence_length, qkv_hidden_size})), common_options);

    // Reshape key to (batch_size, sequence_length, kv_num_heads, head_size) for rotary embedding
    const std::vector<uint32_t> key_reshape_for_rotary = {batch_size, qkv_sequence_length, kv_num_heads, head_size};
    common_options.set("label", node.Name() + "_/GQA/key/reshape_for_rotary");
    emscripten::val reshaped_key_for_rotary = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", key_input, emscripten::val::array(key_reshape_for_rotary), common_options);

    // Apply rotary embedding to key
    emscripten::val rotary_key_output;
    ORT_RETURN_IF_ERROR(ApplyRotaryEmbedding(
        model_builder,
        node.Name() + "_key",
        reshaped_key_for_rotary,
        cos_cache,
        sin_cache,
        position_ids,
        q_type,
        batch_size,
        qkv_sequence_length,
        kv_num_heads,
        head_size,
        rotary_embedding_dim,
        rotary_interleaved,
        true,
        use_position_ids_as_offset,  // position_ids_is_offset
        rotary_key_output));

    // Reshape back to (batch_size, sequence_length, kv_hidden_size)
    const uint32_t kv_hidden_size = kv_num_heads * head_size;
    common_options.set("label", node.Name() + "_/GQA/key/reshape_after_rotary");
    key_input = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", rotary_key_output, emscripten::val::array(std::vector<uint32_t>({batch_size, qkv_sequence_length, kv_hidden_size})), common_options);
  }

  // Reshape and transpose the input "query"
  common_options.set("label", node.Name() + "_/GQA/query/reshape");
  emscripten::val reshaped_query = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", query_input, emscripten::val::array(reshape_tensor_shape), common_options);

  emscripten::val transpose_options = emscripten::val::object();
  transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  transpose_options.set("label", node.Name() + "_/GQA/query/transpose");
  emscripten::val new_query =
      model_builder.GetBuilder().call<emscripten::val>("transpose", reshaped_query, transpose_options);

  // Reshape the inputs "key" and "value" for scatterND
  std::vector<uint32_t> reshape_kv_shape = {batch_size, qkv_sequence_length, kv_num_heads, head_size};
  common_options.set("label", node.Name() + "_/GQA/key/reshape_1");
  emscripten::val key_for_scatter = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", key_input, emscripten::val::array(reshape_kv_shape), common_options);

  common_options.set("label", node.Name() + "_/GQA/value/reshape_1");
  emscripten::val value_for_scatter = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", value_input, emscripten::val::array(reshape_kv_shape), common_options);

  /* Calculate scatter_indices for kv's scatterND
                                                                                 if_prefill (0/1 constant)
                                                                                          |
  scatter_indices_left_constant         scatter_indices_right_constant           0 ---> Where <--- Cast <---seqlens_k
                |                                      |                                  |
                |                                     Add <--------------------------- scatter_pos*
                |                                      |
                +------------------+-------------------+
                                   |
                            scatter_indices
  */
  // Prepare the constant materials for scatter indices
  auto left = generate_indices(batch_size, kv_num_heads, qkv_sequence_length);
  auto right = repeat_sequence(qkv_sequence_length, kv_num_heads, batch_size);

  std::string left_name = "webnn_GQA_left_constant_of_scatter_indices_" + std::to_string(batch_size) + "_" +
                          std::to_string(qkv_sequence_length) + "_" + std::to_string(kv_num_heads) + "_2";
  emscripten::val left_constant = model_builder.CreateOrGetConstant<int32_t>(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, left_name, left,
      std::vector<uint32_t>({batch_size * qkv_sequence_length * kv_num_heads, 2}));

  std::string right_name = "webnn_GQA_right_constant_of_scatter_indices_" + std::to_string(batch_size) + "_" +
                           std::to_string(qkv_sequence_length) + "_" + std::to_string(kv_num_heads) + "_1";
  emscripten::val right_constant = model_builder.CreateOrGetConstant<int32_t>(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, right_name, right,
      std::vector<uint32_t>({batch_size * qkv_sequence_length * kv_num_heads, 1}));

  // The prefilling and decoding stages require different index construction for ScatterND operations.
  // Similar to other EPs like CPU and DirectML, when qkv_sequence_length > 1, the key and value are scattered to the
  // beginning of kv cache.
  std::vector<uint8_t> first_condition({(qkv_sequence_length > 1)});
  std::string condition_name = "webnn_GQA_condition_constant_for_where_1";
  emscripten::val condition_constant = model_builder.CreateOrGetConstant<uint8_t>(
      ONNX_NAMESPACE::TensorProto_DataType_UINT8, condition_name, first_condition, std::vector<uint32_t>({1}));

  emscripten::val value_zero_constant =
      model_builder.CreateOrGetConstant<int>(ONNX_NAMESPACE::TensorProto_DataType_INT32, 0, {1});

  // Use concat and reshape to achieve scatter_indices
  common_options.set("label", node.Name() + "_/GQA/scatter/where");
  emscripten::val scatter_pos = model_builder.GetBuilder().call<emscripten::val>(
      "where", condition_constant, value_zero_constant, seqlens_k_input, common_options);

  common_options.set("label", node.Name() + "_/GQA/right_constant/add");
  right_constant = model_builder.GetBuilder().call<emscripten::val>("add", right_constant, scatter_pos, common_options);

  common_options.set("label", node.Name() + "_/GQA/concat_for_pre_scatter_indices");
  std::vector<emscripten::val> inputs({left_constant, right_constant});
  uint32_t axis = 1;
  emscripten::val pre_scatter_indices =
      model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(inputs), axis, common_options);

  common_options.set("label", node.Name() + "_/GQA/pre_scatter_indices/reshape");
  emscripten::val scatter_indices = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", pre_scatter_indices, emscripten::val::array(scatter_indices_shape), common_options);

  // scatterND for present_key and present_value, or use key/value directly if no past
  emscripten::val present_key;
  emscripten::val present_value;
  if (has_past_key && has_past_value) {
    common_options.set("label", node.Name() + "_/GQA/present_key/ScatterND");
    present_key = model_builder.GetBuilder().call<emscripten::val>(
        "scatterND", past_key_input, scatter_indices, key_for_scatter, common_options);

    common_options.set("label", node.Name() + "_/GQA/present_value/ScatterND");
    present_value = model_builder.GetBuilder().call<emscripten::val>(
        "scatterND", past_value_input, scatter_indices, value_for_scatter, common_options);
  } else {
    // No past_key/past_value, use key/value directly as present_key/present_value (first token case)
    // Transpose key and value to BNSH format: (B, S, kv_N, H) -> (B, kv_N, S, H)
    transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
    transpose_options.set("label", node.Name() + "_/GQA/key/transpose_to_bnsh");
    present_key = model_builder.GetBuilder().call<emscripten::val>("transpose", key_for_scatter, transpose_options);

    transpose_options.set("label", node.Name() + "_/GQA/value/transpose_to_bnsh");
    present_value = model_builder.GetBuilder().call<emscripten::val>("transpose", value_for_scatter, transpose_options);
  }

  emscripten::val true_present_key;
  emscripten::val true_present_value;
  // If no past_key, the sequence length is qkv_sequence_length, otherwise it is past_sequence_length.
  // In prefill stage, sequence_length == total_sequence_length.
  // In decoding stage, past_sequence_length == total_sequence_length.
  // So we can use total_sequence_length (which is provided in input[6]) or derive it from the logic above.
  uint32_t current_total_seq_len;
  if (!has_past_key && !has_past_value) {
    // Prefill: current_total_seq_len is simply qkv_sequence_length (which should == total_sequence_length)
    current_total_seq_len = qkv_sequence_length;
  } else {
    // Decoding: current_total_seq_len is past_sequence_length
    current_total_seq_len = past_sequence_length;
  }

  if (group_size != 1) {
    // Broadcast key and value for group query by reshape, expand and reshape.
    // present kv shape (B,kv_N,P,H) -> (B,kv_N,1,P,H) -> (B,kv_N,N/kv_N,P,H) -> (B,N,P,H) broadcasted kv shape
    const std::vector<uint32_t> group_broadcast_tensor_shape_1 = {batch_size, kv_num_heads, 1, current_total_seq_len,
                                                                  head_size};
    const std::vector<uint32_t> group_broadcast_tensor_shape_2 = {batch_size, kv_num_heads, group_size,
                                                                  current_total_seq_len, head_size};
    const std::vector<uint32_t> group_broadcast_tensor_shape_3 = {batch_size, num_heads, current_total_seq_len,
                                                                  head_size};
    common_options.set("label", node.Name() + "_/GQA/true_present_key/reshape_1");
    true_present_key = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", present_key, emscripten::val::array(group_broadcast_tensor_shape_1), common_options);
    common_options.set("label", node.Name() + "_/GQA/true_present_key/expand");
    true_present_key = model_builder.GetBuilder().call<emscripten::val>(
        "expand", true_present_key, emscripten::val::array(group_broadcast_tensor_shape_2), common_options);
    common_options.set("label", node.Name() + "_/GQA/true_present_key/reshape_2");
    true_present_key = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", true_present_key, emscripten::val::array(group_broadcast_tensor_shape_3), common_options);

    common_options.set("label", node.Name() + "_/GQA/true_present_value/reshape_1");
    true_present_value = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", present_value, emscripten::val::array(group_broadcast_tensor_shape_1), common_options);
    common_options.set("label", node.Name() + "_/GQA/true_present_value/expand");
    true_present_value = model_builder.GetBuilder().call<emscripten::val>(
        "expand", true_present_value, emscripten::val::array(group_broadcast_tensor_shape_2), common_options);
    common_options.set("label", node.Name() + "_/GQA/true_present_value/reshape_2");
    true_present_value = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", true_present_value, emscripten::val::array(group_broadcast_tensor_shape_3), common_options);
  } else {  // no need for broadcast
    true_present_key = present_key;
    true_present_value = present_value;
  }

  // Transpose key for matrix multiplication
  transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 1, 3, 2})));
  transpose_options.set("label", node.Name() + "_/GQA/present_key/transpose");
  true_present_key = model_builder.GetBuilder().call<emscripten::val>("transpose", true_present_key, transpose_options);

  emscripten::val scale_constant = model_builder.CreateOrGetConstant<float>(q_type, scale_value, {1});

  /* Calculate attention_bias for masking softmax
        ones_array (shape=B,N,S,P)                          range_of_qkv_sequence_length_constant (0,1,2,...) (shape=S)
          |                                                                 |
        CumSum (axis=3, exclusive=true, reversed=false)                    Add <--- scatter_pos
          |                                                                 |
          |                                                               Expand (shape=P,S)
          |                                                                 |
          +-------------------------------> Lesser <---------------------Transpose (1,0)
                                                |
                                      1 ---> Where (attn_mask) <--- finfo_min (minimum value of FP32)
                                                |
                                          attention_bias
  */
  emscripten::val value_int_one_constant =
      model_builder.CreateOrGetConstant<int>(ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, {1});

  std::vector<uint32_t> mask_shape_ones_shape = {batch_size, num_heads, qkv_sequence_length, current_total_seq_len};
  common_options.set("label", node.Name() + "_/GQA/GQA_mask_shape_ones/expand");
  emscripten::val mask_shape_ones_shape_constant = model_builder.GetBuilder().call<emscripten::val>(
      "expand", value_int_one_constant, emscripten::val::array(mask_shape_ones_shape), common_options);

  emscripten::val cumsum_options = emscripten::val::object();
  cumsum_options.set("label", node.Name() + "_range_of_mask_shape");
  cumsum_options.set("exclusive", true);
  cumsum_options.set("reversed", false);
  emscripten::val neq_left = model_builder.GetBuilder().call<emscripten::val>(
      "cumulativeSum", mask_shape_ones_shape_constant, gsl::narrow<uint32_t>(3), cumsum_options);

  std::vector<uint32_t> reshape_pre_neq_right = {current_total_seq_len, qkv_sequence_length};
  std::vector<int32_t> pre_neq_right_data_range(qkv_sequence_length);
  std::iota(pre_neq_right_data_range.begin(), pre_neq_right_data_range.end(), 1);

  std::string pre_neq_right_data_range_name =
      "webnn_GQA_pre_neq_right_data_range_" + std::to_string(qkv_sequence_length);
  emscripten::val pre_neq_right_data_range_constant = model_builder.CreateOrGetConstant<int32_t>(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, pre_neq_right_data_range_name, pre_neq_right_data_range,
      std::vector<uint32_t>({qkv_sequence_length}));

  common_options.set("label", node.Name() + "_/GQA/attn_mask/add");
  emscripten::val pre_neq_right = model_builder.GetBuilder().call<emscripten::val>(
      "add", pre_neq_right_data_range_constant, scatter_pos, common_options);

  common_options.set("label", node.Name() + "_/GQA/expand_neq_right");
  emscripten::val expanded_neq_right = model_builder.GetBuilder().call<emscripten::val>(
      "expand", pre_neq_right, emscripten::val::array(reshape_pre_neq_right), common_options);

  transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({1, 0})));
  transpose_options.set("label", node.Name() + "_/GQA/neq_right/transpose");
  emscripten::val neq_right =
      model_builder.GetBuilder().call<emscripten::val>("transpose", expanded_neq_right, transpose_options);

  common_options.set("label", node.Name() + "_/GQA/attn_mask/condition_1");
  emscripten::val condition_1 =
      model_builder.GetBuilder().call<emscripten::val>("lesser", neq_left, neq_right, common_options);

  emscripten::val condition = condition_1;
  // For local window size not equal to -1, new attention mask pattern for applying sliding window
  /*
     condition_1 (old attn_mask) ---> CumSum (axis=3, exclusive=true, reversed=true)
          |                             |
          |                           Lesser <--- local_window_size
          |                             |
      LogicalAnd <----------------- condition_2
          |
    new attn_mask
  */
  if (local_window_size != -1) {
    // Cast condition
    common_options.set("label", node.Name() + "_/GQA/attn_mask/condition_2/cast");
    emscripten::val casted_condition_1 =
        model_builder.GetBuilder().call<emscripten::val>("cast", condition_1, emscripten::val("int32"), common_options);

    cumsum_options = emscripten::val::object();
    cumsum_options.set("label", node.Name() + "_/GQA/attn_mask/condition_2/cumsum");
    cumsum_options.set("exclusive", true);
    cumsum_options.set("reversed", true);
    emscripten::val neq_left_2 = model_builder.GetBuilder().call<emscripten::val>(
        "cumulativeSum", casted_condition_1, gsl::narrow<uint32_t>(3), cumsum_options);

    emscripten::val local_window_size_constant =
        model_builder.CreateOrGetConstant<int>(ONNX_NAMESPACE::TensorProto_DataType_INT32, local_window_size, {1});

    common_options.set("label", node.Name() + "_/GQA/attn_mask/condition_2");
    emscripten::val condition_2 =
        model_builder.GetBuilder().call<emscripten::val>("lesser", neq_left_2, local_window_size_constant, common_options);

    common_options.set("label", node.Name() + "_/GQA/attn_mask/condition/and");
    condition = model_builder.GetBuilder().call<emscripten::val>(
        "logicalAnd", condition_1, condition_2, common_options);
  }

  // For attended positions, use 0.0 (no change to attention scores)
  // For masked positions, use a very large negative number (softmax → 0)
  emscripten::val value_zero_constant_float = model_builder.CreateOrGetConstant<float>(q_type, 0, {1});

  // finfo_min: the minimum value of float32
  emscripten::val finfo_min_constant = model_builder.CreateOrGetConstant<float>(q_type, -3.4028234663852886e+38, {1});

  common_options.set("label", node.Name() + "_/GQA/attn_mask/where");
  emscripten::val attn_mask = model_builder.GetBuilder().call<emscripten::val>("where", condition, value_zero_constant_float,
                                                                               finfo_min_constant, common_options);

  // Execute ScaledDotProductAttention
  emscripten::val output =
      ScaledDotProductAttention(model_builder, node, logger, new_query, true_present_key, true_present_value,
                                scale_constant, attn_mask, reshape_output_shape);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  model_builder.AddOperand(node.OutputDefs()[1]->Name(), std::move(present_key));
  model_builder.AddOperand(node.OutputDefs()[2]->Name(), std::move(present_value));

  return Status::OK();
}

// Operator support related.

bool GroupQueryAttentionOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                                                     const WebnnDeviceType /* device_type */,
                                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  NodeAttrHelper helper(node);

  const int64_t do_rotary = helper.Get("do_rotary", static_cast<int64_t>(0));

  // When do_rotary is true, cos_cache and sin_cache must be provided
  if (do_rotary) {
    if (!TensorExists(input_defs, 7) || !TensorExists(input_defs, 8)) {
      LOGS(logger, VERBOSE) << op_type << " requires cos_cache and sin_cache when do_rotary is true";
      return false;
    }
  }

  const auto& total_sequence_length_name = input_defs[6]->Name();
  const auto* total_sequence_length_initializer = graph_viewer.GetConstantInitializer(total_sequence_length_name);
  if (!total_sequence_length_initializer) {
    LOGS(logger, VERBOSE) << "total_sequence_length must be constant";
    return false;
  }

  const auto total_sequence_length_tensor = *total_sequence_length_initializer;
  emscripten::val total_sequence_length = emscripten::val::undefined();
  if (!ReadScalarTensorData(total_sequence_length_tensor, total_sequence_length, graph_viewer, logger)) {
    return false;
  }

  std::vector<int64_t> query_shape;
  if (!GetShape(*input_defs[0], query_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get query shape.";
    return false;
  }
  const auto sequence_length = query_shape[1];

  // Check if past_key exists to determine past_sequence_length
  const bool has_past_key = TensorExists(input_defs, 3);
  int64_t past_sequence_length = 0;
  if (has_past_key) {
    std::vector<int64_t> past_key_shape;
    if (!GetShape(*input_defs[3], past_key_shape, logger)) {
      LOGS(logger, VERBOSE) << "Cannot get past_key shape.";
      return false;
    }
    past_sequence_length = past_key_shape[2];
  }

  // WebNN EP only supports past_sequence_length of past kv equals to present_sequence_length of present kv
  // According to CPU EP, present_sequence_length = max(past_sequence_length,total_sequence_length)
  // For prefilling stage (the first prompt), it requires sequence_length == total_sequence_length.
  if (sequence_length != 1) {
    if (sequence_length != total_sequence_length.as<int32_t>()) {
      LOGS(logger, VERBOSE) << op_type << " sequence_length != total_sequence_length. Not first prompt.";
      return false;
    }
  } else {  // For decoding stage, it requires past_sequence_length == total_sequence_length.
    if (has_past_key && past_sequence_length != total_sequence_length.as<int32_t>()) {
      LOGS(logger, VERBOSE) << op_type << " past_sequence_length != total_sequence_length.";
      return false;
    }
  }

  const auto& output_defs = node.OutputDefs();
  if (output_defs.size() != 3) {
    LOGS(logger, VERBOSE) << op_type << " output count must be three.";
    return false;
  }

  return true;
}

bool GroupQueryAttentionOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                          const emscripten::val& wnn_limits,
                                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  NodeAttrHelper helper(node);

  const int64_t do_rotary = helper.Get("do_rotary", static_cast<int64_t>(0));

  // Validate required inputs: query(0), seqlens_k(5), total_sequence_length(6) are always required
  // key(1), value(2), past_key(3), past_value(4) are optional
  // cos_cache(7), sin_cache(8) are required when do_rotary is true
  // position_ids(9), attention_bias(10), head_sink(11) are optional

  // Check required inputs
  if (!TensorExists(input_defs, 0)) {
    LOGS(logger, VERBOSE) << op_type << " requires query input (index 0)";
    return false;
  }
  if (!TensorExists(input_defs, 5)) {
    LOGS(logger, VERBOSE) << op_type << " requires seqlens_k input (index 5)";
    return false;
  }
  if (!TensorExists(input_defs, 6)) {
    LOGS(logger, VERBOSE) << op_type << " requires total_sequence_length input (index 6)";
    return false;
  }

  // Check key/value pair consistency
  const bool has_key = TensorExists(input_defs, 1);
  const bool has_value = TensorExists(input_defs, 2);
  if (has_key != has_value) {
    LOGS(logger, VERBOSE) << op_type << " key and value must both be present or both be absent";
    return false;
  }

  // Check past_key/past_value pair consistency
  const bool has_past_key = TensorExists(input_defs, 3);
  const bool has_past_value = TensorExists(input_defs, 4);
  if (has_past_key != has_past_value) {
    LOGS(logger, VERBOSE) << op_type << " past_key and past_value must both be present or both be absent";
    return false;
  }

  // Check do_rotary requirements
  const bool has_cos_cache = TensorExists(input_defs, 7);
  const bool has_sin_cache = TensorExists(input_defs, 8);
  if (do_rotary) {
    if (!has_cos_cache || !has_sin_cache) {
      LOGS(logger, VERBOSE) << op_type << " requires cos_cache and sin_cache when do_rotary is true";
      return false;
    }
  }

  // Get query type (required)
  int32_t q_type = 0;
  if (!GetType(*input_defs[0], q_type, logger)) {
    return false;
  }

  // Check optional key/value types
  if (has_key) {
    int32_t k_type = 0;
    int32_t v_type = 0;
    if (!GetType(*input_defs[1], k_type, logger) || !GetType(*input_defs[2], v_type, logger)) {
      return false;
    }
    std::array<int32_t, 3> qkv_types{q_type, k_type, v_type};
    if (!AreDataTypesSame(op_type, qkv_types, logger)) {
      return false;
    }
  }

  // Check optional past_key/past_value types
  if (has_past_key) {
    int32_t past_k_type = 0;
    int32_t past_v_type = 0;
    if (!GetType(*input_defs[3], past_k_type, logger) || !GetType(*input_defs[4], past_v_type, logger)) {
      return false;
    }
    std::array<int32_t, 3> past_types{q_type, past_k_type, past_v_type};
    if (!AreDataTypesSame(op_type, past_types, logger)) {
      return false;
    }
  }

  // Check seqlens_k and total_sequence_length types
  int32_t seqlens_k_type = 0;
  int32_t total_sequence_length_type = 0;
  if (!GetType(*input_defs[5], seqlens_k_type, logger) || !GetType(*input_defs[6], total_sequence_length_type, logger)) {
    return false;
  }

  if (q_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT && q_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    LOGS(logger, VERBOSE) << op_type << " query type must be float or float16";
    return false;
  }

  if (seqlens_k_type != ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
      total_sequence_length_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    LOGS(logger, VERBOSE) << op_type << " seqlens_k and total_sequence_length must be int32";
    return false;
  }

  // Check cos_cache/sin_cache types when do_rotary is true
  if (do_rotary && has_cos_cache && has_sin_cache) {
    int32_t cos_cache_type = 0;
    int32_t sin_cache_type = 0;
    if (!GetType(*input_defs[7], cos_cache_type, logger) || !GetType(*input_defs[8], sin_cache_type, logger)) {
      return false;
    }
    std::array<int32_t, 3> cache_types{q_type, cos_cache_type, sin_cache_type};
    if (!AreDataTypesSame(op_type, cache_types, logger)) {
      return false;
    }
  }

  // Check shapes
  std::vector<int64_t> input_q_shape;
  if (!GetShape(*input_defs[0], input_q_shape, logger)) {
    return false;
  }
  const auto q_rank = input_q_shape.size();
  if (q_rank != 3) {  // The query shape should be BSW
    LOGS(logger, VERBOSE) << op_type << " query shape is not BSW.";
    return false;
  }

  if (has_key) {
    std::vector<int64_t> input_k_shape, input_v_shape;
    if (!GetShape(*input_defs[1], input_k_shape, logger) || !GetShape(*input_defs[2], input_v_shape, logger)) {
      return false;
    }
    const auto k_rank = input_k_shape.size();
    const auto v_rank = input_v_shape.size();
    if (k_rank != 3 || v_rank != 3) {  // The kv shape should be BSW
      LOGS(logger, VERBOSE) << op_type << " key/value shape is not BSW.";
      return false;
    }
  }

  if (has_past_key) {
    std::vector<int64_t> input_past_k_shape, input_past_v_shape;
    if (!GetShape(*input_defs[3], input_past_k_shape, logger) ||
        !GetShape(*input_defs[4], input_past_v_shape, logger)) {
      return false;
    }
    const auto past_k_rank = input_past_k_shape.size();
    const auto past_v_rank = input_past_v_shape.size();
    if (past_k_rank != 4 || past_v_rank != 4) {  // The past qkv shape should be BNSH
      LOGS(logger, VERBOSE) << op_type << " past qkv shape is not BNSH.";
      return false;
    }
  }

  return true;
}

bool GroupQueryAttentionOpBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                                           const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type = 0;
  int32_t present_k_type = 0;
  int32_t present_v_type = 0;
  if (!GetType(*output_defs[0], output_type, logger) || !GetType(*output_defs[1], present_k_type, logger) ||
      !GetType(*output_defs[2], present_v_type, logger)) {
    return false;
  }

  std::array<int32_t, 3> output_types{output_type, present_k_type, present_v_type};
  if (!AreDataTypesSame(op_type, output_types, logger)) {
    return false;
  }

  // GQA allows float16, bfloat16 and float32, but WebNN only supports float16 and float32.
  if (output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return false;
  }
  return true;
}

void CreateGroupQueryAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GroupQueryAttentionOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime

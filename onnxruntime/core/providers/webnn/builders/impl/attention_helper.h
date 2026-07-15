// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webnn/builders/helper.h"

namespace onnxruntime {
namespace webnn {
/*
    RotaryEmbedding Helper: Apply rotary positional embedding to input tensor.
    This helper function implements rotary embedding that can be reused by GQA and RotaryEmbedding ops.

    The decomposed graph is referenced from DML EP at:
    onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorRotaryEmbedding.cpp

                 Input                            CosCache   PositionIds     SinCache
                   |                                 |           |              |
                   |                                 |  +--------+-----------+  |
                 Split                               |  |                    |  |
                  |  |                              Gather                  Gather
          +-------+  |                                |                        |
          |          |                                |                        |
          |     Identity----------+                   |                        |
          |        |              |                   |                        |
          |        |              |                   |                        |
          |    --Split--          |                   |                        |
          |    \       /          | +-----------------+                        |
          |     \     /           | |                                          |
          |      \   /            Mul                                          |
          |       \ /              |                                           |
          |        X               |                                           |
          |       / \              |                                           |
          |      /   \             |                                           |
          |       Join             |                                           |
          |        |               |                                           |
          |        | +---------------------------------------------------------+
          |        | |             |
          |        Mul             |
          |         |              |
          |         +-----+ +------+
          |               | |
          |               Add
          |                |
          +-------------+  |
                        |  |
                        Join
*/
inline Status ApplyRotaryEmbedding(
    ModelBuilder& model_builder,
    const std::string& node_name,
    emscripten::val input,         // Shape: [batch_size, sequence_length, num_heads, head_size]
    emscripten::val cos_cache,     // Shape: [max_sequence_length, head_size / 2]
    emscripten::val sin_cache,     // Shape: [max_sequence_length, head_size / 2]
    emscripten::val position_ids,  // Shape: [batch_size, sequence_length] or [1]
    int32_t input_data_type,
    uint32_t batch_size,
    uint32_t sequence_length,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t rotary_embedding_dim,
    bool interleaved,
    bool has_position_ids,
    bool position_ids_is_offset,
    emscripten::val& output) {
  emscripten::val wnn_builder = model_builder.GetBuilder();
  ORT_RETURN_IF_NOT(head_size >= rotary_embedding_dim,
                    "Rotary embedding dimension must be less than or equal to head_size");
  const uint32_t half_rotary_embedding_dim = rotary_embedding_dim / 2;

  // Split the input to perform the rotary embedding only on a subregion of the tensor if needed.
  emscripten::val partial_input0 = input;
  emscripten::val partial_input1 = emscripten::val::undefined();
  if (head_size > rotary_embedding_dim) {
    const std::vector<uint32_t> splits{rotary_embedding_dim, head_size - rotary_embedding_dim};
    emscripten::val split_input_options = emscripten::val::object();
    split_input_options.set("label", node_name + "_rotary_split_input");
    split_input_options.set("axis", 3);
    emscripten::val split = wnn_builder.call<emscripten::val>(
        "split", input, emscripten::val::array(splits), split_input_options);
    partial_input0 = split[0];
    partial_input1 = split[1];
  }

  // Split the partial input0 data into 2 equal parts.
  const std::vector<uint32_t> new_partial_input0_shape =
      interleaved ? std::vector<uint32_t>({batch_size, sequence_length, num_heads, half_rotary_embedding_dim, 2})
                  : std::vector<uint32_t>({batch_size, sequence_length, num_heads, 2, half_rotary_embedding_dim});
  emscripten::val reshape_partial_input0_options = emscripten::val::object();
  reshape_partial_input0_options.set("label", node_name + "_rotary_reshape_partial_input0");
  partial_input0 = wnn_builder.call<emscripten::val>(
      "reshape", partial_input0, emscripten::val::array(new_partial_input0_shape), reshape_partial_input0_options);

  // Split partial input0.
  const int split_axis = interleaved ? 4 : 3;
  emscripten::val split_partial_input0_options = emscripten::val::object();
  split_partial_input0_options.set("label", node_name + "_rotary_split_partial_input0");
  split_partial_input0_options.set("axis", split_axis);
  emscripten::val split_partial_input0 = wnn_builder.call<emscripten::val>(
      "split", partial_input0, 2, split_partial_input0_options);

  // Swap the two halves and join them together.
  emscripten::val concat_partial_input0_options = emscripten::val::object();
  concat_partial_input0_options.set("label", node_name + "_rotary_concat_partial_input0");
  emscripten::val concated_partial_input0 = wnn_builder.call<emscripten::val>(
      "concat", split_partial_input0.call<emscripten::val>("reverse"), split_axis, concat_partial_input0_options);

  emscripten::val gather_position_ids = position_ids;
  if (position_ids_is_offset) {
    // Generate a sequence from 0 to sequence_length and add the offset to it.
    const std::vector<uint32_t> position_ids_range_shape = {1, sequence_length};
    std::string typed_array_name = "BigInt64Array";
    int position_ids_data_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    const bool is_int64_supported = model_builder.IsInt64Supported();
    if (!is_int64_supported) {
      typed_array_name = "Int32Array";
      position_ids_data_type = ONNX_NAMESPACE::TensorProto_DataType_INT32;
    }
    emscripten::val position_ids_range_buffer = emscripten::val::global(typed_array_name.c_str()).new_(sequence_length);
    for (uint32_t i = 0; i < sequence_length; i++) {
      position_ids_range_buffer.set(i, is_int64_supported ? emscripten::val::global("BigInt")(i) : emscripten::val(i));
    }
    emscripten::val position_ids_range_desc = emscripten::val::object();
    position_ids_range_desc.set("shape", emscripten::val::array(position_ids_range_shape));
    position_ids_range_desc.set("dimensions", emscripten::val::array(position_ids_range_shape));
    ORT_RETURN_IF_NOT(SetWebnnDataType(position_ids_range_desc, position_ids_data_type),
                      "WebNN backend does not support data type: ", position_ids_data_type);
    emscripten::val position_ids_range = wnn_builder.call<emscripten::val>(
        "constant", position_ids_range_desc, position_ids_range_buffer);
    emscripten::val position_ids_add_range_options = emscripten::val::object();
    position_ids_add_range_options.set("label", node_name + "_rotary_position_ids_add_range");
    gather_position_ids = wnn_builder.call<emscripten::val>(
        "add", position_ids, position_ids_range, position_ids_add_range_options);
  }

  // Gather the cosine/sine values based on the position_ids (if it presents).
  emscripten::val gather_cos = cos_cache;
  emscripten::val gather_sin = sin_cache;
  if (has_position_ids) {
    emscripten::val gather_cos_options = emscripten::val::object();
    emscripten::val gather_sin_options = emscripten::val::object();
    gather_cos_options.set("label", node_name + "_rotary_gather_cos");
    gather_sin_options.set("label", node_name + "_rotary_gather_sin");
    gather_cos_options.set("axis", 0);
    gather_sin_options.set("axis", 0);
    gather_cos = wnn_builder.call<emscripten::val>("gather", gather_cos, gather_position_ids, gather_cos_options);
    gather_sin = wnn_builder.call<emscripten::val>("gather", gather_sin, gather_position_ids, gather_sin_options);
  } else {
    // When position_ids is not provided, slice the cos/sin cache to get the first sequence_length rows.
    // cos_cache/sin_cache shape: [max_sequence_length, half_rotary_embedding_dim]
    // After slice: [sequence_length, half_rotary_embedding_dim]
    emscripten::val slice_cos_options = emscripten::val::object();
    emscripten::val slice_sin_options = emscripten::val::object();
    slice_cos_options.set("label", node_name + "_rotary_slice_cos");
    slice_sin_options.set("label", node_name + "_rotary_slice_sin");
    const std::vector<uint32_t> slice_starts = {0, 0};
    const std::vector<uint32_t> slice_sizes = {sequence_length, half_rotary_embedding_dim};
    gather_cos = wnn_builder.call<emscripten::val>("slice", gather_cos,
                                                   emscripten::val::array(slice_starts),
                                                   emscripten::val::array(slice_sizes),
                                                   slice_cos_options);
    gather_sin = wnn_builder.call<emscripten::val>("slice", gather_sin,
                                                   emscripten::val::array(slice_starts),
                                                   emscripten::val::array(slice_sizes),
                                                   slice_sin_options);
  }

  // Reshape and broadcast them to match the number of heads of the input data.
  const std::vector<uint32_t> reshaped_cos_sin_shape =
      interleaved ? std::vector<uint32_t>({batch_size, sequence_length, 1, half_rotary_embedding_dim, 1})
                  : std::vector<uint32_t>({batch_size, sequence_length, 1, 1, half_rotary_embedding_dim});
  emscripten::val reshape_gather_cos_options = emscripten::val::object();
  emscripten::val reshape_gather_sin_options = emscripten::val::object();
  reshape_gather_cos_options.set("label", node_name + "_rotary_reshape_gather_cos");
  reshape_gather_sin_options.set("label", node_name + "_rotary_reshape_gather_sin");
  gather_cos = wnn_builder.call<emscripten::val>(
      "reshape", gather_cos, emscripten::val::array(reshaped_cos_sin_shape), reshape_gather_cos_options);
  gather_sin = wnn_builder.call<emscripten::val>(
      "reshape", gather_sin, emscripten::val::array(reshaped_cos_sin_shape), reshape_gather_sin_options);

  // Multiply the non-rotated data with the cosine and the rotated data with the sine.
  emscripten::val mul_cos_options = emscripten::val::object();
  mul_cos_options.set("label", node_name + "_rotary_mul_cos");
  emscripten::val mul_cos = wnn_builder.call<emscripten::val>(
      "mul", partial_input0, gather_cos, mul_cos_options);
  emscripten::val mul_sin_options = emscripten::val::object();
  mul_sin_options.set("label", node_name + "_rotary_mul_sin");
  emscripten::val mul_sin = wnn_builder.call<emscripten::val>(
      "mul", concated_partial_input0, gather_sin, mul_sin_options);

  // Create a vector that contains the sign values {-1, 1}.
  emscripten::val sign_buffer = emscripten::val::undefined();
  const std::vector<uint32_t> sign_shape = interleaved ? std::vector<uint32_t>({1, 1, 1, 2})
                                                       : std::vector<uint32_t>({1, 1, 2, 1});
  emscripten::val sign_constant_desc = emscripten::val::object();
  sign_constant_desc.set("shape", emscripten::val::array(sign_shape));
  sign_constant_desc.set("dimensions", emscripten::val::array(sign_shape));
  ORT_RETURN_IF_NOT(SetWebnnDataType(sign_constant_desc, input_data_type),
                    "WebNN backend does not support data type: ", input_data_type);
  if (input_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    sign_buffer = emscripten::val::global("Float32Array").new_(2);
    sign_buffer.set(0, -1.0f);
    sign_buffer.set(1, 1.0f);
  } else if (input_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    if (model_builder.IsFloat16ArrayAvailable()) {
      sign_buffer = emscripten::val::global("Float16Array").new_(2);
      sign_buffer.set(0, -1.0f);
      sign_buffer.set(1, 1.0f);
    } else {
      sign_buffer = emscripten::val::global("Uint16Array").new_(2);
      sign_buffer.set(0, PackFloat32ToUint16AsFloat16(-1.0f));
      sign_buffer.set(1, PackFloat32ToUint16AsFloat16(1.0f));
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported input data type for rotary embedding: ",
                           input_data_type);
  }
  emscripten::val sign_constant = wnn_builder.call<emscripten::val>("constant", sign_constant_desc, sign_buffer);

  // Multiply the broadcasted sign values with the rotated input.
  emscripten::val mul_sign_options = emscripten::val::object();
  mul_sign_options.set("label", node_name + "_rotary_mul_sign");
  mul_sin = wnn_builder.call<emscripten::val>("mul", mul_sin, sign_constant, mul_sign_options);

  // Reshape mul_cos and mul_sin to (batch_size, sequence_length, num_heads, rotary_embedding_dim).
  const std::vector<uint32_t> reshaped_mul_cos_sin_shape =
      {batch_size, sequence_length, num_heads, rotary_embedding_dim};
  emscripten::val reshape_mul_cos_sin_options = emscripten::val::object();
  reshape_mul_cos_sin_options.set("label", node_name + "_rotary_reshape_mul_cos_sin");
  mul_cos = wnn_builder.call<emscripten::val>(
      "reshape", mul_cos, emscripten::val::array(reshaped_mul_cos_sin_shape), reshape_mul_cos_sin_options);
  mul_sin = wnn_builder.call<emscripten::val>(
      "reshape", mul_sin, emscripten::val::array(reshaped_mul_cos_sin_shape), reshape_mul_cos_sin_options);

  // Add the multiplied cos and sin values together.
  emscripten::val add_mul_cos_sin_options = emscripten::val::object();
  add_mul_cos_sin_options.set("label", node_name + "_rotary_add_mul_cos_sin");
  output = wnn_builder.call<emscripten::val>(
      "add", mul_cos, mul_sin, add_mul_cos_sin_options);

  // Join the added values with the rest of the input.
  if (head_size != rotary_embedding_dim) {
    emscripten::val concat_back_input_options = emscripten::val::object();
    concat_back_input_options.set("label", node_name + "_rotary_concat_back_input");
    emscripten::val concat_inputs = emscripten::val::array();
    concat_inputs.call<void>("push", output);
    concat_inputs.call<void>("push", partial_input1);
    output = wnn_builder.call<emscripten::val>("concat", concat_inputs, 3, concat_back_input_options);
  }

  return Status::OK();
}

/*
    ScaledDotProductAttention Subgraph: The basis for MultiHeadAttention and GroupQueryAttention
    inputs: query, key, value, scale, attention mask, and reshape_output_shape (for reshape)
    Abbreviations: B is batch_size, S is query sequence_length, kv_S is key/value sequence length,
                   N is number of attention heads, H is head size, W is hidden_size

  query         key
    |            |
    +---matmul---+    scale
          |             |
          +-----div-----+   attn_mask
                 |             |
                 +-----add-----+        value
                        |                 |
                        +------matmul-----+
                                 |
                   (0,2,1,3) transpose B,H,S,N -> B,S,H,N
                                 |
                              reshape B,S,H,N -> B,S,W
                                 |
                               output
*/
inline emscripten::val ScaledDotProductAttention(ModelBuilder& model_builder, const Node& node,
                                                 const logging::Logger& logger, emscripten::val query,
                                                 emscripten::val key, emscripten::val value, emscripten::val scale,
                                                 emscripten::val attn_mask,
                                                 std::vector<uint32_t> reshape_output_shape) {
  emscripten::val common_options = emscripten::val::object();
  // B,H,S,N * B,H,kv_S,N = B,H,S,kv_S
  common_options.set("label", node.Name() + "_/Attention/qkv/matmul_1");
  emscripten::val matmul_output =
      model_builder.GetBuilder().call<emscripten::val>("matmul", query, key, common_options);

  common_options.set("label", node.Name() + "_/Attention/qkv/div");
  emscripten::val div_output =
      model_builder.GetBuilder().call<emscripten::val>("mul", matmul_output, scale, common_options);

  emscripten::val softmax_input = div_output;
  if (attn_mask != emscripten::val::undefined()) {
    common_options.set("label", node.Name() + "_/Attention/attn_mask/softmax_input");
    softmax_input = model_builder.GetBuilder().call<emscripten::val>("add", div_output, attn_mask, common_options);
  }

  common_options.set("label", node.Name() + "_/Attention/attn_mask/softmax_input");
  int32_t softmax_axis = 3;
  emscripten::val softmax_output =
      model_builder.GetBuilder().call<emscripten::val>("softmax", softmax_input, softmax_axis, common_options);

  // B,H,S,kv_S * B,H,kv_S,N = B,H,S,N
  common_options.set("label", node.Name() + "_/Attention/qkv/matmul_2");
  emscripten::val attn_output =
      model_builder.GetBuilder().call<emscripten::val>("matmul", softmax_output, value, common_options);

  emscripten::val options = emscripten::val::object();
  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  options.set("label", node.Name() + "_/Attention/qkv/transpose");
  attn_output = model_builder.GetBuilder().call<emscripten::val>("transpose", attn_output, options);

  common_options.set("label", node.Name() + "_/Attention/qkv/reshape");
  attn_output = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", attn_output, emscripten::val::array(reshape_output_shape), common_options);

  return attn_output;
}

}  // namespace webnn
}  // namespace onnxruntime

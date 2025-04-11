// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace webnn {
/*
    ScaledDotProductAttention Subgraph: The basis for MultiHeadAttention and GroupQueryAttention
    inputs: query, key, value, scale, attention mask, and reshape_output_shape (for reshape)
    Abbreviatios: B is batch_size, S is query sequence_length, kv_S is key/value sequence length,
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

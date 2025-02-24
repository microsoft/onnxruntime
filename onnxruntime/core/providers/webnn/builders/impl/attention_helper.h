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
emscripten::val ScaledDotProductAttention(ModelBuilder& model_builder, const Node& node, const logging::Logger& logger,
                                          emscripten::val query, emscripten::val key, emscripten::val value,
                                          emscripten::val scale, emscripten::val attn_mask,
                                          std::vector<uint32_t> reshape_output_shape) {
  emscripten::val common_options = emscripten::val::object();
  // B,H,S,N * B,H,kv_S,N = B,H,S,kv_S
  common_options.set("label", node.Name() + "/Attention/qkv/matmul_1");
  emscripten::val matmul_output =
      model_builder.GetBuilder().call<emscripten::val>("matmul", query, key, common_options);

  common_options.set("label", node.Name() + "/Attention/qkv/div");
  emscripten::val div_output =
      model_builder.GetBuilder().call<emscripten::val>("mul", matmul_output, scale, common_options);

  common_options.set("label", node.Name() + "/Attention/attn_mask/softmax_input");
  emscripten::val softmax_input =
      model_builder.GetBuilder().call<emscripten::val>("add", div_output, attn_mask, common_options);

  common_options.set("label", node.Name() + "/Attention/attn_mask/softmax_input");
  int32_t softmax_axis = 3;
  emscripten::val softmax_output =
      model_builder.GetBuilder().call<emscripten::val>("softmax", softmax_input, softmax_axis, common_options);

    // B,H,S,kv_S * B,H,kv_S,N = B,H,S,N
  common_options.set("label", node.Name() + "/Attention/qkv/matmul_2");
  emscripten::val attn_output =
      model_builder.GetBuilder().call<emscripten::val>("matmul", softmax_output, value, common_options);

  emscripten::val options = emscripten::val::object();
  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  options.set("label", node.Name() + "/Attention/qkv/transpose");
  attn_output = model_builder.GetBuilder().call<emscripten::val>("transpose", attn_output, options);

  common_options.set("label", node.Name() + "/Attention/qkv/reshape");
  attn_output = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", attn_output, emscripten::val::array(reshape_output_shape), common_options);

  return attn_output;
}

}  // namespace webnn
}  // namespace onnxruntime

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
#include "attention_helper.h"

namespace onnxruntime {
namespace webnn {

class MultiHeadAttentionOpBuilder : public BaseOpBuilder {
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

/** MultiHeadAttention SubGraph.
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size, P is past_sequence_length
               N is number of attention heads, H is head size, and W=N*H, h=Sqrt(H)
               B and S could be symbolic. ? means it is optional.
    MHA inputs: query, key value, past_key, past_value, seqlens_k, total_sequence_length
    Notes: If the datatype of the inputs (qkv and past kv) is float16, we cast them to float32 to ensure data precision.

                 query     key     value
                   |        |        |
           q_Reshape   k_Reshape   v_Reshape  (shape=B,S,H,N)
                   |        |        |
          q_Transpose  k_Transpose v_Transpose
           (0,2,1,3)    (0,2,3,1)   (perm=0,2,1,3)
             \           /           |     past_key
              \         /            |        |
present_key<---\----ScatterND <------|--------+
               |      |              |        |
               |  opt_k_transpose?   |    seqlens_k
               \  (0,1,3,2)          |        |
                \    /               |        +----past_value
                qk_MatMul            |       /
                     |    [B=h]      |      /
                     |   /           |     /
                  qk_Div         ScatterND -----> present_value
                      |              |
                      |              /
                     Add <----------/---------------finfo_min_mask
                      |            /
                    Softmax       /
                       \         /
                        \       /
                      qkv_MatMul
                             |
                          Transpose (perm=0,2,1,3)
                             |
                          Reshape---(shape=B,P,W)
                             |
                           output
*/

Status MultiHeadAttentionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                          const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  if (op_type != "MultiHeadAttention") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported normalization op: ", op_type);
  }

  const auto& input_defs = node.InputDefs();
  emscripten::val common_options = emscripten::val::object();
  emscripten::val options = emscripten::val::object();
  bool k_reshape_skip, v_reshape_skip, q_transpose_skip, k_transpose_skip, v_transpose_skip;
  emscripten::val query_input, key_input, value_input, attention_bias;

  query_input = model_builder.GetOperand(input_defs[0]->Name());
  if (input_defs[0]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/MHA/preprocess/cast/query_input");
    query_input = model_builder.GetBuilder().call<emscripten::val>("cast", query_input, emscripten::val("float32"),
                                                                   common_options);
  }

  std::vector<int64_t> input_q_shape, input_k_shape, input_v_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_q_shape, logger), "Cannot get query shape");
  const auto q_rank = input_q_shape.size();
  if (q_rank == 3) {  // Query with shape (batch_size, sequence_length, hidden_size)
    key_input = model_builder.GetOperand(input_defs[1]->Name());
    if (input_defs[1]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
      common_options.set("label", node.Name() + "/MHA/preprocess/cast/key_input");
      key_input = model_builder.GetBuilder().call<emscripten::val>("cast", key_input, emscripten::val("float32"),
                                                                   common_options);
    }

    ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_k_shape, logger), "Cannot get key shape");
    const auto k_rank = input_k_shape.size();

    if (k_rank == 5) {  // packed KV with shape (batch_size, kv_sequence_length, num_heads, 2, head_size)
      k_reshape_skip = false;
      v_reshape_skip = false;
      options.set("axis", 3);
      emscripten::val output_array = model_builder.GetBuilder().call<emscripten::val>("split", key_input, 2, options);
      key_input = output_array[0];
      value_input = output_array[1];
    } else {
      if (k_rank == 3) {  // Key with shape (batch_size, kv_sequence_length, hidden_size)
        k_reshape_skip = false;
      } else if (k_rank == 4) {  // past_key with shape (batch_size, num_heads, kv_sequence_length, head_size)
        k_reshape_skip = true;
      }
      value_input = model_builder.GetOperand(input_defs[2]->Name());
      if (input_defs[2]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
        common_options.set("label", node.Name() + "/MHA/preprocess/cast/value_input");
        value_input = model_builder.GetBuilder().call<emscripten::val>("cast", value_input, emscripten::val("float32"),
                                                                       common_options);
      }

      ORT_RETURN_IF_NOT(GetShape(*input_defs[2], input_v_shape, logger), "Cannot get value shape");
      const auto v_rank = input_v_shape.size();
      if (v_rank == 3) {  // Value with shape (batch_size, kv_sequence_length, v_hidden_size)
        v_reshape_skip = false;
      } else if (v_rank == 4) {  // past_value with shape (batch_size, num_heads, kv_sequence_length, head_size)
        v_reshape_skip = true;
      }
    }
  } else {  // packed QKV with shape (batch_size, kv_sequence_length, num_heads, 3, head_size)
    k_reshape_skip = false;
    v_reshape_skip = false;
    options.set("axis", 3);
    emscripten::val output_array = model_builder.GetBuilder().call<emscripten::val>("split", query_input, 3, options);
    query_input = output_array[0];
    key_input = output_array[1];
    value_input = output_array[2];
  }

  emscripten::val attention_bias = model_builder.GetOperand(input_defs[5]->Name());

  NodeAttrHelper helper(node);
  uint32_t num_heads = helper.Get("num_heads", 32);

  uint32_t qkv_batch_size = SafeInt<uint32_t>(input_q_shape[0]);
  uint32_t qkv_sequence_length = SafeInt<uint32_t>(input_q_shape[1]);
  uint32_t qkv_hidden_size = SafeInt<uint32_t>(input_q_shape[2]);
  uint32_t head_size = SafeInt<uint32_t>(qkv_hidden_size / num_heads);
  uint32_t past_sequence_length = SafeInt<uint32_t>(input_past_k_shape[2]);

  float scale = helper.Get("scale", static_cast<float>(sqrt(head_size)));

  std::vector<uint32_t> reshape_output_shape = {qkv_batch_size, qkv_sequence_length, qkv_hidden_size};
  std::vector<uint32_t> reshape_tensor_shape = {qkv_batch_size, qkv_sequence_length, num_heads, head_size};

  // query_input -> reshape(B,S,N,H) -> transpose(B,N,S,H) -> new_query
  common_options.set("label", node.Name() + "/MHA/query/reshape");
  emscripten::val reshaped_query = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", query_input, emscripten::val::array(reshape_tensor_shape), common_options);

  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  options.set("label", node.Name() + "/MHA/query/transpose");
  emscripten::val new_query = model_builder.GetBuilder().call<emscripten::val>("transpose", reshaped_query, options);

  emscripten::val present_key, present_value;
  if (!k_reshape_skip) {
    common_options.set("label", node.Name() + "/MHA/key/reshape_1");
    present_key = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", key_input, emscripten::val::array(reshape_tensor_shape), common_options);

    options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
    options.set("label", node.Name() + "/MHA/key/transpose");
    present_key = model_builder.GetBuilder().call<emscripten::val>("transpose", present_key, options);

    if (input_defs[6]->Name() != "") {
      emscripten::val past_key_input = model_builder.GetOperand(input_defs[6]->Name());
      if (input_defs[6]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
        common_options.set("label", node.Name() + "/MHA/preprocess/cast/past_key_input");
        past_key_input = model_builder.GetBuilder().call<emscripten::val>("cast", past_key_input,
                                                                          emscripten::val("float32"), common_options);
      }

      common_options.set("label", node.Name() + "/MHA/key/concat");
      std::vector<emscripten::val> inputs({present_key, past_key_input});
      uint32_t axis = 2;
      present_key = model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(inputs), axis,
                                                                     common_options);
    }
  } else {
    present_key = key_input;
  }
  if (node.OutputDefs()[1]->Name() != "") {
    if (node.OutputDefs()[1]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
      common_options.set("label", node.Name() + "/MHA/postprocess/cast/present_key");
      present_key = model_builder.GetBuilder().call<emscripten::val>("cast", present_key, emscripten::val("float16"),
                                                                     common_options);
    }
    model_builder.AddOperand(node.OutputDefs()[1]->Name(), std::move(present_key));
  }

  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 1, 3, 2})));
  options.set("label", node.Name() + "/MHA/key/transpose");
  emscripten::val new_key = model_builder.GetBuilder().call<emscripten::val>("transpose", present_key, options);

  if (!v_reshape_skip) {
    common_options.set("label", node.Name() + "/MHA/value/reshape_1");
    emscripten::val present_value = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", value_input, emscripten::val::array(reshape_tensor_shape), common_options);

    options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
    options.set("label", node.Name() + "/MHA/value/transpose");
    present_value = model_builder.GetBuilder().call<emscripten::val>("transpose", present_value, options);

    if (input_defs[7]->Name() != "") {
      emscripten::val past_value_input = model_builder.GetOperand(input_defs[7]->Name());
      if (input_defs[7]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
        common_options.set("label", node.Name() + "/MHA/preprocess/cast/past_value_input");
        past_value_input = model_builder.GetBuilder().call<emscripten::val>("cast", past_value_input,
                                                                            emscripten::val("float32"), common_options);
      }

      common_options.set("label", node.Name() + "/MHA/value/concat");
      std::vector<emscripten::val> inputs({present_value, past_value_input});
      uint32_t axis = 2;
      present_value = model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(inputs), axis,
                                                                       common_options);
    }
  } else {
    present_value = value_input;
  }
  if (node.OutputDefs()[2]->Name() != "") {
    if (node.OutputDefs()[2]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
      common_options.set("label", node.Name() + "/MHA/postprocess/cast/present_value");
      present_value = model_builder.GetBuilder().call<emscripten::val>("cast", present_value,
                                                                       emscripten::val("float16"), common_options);
    }
    model_builder.AddOperand(node.OutputDefs()[2]->Name(), std::move(present_value));
  }

  // common_options.set("label", node.Name() + "/MHA/qkv/matmul_1");
  // emscripten::val matmul_output =
  //     model_builder.GetBuilder().call<emscripten::val>("matmul", new_query, new_key, common_options);

  emscripten::val desc_scale = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc_scale, ONNX_NAMESPACE::TensorProto_DataType_FLOAT), "Unsupported data type");
  emscripten::val dims_scale = emscripten::val::array(std::vector<uint32_t>({1}));
  desc_scale.set("dimensions", dims_scale);
  desc_scale.set("shape", dims_scale);
  emscripten::val scale_buffer =
      emscripten::val::global("Float32Array").new_(emscripten::val::array(std::vector<float>({scale})));
  emscripten::val scale_constant =
      model_builder.GetBuilder().call<emscripten::val>("constant", desc_scale, scale_buffer);

  emscripten::val output =
      ScaledDotProductAttention(model_builder, node, logger, new_query, new_key, present_value, scale_constant,
                                attention_bias, reshape_output_shape);

  // common_options.set("label", node.Name() + "/MHA/qkv/div");
  // emscripten::val div_output =
  //     model_builder.GetBuilder().call<emscripten::val>("div", matmul_output, scale_constant, common_options);

  // common_options.set("label", node.Name() + "/MHA/attn_mask/softmax_input");
  // emscripten::val softmax_input =
  //     model_builder.GetBuilder().call<emscripten::val>("add", div_output, attention_bias, common_options);

  // common_options.set("label", node.Name() + "/MHA/attn_mask/softmax_input");
  // int32_t softmax_axis = 3;
  // emscripten::val softmax_output =
  //     model_builder.GetBuilder().call<emscripten::val>("softmax", softmax_input, softmax_axis, common_options);

  // common_options.set("label", node.Name() + "/MHA/qkv/matmul_2");
  // emscripten::val attn_output =
  //     model_builder.GetBuilder().call<emscripten::val>("matmul", softmax_output, present_value, common_options);

  // options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  // options.set("label", node.Name() + "/MHA/qkv/transpose");
  // emscripten::val transposed_attn_output =
  //     model_builder.GetBuilder().call<emscripten::val>("transpose", attn_output, options);

  // common_options.set("label", node.Name() + "/MHA/qkv/reshape");
  // emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
  //     "reshape", transposed_attn_output, emscripten::val::array(reshape_output_shape), common_options);

  if (node.OutputDefs()[0]->Type() == onnx::Utils::DataTypeUtils::ToType("float16")) {
    common_options.set("label", node.Name() + "/MHA/postprocess/cast/output");
    output =
        model_builder.GetBuilder().call<emscripten::val>("cast", output, emscripten::val("float16"), common_options);
  }
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));

  return Status::OK();
}

// Operator support related.

bool MultiHeadAttentionOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
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

bool MultiHeadAttentionOpBuilder::HasSupportedInputsImpl(const InitializedTensorSet& /* initializers */,
                                                         const Node& node, const emscripten::val& wnn_limits,
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
      "MultiHeadAttention",
  };

  op_registrations.builders.push_back(std::make_unique<MultiHeadAttentionOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime

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

class MultiHeadAttentionOpBuilder : public BaseOpBuilder {
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

/** MultiHeadAttention SubGraph.
 Abbreviations: B is batch_size, S is sequence_length, W is hidden_size
                N is number of attention heads, H is head size
    Notes: If the datatype of the inputs (qkv and past kv) is float16, we cast them to float32 to ensure data precision.

         query       key              value
           |          |                 |
        Reshape    Reshape           Reshape  (shape=B,S,H,N)
           |          |                 |
       Transpose  Transpose         Transpose  (perm=0,2,1,3)
            \         |  past_key       |
             \        |  /              |
present_key<--\-----Concat              |  past_value
               \      |                 |   /
               |      |               Concat----> present_value
               |      |                 |
               |  k_Transpose           |       attention_bias
               |   (0,1,3,2)            |          /
               |      |                 |         /
            +---------------------------------------+
            |        ScaledDotProductAttention      |
            +---------------------------------------+
                             |
                           output
*/

Status MultiHeadAttentionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val common_options = emscripten::val::object();
  emscripten::val options = emscripten::val::object();
  emscripten::val transpose_options = emscripten::val::object();
  emscripten::val split_options = emscripten::val::object();

  bool k_reshape_skip, v_reshape_skip;
  emscripten::val query_input, key_input, value_input;

  NodeAttrHelper helper(node);
  const uint32_t num_heads = helper.Get("num_heads", 0);

  query_input = model_builder.GetOperand(input_defs[0]->Name());

  int32_t input_query_type = 0;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_query_type, logger), "Could not get input data type.");
  int32_t output_type = 0;
  ORT_RETURN_IF_NOT(GetType(*node.OutputDefs()[0], output_type, logger), "Could not get input data type.");

  if (input_query_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    common_options.set("label", node.Name() + "_/MHA/preprocess/cast/query_input");
    query_input = model_builder.GetBuilder().call<emscripten::val>("cast", query_input, emscripten::val("float32"),
                                                                   common_options);
  }

  std::vector<int64_t> input_q_shape, input_k_shape, input_v_shape;
  uint32_t batch_size, sequence_length, kv_sequence_length, hidden_size, head_size;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_q_shape, logger), "Cannot get query shape");
  const auto q_rank = input_q_shape.size();
  if (q_rank == 3) {  // Query with shape (batch_size, sequence_length, hidden_size)
    hidden_size = SafeInt<uint32_t>(input_q_shape[2]);
    head_size = hidden_size / num_heads;
    key_input = model_builder.GetOperand(input_defs[1]->Name());
    if (input_query_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      common_options.set("label", node.Name() + "_/MHA/preprocess/cast/key_input");
      key_input = model_builder.GetBuilder().call<emscripten::val>("cast", key_input, emscripten::val("float32"),
                                                                   common_options);
    }

    ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_k_shape, logger), "Cannot get key shape");
    const auto k_rank = input_k_shape.size();

    if (k_rank == 5) {  // packed KV with shape (batch_size, kv_sequence_length, num_heads, 2, head_size)
      kv_sequence_length = SafeInt<uint32_t>(input_k_shape[1]);
      k_reshape_skip = false;
      v_reshape_skip = false;
      split_options.set("axis", 3);
      split_options.set("label", node.Name() + "_/MHA/key/split");
      emscripten::val output_array =
          model_builder.GetBuilder().call<emscripten::val>("split", key_input, 2, split_options);
      key_input = output_array[0];
      value_input = output_array[1];
    } else {
      if (k_rank == 3) {  // Key with shape (batch_size, kv_sequence_length, hidden_size)
        kv_sequence_length = SafeInt<uint32_t>(input_k_shape[1]);
        k_reshape_skip = false;
      } else {  // past_key with shape (batch_size, num_heads, kv_sequence_length, head_size)
        kv_sequence_length = SafeInt<uint32_t>(input_k_shape[2]);
        k_reshape_skip = true;
      }
      value_input = model_builder.GetOperand(input_defs[2]->Name());
      if (input_query_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        common_options.set("label", node.Name() + "_/MHA/preprocess/cast/value_input");
        value_input = model_builder.GetBuilder().call<emscripten::val>("cast", value_input, emscripten::val("float32"),
                                                                       common_options);
      }

      ORT_RETURN_IF_NOT(GetShape(*input_defs[2], input_v_shape, logger), "Cannot get value shape");
      const auto v_rank = input_v_shape.size();
      if (v_rank == 3) {  // Value with shape (batch_size, kv_sequence_length, v_hidden_size)
        v_reshape_skip = false;
      } else {  // past_value with shape (batch_size, num_heads, kv_sequence_length, head_size)
        v_reshape_skip = true;
      }
    }
  } else {  // packed QKV with shape (batch_size, kv_sequence_length, num_heads, 3, head_size)
    kv_sequence_length = SafeInt<uint32_t>(input_q_shape[2]);
    head_size = SafeInt<uint32_t>(input_q_shape[4]);
    hidden_size = num_heads * head_size;
    k_reshape_skip = false;
    v_reshape_skip = false;
    split_options.set("axis", 3);
    split_options.set("label", node.Name() + "_/MHA/query/split");
    emscripten::val output_array =
        model_builder.GetBuilder().call<emscripten::val>("split", query_input, 3, split_options);
    query_input = output_array[0];
    key_input = output_array[1];
    value_input = output_array[2];
  }

  emscripten::val attention_bias = emscripten::val::undefined();
  if (!TensorExists(input_defs, 5)) {
    attention_bias = model_builder.GetOperand(input_defs[5]->Name());
    if (input_query_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      common_options.set("label", node.Name() + "_/MHA/preprocess/cast/attention_bias");
      attention_bias = model_builder.GetBuilder().call<emscripten::val>("cast", attention_bias,
                                                                        emscripten::val("float32"), common_options);
    }
  }

  batch_size = SafeInt<uint32_t>(input_q_shape[0]);
  sequence_length = SafeInt<uint32_t>(input_q_shape[1]);

  const float scale_value = helper.Get("scale", 1 / sqrt(static_cast<float>(head_size)));

  const std::vector<uint32_t> reshape_output_shape = {batch_size, sequence_length, hidden_size};
  const std::vector<uint32_t> q_reshape_tensor_shape = {batch_size, sequence_length, num_heads, head_size};
  const std::vector<uint32_t> reshape_tensor_shape = {batch_size, kv_sequence_length, num_heads, head_size};

  // query_input -> reshape(B,S,N,H) -> transpose(B,N,S,H) -> new_query
  common_options.set("label", node.Name() + "_/MHA/query/reshape");
  emscripten::val reshaped_query = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", query_input, emscripten::val::array(q_reshape_tensor_shape), common_options);

  transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  transpose_options.set("label", node.Name() + "_/MHA/query/transpose");
  emscripten::val new_query =
      model_builder.GetBuilder().call<emscripten::val>("transpose", reshaped_query, transpose_options);

  emscripten::val present_key, present_value;
  if (!k_reshape_skip) {
    common_options.set("label", node.Name() + "_/MHA/key/reshape_1");
    present_key = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", key_input, emscripten::val::array(reshape_tensor_shape), common_options);

    transpose_options.set("label", node.Name() + "_/MHA/key/transpose");
    present_key = model_builder.GetBuilder().call<emscripten::val>("transpose", present_key, transpose_options);

    if (TensorExists(input_defs, 6)) {
      emscripten::val past_key_input = model_builder.GetOperand(input_defs[6]->Name());
      if (input_query_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        common_options.set("label", node.Name() + "_/MHA/preprocess/cast/past_key_input");
        past_key_input = model_builder.GetBuilder().call<emscripten::val>("cast", past_key_input,
                                                                          emscripten::val("float32"), common_options);
      }

      common_options.set("label", node.Name() + "_/MHA/key/concat");
      std::vector<emscripten::val> inputs({past_key_input, present_key});
      uint32_t axis = 2;
      present_key = model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(inputs), axis,
                                                                     common_options);
    }
  } else {
    present_key = key_input;
  }

  transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 1, 3, 2})));
  transpose_options.set("label", node.Name() + "_/MHA/key/transpose");
  emscripten::val new_key =
      model_builder.GetBuilder().call<emscripten::val>("transpose", present_key, transpose_options);

  if (!v_reshape_skip) {
    common_options.set("label", node.Name() + "_/MHA/value/reshape_1");
    present_value = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", value_input, emscripten::val::array(reshape_tensor_shape), common_options);

    transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
    transpose_options.set("label", node.Name() + "_/MHA/value/transpose");
    present_value = model_builder.GetBuilder().call<emscripten::val>("transpose", present_value, transpose_options);

    if (TensorExists(input_defs, 7)) {
      emscripten::val past_value_input = model_builder.GetOperand(input_defs[7]->Name());
      if (input_query_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        common_options.set("label", node.Name() + "_/MHA/preprocess/cast/past_value_input");
        past_value_input = model_builder.GetBuilder().call<emscripten::val>("cast", past_value_input,
                                                                            emscripten::val("float32"), common_options);
      }

      common_options.set("label", node.Name() + "_/MHA/value/concat");
      std::vector<emscripten::val> inputs({past_value_input, present_value});
      uint32_t axis = 2;
      present_value = model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(inputs), axis,
                                                                       common_options);
    }
  } else {
    present_value = value_input;
  }

  emscripten::val scale_constant =
      model_builder.CreateOrGetConstant<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, scale_value, {1});

  emscripten::val output = ScaledDotProductAttention(model_builder, node, logger, new_query, new_key, present_value,
                                                     scale_constant, attention_bias, reshape_output_shape);

  if (output_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    common_options.set("label", node.Name() + "_/MHA/postprocess/cast/output");
    output =
        model_builder.GetBuilder().call<emscripten::val>("cast", output, emscripten::val("float16"), common_options);
  }
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));

  if (TensorExists(node.OutputDefs(), 1)) {
    if (output_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      common_options.set("label", node.Name() + "_/MHA/postprocess/cast/present_key");
      present_key = model_builder.GetBuilder().call<emscripten::val>("cast", present_key, emscripten::val("float16"),
                                                                     common_options);
    }
    model_builder.AddOperand(node.OutputDefs()[1]->Name(), std::move(present_key));
  }

  if (TensorExists(node.OutputDefs(), 2)) {
    if (output_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      common_options.set("label", node.Name() + "_/MHA/postprocess/cast/present_value");
      present_value = model_builder.GetBuilder().call<emscripten::val>("cast", present_value,
                                                                       emscripten::val("float16"), common_options);
    }
    model_builder.AddOperand(node.OutputDefs()[2]->Name(), std::move(present_value));
  }

  return Status::OK();
}

// Operator support related.

bool MultiHeadAttentionOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                                                    const WebnnDeviceType /* device_type */,
                                                    const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  NodeAttrHelper helper(node);
  const uint32_t num_heads = helper.Get("num_heads", 0);
  if (num_heads == 0) {
    LOGS(logger, VERBOSE) << "Attributes num_heads is required.";
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get input shape.";
    return false;
  }

  return true;
}

bool MultiHeadAttentionOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                         const emscripten::val& wnn_limits,
                                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();

  std::vector<int32_t> input_types;
  for (int i = 0; i < 10; i++) {
    if (i == 3 || i == 4 || i == 8 || i == 9) {
      if (TensorExists(input_defs, i)) {
        LOGS(logger, VERBOSE)
            << op_type << " does not support input " << i
            << ", as it does not support inputs bias, key_padding_mask, past_sequence_length, or cache_indirection.";
        return false;
      }
    } else {
      if (TensorExists(input_defs, i)) {
        int32_t input_type = 0;
        if (!GetType(*input_defs[i], input_type, logger)) {
          return false;
        }
        input_types.push_back(input_type);
      }
    }
  }

  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }
  return true;
}

bool MultiHeadAttentionOpBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                                          const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  if (TensorExists(output_defs, 3)) {
    LOGS(logger, VERBOSE) << op_type << " does not support output qk.";
    return false;
  }

  bool has_present_k = TensorExists(output_defs, 1);
  bool has_present_v = TensorExists(output_defs, 2);
  if (has_present_k != has_present_v) {  // present_k and present_v must appear together.
    return false;
  }

  int32_t output_type = 0;
  if (has_present_k) {
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
  }

  if (output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return false;
  }
  return true;
}

void CreateMultiHeadAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<MultiHeadAttentionOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}
}  // namespace webnn
}  // namespace onnxruntime

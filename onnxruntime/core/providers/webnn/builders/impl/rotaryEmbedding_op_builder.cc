// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

// WebNN doesn't provide a dedicated op for RotaryEmbedding. Instead, we implement it by using a
// combination of WebNN ops. The decomposed graph is referenced from DML EP at:
// onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorRotaryEmbedding.cpp
/*
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
namespace onnxruntime {
namespace webnn {

class RotaryEmbeddingOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

Status RotaryEmbeddingOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t input_data_type;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_data_type, logger), "Cannot get input type");
  std::vector<int64_t> input_shape;
  std::vector<int64_t> position_ids_shape;
  std::vector<int64_t> cos_cache_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], position_ids_shape, logger), "Cannot get position_ids shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[2], cos_cache_shape, logger), "Cannot get cos_cache shape");
  const bool input_is_4d = input_shape.size() == 4;
  // When position_ids is a 1D tensor, it represents the start offset for each sequence.
  const bool position_ids_is_offset = position_ids_shape.size() == 1;

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val position_ids = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val cos_cache = model_builder.GetOperand(input_defs[2]->Name());
  emscripten::val sin_cache = model_builder.GetOperand(input_defs[3]->Name());

  const auto node_name = node.Name();
  emscripten::val wnn_builder = model_builder.GetBuilder();

  NodeAttrHelper helper(node);
  const bool interleaved = gsl::narrow_cast<bool>(helper.Get("interleaved", 0));
  uint32_t num_heads = helper.Get("num_heads", 0);
  uint32_t rotary_embedding_dim = helper.Get("rotary_embedding_dim", 0);

  // The input is either with 3D tensor shape (batch_size, sequence_length, hidden_size) or
  // 4D tensor shape (batch_size, num_heads, sequence_length, head_size)
  const uint32_t batch_size = static_cast<uint32_t>(input_shape[0]);
  const uint32_t sequence_length = input_is_4d ? static_cast<uint32_t>(input_shape[2])
                                               : static_cast<uint32_t>(input_shape[1]);
  const uint32_t hidden_size = input_is_4d ? static_cast<uint32_t>(input_shape[1] * input_shape[3])
                                           : static_cast<uint32_t>(input_shape[2]);
  const uint32_t head_size = num_heads == 0 ? static_cast<uint32_t>(cos_cache_shape[1]) * 2
                                            : hidden_size / num_heads;
  if (num_heads == 0) {
    num_heads = hidden_size / head_size;
  }
  if (rotary_embedding_dim == 0) {
    rotary_embedding_dim = head_size;
  }

  // First ensure the input has shape (batch_size, num_heads, sequence_length, head_size).
  if (!input_is_4d) {
    const std::vector<uint32_t> new_shape{batch_size, num_heads, sequence_length, head_size};
    emscripten::val reshape_input_options = emscripten::val::object();
    reshape_input_options.set("label", node_name + "_reshape_input");
    input = wnn_builder.call<emscripten::val>(
        "reshape", input, emscripten::val::array(new_shape), reshape_input_options);
  }

  // Split the input to perform the rotary embedding only on a subregion of the tensor if needed.
  // The split inputs will be joined back together at the end.
  emscripten::val partial_input0 = input;
  emscripten::val partial_input1 = emscripten::val::undefined();
  if (head_size != rotary_embedding_dim) {
    const std::vector<uint32_t> splits{rotary_embedding_dim, head_size - rotary_embedding_dim};
    emscripten::val split_input_options = emscripten::val::object();
    split_input_options.set("label", node_name + "_split_input");
    split_input_options.set("axis", 3);
    emscripten::val split = wnn_builder.call<emscripten::val>(
        "split", input, emscripten::val::array(splits), split_input_options);
    partial_input0 = split[0];
    partial_input1 = split[1];
  }

  // Split the partial input0 data into 2 equal parts.
  // Firstly reshape the partial input0.
  const std::vector<uint32_t> new_partial_input0_shape =
      interleaved ? std::vector<uint32_t>({batch_size, sequence_length, num_heads, rotary_embedding_dim / 2, 2})
                  : std::vector<uint32_t>({batch_size, sequence_length, num_heads, 2, rotary_embedding_dim / 2});
  emscripten::val reshape_partial_input0_options = emscripten::val::object();
  reshape_partial_input0_options.set("label", node_name + "_reshape_partial_input0");
  partial_input0 = wnn_builder.call<emscripten::val>(
      "reshape", partial_input0, emscripten::val::array(new_partial_input0_shape), reshape_partial_input0_options);
  // Split partial input0.
  const int split_axis = interleaved ? 4 : 3;
  emscripten::val split_partial_input0_options = emscripten::val::object();
  split_partial_input0_options.set("label", node_name + "_split_partial_input0");
  split_partial_input0_options.set("axis", split_axis);
  emscripten::val split_partial_input0 = wnn_builder.call<emscripten::val>(
      "split", partial_input0, 2, split_partial_input0_options);

  // Swap the two halves and join them together.
  emscripten::val concat_partial_input0_options = emscripten::val::object();
  concat_partial_input0_options.set("label", node_name + "_concat_partial_input0");
  emscripten::val concated_partial_input0 = wnn_builder.call<emscripten::val>(
      "concat", split_partial_input0.call<emscripten::val>("reverse"), split_axis, concat_partial_input0_options);

  if (position_ids_is_offset) {
    // We generate a sequence from 0 to sequence_length and add the offset to it.
    const std::vector<uint32_t> position_ids_range_shape = {1, sequence_length};
    emscripten::val position_ids_range_buffer = emscripten::val::global("BigInt64Array").new_(sequence_length);
    for (uint32_t i = 0; i < sequence_length; i++) {
      position_ids_range_buffer.set(i, emscripten::val::global("BigInt")(i));
    }
    emscripten::val position_ids_range_desc = emscripten::val::object();
    position_ids_range_desc.set("shape", emscripten::val::array(position_ids_range_shape));
    position_ids_range_desc.set("dimensions", emscripten::val::array(position_ids_range_shape));
    position_ids_range_desc.set("dataType", emscripten::val("int64"));
    emscripten::val position_ids_range = wnn_builder.call<emscripten::val>(
        "constant", position_ids_range_desc, position_ids_range_buffer);
    // Add the offset to the sequence.
    emscripten::val position_ids_add_range_options = emscripten::val::object();
    position_ids_add_range_options.set("label", node_name + "_position_ids_add_range");
    position_ids = wnn_builder.call<emscripten::val>(
        "add", position_ids, position_ids_range, position_ids_add_range_options);
  }

  // Gather the cosine/sine values based on the position_ids.
  emscripten::val gather_cos_sin_options = emscripten::val::object();
  gather_cos_sin_options.set("label", node_name + "_gather_cos_sin");
  gather_cos_sin_options.set("axis", 0);
  emscripten::val gather_cos = wnn_builder.call<emscripten::val>(
      "gather", cos_cache, position_ids, gather_cos_sin_options);
  emscripten::val gather_sin = wnn_builder.call<emscripten::val>(
      "gather", sin_cache, position_ids, gather_cos_sin_options);

  // After gathering cosine/sine, reshape and broadcast them to match the number of heads of the input data.
  const std::vector<uint32_t> reshaped_cos_sin_shape =
      interleaved ? std::vector<uint32_t>({batch_size, sequence_length, 1, rotary_embedding_dim / 2, 1})
                  : std::vector<uint32_t>({batch_size, sequence_length, 1, 1, rotary_embedding_dim / 2});
  emscripten::val reshape_gather_cos_sin_options = emscripten::val::object();
  reshape_gather_cos_sin_options.set("label", node_name + "_reshape_gather_cos_sin");
  gather_cos = wnn_builder.call<emscripten::val>(
      "reshape", gather_cos, emscripten::val::array(reshaped_cos_sin_shape), reshape_gather_cos_sin_options);
  gather_sin = wnn_builder.call<emscripten::val>(
      "reshape", gather_sin, emscripten::val::array(reshaped_cos_sin_shape), reshape_gather_cos_sin_options);

  // Multiply the non-rotated data with the cosine and the rotated data with the sine.
  emscripten::val mul_cos_options = emscripten::val::object();
  mul_cos_options.set("label", node_name + "_mul_cos");
  emscripten::val mul_cos = wnn_builder.call<emscripten::val>(
      "mul", partial_input0, gather_cos, mul_cos_options);
  emscripten::val mul_sin_options = emscripten::val::object();
  mul_sin_options.set("label", node_name + "_mul_sin");
  emscripten::val mul_sin = wnn_builder.call<emscripten::val>(
      "mul", concated_partial_input0, gather_sin, mul_sin_options);

  // Create a vector that contains the sign values {-1, 1}.
  emscripten::val sign_buffer = emscripten::val::undefined();
  const std::vector<uint32_t> sign_shape = interleaved ? std::vector<uint32_t>({1, 1, 1, 2})
                                                       : std::vector<uint32_t>({1, 1, 2, 1});
  emscripten::val sign_constant_desc = emscripten::val::object();
  sign_constant_desc.set("shape", emscripten::val::array(sign_shape));
  sign_constant_desc.set("dimensions", emscripten::val::array(sign_shape));
  ORT_RETURN_IF_NOT(SetWebnnDataType(sign_constant_desc, input_data_type), "Unsupported data type");
  if (input_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    sign_buffer = emscripten::val::global("Float32Array").new_(2);
    sign_buffer.set(0, -1.0f);
    sign_buffer.set(1, 1.0f);
  } else if (input_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    sign_buffer = emscripten::val::global("Uint16Array").new_(2);
    sign_buffer.set(0, PackFloat32ToUint16AsFloat16(-1.0f));
    sign_buffer.set(1, PackFloat32ToUint16AsFloat16(1.0f));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported input data type: ", input_data_type);
  }
  emscripten::val sign_constant = wnn_builder.call<emscripten::val>("constant", sign_constant_desc, sign_buffer);

  // Multiply the broadcasted sign values with the rotated input.
  emscripten::val mul_sign_options = emscripten::val::object();
  mul_sign_options.set("label", node_name + "_mul_sign");
  mul_sin = wnn_builder.call<emscripten::val>("mul", mul_sin, sign_constant, mul_sign_options);

  // Reshape mul_cos and mul_sin to (batch_size, sequence_length, num_heads, rotary_embedding_dim).
  const std::vector<uint32_t> reshaped_mul_cos_sin_shape =
      {batch_size, sequence_length, num_heads, rotary_embedding_dim};
  emscripten::val reshape_mul_cos_sin_options = emscripten::val::object();
  reshape_mul_cos_sin_options.set("label", node_name + "_reshape_mul_cos_sign");
  mul_cos = wnn_builder.call<emscripten::val>(
      "reshape", mul_cos, emscripten::val::array(reshaped_mul_cos_sin_shape), reshape_mul_cos_sin_options);
  mul_sin = wnn_builder.call<emscripten::val>(
      "reshape", mul_sin, emscripten::val::array(reshaped_mul_cos_sin_shape), reshape_mul_cos_sin_options);

  // Add the multiplied cos and sin values together.
  emscripten::val add_mul_cos_sin_options = emscripten::val::object();
  add_mul_cos_sin_options.set("label", node_name + "_add_mul_cos_sin");
  emscripten::val output = wnn_builder.call<emscripten::val>(
      "add", mul_cos, mul_sin, add_mul_cos_sin_options);

  // Join the added values with the rest of the input.
  if (head_size != rotary_embedding_dim) {
    emscripten::val concat_back_input_options = emscripten::val::object();
    concat_back_input_options.set("label", node_name + "_concat_back_input");
    emscripten::val concat_inputs = emscripten::val::array();
    concat_inputs.call<void>("push", output);
    concat_inputs.call<void>("push", partial_input1);
    output = wnn_builder.call<emscripten::val>("concat", concat_inputs, 3, concat_back_input_options);
  }

  // Reshape the output to the original shape. The output shape is the same as the input shape.
  const std::vector<uint32_t> output_shape = GetVecUint32FromVecInt64(input_shape);
  emscripten::val reshape_output_options = emscripten::val::object();
  reshape_output_options.set("label", node_name + "_reshape_output");
  output = wnn_builder.call<emscripten::val>(
      "reshape", output, emscripten::val::array(output_shape), reshape_output_options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool RotaryEmbeddingOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                                 const WebnnDeviceType /* device_type */,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  std::vector<int64_t> cos_cache_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) return false;
  if (!GetShape(*input_defs[2], cos_cache_shape, logger)) return false;
  const auto input_size = input_shape.size();
  if (input_size != 3 && input_size != 4) {
    LOGS(logger, VERBOSE) << "RotaryEmbedding only supports 3D or 4D input shape, input is " << input_size << "D shape";
    return false;
  }

  NodeAttrHelper helper(node);
  const int is_packed_batching = helper.Get("is_packed_batching", 0);
  const int num_heads = helper.Get("num_heads", 0);
  const int rotary_embedding_dim = helper.Get("rotary_embedding_dim", 0);

  const auto sequence_length = input_size == 4 ? input_shape[2] : input_shape[1];
  if (is_packed_batching == 0 && sequence_length > cos_cache_shape[0]) {
    LOGS(logger, VERBOSE) << "RotaryEmbedding: updating cos_cache and sin_cache is not currently supported";
    return false;
  }

  if (input_size == 4 && num_heads != 0 && num_heads != input_shape[1]) {
    LOGS(logger, VERBOSE) << "RotaryEmbedding: when input has 4 dimensions, num_heads must be 0 or have the same value "
                             "as the second dimension of the input";
    return false;
  }

  if (rotary_embedding_dim > 0 && num_heads == 0) {
    LOGS(logger, VERBOSE) << "RotaryEmbedding: num_heads must be provided if rotary_embedding_dim is specified";
    return false;
  }

  return true;
}

void CreateRotaryEmbeddingOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<RotaryEmbeddingOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime

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
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size, P is past_sequence_length
               N is number of attention heads, kv_N is number of attention heads for kv, H is head size
               G is group size, and G=N/kv_N, W=N*H, h=Sqrt(H).
    GQA inputs: query, key, value, past_key, past_value, seqlens_k, total_sequence_length
    Notes: cos_cache, sin_cache inputs are not supported. If the data type of the inputs (qkv and past kv) is float16,
    we cast them to float32 to ensure data precision.

          query      key               value
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

  emscripten::val query_input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val key_input = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val value_input = model_builder.GetOperand(input_defs[2]->Name());
  emscripten::val past_key_input = model_builder.GetOperand(input_defs[3]->Name());
  emscripten::val past_value_input = model_builder.GetOperand(input_defs[4]->Name());
  emscripten::val seqlens_k_input = model_builder.GetOperand(input_defs[5]->Name());

  std::vector<int64_t> input_q_shape, input_past_k_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_q_shape, logger), "Cannot get query shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[3], input_past_k_shape, logger), "Cannot get past_key shape");

  NodeAttrHelper helper(node);
  const uint32_t kv_num_heads = helper.Get("kv_num_heads", 0);
  const uint32_t num_heads = helper.Get("num_heads", 0);

  const uint32_t batch_size = SafeInt<uint32_t>(input_q_shape[0]);
  const uint32_t qkv_sequence_length = SafeInt<uint32_t>(input_q_shape[1]);
  const uint32_t qkv_hidden_size = SafeInt<uint32_t>(input_q_shape[2]);
  const uint32_t head_size = SafeInt<uint32_t>(qkv_hidden_size / num_heads);
  const uint32_t past_sequence_length = SafeInt<uint32_t>(input_past_k_shape[2]);
  const uint32_t group_size = SafeInt<uint32_t>(num_heads / kv_num_heads);

  const float scale_value = helper.Get("scale", 1 / sqrt(static_cast<float>(head_size)));

  const std::vector<uint32_t> reshape_output_shape = {batch_size, qkv_sequence_length, qkv_hidden_size};
  const std::vector<uint32_t> scatter_indices_shape = {batch_size, qkv_sequence_length, kv_num_heads, 3};
  const std::vector<uint32_t> reshape_tensor_shape = {batch_size, qkv_sequence_length, num_heads, head_size};

  emscripten::val common_options = emscripten::val::object();
  emscripten::val common_desc = emscripten::val::object();

  int32_t q_type = 0;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], q_type, logger), "Could not get input data type.");

  // Check whether inputs' data type is fp16, if so, we should cast them to fp32 to ensure the calculation precision.
  if (q_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    common_options.set("label", node.Name() + "_/GQA/preprocess/cast/query_input");
    query_input = model_builder.GetBuilder().call<emscripten::val>("cast", query_input, emscripten::val("float32"),
                                                                   common_options);

    common_options.set("label", node.Name() + "_/GQA/preprocess/cast/key_input");
    key_input =
        model_builder.GetBuilder().call<emscripten::val>("cast", key_input, emscripten::val("float32"), common_options);

    common_options.set("label", node.Name() + "_/GQA/preprocess/cast/value_input");
    value_input = model_builder.GetBuilder().call<emscripten::val>("cast", value_input, emscripten::val("float32"),
                                                                   common_options);

    common_options.set("label", node.Name() + "_/GQA/preprocess/cast/past_key_input");
    past_key_input = model_builder.GetBuilder().call<emscripten::val>("cast", past_key_input,
                                                                      emscripten::val("float32"), common_options);

    common_options.set("label", node.Name() + "_/GQA/preprocess/cast/past_value_input");
    past_value_input = model_builder.GetBuilder().call<emscripten::val>("cast", past_value_input,
                                                                        emscripten::val("float32"), common_options);
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

  // scatterND for present_key and present_value
  common_options.set("label", node.Name() + "_/GQA/present_key/ScatterND");
  emscripten::val present_key = model_builder.GetBuilder().call<emscripten::val>(
      "scatterND", past_key_input, scatter_indices, key_for_scatter, common_options);

  common_options.set("label", node.Name() + "_/GQA/present_value/ScatterND");
  emscripten::val present_value = model_builder.GetBuilder().call<emscripten::val>(
      "scatterND", past_value_input, scatter_indices, value_for_scatter, common_options);

  emscripten::val true_present_key;
  emscripten::val true_present_value;
  if (group_size != 1) {
    // Broadcast key and value for group query by reshape, expand and reshape.
    // present kv shape (B,kv_N,P,H) -> (B,kv_N,1,P,H) -> (B,kv_N,N/kv_N,P,H) -> (B,N,P,H) broadcasted kv shape
    const std::vector<uint32_t> group_broadcast_tensor_shape_1 = {batch_size, kv_num_heads, 1, past_sequence_length,
                                                                  head_size};
    const std::vector<uint32_t> group_broadcast_tensor_shape_2 = {batch_size, kv_num_heads, group_size,
                                                                  past_sequence_length, head_size};
    const std::vector<uint32_t> group_broadcast_tensor_shape_3 = {batch_size, num_heads, past_sequence_length,
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

  emscripten::val scale_constant =
      model_builder.CreateOrGetConstant<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, scale_value, {1});

  /* Calculate attention_bias for masking softmax
        ones_array (shape=B,N,S,P)                          range_of_qkv_sequence_length_constant (0,1,2,...) (shape=S)
          |                                                                 |
        CumSum (axis=3, exclusive=true, reversed=false)                    Add <--- scatter_pos
          |                                                                 |
          |                                                               Expand (shape=P,S)
          |                                                                 |
          +-------------------------------> Lesser <---------------------Transpose (1,0)
                                                |
                                      1 ---> Where <--- finfo_min (minimum value of FP32)
                                                |
                                          attention_bias
  */
  const std::vector<int32_t> mask_shape_ones_shape(batch_size * num_heads * qkv_sequence_length * past_sequence_length,
                                                   1);
  std::string mask_shape_ones_shape_name = "webnn_GQA_left_constant_of_scatter_indices_" + std::to_string(batch_size) +
                                           "_" + std::to_string(num_heads) + "_" + std::to_string(qkv_sequence_length) +
                                           "_" + std::to_string(past_sequence_length);
  emscripten::val mask_shape_ones_shape_constant = model_builder.CreateOrGetConstant<int32_t>(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, mask_shape_ones_shape_name, mask_shape_ones_shape,
      std::vector<uint32_t>({batch_size, num_heads, qkv_sequence_length, past_sequence_length}));

  emscripten::val cumsum_options = emscripten::val::object();
  cumsum_options.set("label", node.Name() + "_range_of_mask_shape");
  cumsum_options.set("exclusive", true);
  cumsum_options.set("reversed", false);
  emscripten::val neq_left = model_builder.GetBuilder().call<emscripten::val>(
      "cumulativeSum", mask_shape_ones_shape_constant, gsl::narrow<uint32_t>(3), cumsum_options);

  std::vector<uint32_t> reshape_pre_neq_right = {past_sequence_length, qkv_sequence_length};
  std::vector<int32_t> pre_neq_right_data_range(qkv_sequence_length);
  std::iota(pre_neq_right_data_range.begin(), pre_neq_right_data_range.end(), 1);

  std::string pre_neq_right_data_range_name =
      "webnn_GQA_left_constant_of_scatter_indices_" + std::to_string(qkv_sequence_length);
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

  common_options.set("label", node.Name() + "_/GQA/attn_mask/condition");
  emscripten::val condition =
      model_builder.GetBuilder().call<emscripten::val>("lesser", neq_left, neq_right, common_options);

  emscripten::val value_one_constant =
      model_builder.CreateOrGetConstant<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1, {1});

  // finfo_min: the minimum value of float32
  emscripten::val finfo_min_constant = model_builder.CreateOrGetConstant<float>(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, -3.4028234663852886e+38, {1});

  common_options.set("label", node.Name() + "_/GQA/attn_mask/where");
  emscripten::val attn_mask = model_builder.GetBuilder().call<emscripten::val>("where", condition, value_one_constant,
                                                                               finfo_min_constant, common_options);

  // Execute ScaledDotProductAttention
  emscripten::val output =
      ScaledDotProductAttention(model_builder, node, logger, new_query, true_present_key, true_present_value,
                                scale_constant, attn_mask, reshape_output_shape);

  if (q_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    common_options.set("label", node.Name() + "_/GQA/postprocess/cast/output");
    output =
        model_builder.GetBuilder().call<emscripten::val>("cast", output, emscripten::val("float16"), common_options);

    common_options.set("label", node.Name() + "_/GQA/postprocess/cast/present_key");
    present_key = model_builder.GetBuilder().call<emscripten::val>("cast", present_key, emscripten::val("float16"),
                                                                   common_options);

    common_options.set("label", node.Name() + "_/GQA/postprocess/cast/present_value");
    present_value = model_builder.GetBuilder().call<emscripten::val>("cast", present_value, emscripten::val("float16"),
                                                                     common_options);
  }

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

  const auto& total_sequence_length_name = input_defs[6]->Name();
  const auto* total_sequence_length_initializer = graph_viewer.GetConstantInitializer(total_sequence_length_name);
  if (!total_sequence_length_initializer) {
    LOGS(logger, VERBOSE) << "total_sequence_length must be constant";
    return false;
  }

  const auto total_sequence_length_tensor = *total_sequence_length_initializer;
  emscripten::val total_sequence_length = emscripten::val::undefined();
  if (!ReadScalarTensorData(total_sequence_length_tensor, total_sequence_length, logger)) {
    return false;
  }

  std::vector<int64_t> query_shape;
  if (!GetShape(*input_defs[0], query_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get query shape.";
    return false;
  }
  const auto sequence_length = query_shape[1];

  std::vector<int64_t> past_key_shape;
  if (!GetShape(*input_defs[3], past_key_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get past_key shape.";
    return false;
  }
  const auto past_sequence_length = past_key_shape[2];

  // WebNN EP only supports past_sequence_length of past kv equals to present_sequence_length of present kv
  // According to CPU EP, present_sequence_length = max(past_sequence_length,total_sequence_length)
  // For prefilling stage (the first prompt), it requires sequence_length == total_sequence_length.
  if (sequence_length != 1) {
    if (sequence_length != total_sequence_length.as<int32_t>()) {
      LOGS(logger, VERBOSE) << op_type << " sequence_length != total_sequence_length. Not first prompt.";
      return false;
    }
  } else {  // For decoding stage, it requires past_sequence_length == total_sequence_length.
    if (past_sequence_length != total_sequence_length.as<int32_t>()) {
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

  for (int i = 0; i < 9; i++) {
    if (i < 7) {
      if (!TensorExists(input_defs, i)) {
        LOGS(logger, VERBOSE) << op_type << " requires input " << i;
        return false;
      }
    } else {  // cos_cache and sin_cache are not supported
      if (TensorExists(input_defs, i)) {
        LOGS(logger, VERBOSE) << op_type << " does not support input " << i;
        return false;
      }
    }
  }

  int32_t q_type = 0;
  int32_t k_type = 0;
  int32_t v_type = 0;
  int32_t past_k_type = 0;
  int32_t past_v_type = 0;
  int32_t seqlens_k_type = 0;
  int32_t total_sequence_length_type = 0;
  if (!GetType(*input_defs[0], q_type, logger) || !GetType(*input_defs[1], k_type, logger) ||
      !GetType(*input_defs[2], v_type, logger) || !GetType(*input_defs[3], past_k_type, logger) ||
      !GetType(*input_defs[4], past_v_type, logger) || !GetType(*input_defs[5], seqlens_k_type, logger) ||
      !GetType(*input_defs[6], total_sequence_length_type, logger)) {
    return false;
  }

  std::array<int32_t, 5> input_types{q_type, k_type, v_type, past_k_type, past_v_type};
  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  if (q_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT && q_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return false;
  }

  if (seqlens_k_type != ONNX_NAMESPACE::TensorProto_DataType_INT32 &&
      total_sequence_length_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    return false;
  }

  std::vector<int64_t> input_q_shape, input_k_shape, input_v_shape, input_past_k_shape, input_past_v_shape;
  if (!GetShape(*input_defs[0], input_q_shape, logger) || !GetShape(*input_defs[1], input_k_shape, logger) ||
      !GetShape(*input_defs[2], input_v_shape, logger) || !GetShape(*input_defs[3], input_past_k_shape, logger) ||
      !GetShape(*input_defs[4], input_past_v_shape, logger)) {
    return false;
  }
  const auto q_rank = input_q_shape.size();
  const auto k_rank = input_k_shape.size();
  const auto v_rank = input_v_shape.size();
  const auto past_k_rank = input_past_k_shape.size();
  const auto past_v_rank = input_past_v_shape.size();
  if (q_rank != 3 || k_rank != 3 || v_rank != 3) {  // The qkv shape should be BSW
    LOGS(logger, VERBOSE) << op_type << " qkv shape is not BSW.";
    return false;
  }

  if (past_k_rank != 4 || past_v_rank != 4) {  // The past qkv shape should be BNSH
    LOGS(logger, VERBOSE) << op_type << " past qkv shape is not BNSH.";
    return false;
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/attention_helper.h"
#include "core/util/shape_checker.h"

namespace onnxruntime {
namespace attention_helper {

void AttentionParameters::checkParameters() const {
  ORT_ENFORCE(batch_size > 0, "Batch size must be greater than 0");
  ORT_ENFORCE(q_sequence_length > 0, "Q sequence length must be greater than 0");
  ORT_ENFORCE(kv_sequence_length > 0, "KV sequence length must be greater than 0");
  ORT_ENFORCE(head_size > 0, "Head size must be greater than 0");
  ORT_ENFORCE(v_head_size > 0, "V head size must be greater than 0");
  ORT_ENFORCE(past_sequence_length >= 0, "Past sequence length must be non-negative");
  ORT_ENFORCE(total_sequence_length > 0, "Total sequence length must be greater than 0");
  ORT_ENFORCE(kv_num_heads > 0, "KV number of heads must be greater than 0");
  ORT_ENFORCE(q_num_heads > 0, "Q number of heads must be greater than 0");
  ORT_ENFORCE(total_sequence_length == past_sequence_length + kv_sequence_length,
              "Total sequence length must be equal to past sequence length plus KV sequence length");
}

Status ComputeOutputShapeForAttention(
    const Tensor* Q,
    const Tensor* K,
    const Tensor* V,
    const Tensor* attn_mask,
    const Tensor* past_key,
    const Tensor* past_value,
    bool is_causal,
    float softcap,
    int softmax_precision,
    attention_helper::QKMatMulOutputMode qk_matmul_output_mode,
    int kv_num_heads,
    int q_num_heads,
    float scale,
    AttentionParameters& parameters,
    std::vector<int64_t>& y_shape,
    std::vector<int64_t>& present_key_shape,
    std::vector<int64_t>& present_value_shape,
    std::vector<int64_t>& output_qk_shape) {
  ORT_ENFORCE(Q != nullptr && K != nullptr && V != nullptr,
              "Q, K, and V inputs must not be null");
  int q_dims = onnxruntime::narrow<int>(Q->Shape().NumDimensions());
  int k_dims = onnxruntime::narrow<int>(K->Shape().NumDimensions());
  int v_dims = onnxruntime::narrow<int>(V->Shape().NumDimensions());
  ORT_ENFORCE(q_dims == 3 || q_dims == 4, "Q must be a 3D or 4D tensor");
  ORT_ENFORCE(q_dims == k_dims, "Q and K must have the same rank.");
  ORT_ENFORCE(q_dims == v_dims, "Q and V must have the same rank.");

  ORT_ENFORCE((past_key == nullptr) == (past_value == nullptr), "past_key and past_value must be both null or both not null");
  ORT_ENFORCE(Q->Shape()[0] == K->Shape()[0], "inconsistent batch_size (between Q and K)");
  ORT_ENFORCE(Q->Shape()[0] == V->Shape()[0], "inconsistent batch_size (between Q and V)");
  ORT_ENFORCE(past_key == nullptr || Q->Shape()[0] == past_key->Shape()[0], "inconsistent batch_size (between Q and past_key)");
  ORT_ENFORCE(past_value == nullptr || Q->Shape()[0] == past_value->Shape()[0], "inconsistent batch_size (between Q and past_value)");
  ORT_ENFORCE(past_value == nullptr || past_value->Shape()[2] == past_key->Shape()[2], "inconsistent past_sequence_length (between past_key and past_value)");

  parameters.is_causal = is_causal;
  parameters.softcap = softcap;
  parameters.softmax_precision = softmax_precision;
  parameters.qk_matmul_output_mode = qk_matmul_output_mode;         // output mode for Q*K matmul
  parameters.batch_size = onnxruntime::narrow<int>(Q->Shape()[0]);  // Q.shape[0], K.shape[0], V.shape[0] (4D)

  ORT_ENFORCE(parameters.batch_size > 0, "Batch size must be greater than 0");
  ORT_ENFORCE(attn_mask == nullptr || (attn_mask->Shape().NumDimensions() >= 2 && attn_mask->Shape().NumDimensions() <= 4), "attn_mask must be 2D or 3D or 4D tensor");

  if (q_dims == 4) {
    // 4D
    parameters.kv_num_heads = kv_num_heads > 0 ? kv_num_heads : onnxruntime::narrow<int>(K->Shape()[1]);  // K.shape[1] or V.shape[1] (4D)
    parameters.q_num_heads = q_num_heads > 0 ? q_num_heads : onnxruntime::narrow<int>(Q->Shape()[1]);     // Q.shape[1] (4D)

    ORT_ENFORCE(parameters.kv_num_heads == onnxruntime::narrow<int>(K->Shape()[1]), "kv_num_heads different from K.shape[1]");
    ORT_ENFORCE(parameters.kv_num_heads == onnxruntime::narrow<int>(V->Shape()[1]), "kv_num_heads different from V.shape[1]");
    ORT_ENFORCE(parameters.q_num_heads == onnxruntime::narrow<int>(Q->Shape()[1]), "q_num_heads different from Q.shape[1]");
    ORT_ENFORCE(Q->Shape()[3] == K->Shape()[3], "inconsistent head_size");
    ORT_ENFORCE(K->Shape()[2] == V->Shape()[2], "inconsistent kv_sequence_length");
    ORT_ENFORCE(attn_mask == nullptr || attn_mask->Shape()[attn_mask->Shape().NumDimensions() - 2] == Q->Shape()[2], "inconsistent q_sequence_length (between attn_mask and Q)");

    // From shapes
    parameters.transpose_output = false;                                      // whether to transpose the output from BxNxSxH to BxSxNxH
    parameters.q_sequence_length = onnxruntime::narrow<int>(Q->Shape()[2]);   // Q.shape[2] (4D)
    parameters.head_size = onnxruntime::narrow<int>(Q->Shape()[3]);           // Q.shape[3] (4D)
    parameters.kv_sequence_length = onnxruntime::narrow<int>(K->Shape()[2]);  // K.shape[2] or V.shape[2] (4D)
    parameters.v_head_size = onnxruntime::narrow<int>(V->Shape()[3]);         // V.shape[3] (4D)
    parameters.past_sequence_length = past_key == nullptr                     // past_key.shape[2] or past_value.shape[2] (4D) or given by the mask
                                          ? 0
                                          : onnxruntime::narrow<int>(past_key->Shape()[2]);

    y_shape = {static_cast<int64_t>(parameters.batch_size),
               static_cast<int64_t>(parameters.q_num_heads),
               static_cast<int64_t>(parameters.q_sequence_length),
               static_cast<int64_t>(parameters.v_head_size)};
  } else {
    // 3D
    parameters.kv_num_heads = kv_num_heads;
    parameters.q_num_heads = q_num_heads;

    // From shapes
    ORT_ENFORCE(Q->Shape()[2] % parameters.q_num_heads == 0, "inconsistent q_hidden_size, it should be a multiple of q_num_heads");
    ORT_ENFORCE(V->Shape()[2] % parameters.kv_num_heads == 0, "inconsistent v_hidden_size, it should be a multiple of kv_num_heads");

    parameters.transpose_output = true;  // whether to transpose the output from BxNxSxH to BxSxNxH
    parameters.q_sequence_length = onnxruntime::narrow<int>(Q->Shape()[1]);
    parameters.head_size = onnxruntime::narrow<int>(Q->Shape()[2]) / parameters.q_num_heads;
    parameters.kv_sequence_length = onnxruntime::narrow<int>(K->Shape()[1]);
    parameters.v_head_size = onnxruntime::narrow<int>(V->Shape()[2]) / parameters.kv_num_heads;
    parameters.past_sequence_length = past_key == nullptr
                                          ? 0
                                          : onnxruntime::narrow<int>(past_key->Shape()[2]);

    y_shape = {static_cast<int64_t>(parameters.batch_size),
               static_cast<int64_t>(parameters.q_sequence_length),
               static_cast<int64_t>(parameters.q_num_heads * parameters.v_head_size)};
  }
  parameters.total_sequence_length = parameters.past_sequence_length + parameters.kv_sequence_length;

  ORT_ENFORCE(attn_mask == nullptr || attn_mask->Shape()[attn_mask->Shape().NumDimensions() - 1] == parameters.total_sequence_length,
              "inconsistent total_sequence_length (between attn_mask and past_key and past_value)");
  ORT_ENFORCE(attn_mask == nullptr ||
                  attn_mask->Shape().NumDimensions() < 3 ||
                  attn_mask->Shape()[attn_mask->Shape().NumDimensions() - 3] == 1 ||
                  attn_mask->Shape()[attn_mask->Shape().NumDimensions() - 3] == parameters.kv_num_heads,
              "attn_mask must be broadcastable to (batch_size, kv_num_heads, q_sequence_length, total_sequence_length) but is not compatible with kv_num_heads");
  ORT_ENFORCE(attn_mask == nullptr ||
                  attn_mask->Shape().NumDimensions() < 4 ||
                  attn_mask->Shape()[0] == 1 ||
                  attn_mask->Shape()[0] == parameters.batch_size,
              "attn_mask must be broadcastable to (batch_size, kv_num_heads, q_sequence_length, total_sequence_length) but is not compatible with batch_size");
  ASSERT_TENSOR_DIMS(past_key, parameters.batch_size, parameters.kv_num_heads, parameters.past_sequence_length, parameters.head_size);
  ASSERT_TENSOR_DIMS(past_value, parameters.batch_size, parameters.kv_num_heads, parameters.past_sequence_length, parameters.v_head_size);

  parameters.scale = std::isnan(scale) ? static_cast<float>(1.0 / sqrt(parameters.head_size)) : scale;
  parameters.checkParameters();

  present_key_shape = {static_cast<int64_t>(parameters.batch_size),
                       static_cast<int64_t>(parameters.kv_num_heads),
                       static_cast<int64_t>(parameters.past_sequence_length + parameters.kv_sequence_length),
                       static_cast<int64_t>(parameters.head_size)};
  present_value_shape = {static_cast<int64_t>(parameters.batch_size),
                         static_cast<int64_t>(parameters.kv_num_heads),
                         static_cast<int64_t>(parameters.past_sequence_length + parameters.kv_sequence_length),
                         static_cast<int64_t>(parameters.v_head_size)};
  if (qk_matmul_output_mode == QKMatMulOutputMode::kNone) {
    output_qk_shape.clear();
  } else {
    output_qk_shape = {static_cast<int64_t>(parameters.batch_size),
                       static_cast<int64_t>(parameters.q_num_heads),
                       static_cast<int64_t>(parameters.q_sequence_length),
                       static_cast<int64_t>(parameters.past_sequence_length + parameters.kv_sequence_length)};
  }
  return Status::OK();
}
}  // namespace attention_helper
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/attention_base.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                  const TensorShape* weights_shape,
                                  const TensorShape& bias_shape,
                                  const Tensor*& mask_index,
                                  const Tensor* past,
                                  const Tensor* extra_add_qk,
                                  const Tensor* key,
                                  const Tensor* value,
                                  void* parameters,
                                  const Tensor* past_seq_len) const {
  // Abbreviation and Meanings:
  //   B:    batch_size
  //   S:    sequence_length (input sequence length of query)
  //   P:    past_sequence_length (past sequence length of key or value)
  //   L:    kv_sequence_length (input sequence length of key or value)
  //   M:    max_sequence_length
  //   T:    total_sequence_length = past_sequence_length + kv_sequence_length
  //   N:    num_heads
  //   H:    head size for Q and K, aka q_head_size or v_head_size or qk_head_size
  //   H_v:  v_head_size
  //   D_i:  input hidden size
  //   D:    hidden size for Q and K (D = N * H), aka q_hidden_size or k_hidden_size or qk_hidden_size
  //   D_v:  v_hidden_size = num_heads * v_head_size

  // When past state is used, Q, K and V should have same hidden size (unless we split it into past_key and past_value).

  // Input shapes with weights:
  //   input        (Q/K/V)    : (B, S, D_i)
  //   weights      (Q/K/V)    : (D_i, D + D + D_v)
  //   bias         (Q/K/V)    : (D + D + D_v)
  //   mask_index              : see below
  //   past         (K/V)      : (2, B, N, P, H) or NULL
  //   extra_add_qk            : (B, N, S, T) or NULL
  //   key                     : NULL
  //   value                   : NULL

  // Input shapes without weights (only bias is provided):
  //   input         (Q)       : (B, S, D)
  //   weights                 : NULL
  //   bias          (Q/K/V)   : (D + D + D_v)
  //   mask_index              : see below
  //   past          (K/V)     : (2, B, N, P, H) or NULL
  //   extra_add_qk            : (B, N, S, T) or NULL
  //   key           (K)       : (B, L, D)
  //   value         (V)       : (B, L, D_v)

  // For mask_index, the following shapes are supported:
  //     NULL, (B, 1), (1, 1)
  //     (B), (2 * B),
  //     (B, T)
  //     (B, S, T)
  //     (B, 1, M, M)
  //
  // When a model is pruned (like some attention heads are removed in Q/K/V), input_hidden_size could be larger
  // than hidden dimension of Q, K and V.

  if (past != nullptr && extra_add_qk != nullptr) {
    // past is used on GPT-2 model with past state, we don't have a case for extra add qk yet
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Attention cannot have both past and extra_add_qk");
  }

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }

  auto& batch_size = dims[0];
  auto& sequence_length = dims[1];
  int64_t input_hidden_size = dims[2];

  const auto& bias_dims = bias_shape.GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }

  if (weights_shape != nullptr) {
    const auto& weights_dims = weights_shape->GetDims();
    if (weights_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'weights' is expected to have 2 dimensions, got ",
                             weights_dims.size());
    }
    if (weights_dims[0] != input_hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 1 dimension 0 should have same length as dimension 2 of input 0");
    }

    if (bias_dims[0] != weights_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'bias' dimension 0 should have same length as dimension 1 of input 'weights'");
    }
  }

  int64_t q_hidden_size = bias_dims[0] / static_cast<int64_t>(3);
  int64_t k_hidden_size = q_hidden_size;
  int64_t v_hidden_size = k_hidden_size;
  if (qkv_hidden_sizes_.size() != 0) {
    if (qkv_hidden_sizes_.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "qkv_hidden_sizes attribute should have 3 elements");
    }

    for (size_t i = 0; i < qkv_hidden_sizes_.size(); i++) {
      if (qkv_hidden_sizes_[i] % num_heads_ != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "hidden_size should be divisible by num_heads:", qkv_hidden_sizes_[i]);
      }
    }

    q_hidden_size = qkv_hidden_sizes_[0];
    k_hidden_size = qkv_hidden_sizes_[1];
    v_hidden_size = qkv_hidden_sizes_[2];
  }

  int64_t kv_sequence_length = sequence_length;
  if (weights_shape == nullptr) {  // no weights
    if (this->require_weights_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "This operator requires weights");
    }

    if (key == nullptr || value == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "When weights is not provided, key and value are required");
    }

    const auto& key_dims = key->Shape().GetDims();
    if (key_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 dimensions, got ",
                             key_dims.size());
    }
    if (key_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key' dimension 0 should have same length as dimension 0 of input 0");
    }

    const auto& value_dims = value->Shape().GetDims();
    if (value_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 3 dimensions, got ",
                             value_dims.size());
    }
    if (value_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'value' dimension 0 should have same length as dimension 0 of input 0");
    }

    if (value_dims[1] != key_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key' and 'value' dimension 1 should have same length");
    }

    q_hidden_size = dims[2];
    k_hidden_size = key_dims[2];
    v_hidden_size = value_dims[2];
    kv_sequence_length = key_dims[1];

    if (qkv_hidden_sizes_.size() != 0 &&
        (q_hidden_size != qkv_hidden_sizes_[0] ||
         k_hidden_size != qkv_hidden_sizes_[1] ||
         v_hidden_size != qkv_hidden_sizes_[2])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "qkv_hidden_sizes does not match with query, key and value input shape",
                             " q_hidden_size=", q_hidden_size,
                             " k_hidden_size=", k_hidden_size,
                             " v_hidden_size=", v_hidden_size,
                             "qkv_hidden_sizes[0]=", qkv_hidden_sizes_[0],
                             "qkv_hidden_sizes[1]=", qkv_hidden_sizes_[1],
                             "qkv_hidden_sizes[2]=", qkv_hidden_sizes_[2]);
    }
  }

  if (q_hidden_size != k_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "qkv_hidden_sizes first element should be same as the second");
  }

  if (this->require_same_hidden_size_ && k_hidden_size != v_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Hidden size of Q, K and V shall be same");
  }

  if (bias_dims[0] != q_hidden_size + k_hidden_size + v_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'bias' dimension 0 should have same length as sum of Q/K/V hidden sizes:",
                           " q_hidden_size=", q_hidden_size, " k_hidden_size=", k_hidden_size, " v_hidden_size=",
                           v_hidden_size, "bias_dims[0]=", bias_dims[0]);
  }

  int64_t past_sequence_length = 0;
  if (past != nullptr) {  // past is optional
    if (k_hidden_size != v_hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'past' expect k_hidden_size == v_hidden_size");
    }

    const auto& past_dims = past->Shape().GetDims();
    if (past_dims.size() != 5) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'past' is expected to have 5 dimension, got ",
                             past_dims.size());
    }

    if (past_dims[0] != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'past' dimension 0 shall have length of 2");
    }

    if (past_dims[1] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'past' dimension 1 shall have same length as dimension 0 of input 0");
    }

    if (static_cast<int>(past_dims[2]) != num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'past' dimension 2 shall have length of num_heads", num_heads_);
    }

    if (static_cast<int>(past_dims[4]) != k_hidden_size / num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'past' dimension 2 shall have length of ", k_hidden_size / num_heads_);
    }

    if (!past_present_share_buffer_) {
      past_sequence_length = past_dims[3];
    } else {
      if (past_seq_len == nullptr || !onnxruntime::IsScalarOr1ElementVector(past_seq_len)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "past_sequence_length tensor must be of one element when past_present_share_buffer is set");
      }
      past_sequence_length = *past_seq_len->Data<int32_t>();
    }
  }

  int64_t total_sequence_length = kv_sequence_length + past_sequence_length;
  if (past != nullptr && past_present_share_buffer_) {
    const auto& past_dims = past->Shape().GetDims();
    if (past_dims[3] < total_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "when past_present_share_buffer, past tensor sequence must not smaller than total_sequqnce_length ");
    }
  }

  int64_t max_sequence_length = -1;
  if (mask_index != nullptr) {  // mask_index is optional
    bool is_dummy = false;
    auto status = this->CheckMask(mask_index, is_dummy,
                                  max_sequence_length, batch_size, sequence_length, total_sequence_length);
    if (status != Status::OK()) {
      return status;
    }
    if (is_dummy) {
      mask_index = nullptr;
    }
  }

  if (extra_add_qk != nullptr) {
    const auto& extra_add_qk_dims = extra_add_qk->Shape().GetDims();

    if (extra_add_qk_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' is expected to have 4 dimensions, got ",
                             extra_add_qk_dims.size());
    }

    if (extra_add_qk_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' dimension 0 should be same as batch_size, got ",
                             extra_add_qk_dims[0]);
    }
    if (extra_add_qk_dims[1] != num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' dimension 1 should be same as number of heads, got ",
                             extra_add_qk_dims[1]);
    }
    if (extra_add_qk_dims[2] != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' dimension 2 should be same as sequence_length, got ",
                             extra_add_qk_dims[2]);
    }
    if (extra_add_qk_dims[3] != total_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' dimension 3 should be same as total_sequence_length, got ",
                             extra_add_qk_dims[3]);
    }
  }

  if (past != nullptr && past_present_share_buffer_) {
    if (max_sequence_length <= 0) {
      max_sequence_length = past->Shape().GetDims()[3];
    }
    if (max_sequence_length != past->Shape().GetDims()[3]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "max_sequence_length not matching from mask and past when past_present_share_buffer_ is set");
    }
  }

  if (parameters != nullptr) {
    AttentionParameters* output_parameters = reinterpret_cast<AttentionParameters*>(parameters);
    output_parameters->batch_size = static_cast<int>(batch_size);
    output_parameters->sequence_length = static_cast<int>(sequence_length);
    output_parameters->past_sequence_length = static_cast<int>(past_sequence_length);
    output_parameters->kv_sequence_length = static_cast<int>(kv_sequence_length);
    output_parameters->total_sequence_length = static_cast<int>(total_sequence_length);
    output_parameters->max_sequence_length = static_cast<int>(max_sequence_length);
    output_parameters->input_hidden_size = static_cast<int>(input_hidden_size);
    output_parameters->hidden_size = static_cast<int>(q_hidden_size);
    output_parameters->v_hidden_size = static_cast<int>(v_hidden_size);
    output_parameters->head_size = static_cast<int>(q_hidden_size) / num_heads_;
    output_parameters->v_head_size = static_cast<int>(v_hidden_size) / num_heads_;
    output_parameters->num_heads = num_heads_;
    output_parameters->is_unidirectional = is_unidirectional_;
    output_parameters->past_present_share_buffer = (past_present_share_buffer_ != 0);
  }

  return Status::OK();
}

Status AttentionBase::CheckMask(const Tensor* mask_index,
                                bool& is_dummy,
                                int64_t& max_sequence_length,
                                int64_t batch_size,
                                int64_t sequence_length,
                                int64_t total_sequence_length) const {
  is_dummy = false;
  const auto& mask_dims = mask_index->Shape().GetDims();
  if (mask_dims.size() == 1) {
    if (mask_dims[0] != batch_size && mask_dims[0] != 2 * batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'mask_index' with 1D data shall have length of batch_size or 2 * batch_size");
    }
  } else if (mask_dims.size() == 2) {
    if (mask_dims[0] != batch_size || mask_dims[1] != total_sequence_length) {
      // Add operator supports broadcasting. Here we handle a case with only one element in the 2nd dimension.
      if ((mask_dims[0] == batch_size || mask_dims[0] == 1) && mask_dims[1] == 1) {
        // Mask will have same value after propagation, which has same effect as no mask.
        is_dummy = true;
      } else {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Inputs 'mask_index' with 2D data shall have shape "
            "batch_size x total_sequence_length");
      }
    }
  } else if (mask_dims.size() == 3) {
    if (mask_dims[0] != batch_size || mask_dims[1] != sequence_length || mask_dims[2] != total_sequence_length) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Inputs 'mask_index' with 3D data shall have shape "
          "batch_size x sequence_length x total_sequence_length");
    }
  } else if (mask_dims.size() == 4) {
    if (mask_dims[0] != batch_size || mask_dims[1] != 1 || mask_dims[2] != mask_dims[3] ||
        mask_dims[2] < total_sequence_length) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Inputs 'mask_index' with 4D data shall have shape "
          "batch_size x 1 x max_sequence_length x max_sequence_length)");
    }
    max_sequence_length = mask_dims[3];

    if (is_unidirectional_ == true) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'mask_index' with 4D data shall have is_unidirectional_ set to false");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'mask_index' is expected to have 1, 2, 3 or 4 dimensions, got ",
                           mask_dims.size());
  }

  return Status::OK();
}

Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                  const TensorShape* weights_shape,
                                  const TensorShape& bias_shape,
                                  const Tensor*& mask_index,
                                  const Tensor* past,
                                  const Tensor* extra_add_qk,
                                  const Tensor* key,
                                  const Tensor* value,
                                  void* parameters,
                                  const int max_threads_per_block,
                                  const Tensor* past_seq_len) const {
  if (num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(input_shape, weights_shape, bias_shape, mask_index, past, extra_add_qk, key, value, parameters, past_seq_len);
}

Tensor* AttentionBase::GetPresent(OpKernelContext* context,
                                  const Tensor* past,
                                  int batch_size,
                                  int head_size,
                                  int kv_sequence_length,
                                  int& past_sequence_length) const {
  // Input and output shapes:
  //   past        : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   present     : (2, batch_size, num_heads, past_sequence_length + kv_sequence_length, head_size)

  past_sequence_length = (nullptr != past) ? static_cast<int>(past->Shape().GetDims()[3]) : 0;
  std::array<int64_t, 5> present_dims{2, batch_size, num_heads_, static_cast<int64_t>(kv_sequence_length) + past_sequence_length, head_size};

  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(1, present_shape);
  if (nullptr != past && nullptr == present) {
    ORT_THROW("Expect to have present state output when past state input is given");
  }

  return present;
}

}  // namespace contrib
}  // namespace onnxruntime

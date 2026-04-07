// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <vector>
#include "core/common/common.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#endif
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#ifndef SHARED_PROVIDER
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#endif

namespace onnxruntime {
namespace contrib {

class AttentionBase {
 public:
  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const Tensor*& mask_index,  // Dummy mask of shape (1 or batch_size, 1) will be updated to nullptr.
                     const Tensor* past,
                     const Tensor* attention_bias,
                     void* parameters,
                     const int max_threads_per_block,  // for CUDA
                     const Tensor* past_seq_len = nullptr) const;

#ifdef SHARED_PROVIDER
  Tensor* GetPresent(OpKernelContext* context,
                     const Tensor* past,
                     int batch_size,
                     int head_size,
                     int kv_sequence_length,
                     int& past_sequence_length) const;
#else
  template <typename TOpKernelContext>
  Tensor* GetPresent(TOpKernelContext* context,
                     const Tensor* past,
                     int batch_size,
                     int head_size,
                     int kv_sequence_length,
                     int& past_sequence_length) const;
#endif

 protected:
  // Keep the class layout identical in SHARED_PROVIDER and non-SHARED_PROVIDER builds.
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;

  template <typename KernelInfoType>
  AttentionBase(const KernelInfoType& info, bool require_same_hidden_size) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    is_unidirectional_ = info.template GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;
    do_rotary_ = info.template GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
    rotary_embedding_ = static_cast<int>(info.template GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0));
    mask_filter_value_ = info.template GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
    scale_ = info.template GetAttrOrDefault<float>("scale", 0.0f);
    if (!info.template GetAttrs<int64_t>("qkv_hidden_sizes", qkv_hidden_sizes_).IsOK()) {
      qkv_hidden_sizes_.clear();
    }

    past_present_share_buffer_ = info.template GetAttrOrDefault<int64_t>("past_present_share_buffer", 0LL);

    require_same_hidden_size_ = require_same_hidden_size;

#ifndef SHARED_PROVIDER
    SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
#endif
  }

  Status CheckMask(const Tensor* mask_index,
                   AttentionMaskType& mask_type,
                   int64_t& max_sequence_length,  // output: max_sequence_length when mask_index is 4D tensor
                   int64_t batch_size,
                   int64_t sequence_length,
                   int64_t total_sequence_length) const;

  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const Tensor*& mask_index,  // Dummy mask of shape (1 or batch_size, 1) will be updated to nullptr.
                     const Tensor* past,
                     const Tensor* attention_bias,
                     void* parameters,
                     const Tensor* past_seq_len = nullptr) const;

  int num_heads_;                          // number of attention heads
  bool is_unidirectional_;                 // whether every token can only attend to previous tokens.
  std::vector<int64_t> qkv_hidden_sizes_;  // Q, K, V hidden sizes parsed from the qkv_hidden_sizes attribute.
  bool require_same_hidden_size_;          // whether the implementation supports different hidden sizes of Q/K/V.
  bool past_present_share_buffer_;         // whether or not the past (if used) and present tensor share the same buffer
  bool do_rotary_;                         // whether or not to use rotary embeddings
  int rotary_embedding_;                   // rotary embedding dimension
  float mask_filter_value_;                // the value to be used for filtered out positions
  float scale_;                            // the scale to be used for softmax
};

#ifndef SHARED_PROVIDER
// Inline implementations of out-of-line methods for non-SHARED_PROVIDER builds
// (attention_base.cc definitions are used only in the SHARED_PROVIDER bridge path).
inline Status AttentionBase::CheckMask(const Tensor* mask_index,
                                       AttentionMaskType& mask_type,
                                       int64_t& max_sequence_length,
                                       int64_t batch_size,
                                       int64_t sequence_length,
                                       int64_t total_sequence_length) const {
  const auto& mask_dims = mask_index->Shape().GetDims();
  if (mask_dims.size() == 1) {
    if (mask_dims[0] != batch_size && mask_dims[0] != 2 * batch_size && mask_dims[0] != 3 * batch_size + 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'mask_index' with 1D data shall have length of batch_size or 2 * batch_size or 3 * batch_size + 2");
    }
    mask_type = (mask_dims[0] == batch_size ? AttentionMaskType::MASK_1D_KEY_SEQ_LEN : mask_dims[0] == 2 * batch_size ? AttentionMaskType::MASK_1D_END_START
                                                                                                                      : AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START);

    // Validate that end_position values (first batch_size elements) are non-negative.
    // Negative end_position causes out-of-bounds writes in PrepareMask.
    // Only validate when mask_index is on CPU; GPU tensors are clamped in the CUDA kernel.
    if (mask_index->Location().device.Type() == OrtDevice::CPU) {
      const int32_t* mask_data = mask_index->Data<int32_t>();
      for (int64_t i = 0; i < batch_size; i++) {
        if (mask_data[i] < 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "mask_index value ", mask_data[i], " at index ", i,
                                 " is negative. mask_index end_position values must be non-negative.");
        }
      }
    }
  } else if (mask_dims.size() == 2) {
    if (mask_dims[0] == batch_size && mask_dims[1] == total_sequence_length) {
      mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
    } else {
      if ((mask_dims[0] == batch_size || mask_dims[0] == 1) && mask_dims[1] == 1) {
        mask_type = AttentionMaskType::MASK_2D_DUMMY;
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
    mask_type = AttentionMaskType::MASK_3D_ATTENTION;
  } else if (mask_dims.size() == 4) {
    if (mask_dims[0] != batch_size || mask_dims[1] != 1 || mask_dims[2] != mask_dims[3] ||
        mask_dims[2] < total_sequence_length) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Inputs 'mask_index' with 4D data shall have shape "
          "batch_size x 1 x max_sequence_length x max_sequence_length)");
    }
    max_sequence_length = mask_dims[3];
    mask_type = AttentionMaskType::MASK_4D_MEGATRON;
    if (this->is_unidirectional_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'mask_index' with 4D data shall have is_unidirectional set to false");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'mask_index' is expected to have 1, 2, 3 or 4 dimensions, got ",
                           mask_dims.size());
  }

  return Status::OK();
}

inline Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                         const TensorShape& weights_shape,
                                         const TensorShape& bias_shape,
                                         const Tensor*& mask_index,
                                         const Tensor* past,
                                         const Tensor* attention_bias,
                                         void* parameters,
                                         const Tensor* past_seq_len) const {
  if (past != nullptr && attention_bias != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Attention cannot have both past and attention_bias");
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

  const auto& weights_dims = weights_shape.GetDims();
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
  if (past != nullptr) {
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
      if (past_seq_len == nullptr || !::onnxruntime::IsScalarOr1ElementVector(past_seq_len)) {
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
                             "when past_present_share_buffer, past tensor sequence must not smaller than total_sequence_length ");
    }
  }

  int64_t max_sequence_length = -1;
  AttentionMaskType mask_type = AttentionMaskType::MASK_NONE;
  if (mask_index != nullptr) {
    mask_type = AttentionMaskType::MASK_UNKNOWN;
    auto status = this->CheckMask(mask_index, mask_type,
                                  max_sequence_length, batch_size, sequence_length, total_sequence_length);
    if (status != Status::OK()) {
      return status;
    }

    if (mask_type == AttentionMaskType::MASK_2D_DUMMY) {
      mask_index = nullptr;
      mask_type = AttentionMaskType::MASK_NONE;
    }
  }

  gsl::span<const int64_t> attention_bias_dims;
  if (attention_bias != nullptr) {
    attention_bias_dims = attention_bias->Shape().GetDims();

    ORT_RETURN_IF_ERROR(::onnxruntime::contrib::multihead_attention_helper::CheckAttentionBias(
        attention_bias_dims, batch_size, num_heads_, sequence_length, total_sequence_length));
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
    output_parameters->past_present_share_buffer = (past_present_share_buffer_ != 0 && past != nullptr);
    output_parameters->do_rotary = do_rotary_;
    output_parameters->rotary_dim = rotary_embedding_ == 0 ? (int)(output_parameters->head_size) : rotary_embedding_;
    output_parameters->mask_filter_value = mask_filter_value_;
    output_parameters->scale = scale_;
    output_parameters->mask_type = mask_type;
    output_parameters->broadcast_attn_bias_dim_0 = attention_bias_dims.size() > 0 && attention_bias_dims[0] == 1;
    output_parameters->broadcast_attn_bias_dim_1 = attention_bias_dims.size() > 1 && attention_bias_dims[1] == 1;
    output_parameters->qkv_format = Q_K_V_BNSH;
  }

  return Status::OK();
}

inline Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                         const TensorShape& weights_shape,
                                         const TensorShape& bias_shape,
                                         const Tensor*& mask_index,
                                         const Tensor* past,
                                         const Tensor* attention_bias,
                                         void* parameters,
                                         const int max_threads_per_block,
                                         const Tensor* past_seq_len) const {
  if (num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(input_shape, weights_shape, bias_shape, mask_index, past, attention_bias, parameters, past_seq_len);
}

template <typename TOpKernelContext>
inline Tensor* AttentionBase::GetPresent(TOpKernelContext* context,
                                         const Tensor* past,
                                         int batch_size,
                                         int head_size,
                                         int kv_sequence_length,
                                         int& past_sequence_length) const {
  past_sequence_length = (nullptr != past) ? static_cast<int>(past->Shape().GetDims()[3]) : 0;
  std::array<int64_t, 5> present_dims{2, batch_size, num_heads_,
                                      static_cast<int64_t>(kv_sequence_length) + past_sequence_length, head_size};

  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(1, present_shape);
  if (nullptr != past && nullptr == present) {
    ORT_THROW("Expect to have present state output when past state input is given");
  }

  return present;
}
#endif  // SHARED_PROVIDER

}  // namespace contrib
}  // namespace onnxruntime

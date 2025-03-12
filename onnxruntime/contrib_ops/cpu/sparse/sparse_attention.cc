// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/sparse/sparse_attention.h"
#include "contrib_ops/cpu/sparse/sparse_attention_helper.h"
#include "contrib_ops/cpu/bert/rotary_helper.h"
#include "contrib_ops/cpu/bert/attention_utils.h"
#include "contrib_ops/cpu/bert/rotary_embedding.h"
#include "contrib_ops/cpu/bert/rotary_embedding_helper.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

#include <unsupported/Eigen/SpecialFunctions>
#include <vector>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      SparseAttention,                                                  \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()), \
      SparseAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
SparseAttention<T>::SparseAttention(const OpKernelInfo& info) : OpKernel(info), SparseAttentionBase(info) {
}

template <typename T>
Status SparseAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* block_row_indices = context->Input<Tensor>(5);
  const Tensor* block_col_indices = context->Input<Tensor>(6);
  const Tensor* total_seq_len = context->Input<Tensor>(7);
  const Tensor* total_key_lengths = context->Input<Tensor>(8);
  const Tensor* cos_cache = context->Input<Tensor>(9);
  const Tensor* sin_cache = context->Input<Tensor>(10);

  SparseAttentionParameters parameters = {};

  // Parameters from node attribute shall be set before calling CheckInputs
  parameters.sparse_block_size = sparse_block_size_;
  parameters.num_heads = num_heads_;
  parameters.kv_num_heads = kv_num_heads_;
  parameters.scale = scale_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;
  ORT_RETURN_IF_ERROR(sparse_attention_helper::CheckInputs(&parameters,
                                                           query,
                                                           key,
                                                           value,
                                                           past_key,
                                                           past_value,
                                                           cos_cache,
                                                           sin_cache,
                                                           block_row_indices,
                                                           block_col_indices,
                                                           total_key_lengths,
                                                           total_seq_len));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  int q_hidden_size = parameters.hidden_size;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(q_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  constexpr bool past_present_share_buffer = true;  // Only supports share buffer for past and present for now.
  parameters.past_present_share_buffer = past_present_share_buffer;

  int head_size = parameters.head_size;
  const int cache_length = past_present_share_buffer
                               ? parameters.max_cache_sequence_length
                               : parameters.total_sequence_length;
  std::vector<int64_t> present_k_shape({static_cast<int64_t>(batch_size),
                                        static_cast<int64_t>(kv_num_heads_),
                                        static_cast<int64_t>(cache_length),
                                        static_cast<int64_t>(head_size)});
  std::vector<int64_t> present_v_shape({static_cast<int64_t>(batch_size),
                                        static_cast<int64_t>(kv_num_heads_),
                                        static_cast<int64_t>(cache_length),
                                        static_cast<int64_t>(head_size)});
  Tensor* present_key = context->Output(1, present_k_shape);
  Tensor* present_value = context->Output(2, present_v_shape);

  // Check past and present share buffer.
  if (past_present_share_buffer) {
    ORT_ENFORCE(past_key->DataRaw() == present_key->DataRaw() && past_value->DataRaw() == present_value->DataRaw());
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue Q;
  OrtValue K;
  OrtValue V;

  const bool packed_qkv = parameters.is_packed_qkv;
  if (packed_qkv) {
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        allocator, batch_size, num_heads_ + 2 * kv_num_heads_, sequence_length, head_size, query, Q));
  } else {
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        allocator, batch_size, num_heads_, sequence_length, head_size, query, Q));
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        allocator, batch_size, kv_num_heads_, sequence_length, head_size, key, K));
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        allocator, batch_size, kv_num_heads_, sequence_length, head_size, value, V));
  }

  if (do_rotary_) {
    rotary_embedding_helper::RotaryParameters rotary_params = {};
    rotary_params.batch_size = batch_size;
    rotary_params.sequence_length = sequence_length;
    rotary_params.hidden_size = q_hidden_size;
    rotary_params.head_size = head_size;
    rotary_params.rotary_embedding_dim = parameters.rotary_dim;
    rotary_params.num_heads = num_heads_;
    rotary_params.max_sequence_length = sequence_length;  // unused
    rotary_params.seq_stride = head_size;
    rotary_params.head_stride = sequence_length * rotary_params.seq_stride;
    rotary_params.batch_stride = (packed_qkv ? (num_heads_ + 2 * kv_num_heads_) : num_heads_) *
                                 rotary_params.head_stride;
    rotary_params.position_ids_format = sequence_length == 1 ? 1 : 0;
    rotary_params.transposed = true;
    auto* tp = context->GetOperatorThreadPool();

    const bool is_prompt = parameters.total_sequence_length == parameters.sequence_length;
    std::vector<int64_t> pos_ids(is_prompt ? 1 : batch_size * sequence_length);
    if (is_prompt) {
      pos_ids[0] = static_cast<int64_t>(0);
    } else if (sequence_length == 1) {
      for (int b = 0; b < batch_size; b++) {
        pos_ids[b] = static_cast<int64_t>(total_key_lengths->Data<int32_t>()[b]) - 1;
      }
    } else {
      // This supports a rare case that sequence_length > 1 when it is not prompt.
      for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < sequence_length; s++) {
          pos_ids[b * sequence_length + s] = static_cast<int64_t>(total_key_lengths->Data<int32_t>()[b]) -
                                             (sequence_length - s);
        }
      }
    }

    const T* q_input;
    const T* k_input;
    T* q_rotary;
    T* k_rotary;
    if (packed_qkv) {
      OrtValue RotaryQKV;
      TensorShape qkv_shape({batch_size, num_heads_ + 2 * kv_num_heads_, sequence_length, head_size});
      Tensor::InitOrtValue(element_type, qkv_shape, allocator, RotaryQKV);
      q_input = Q.Get<Tensor>().Data<T>();
      k_input = q_input + num_heads_ * sequence_length * head_size;
      q_rotary = RotaryQKV.GetMutable<Tensor>()->MutableData<T>();
      k_rotary = q_rotary + num_heads_ * sequence_length * head_size;
      Q = RotaryQKV;
    } else {
      OrtValue RotaryQ;
      TensorShape q_shape({batch_size, num_heads_, sequence_length, head_size});
      Tensor::InitOrtValue(element_type, q_shape, allocator, RotaryQ);
      OrtValue RotaryK;
      TensorShape k_shape({batch_size, kv_num_heads_, sequence_length, head_size});
      Tensor::InitOrtValue(element_type, k_shape, allocator, RotaryK);
      q_input = Q.Get<Tensor>().Data<T>();
      k_input = K.Get<Tensor>().Data<T>();
      q_rotary = RotaryQ.GetMutable<Tensor>()->MutableData<T>();
      k_rotary = RotaryK.GetMutable<Tensor>()->MutableData<T>();
      Q = RotaryQ;
      K = RotaryK;
    }

    ORT_RETURN_IF_ERROR(RunRotaryEmbedding<T>(tp, rotary_params, q_input,
                                              pos_ids.data(), cos_cache->Data<T>(),
                                              sin_cache->Data<T>(), q_rotary, rotary_interleaved_));

    rotary_params.num_heads = kv_num_heads_;
    rotary_params.hidden_size = parameters.kv_hidden_size;
    if (!packed_qkv) {
      rotary_params.batch_stride = kv_num_heads_ * rotary_params.head_stride;
    }
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding<T>(tp, rotary_params, k_input,
                                              pos_ids.data(), cos_cache->Data<T>(),
                                              sin_cache->Data<T>(), k_rotary, rotary_interleaved_));
    if (packed_qkv) {
      const T* v_input = k_input + kv_num_heads_ * sequence_length * head_size;
      T* v_rotary = k_rotary + kv_num_heads_ * sequence_length * head_size;
      ORT_RETURN_IF_ERROR(rotary_helper::PackVIntoRotaryQKV<T>(tp,
                                                               parameters.batch_size,
                                                               parameters.sequence_length,
                                                               parameters.num_heads,
                                                               parameters.kv_num_heads,
                                                               parameters.head_size,
                                                               v_input,
                                                               v_rotary));
    }
  }

  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  // Compute the attention score and apply the score to V
  return ApplyAttention(Q.Get<Tensor>().Data<T>(), packed_qkv ? nullptr : K.Get<Tensor>().Data<T>(),
                        packed_qkv ? nullptr : V.Get<Tensor>().Data<T>(), past_key, past_value,
                        output, present_key, present_value,
                        total_key_lengths, block_row_indices, block_col_indices, parameters, allocator, context);
}
}  // namespace contrib
}  // namespace onnxruntime

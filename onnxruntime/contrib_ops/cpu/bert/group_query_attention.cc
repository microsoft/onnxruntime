// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "group_query_attention.h"
#include "group_query_attention_helper.h"
#include "attention_utils.h"
#include "rotary_embedding.h"
#include "rotary_embedding_helper.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

#include <unsupported/Eigen/SpecialFunctions>
#include <vector>
#include <iostream>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    GroupQueryAttention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>()}),
    GroupQueryAttention<float>);

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info) : OpKernel(info), GQAAttentionBase(info, false) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
}

template <typename T>
Status GroupQueryAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* seqlens_k = context->Input<Tensor>(5);
  const Tensor* total_seqlen = context->Input<Tensor>(6);
  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);

  GroupQueryAttentionParameters parameters = {};
  constexpr float scale = 1.0f;
  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &parameters,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                seqlens_k,
                                                                total_seqlen,
                                                                /*is_past_bsnh_*/ false,
                                                                scale));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int present_kv_seqlen = parameters.seqlen_present_kv_cache;
  int head_size = parameters.head_size;
  int q_hidden_size = parameters.hidden_size;
  const bool packed_qkv = parameters.is_packed_qkv;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(q_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_k_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_), static_cast<int64_t>(present_kv_seqlen), static_cast<int64_t>(head_size)});
  std::vector<int64_t> present_v_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_), static_cast<int64_t>(present_kv_seqlen), static_cast<int64_t>(head_size)});
  Tensor* present_k = context->Output(1, present_k_shape);
  Tensor* present_v = context->Output(2, present_v_shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  OrtValue Q;
  OrtValue K;
  OrtValue V;
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
    rotary_params.batch_stride = (packed_qkv ? (num_heads_ + 2 * kv_num_heads_) : num_heads_) * rotary_params.head_stride;
    rotary_params.position_ids_format = 1;
    rotary_params.transposed = true;
    auto* tp = context->GetOperatorThreadPool();
    OrtValue pos_ids;
    auto element_type = DataTypeImpl::GetType<int64_t>();
    std::vector<int64_t> pos_id_dim({batch_size, sequence_length});
    gsl::span<const int64_t> pos_id_dim_span{pos_id_dim};
    TensorShape pos_id_shape(pos_id_dim_span);
    Tensor::InitOrtValue(element_type, pos_id_shape, allocator, pos_ids);
    ORT_RETURN_IF_ERROR(group_query_attention_helper::GeneratePositionIds(seqlens_k->Data<int32_t>(), batch_size,
                                                                          sequence_length, pos_ids.GetMutable<Tensor>()->MutableData<int64_t>(), context));

    const T* q_input;
    const T* k_input;
    T* q_rotary;
    T* k_rotary;
    if (packed_qkv) {
      OrtValue RotaryQKV;
      element_type = DataTypeImpl::GetType<T>();
      std::vector<int64_t> qkv_dim({batch_size, num_heads_ + 2 * kv_num_heads_, sequence_length, head_size});
      gsl::span<const int64_t> qkv_dim_span{qkv_dim};
      TensorShape qkv_shape(qkv_dim_span);
      Tensor::InitOrtValue(element_type, qkv_shape, allocator, RotaryQKV);
      q_input = Q.Get<Tensor>().Data<T>();
      k_input = q_input + num_heads_ * sequence_length * head_size;
      q_rotary = RotaryQKV.GetMutable<Tensor>()->MutableData<T>();
      k_rotary = q_rotary + num_heads_ * sequence_length * head_size;
      Q = RotaryQKV;
    } else {
      OrtValue RotaryQ;
      element_type = DataTypeImpl::GetType<T>();
      std::vector<int64_t> q_dim({batch_size, num_heads_, sequence_length, head_size});
      gsl::span<const int64_t> q_dim_span{q_dim};
      TensorShape q_shape(q_dim_span);
      Tensor::InitOrtValue(element_type, q_shape, allocator, RotaryQ);
      OrtValue RotaryK;
      std::vector<int64_t> k_dim({batch_size, kv_num_heads_, sequence_length, head_size});
      gsl::span<const int64_t> k_dim_span{k_dim};
      TensorShape k_shape(k_dim_span);
      Tensor::InitOrtValue(element_type, k_shape, allocator, RotaryK);
      q_input = Q.Get<Tensor>().Data<T>();
      k_input = K.Get<Tensor>().Data<T>();
      q_rotary = RotaryQ.GetMutable<Tensor>()->MutableData<T>();
      k_rotary = RotaryK.GetMutable<Tensor>()->MutableData<T>();
      Q = RotaryQ;
      K = RotaryK;
    }
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding<T>(tp, rotary_params, q_input,
                                              pos_ids.Get<Tensor>().Data<int64_t>(), cos_cache->Data<T>(),
                                              sin_cache->Data<T>(), q_rotary, rotary_interleaved_));
    rotary_params.num_heads = kv_num_heads_;
    rotary_params.hidden_size = parameters.kv_hidden_size;
    if (!packed_qkv) {
      rotary_params.batch_stride = kv_num_heads_ * rotary_params.head_stride;
    }
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding<T>(tp, rotary_params, k_input,
                                              pos_ids.Get<Tensor>().Data<int64_t>(), cos_cache->Data<T>(),
                                              sin_cache->Data<T>(), k_rotary, rotary_interleaved_));
    if (packed_qkv) {
      const T* v_input = k_input + kv_num_heads_ * sequence_length * head_size;
      T* v_rotary = k_rotary + kv_num_heads_ * sequence_length * head_size;
      ORT_RETURN_IF_ERROR(group_query_attention_helper::PackVIntoRotaryQKV<T>(tp, parameters, v_input, v_rotary));
    }
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q.Get<Tensor>().Data<T>(), packed_qkv ? nullptr : K.Get<Tensor>().Data<T>(),
                        packed_qkv ? nullptr : V.Get<Tensor>().Data<T>(), past_key, past_value, output, present_k, present_v,
                        seqlens_k, parameters, context);
}
}  // namespace contrib
}  // namespace onnxruntime

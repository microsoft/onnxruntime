// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "group_query_attention.h"
#include "group_query_attention_helper.h"
#include "attention_utils.h"
// #include "rotary_embedding.h"
// #include "rotary_embedding_helper.h"

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

// TODO: get this right
// TODO: How can I specify float32 for cpu only
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

  // if (query->Shape().GetDims().size() == 5) {
  //   ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for CPU");
  // }
  // if (key != nullptr && key->Shape().GetDims().size() == 5) {
  //   ORT_NOT_IMPLEMENTED("Packed KV not implemented for CPU");
  // }

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

  // TODO: figure out parameters
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

  // TODO: update this comment
  // For each of Q/K/V, there are multiple scenarios:
  // 1) Combined QKV bias is null
  //    a) Q/K/V is (B, S, D)
  //    b) Q/K/V is (B, S, N, H)
  // 2) No packed QKV in Q
  //    a) Q/K/V has seq_len = 1
  //    b) Q/K/V has seq_len > 1

  // TODO: what's with the maybe?
  // TODO: account for packed qkv
  // TODO: make kernel take in BxSxNxH
  OrtValue Q;
  OrtValue K;
  OrtValue V;
  // TODO: refactor and organize
  if (packed_qkv) {
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        context, allocator, batch_size, num_heads_ + 2 * kv_num_heads_, sequence_length, head_size, query, Q));
    if (do_rotary_) {
      auto* tp = context->GetOperatorThreadPool();
      OrtValue pos_ids;
      auto element_type = DataTypeImpl::GetType<int64_t>();
      std::vector<int64_t> pos_id_dim({batch_size, sequence_length});
      gsl::span<const int64_t> pos_id_dim_span{pos_id_dim};
      TensorShape pos_id_shape(pos_id_dim_span);
      Tensor::InitOrtValue(element_type, pos_id_shape, allocator, pos_ids);
      ORT_RETURN_IF_ERROR(group_query_attention_helper::GeneratePositionIds(seqlens_k->Data<int32_t>(), batch_size,
                          sequence_length, pos_ids.GetMutable<Tensor>()->MutableData<int64_t>(), context));

      OrtValue RotaryQKV;
      element_type = DataTypeImpl::GetType<T>();
      std::vector<int64_t> qkv_dim({batch_size, num_heads_ + 2 * kv_num_heads_, sequence_length, head_size});
      gsl::span<const int64_t> qkv_dim_span{qkv_dim};
      TensorShape qkv_shape(qkv_dim_span);
      Tensor::InitOrtValue(element_type, qkv_shape, allocator, RotaryQKV);
      T* rotary_q = RotaryQKV.GetMutable<Tensor>()->MutableData<T>();
      ORT_RETURN_IF_ERROR(group_query_attention_helper::RunRotaryEmbedding<T>(tp, parameters, Q.Get<Tensor>().Data<T>(),
                          pos_ids.Get<Tensor>().Data<int64_t>(), cos_cache->Data<T>(),
                          sin_cache->Data<T>(), rotary_q, rotary_interleaved_, true));
      T* k = Q.GetMutable<Tensor>()->MutableData<T>() + num_heads_ * sequence_length * head_size;
      T* rotary_k = rotary_q + num_heads_ * sequence_length * head_size;
      ORT_RETURN_IF_ERROR(group_query_attention_helper::RunRotaryEmbedding<T>(tp, parameters, k,
                          pos_ids.Get<Tensor>().Data<int64_t>(), cos_cache->Data<T>(),
                          sin_cache->Data<T>(), rotary_k, rotary_interleaved_, false));
      // TODO: copy v into rotary_qkv
      T* v = k + kv_num_heads_ * sequence_length * head_size;
      T* rotary_v = rotary_k + kv_num_heads_ * sequence_length * head_size;
      ORT_RETURN_IF_ERROR(group_query_attention_helper::PackVIntoRotaryQKV<T>(tp, parameters, v, rotary_v));
      Q = RotaryQKV;
    }
  } else {
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        context, allocator, batch_size, num_heads_, sequence_length, head_size, query, Q));
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        context, allocator, batch_size, kv_num_heads_, sequence_length, head_size, key, K));
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        context, allocator, batch_size, kv_num_heads_, sequence_length, head_size, value, V));

    if (do_rotary_) {
      auto* tp = context->GetOperatorThreadPool();
      OrtValue pos_ids;
      auto element_type = DataTypeImpl::GetType<int64_t>();
      std::vector<int64_t> pos_id_dim({batch_size, sequence_length});
      gsl::span<const int64_t> pos_id_dim_span{pos_id_dim};
      TensorShape pos_id_shape(pos_id_dim_span);
      Tensor::InitOrtValue(element_type, pos_id_shape, allocator, pos_ids);
      ORT_RETURN_IF_ERROR(group_query_attention_helper::GeneratePositionIds(seqlens_k->Data<int32_t>(), batch_size,
                          sequence_length, pos_ids.GetMutable<Tensor>()->MutableData<int64_t>(), context));

      OrtValue RotaryQ;
      element_type = DataTypeImpl::GetType<T>();
      std::vector<int64_t> q_dim({batch_size, num_heads_, sequence_length, head_size});
      gsl::span<const int64_t> q_dim_span{q_dim};
      TensorShape q_shape(q_dim_span);
      Tensor::InitOrtValue(element_type, q_shape, allocator, RotaryQ);
      ORT_RETURN_IF_ERROR(group_query_attention_helper::RunRotaryEmbedding<T>(tp, parameters, Q.Get<Tensor>().Data<T>(),
                          pos_ids.Get<Tensor>().Data<int64_t>(), cos_cache->Data<T>(),
                          sin_cache->Data<T>(), RotaryQ.GetMutable<Tensor>()->MutableData<T>(), rotary_interleaved_, true));

      OrtValue RotaryK;
      std::vector<int64_t> k_dim({batch_size, num_heads_, sequence_length, head_size});
      gsl::span<const int64_t> k_dim_span{k_dim};
      TensorShape k_shape(k_dim_span);
      Tensor::InitOrtValue(element_type, k_shape, allocator, RotaryK);
      ORT_RETURN_IF_ERROR(group_query_attention_helper::RunRotaryEmbedding<T>(tp, parameters, K.Get<Tensor>().Data<T>(),
                          pos_ids.Get<Tensor>().Data<int64_t>(),
                          cos_cache->Data<T>(),
                          sin_cache->Data<T>(), RotaryK.GetMutable<Tensor>()->MutableData<T>(), rotary_interleaved_, false));
      Q = RotaryQ;
      K = RotaryK;
    }
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q.Get<Tensor>().Data<T>(), packed_qkv ? nullptr : K.Get<Tensor>().Data<T>(),
                        packed_qkv ? nullptr : V.Get<Tensor>().Data<T>(), past_key, past_value, output, present_k, present_v,
                        seqlens_k, parameters, context);
}
}  // namespace contrib
}  // namespace onnxruntime

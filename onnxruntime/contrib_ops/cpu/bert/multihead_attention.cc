// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_cpu_base.h"
#include "multihead_attention.h"
#include "multihead_attention_helper.h"
#include "attention_utils.h"

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/common/safeint.h"
#include "core/platform/env_var_utils.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

#include <algorithm>
#include <type_traits>
#include <unsupported/Eigen/SpecialFunctions>
#include <vector>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    MultiHeadAttention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MultiHeadAttention<float>);

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(const OpKernelInfo& info) : OpKernel(info), AttentionCPUBase(info, false) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;

  const auto& env = Env::Default();
  l2_cache_size_ = env.GetL2CacheSize();

  disable_flash_ = ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFlashAttention, false);
}

template <typename T>
Status MultiHeadAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* key_padding_mask = context->Input<Tensor>(4);
  const Tensor* attn_bias = context->Input<Tensor>(5);
  const Tensor* past_key = context->Input<Tensor>(6);
  const Tensor* past_value = context->Input<Tensor>(7);

  if (query->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for CPU");
  }
  if (key != nullptr && key->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed KV not implemented for CPU");
  }

  AttentionParameters parameters = {};
  bool past_present_share_buffer = false;
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      key_padding_mask,
                                                                      attn_bias,
                                                                      past_key,
                                                                      past_value,
                                                                      nullptr,
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_,
                                                                      scale_,
                                                                      is_unidirectional_,
                                                                      past_present_share_buffer,
                                                                      kMultiHeadAttention));

  const int batch_size = parameters.batch_size;
  const int q_sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int total_kv_sequence_length = parameters.total_sequence_length;
  int qk_head_size = parameters.head_size;
  int v_head_size = parameters.v_head_size;
  int qk_hidden_size = parameters.hidden_size;
  int v_hidden_size = parameters.v_hidden_size;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(q_sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  constexpr int q_bias_offset = 0;
  const int k_bias_offset = qk_hidden_size;
  const int v_bias_offset = 2 * qk_hidden_size;

  // If optional outputs aren't needed, present_k and present_v will be null
  std::vector<int64_t> present_k_shape({static_cast<int64_t>(batch_size),
                                        static_cast<int64_t>(num_heads_),
                                        static_cast<int64_t>(total_kv_sequence_length),
                                        static_cast<int64_t>(qk_head_size)});
  std::vector<int64_t> present_v_shape({static_cast<int64_t>(batch_size),
                                        static_cast<int64_t>(num_heads_),
                                        static_cast<int64_t>(total_kv_sequence_length),
                                        static_cast<int64_t>(v_head_size)});
  Tensor* present_k = context->Output(1, present_k_shape);
  Tensor* present_v = context->Output(2, present_v_shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  OrtValue Q;
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSHAndAddBias<T>(
      context, allocator, batch_size, num_heads_, q_sequence_length, qk_head_size, query, bias, q_bias_offset, Q));

  if (parameters.qkv_format == Q_K_V_BSNH_BNSH_BNSH) {
    // For cross attention with k and v in BNSH format, we assume that bias for key and value are zeros.
    // So we don't need to add bias for key and value here.
    assert(past_key == nullptr);
    assert(past_value == nullptr);
    return ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(),
                          key->Data<T>(),
                          value->Data<T>(),
                          key_padding_mask, nullptr /* past */, past_key, past_value, output, present_k, present_v,
                          batch_size, q_sequence_length, kv_sequence_length,
                          qk_head_size, v_head_size, v_hidden_size, attn_bias, context);
  }

  OrtValue K;
  OrtValue V;
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSHAndAddBias<T>(
      context, allocator, batch_size, num_heads_, kv_sequence_length, qk_head_size, key, bias, k_bias_offset, K));
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSHAndAddBias<T>(
      context, allocator, batch_size, num_heads_, kv_sequence_length, v_head_size, value, bias, v_bias_offset, V));

  if (std::is_same_v<T, float> &&
      !disable_flash_ &&
      !is_unidirectional_ &&
      key_padding_mask == nullptr &&
      attn_bias == nullptr &&
      past_key == nullptr &&
      past_value == nullptr &&
      present_k == nullptr &&
      present_v == nullptr &&
      l2_cache_size_ > 0) {
    MlasFlashAttentionThreadedArgs args;
    args.batch_size = batch_size;
    args.num_heads = num_heads_;
    args.q_sequence_length = q_sequence_length;
    args.kv_sequence_length = kv_sequence_length;
    args.qk_head_size = qk_head_size;
    args.v_head_size = v_head_size;
    args.scale = (scale_ == 0.0f) ? 1.0f / sqrt(static_cast<float>(qk_head_size)) : scale_;
    /*
      q_block_size, kv_block_size correspond to Br, Bc in the FlashAttention paper.
      Let M = l2_cache_size / sizeof(float)
      In the FlashAttention kernel, there are 5 big matrices that we need to keep in L2 cache:
        slice of Q -- [Br, qk_head_size]
        slice of K -- [Bc, qk_head_size]
        slice of V -- [Bc, v_head_size]
        result of QK -- [Br, Bc]
        temporary output (same shape as QKV) -- [Br, v_head_size]
      The total size of these matrices is (Br + Bc) * (qk_head_size + v_head_size) + Br * Bc
      By taking Bc = M / (4 * (qk_head_size + v_head_size)), and Br = min(Bc, qk_head_size + v_head_size), we have
        (Br + Bc) * (qk_head_size + v_head_size) + Br * Bc
        <= 2 * Bc * (qk_head_size + v_head_size) + Br * Bc
        <= 2 * Bc * (qk_head_size + v_head_size) + M/4
        <= 2 * M/4 + M/4 = M * (3/4)

      We leave 1/4 of the L2 cache for
        1. storing small tensors l and m
        2. instruction (code)
    */
    args.kv_block_size = l2_cache_size_ / (static_cast<int>(sizeof(float)) * 4 * (qk_head_size + v_head_size));
    args.kv_block_size = std::max(args.kv_block_size, 1);  // avoid kv_block_size = 0
    args.q_block_size = std::min(args.kv_block_size, qk_head_size + v_head_size);
    args.kv_block_size = std::min(args.kv_block_size, kv_sequence_length);  // No point to have kv_block_size > kv_sequence_length
    args.q_block_size = std::min(args.q_block_size, q_sequence_length);     // No point to have q_block_size > q_sequence_length

    auto* tp = context->GetOperatorThreadPool();
    args.thread_count = concurrency::ThreadPool::DegreeOfParallelism(tp);
    args.buffer_size_per_thread = (static_cast<size_t>(args.q_block_size) * 2 +
                                   static_cast<size_t>(args.q_block_size) * static_cast<size_t>(args.kv_block_size) +
                                   static_cast<size_t>(args.q_block_size) * static_cast<size_t>(args.v_head_size)) *
                                  sizeof(float);
    size_t buffer_bytes = args.buffer_size_per_thread * args.thread_count;
    IAllocatorUniquePtr<void> buffer = IAllocator::MakeUniquePtr<void>(allocator, buffer_bytes);

    args.buffer = reinterpret_cast<float*>(buffer.get());

    args.query = Q.Get<Tensor>().Data<float>();
    args.key = K.Get<Tensor>().Data<float>();
    args.value = V.Get<Tensor>().Data<float>();
    args.output = output->MutableData<float>();

    MlasFlashAttention(&args, tp);
    return Status::OK();
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(),
                        K.GetMutable<Tensor>()->MutableData<T>(),
                        V.GetMutable<Tensor>()->MutableData<T>(),
                        key_padding_mask, nullptr /* past */, past_key, past_value, output, present_k, present_v,
                        batch_size, q_sequence_length, kv_sequence_length,
                        qk_head_size, v_head_size, v_hidden_size, attn_bias, context);
}
}  // namespace contrib
}  // namespace onnxruntime

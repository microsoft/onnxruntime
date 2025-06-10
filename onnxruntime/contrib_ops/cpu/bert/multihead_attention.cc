// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_cpu_base.h"
#include "contrib_ops/cpu/bert/multihead_attention.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/cpu/bert/attention_utils.h"

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

  disable_decoder_attention_ = ParseEnvironmentVariableWithDefault<bool>(attention::kDisableDecoderAttention, false);
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
  const Tensor* past_sequence_length = context->Input<Tensor>(8);
  const Tensor* cache_indirection = context->Input<Tensor>(9);

  if (query->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for CPU");
  }
  if (key != nullptr && key->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed KV not implemented for CPU");
  }

  bool past_present_share_buffer = past_key != nullptr && past_sequence_length != nullptr;
  if (past_key != nullptr && past_sequence_length != nullptr && cache_indirection != nullptr) {
    ORT_ENFORCE(past_present_share_buffer);
  }

  AttentionParameters parameters = {};
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      key_padding_mask,
                                                                      attn_bias,
                                                                      past_key,
                                                                      past_value,
                                                                      cache_indirection,
                                                                      past_sequence_length,
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_,
                                                                      scale_,
                                                                      is_unidirectional_,
                                                                      past_present_share_buffer,
                                                                      kMultiHeadAttention));
  DUMP_CPU_STRING_INIT();
  DUMP_CPU_STRING("Batch size = ", parameters.batch_size);
  DUMP_CPU_STRING("Sequence length = ", parameters.sequence_length);
  DUMP_CPU_STRING("Past sequence length = ", parameters.past_sequence_length);
  DUMP_CPU_STRING("KV sequence length = ", parameters.kv_sequence_length);
  DUMP_CPU_STRING("Total sequence length = ", parameters.total_sequence_length);
  DUMP_CPU_STRING("Max sequence length = ", parameters.max_sequence_length);
  DUMP_CPU_STRING("Hidden size = ", parameters.hidden_size);
  DUMP_CPU_STRING("Head size = ", parameters.head_size);
  DUMP_CPU_STRING("Num heads = ", parameters.num_heads);
  DUMP_CPU_STRING("Buffer sharing = ", (parameters.past_present_share_buffer == true));
  DUMP_CPU_STRING("QKV format = ", parameters.qkv_format);
  DUMP_CPU_STRING("Beam width = ", parameters.beam_width);

  const int batch_size = parameters.batch_size;
  const int q_sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int total_sequence_length = parameters.total_sequence_length;
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

  // If optional outputs aren't needed, present_key, present_value, and output_qk will be null
  std::vector<int64_t> present_key_shape({static_cast<int64_t>(batch_size),
                                          static_cast<int64_t>(num_heads_),
                                          static_cast<int64_t>(parameters.max_sequence_length),
                                          static_cast<int64_t>(qk_head_size)});
  std::vector<int64_t> present_value_shape({static_cast<int64_t>(batch_size),
                                            static_cast<int64_t>(num_heads_),
                                            static_cast<int64_t>(parameters.max_sequence_length),
                                            static_cast<int64_t>(v_head_size)});
  std::vector<int64_t> output_qk_shape({static_cast<int64_t>(batch_size),
                                        static_cast<int64_t>(num_heads_),
                                        static_cast<int64_t>(q_sequence_length),
                                        static_cast<int64_t>(total_sequence_length)});
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = context->Output(3, output_qk_shape);

  bool use_decoder_masked_multihead_attention = false;
  if (cache_indirection != nullptr) {
    bool use_dmmha_self_attention = parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH &&
                                    parameters.past_present_share_buffer &&
                                    parameters.past_sequence_length > 0;
    bool use_dmmha_cross_attention = parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH &&
                                     past_key == nullptr && past_value == nullptr && nullptr != past_sequence_length &&
                                     parameters.past_sequence_length != *((*past_sequence_length).template Data<int32_t>());
    use_decoder_masked_multihead_attention = !disable_decoder_attention_ &&
                                             (use_dmmha_self_attention || use_dmmha_cross_attention) &&
                                             parameters.sequence_length == 1 &&
                                             parameters.head_size == parameters.v_head_size &&
                                             (parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING || parameters.mask_type == AttentionMaskType::MASK_NONE) &&
                                             nullptr != past_sequence_length && nullptr != cache_indirection;
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  OrtValue Q;
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSHAndAddBias<T>(
      context, allocator, batch_size, num_heads_, q_sequence_length, qk_head_size, query, bias, q_bias_offset, Q));

  if (parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH) {
    // For cross attention with k and v in BNSH format, we assume that bias for key and value are zeros.
    // So we don't need to add bias for key and value here.
    assert(past_key == nullptr);
    assert(past_value == nullptr);

    if (use_decoder_masked_multihead_attention) {
      parameters.total_sequence_length = parameters.kv_sequence_length;
      parameters.max_sequence_length = parameters.kv_sequence_length;
    }

    return ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(),
                          key->Data<T>(),
                          value->Data<T>(),
                          key_padding_mask, nullptr /* past */, past_key, past_value,
                          output, present_key, present_value, output_qk,
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
      past_sequence_length == nullptr &&
      cache_indirection == nullptr &&
      present_key == nullptr &&
      present_value == nullptr &&
      output_qk == nullptr &&
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

  if (use_decoder_masked_multihead_attention) {
    // No production use-case will incur this copy cost as the implementation of
    // DecoderMaskedMultiHeadAttention is written in such a way that the past and present buffers
    // must be shared to have parity in the outputs.
    // This is just to circumvent the OpTester's limitation of not being able to bind a specific
    // buffer to inputs/outputs.
    auto* past_key_data = (past_key == nullptr) ? nullptr : past_key->Data<T>();
    auto* past_value_data = (past_value == nullptr) ? nullptr : past_value->Data<T>();
    auto* present_key_data = (present_key == nullptr) ? nullptr : present_key->MutableData<T>();
    auto* present_value_data = (present_value == nullptr) ? nullptr : present_value->MutableData<T>();

    if (present_key_data != past_key_data) {
      DUMP_CPU_STRING("Copying past_key to present_key for OpTester");
      memcpy(present_key_data, past_key_data, past_key->SizeInBytes());
    }
    if (present_value_data != past_value_data) {
      DUMP_CPU_STRING("Copying past_value to present_value for OpTester");
      memcpy(present_value_data, past_value_data, past_value->SizeInBytes());
    }

    return ApplyAttentionWithBeams(Q.GetMutable<Tensor>()->MutableData<T>(),
                                   K.GetMutable<Tensor>()->MutableData<T>(),
                                   V.GetMutable<Tensor>()->MutableData<T>(),
                                   key_padding_mask, past_key, past_value, output, present_key, present_value,
                                   batch_size, *((*past_sequence_length).template Data<int32_t>()), parameters.max_sequence_length,
                                   qk_head_size, v_head_size, attn_bias, parameters.broadcast_attn_bias_dim_0,
                                   parameters.broadcast_attn_bias_dim_1, cache_indirection, context,
                                   parameters.beam_width, output_qk);
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(),
                        K.GetMutable<Tensor>()->MutableData<T>(),
                        V.GetMutable<Tensor>()->MutableData<T>(),
                        key_padding_mask, nullptr /* past */, past_key, past_value,
                        output, present_key, present_value, output_qk,
                        batch_size, q_sequence_length, kv_sequence_length,
                        qk_head_size, v_head_size, v_hidden_size, attn_bias, context);
}
}  // namespace contrib
}  // namespace onnxruntime

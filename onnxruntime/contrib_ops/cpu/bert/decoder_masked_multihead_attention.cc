// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_cpu_base.h"
#include "attention_utils.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/cpu/bert/decoder_masked_multihead_attention.h"

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {

// TODO: refactor
static constexpr int kPastSequenceLengthInputIndex = 7;
static constexpr int kBeamWidthInputIndex = 8;
static constexpr int kCacheIndirectionInputIndex = 9;
static constexpr int kPastInputIndex = 5;
static constexpr int kPresentOutputIndex = 1;
static constexpr int kQKOutputIndex = 3;
static constexpr int kBiasIndex = 10;

#define REGISTER_KERNEL_TYPED(T)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      DecoderMaskedMultiHeadAttention,                                        \
      kMSDomain,                                                              \
      1,                                                                      \
      T,                                                                      \
      kCpuExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                           \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                   \
          .MayInplace(kPastInputIndex + 1, kPresentOutputIndex + 1)           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex) \
          .InputMemoryType(OrtMemTypeCPUInput, kBeamWidthInputIndex),         \
      DecoderMaskedMultiHeadAttention<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
DecoderMaskedMultiHeadAttention<T>::DecoderMaskedMultiHeadAttention(const OpKernelInfo& info)
    : OpKernel(info), AttentionCPUBase(info, false) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  past_present_share_buffer_ = info.GetAttrOrDefault<int64_t>("past_present_share_buffer", 0LL);
  output_qk_ = info.GetAttrOrDefault<int64_t>("output_qk", 0LL);
}

template <typename T>
Status DecoderMaskedMultiHeadAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* attention_bias = context->Input<Tensor>(4);
  const Tensor* past_key = context->Input<Tensor>(kPastInputIndex);
  const Tensor* past_value = context->Input<Tensor>(kPastInputIndex + 1);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);
  const Tensor* beam_width = context->Input<Tensor>(kBeamWidthInputIndex);
  const Tensor* cache_indir = context->Input<Tensor>(kCacheIndirectionInputIndex);
  const Tensor* bias = context->Input<Tensor>(kBiasIndex);

  DecoderMaskedMultiHeadAttentionParams parameters;

  bool is_unidirectional = false;
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      mask_index,
                                                                      attention_bias,
                                                                      past_key,
                                                                      past_value,
                                                                      past_seq_len,
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_,
                                                                      scale_,
                                                                      is_unidirectional,
                                                                      past_present_share_buffer_,
                                                                      kDecoderMaskedMultiHeadAttention));

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;
  int head_size = parameters.head_size;
  int v_head_size = parameters.v_head_size;
  int hidden_size = parameters.hidden_size;
  int v_hidden_size = parameters.v_hidden_size;

  // This kernel is for decoding only (i.e.) sequence length has to be 1
  if (sequence_length != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input sequence length should be 1 to use DecoderMaskedMultiHeadAttention. "
                           "Actual length is ",
                           sequence_length);
  }

  if (head_size != v_head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "QK head size should be same as V head size to use DecoderMaskedMultiHeadAttention");
  }

  if (parameters.mask_type != AttentionMaskType::MASK_2D_KEY_PADDING &&
      parameters.mask_type != AttentionMaskType::MASK_NONE) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DecoderMaskedMultiHeadAttention only supports no mask or 2D key "
                           "padding mask of shape [batch, total_seq_length] currently");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      parameters.batch_size, parameters.num_heads,
      past_present_share_buffer_ ? parameters.max_sequence_length : parameters.total_sequence_length,
      head_size};
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(kPresentOutputIndex, present_shape);
  Tensor* present_value = context->Output(kPresentOutputIndex + 1, present_shape);
  Tensor* output_qk = nullptr;

  // Decoder cross-attention
  if (past_key == nullptr && present_key == nullptr) {
    if (attention_bias != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "DecoderMaskedMultiHeadAttention does not support attention bias for cross-attention");
    }

    parameters.is_cross_attention = true;
    parameters.total_sequence_length = parameters.kv_sequence_length;
    parameters.max_sequence_length = parameters.kv_sequence_length;
  } else {
    // Sanity check
    ORT_ENFORCE(past_present_share_buffer_);
    ORT_ENFORCE(past_key != nullptr && past_value != nullptr);

    auto* present_key_data = present_key->MutableData<T>();
    auto* present_value_data = present_value->MutableData<T>();
    auto* past_key_data = past_key->Data<T>();
    auto* past_value_data = past_value->Data<T>();

    if (present_key_data != past_key_data) {
      std::memcpy(present_key_data, past_key_data, past_key->SizeInBytes());
    }
    if (present_value_data != past_value_data) {
      std::memcpy(present_value_data, past_value_data, past_value->SizeInBytes());
    }

    parameters.is_cross_attention = false;
  }

  if (output_qk_) {
    int64_t qk_dims[] = {parameters.batch_size, parameters.num_heads, 1, parameters.total_sequence_length};
    TensorShape qk_shape(&qk_dims[0], sizeof(qk_dims) / sizeof(qk_dims[0]));
    output_qk = context->Output(kQKOutputIndex, qk_shape);
  }

  // Beam width (in case we are using this op inside BeamSearch)
  int beam_width_value = 1;
  if (beam_width != nullptr) {
    beam_width_value = static_cast<int>(*beam_width->Data<int32_t>());
  }

  // Cache indirection (in case we are using this op inside BeamSearch)
  if (beam_width_value > 1 && cache_indir == nullptr) {
    // If beam width > 1, then cache indirection buffer MUST be present
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "If beam width is greater than 1, then cache indirection buffer MUST be present");
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  OrtValue Q;
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSHAndAddBias<T>(
      context, allocator, batch_size, num_heads_, 1, head_size, query, bias, 0, Q));

  // Cross-attention case
  if (parameters.is_cross_attention) {
    return ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(),
                          key->Data<T>(),
                          value->Data<T>(),
                          mask_index, nullptr /* past */, past_key, past_value, output, present_key, present_value,
                          batch_size, 1 /* sequence_length */, parameters.kv_sequence_length,
                          head_size, v_head_size, v_hidden_size, attention_bias, context, output_qk);
  }

  OrtValue K, V;
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSHAndAddBias<T>(
      context, allocator, batch_size, num_heads_, 1, head_size, key, bias, hidden_size, K));
  ORT_RETURN_IF_ERROR(MaybeTransposeToBNSHAndAddBias<T>(
      context, allocator, batch_size, num_heads_, 1, v_head_size, value, bias, 2 * hidden_size, V));

  // Self-attention, !has_beams
  if (cache_indir == nullptr) {
    return ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(),
                          K.GetMutable<Tensor>()->MutableData<T>(),
                          V.GetMutable<Tensor>()->MutableData<T>(),
                          mask_index, nullptr /* past */, past_key, past_value, output, present_key, present_value,
                          batch_size, 1 /* sequence_length */, parameters.kv_sequence_length,
                          head_size, v_head_size, v_hidden_size, attention_bias, context, output_qk,
                          parameters.past_sequence_length, true /* past_present_share_buffer */);
  }

  // Self-attention, has_beams
  return ApplyAttentionWithBeams(Q.GetMutable<Tensor>()->MutableData<T>(),
                                 K.GetMutable<Tensor>()->MutableData<T>(),
                                 V.GetMutable<Tensor>()->MutableData<T>(),
                                 mask_index, past_key, past_value, output, present_key, present_value,
                                 batch_size, parameters.past_sequence_length, parameters.max_sequence_length,
                                 head_size, v_head_size, attention_bias, parameters.broadcast_attn_bias_dim_0,
                                 parameters.broadcast_attn_bias_dim_1, cache_indir, context,
                                 beam_width_value, output_qk);
}

template <typename T>
Status DecoderMaskedMultiHeadAttention<T>::ApplyAttentionWithBeams(
    const T* Q,
    const T* K,
    const T* V,
    const Tensor* mask_index,
    const Tensor* past_key,
    const Tensor* past_value,
    Tensor* output,
    Tensor* present_key,
    Tensor* present_value,
    int batch_size,
    int past_sequence_length,
    int max_sequence_length,
    int head_size,
    int v_head_size,
    const Tensor* attn_bias,
    bool broadcast_attn_bias_dim_0,
    bool broadcast_attn_bias_dim_1,
    const Tensor* cache_indir,
    OpKernelContext* context,
    int beam_width,
    Tensor* output_qk) const {
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();

  int total_sequence_length = past_sequence_length + 1;
  size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * total_sequence_length * sizeof(T);
  auto attention_probs = allocator->Alloc(bytes);
  BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

  T* output_qk_data = (output_qk != nullptr) ? output_qk->MutableData<T>() : nullptr;

  const int32_t* mask_index_data = mask_index != nullptr ? mask_index->Data<int32_t>() : nullptr;
  const T* attn_bias_data = attn_bias != nullptr ? attn_bias->Data<T>() : nullptr;

  ComputeAttentionProbsWithBeams(static_cast<T*>(attention_probs), Q, K, mask_index_data, batch_size,
                                 past_sequence_length, max_sequence_length, head_size, past_key->Data<T>(),
                                 present_key->MutableData<T>(), tp, attn_bias_data, broadcast_attn_bias_dim_0,
                                 broadcast_attn_bias_dim_1, cache_indir->Data<int32_t>(), beam_width, output_qk_data);

  // Compute the attentionScore * Value: out_tmp(B, N, 1, H_v) = attention_probs(B, N, 1, T) x V(B, N, T, H_v)
  auto out_tmp_data = allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * v_head_size * sizeof(T));
  BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(std::move(allocator)));

  ComputeVxAttentionScoreWithBeams(output->MutableData<T>(), static_cast<T*>(out_tmp_data),
                                   static_cast<const T*>(attention_probs), V, batch_size,
                                   past_sequence_length, max_sequence_length, v_head_size, past_value->Data<T>(),
                                   present_value->MutableData<T>(), cache_indir->Data<int32_t>(), beam_width, tp);

  return Status::OK();
}

template <typename T>
void DecoderMaskedMultiHeadAttention<T>::ComputeAttentionProbsWithBeams(
    T* attention_probs,
    const T* Q,
    const T* K,
    const int32_t* mask_index_data,
    int batch_size,
    int past_sequence_length,
    int max_sequence_length,
    int head_size,
    const T* past_key_data,
    T* present_key_data,
    ThreadPool* tp,
    const T* attn_bias_data,
    bool broadcast_attn_bias_dim_0,
    bool broadcast_attn_bias_dim_1,
    const int32_t* cache_indir_data,
    int beam_width,
    T* output_qk_data) const {
  float scale = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;

  TensorOpCost unit_cost;
  auto total_sequence_length = past_sequence_length + 1;
  const ptrdiff_t probs_matrix_size = total_sequence_length;
  const ptrdiff_t probs_matrix_bytes = probs_matrix_size * sizeof(T);

  unit_cost.compute_cycles = static_cast<double>((SafeInt<ptrdiff_t>(2) * head_size - 1) * total_sequence_length);
  unit_cost.bytes_loaded = static_cast<double>(SafeInt<ptrdiff_t>(2) * head_size * total_sequence_length * sizeof(T));
  unit_cost.bytes_stored = static_cast<double>(SafeInt<ptrdiff_t>(head_size) * total_sequence_length * sizeof(T));

  if (attn_bias_data != nullptr) {
    unit_cost.bytes_loaded += static_cast<double>(probs_matrix_bytes) * 2;
    unit_cost.bytes_stored += probs_matrix_bytes;
  }

  if (mask_index_data != nullptr) {
    unit_cost.bytes_stored += probs_matrix_bytes;
  }

  // Cost of appending current key to present key
  unit_cost.compute_cycles += static_cast<double>(head_size);
  unit_cost.bytes_loaded += static_cast<double>(head_size);

  // Parallel for loop
  const int loop_len = batch_size * num_heads_;
  ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t i = begin; i != end; ++i) {
      const std::ptrdiff_t batch_index = i / num_heads_;
      const std::ptrdiff_t head_index = i % num_heads_;
      const std::ptrdiff_t beam_batch_index = batch_index / beam_width;
      const T* q_vec = Q + i * head_size;
      const std::ptrdiff_t attn_bias_base_offset = ((broadcast_attn_bias_dim_0 ? 0 : (beam_batch_index * num_heads_)) +
                                                    (broadcast_attn_bias_dim_1 ? 0 : head_index)) *
                                                   probs_matrix_size;

      {
        // Calculate the latest position of the attention_probs
        // (1, H) x (T, H)^T -> (1, T)
        // Decompose into T (1, H) x (1, H)^T -> (1, 1) operations
        auto last_offset = past_sequence_length + i * probs_matrix_size;
        T* attention_probs_ptr = reinterpret_cast<T*>(attention_probs) + last_offset;
        math::Dot<float, CPUMathUtil>(head_size, q_vec, K + i * head_size, attention_probs_ptr, nullptr);

        // Apply the attention bias and mask
        if (attn_bias_data != nullptr) {
          *attention_probs_ptr += attn_bias_data[attn_bias_base_offset + past_sequence_length];
        }
        bool is_masked = (mask_index_data != nullptr) &&
                         (mask_index_data[(batch_index + 1) * total_sequence_length - 1] == 0);
        if (is_masked) {
          *attention_probs_ptr += mask_filter_value_;
        }
        *attention_probs_ptr *= scale;
      }

      {
        // Calculate the rest of the attention_probs
        for (std::ptrdiff_t j = 0; j < past_sequence_length; ++j) {
          const int* beam_indices = &cache_indir_data[batch_index * max_sequence_length];
          const std::ptrdiff_t beam_offset = static_cast<std::ptrdiff_t>(beam_indices[j]) * num_heads_ *
                                             max_sequence_length * head_size;
          const std::ptrdiff_t beam_batch_offset = (beam_batch_index * beam_width * num_heads_ + head_index) *
                                                   max_sequence_length * head_size;
          const T* past_k_vec = past_key_data + beam_batch_offset + beam_offset + j * head_size;
          T* output = reinterpret_cast<T*>(attention_probs) + j + i * probs_matrix_size;
          math::Dot<float, CPUMathUtil>(head_size, q_vec, past_k_vec, output, nullptr);
          // Apply the attention bias and mask
          if (attn_bias_data != nullptr) {
            *output += attn_bias_data[attn_bias_base_offset + j];
          }
          bool is_masked = (mask_index_data != nullptr) &&
                           (mask_index_data[batch_index * total_sequence_length + j] == 0);
          if (is_masked) {
            *output += mask_filter_value_;
          }
          *output *= scale;
        }
      }
      // Append current key to present key (past_present_share_buffer_ is true)
      memcpy(present_key_data + i * max_sequence_length * head_size, K + i * head_size, head_size * sizeof(T));
    }
  });

  if (output_qk_data != nullptr) {
    // Output the scaled Q*K^T if needed.
    memcpy(output_qk_data, attention_probs,
           SafeInt<size_t>(batch_size) * num_heads_ * total_sequence_length * sizeof(T));
  }

  // attention_probs(B, N, 1, T) = Softmax(attention_probs)
  {
    const int N = batch_size * num_heads_;
    const int D = total_sequence_length;
    ComputeAttentionSoftmaxInplace(attention_probs, N, D, tp);
  }
}

template <typename T>
void DecoderMaskedMultiHeadAttention<T>::ComputeVxAttentionScoreWithBeams(
    T* output,
    T* tmp_buffer,
    const T* attention_probs,
    const T* V,
    int batch_size,
    int past_sequence_length,
    int max_sequence_length,
    int v_head_size,
    const T* past_value_data,
    T* present_value_data,
    const int32_t* cache_indir_data,
    int beam_width,
    ThreadPool* tp) const {
  const int total_sequence_length = past_sequence_length + 1;

  TensorOpCost unit_cost;
  unit_cost.compute_cycles = static_cast<double>(SafeInt<ptrdiff_t>(2) * v_head_size * total_sequence_length);
  unit_cost.bytes_loaded = static_cast<double>(SafeInt<ptrdiff_t>(3) * v_head_size * total_sequence_length * sizeof(T));
  unit_cost.bytes_stored = static_cast<double>(SafeInt<ptrdiff_t>(2) * v_head_size * total_sequence_length * sizeof(T));

  // Cost of appending current value to present value
  unit_cost.compute_cycles += static_cast<double>(v_head_size);
  unit_cost.bytes_loaded += static_cast<double>(v_head_size);

  ThreadPool::TryParallelFor(
      tp, SafeInt<ptrdiff_t>(batch_size) * num_heads_, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          const std::ptrdiff_t batch_index = i / num_heads_;
          const std::ptrdiff_t head_index = i % num_heads_;
          const std::ptrdiff_t beam_batch_index = batch_index / beam_width;

          // Compute the attention score
          // (1, T) x (T, H_v) -> (1, H_v)
          // Decompose into T (1, 1) x (1, H_v) -> (1, H_v) operations and accumulate.
          {
            const T* attn_probs_ptr = attention_probs + (i + 1) * total_sequence_length - 1;
            math::Scale<T, CPUMathUtil>(v_head_size,
                                        static_cast<float>(*attn_probs_ptr),
                                        V + i * v_head_size,
                                        output + i * v_head_size,
                                        nullptr);
          }
          {
            for (std::ptrdiff_t j = 0; j < past_sequence_length; ++j) {
              const int* beam_indices = &cache_indir_data[batch_index * max_sequence_length];
              const std::ptrdiff_t beam_offset = static_cast<std::ptrdiff_t>(beam_indices[j]) * num_heads_ *
                                                 max_sequence_length * v_head_size;
              const std::ptrdiff_t beam_batch_offset = (beam_batch_index * beam_width * num_heads_ + head_index) *
                                                       max_sequence_length * v_head_size;
              const T* past_value_vec = past_value_data + beam_offset + beam_batch_offset;
              const T* attn_probs_ptr = attention_probs + j + i * total_sequence_length;

              math::Scale<T, CPUMathUtil>(v_head_size,
                                          static_cast<float>(*attn_probs_ptr),
                                          past_value_vec + j * v_head_size,
                                          tmp_buffer + i * v_head_size,
                                          nullptr);
              math::Add<T, CPUMathUtil>(v_head_size,
                                        output + i * v_head_size,
                                        tmp_buffer + i * v_head_size,
                                        output + i * v_head_size,
                                        nullptr);
            }
          }
          // Append current value to present value (past_present_share_buffer_ is true)
          memcpy(present_value_data + i * max_sequence_length * v_head_size,
                 V + i * v_head_size,
                 v_head_size * sizeof(T));
        }
      });
}

}  // namespace contrib
}  // namespace onnxruntime

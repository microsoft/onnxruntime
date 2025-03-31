// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/attention_cpu_base.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cpu/bert/attention_utils.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/cpu/bert/decoder_masked_multihead_attention.h"
#include "core/platform/env_var_utils.h"

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

  DecoderMaskedMultiHeadAttentionParameters parameters;

  bool is_unidirectional = false;
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      mask_index,
                                                                      attention_bias,
                                                                      past_key,
                                                                      past_value,
                                                                      cache_indir,
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
                          mask_index, nullptr /* past */, past_key, past_value, output, present_key, present_value, output_qk,
                          batch_size, 1 /* sequence_length */, parameters.kv_sequence_length,
                          head_size, v_head_size, v_hidden_size, attention_bias, context);
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
                          mask_index, nullptr /* past */, past_key, past_value, output, present_key, present_value, output_qk,
                          batch_size, 1 /* sequence_length */, parameters.kv_sequence_length,
                          head_size, v_head_size, v_hidden_size, attention_bias, context,
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

}  // namespace contrib
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/cuda/bert/decoder_masked_multihead_attention.h"
#include "contrib_ops/cuda/bert/fastertransformer_decoder_attention/decoder_masked_multihead_attention_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// TODO: refactor
static constexpr int kPastSequenceLengthInputIndex = 7;
static constexpr int kBeamWidthInputIndex = 8;
static constexpr int kCacheIndirectionInputIndex = 9;
static constexpr int kPastInputIndex = 5;
static constexpr int kPresentOutputIndex = 1;
static constexpr int kBiasIndex = 10;

#define REGISTER_KERNEL_TYPED(T1, T2)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      DecoderMaskedMultiHeadAttention,                                        \
      kMSDomain,                                                              \
      1,                                                                      \
      T1,                                                                     \
      kCudaExecutionProvider,                                                 \
      (*KernelDefBuilder::Create())                                           \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                   \
          .MayInplace(kPastInputIndex + 1, kPresentOutputIndex + 1)           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())             \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex) \
          .InputMemoryType(OrtMemTypeCPUInput, kBeamWidthInputIndex),         \
      DecoderMaskedMultiHeadAttention<T1, T2>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(MLFloat16, uint16_t)

template <typename T1, typename T2>
DecoderMaskedMultiHeadAttention<T1, T2>::DecoderMaskedMultiHeadAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  past_present_share_buffer_ = info.GetAttrOrDefault<int64_t>("past_present_share_buffer", 0LL);
}

template <typename T1, typename T2>
Status DecoderMaskedMultiHeadAttention<T1, T2>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* relative_position_bias = context->Input<Tensor>(4);
  const Tensor* past_key = context->Input<Tensor>(kPastInputIndex);
  const Tensor* past_value = context->Input<Tensor>(kPastInputIndex + 1);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);
  const Tensor* beam_width = context->Input<Tensor>(kBeamWidthInputIndex);
  const Tensor* cache_indir = context->Input<Tensor>(kCacheIndirectionInputIndex);
  const Tensor* bias = context->Input<Tensor>(kBiasIndex);

  auto& device_prop = GetDeviceProp();
  DecoderMaskedMultiHeadAttentionParams parameters;
  bool is_dmmha_packing = (key == nullptr && value == nullptr);
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      mask_index,
                                                                      relative_position_bias,
                                                                      past_key,
                                                                      past_value,
                                                                      past_seq_len,
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_,
                                                                      scale_,
                                                                      past_present_share_buffer_,
                                                                      is_dmmha_packing,  // dmmha_packing
                                                                      device_prop.maxThreadsPerBlock));

  if (bias) {
    const T1* bias_data = bias->Data<T1>();
    parameters.q_bias = const_cast<T1*>(bias_data);
    parameters.k_bias = const_cast<T1*>(bias_data + parameters.hidden_size);
    parameters.v_bias = const_cast<T1*>(bias_data + 2LL * parameters.hidden_size);
  }

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  // This kernel is for decoding only (i.e.) sequence length has to be 1
  if (sequence_length != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input sequence length should be 1 to use DecoderMaskedMultiHeadAttention");
  }

  if (parameters.head_size != parameters.v_head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QK head size should be same as V head size to use DecoderMaskedMultiHeadAttention");
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
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      parameters.batch_size, parameters.num_heads,
      past_present_share_buffer_ ? parameters.max_sequence_length : parameters.total_sequence_length,
      parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(kPresentOutputIndex, present_shape);
  Tensor* present_value = context->Output(kPresentOutputIndex + 1, present_shape);

  auto cuda_stream = Stream(context);

  parameters.is_mha = true;

  // Update the q buffers
  parameters.q = const_cast<T1*>(query->Data<T1>());

  // Update the relative position bias for self attention
  if (relative_position_bias != nullptr) {
    parameters.relative_attention_bias = const_cast<T1*>(relative_position_bias->Data<T1>());
  }

  // Decoder cross-attention
  if (past_key == nullptr && present_key == nullptr) {
    if (relative_position_bias != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "DecoderMaskedMultiHeadAttention does not support relative position bias for cross-attention");
    }

    parameters.is_cross_attention = true;
    parameters.total_sequence_length = parameters.kv_sequence_length;
    parameters.max_sequence_length = parameters.kv_sequence_length;
    // parameters.k and paraneters.v are nullptr
    parameters.k_cache = const_cast<T1*>(key->Data<T1>());
    parameters.v_cache = const_cast<T1*>(value->Data<T1>());
    parameters.k_bias = nullptr;
    parameters.v_bias = nullptr;

  } else {
    // Sanity check
    ORT_ENFORCE(past_present_share_buffer_);
    ORT_ENFORCE(past_key != nullptr && past_value != nullptr);

    auto* present_key_data = present_key->MutableData<T1>();
    auto* present_value_data = present_value->MutableData<T1>();
    auto* past_key_data = past_key->Data<T1>();
    auto* past_value_data = past_value->Data<T1>();

    // No production use-case will incur this copy cost as the implementation of
    // GreedySearch/BeamSearch is written in such a way that the past and present buffers
    // will be shared.
    // This is just to circumvent the OpTester's limitation of not being able to bind a specific
    // buffer to inputs/outputs.
    if (present_key_data != past_key_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(present_key_data, past_key_data, past_key->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, cuda_stream));
    }
    if (present_value_data != past_value_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(present_value_data, past_value_data, past_value->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, cuda_stream));
    }

    parameters.is_cross_attention = false;
    parameters.is_packed_qkv = is_dmmha_packing;

    parameters.k = is_dmmha_packing
                       ? const_cast<T1*>(query->Data<T1>() + parameters.hidden_size)
                       : const_cast<T1*>(key->Data<T1>());
    parameters.v = is_dmmha_packing
                       ? const_cast<T1*>(query->Data<T1>() + 2 * static_cast<size_t>(parameters.hidden_size))
                       : const_cast<T1*>(value->Data<T1>());
    parameters.k_cache = present_key_data;
    parameters.v_cache = present_value_data;
  }

  parameters.out = output->MutableDataRaw();

  // Scale
  // If the scale is not provided - use `1/sqrt(head_size)`
  if (parameters.scale == 0.f) {
    parameters.scale = 1.f / sqrtf(static_cast<float>(parameters.head_size));
  }

  // Mask
  if (parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING) {
    parameters.mask = mask_index->Data<int32_t>();
  }

  // Beam width (in case we are using this op inside BeamSearch)
  if (beam_width != nullptr) {
    parameters.beam_width = static_cast<int>(*beam_width->Data<int32_t>());
  }

  // Cache indirection (in case we are using this op inside BeamSearch)
  if (parameters.beam_width > 1) {
    // If beam width > 1, then cache indirection buffer MUST be present
    if (cache_indir == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "If beam width is greater than 1, then cache indirection buffer MUST be present");
    }

    parameters.cache_indir = cache_indir->Data<int32_t>();
  }

  switch (parameters.head_size) {
    case 32:
      mmha_launch_kernel<T2, 32>(parameters, cuda_stream);
      break;

    case 64:
      mmha_launch_kernel<T2, 64>(parameters, cuda_stream);
      break;

    case 128:
      mmha_launch_kernel<T2, 128>(parameters, cuda_stream);
      break;

    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Unsupported head size in DecoderMaskedMultiHeadAttention. "
                             "Got head size: ",
                             parameters.head_size);
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

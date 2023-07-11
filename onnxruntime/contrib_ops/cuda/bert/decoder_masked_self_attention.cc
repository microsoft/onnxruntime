// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/decoder_masked_self_attention.h"
#include "contrib_ops/cuda/bert/fastertransformer_decoder_attention/decoder_masked_multihead_attention_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

static constexpr int kPastSequenceLengthInputIndex = 6;
static constexpr int kBeamWidthInputIndex = 7;
static constexpr int kCacheIndirectionInputIndex = 8;
static constexpr int kPastInputIndex = 4;
static constexpr int kPresentOutputIndex = 1;

#define REGISTER_KERNEL_TYPED(T1, T2)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      DecoderMaskedSelfAttention,                                             \
      kMSDomain,                                                              \
      1,                                                                      \
      T1,                                                                     \
      kCudaExecutionProvider,                                                 \
      (*KernelDefBuilder::Create())                                           \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())             \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex) \
          .InputMemoryType(OrtMemTypeCPUInput, kBeamWidthInputIndex),         \
      DecoderMaskedSelfAttention<T1, T2>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(MLFloat16, uint16_t)

template <typename T1, typename T2>
Status DecoderMaskedSelfAttention<T1, T2>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(kPastInputIndex);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);
  const Tensor* beam_width = context->Input<Tensor>(kBeamWidthInputIndex);
  const Tensor* cache_indir = context->Input<Tensor>(kCacheIndirectionInputIndex);

  auto& device_prop = GetDeviceProp();
  DecoderMaskedMultiHeadAttentionParams parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  relative_position_bias,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock,
                                  past_seq_len));

  // Sanity check
  ORT_ENFORCE(past_present_share_buffer_);

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  // This kernel is for decoding only (i.e.) sequence length has to be 1
  if (sequence_length != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input sequence length should be 1 to use DecoderMaskedSelfAttention");
  }

  ORT_ENFORCE(parameters.sequence_length == parameters.kv_sequence_length);

  // TODO(hasesh): If there is a need, we will support this later
  if (parameters.head_size != parameters.v_head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QK head size should be same as V head size to use DecoderMaskedSelfAttention");
  }

  // TODO(hasesh): If there is a need, we will support this later
  if (relative_position_bias != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "DecoderMaskedSelfAttention does not support relative position bias currently");
  }

  // TODO(hasesh): Support more mask types. Currently, it only supports the HuggingFace GreedySearch/BeamSearch pattern.
  if (parameters.mask_type != AttentionMaskType::MASK_2D_KEY_PADDING &&
      parameters.mask_type != AttentionMaskType::MASK_NONE) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "DecoderMaskedSelfAttention only supports no mask or 2D key "
                           "padding mask of shape [batch, total_seq_length] currently");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  // Present input will have the same shape as the past input
  Tensor* present = context->Output(kPresentOutputIndex, past->Shape());

  auto cuda_stream = Stream(context);

  auto* present_data = present->MutableData<T1>();
  auto* past_data = past->Data<T1>();

  // No production use-case will incur this copy cost as the implementation of
  // GreedySearch/BeamSearch is written in such a way that the past and present buffers
  // will be shared.
  // This is just to circumvent the OpTester's limitation of not being able to bind a specific
  // buffer to inputs/outputs.
  // TODO(hasesh): Enhance the OpTester to support binding buffers for inputs/outputs
  // If ever a production use-case using GreedySearch/BeamSearch incurs this copy test, we
  // will have to debug the GreedySearch/BeamSearch operator
  if (present_data != past_data) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(present_data, past_data, past->SizeInBytes(),
                                         cudaMemcpyDeviceToDevice, cuda_stream));
  }

  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T1>::MappedType CudaT;

  IAllocatorUniquePtr<T1> gemm_buffer;
  int m = batch_size * sequence_length;
  int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
  int k = parameters.input_hidden_size;
  gemm_buffer = GetScratchBuffer<T1>(static_cast<size_t>(m) * n, context->GetComputeStream());

  CudaT one = ToCudaType<T1>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T1>::FromFloat(0.0f);

  // QKV Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrices into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T1>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T1>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // Update the q, k, and v buffers
  parameters.q = gemm_buffer.get();
  parameters.k = reinterpret_cast<CudaT*>(gemm_buffer.get()) + parameters.hidden_size;
  parameters.v = reinterpret_cast<CudaT*>(gemm_buffer.get()) + 2 * static_cast<int64_t>(parameters.hidden_size);

  // Update the q, k, and v bias
  const T1* bias_data = bias->Data<T1>();
  parameters.q_bias = const_cast<T1*>(bias_data);
  parameters.k_bias = const_cast<T1*>(bias_data + parameters.hidden_size);
  parameters.v_bias = const_cast<T1*>(bias_data + 2 * static_cast<int64_t>(parameters.hidden_size));

  // Half of the past/present buffer correspond to K - the other half is V.
  auto k_size = present->Shape().Size() / 2;
  parameters.k_cache = present->MutableDataRaw();
  parameters.v_cache = present->MutableData<T1>() + k_size;
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

  // NeoX rotary embedding
  if (do_rotary_) {
    ORT_ENFORCE(parameters.head_size == 64 || parameters.head_size == 128,
                "Current implementation of rotary embedding only supports head size of 64 or 128");
    parameters.rotary_embedding_dim = parameters.head_size;
    parameters.t_step = parameters.past_sequence_length;
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
                             "Unsupported head size in DecoderMaskedSelfAttention. "
                             "Got head size: ",
                             parameters.head_size);
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

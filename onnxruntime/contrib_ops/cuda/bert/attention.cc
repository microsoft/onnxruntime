// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kPastSequenceLengthInputIndex = 6;
constexpr int kPastInputIndex = 4;
constexpr int kPresentOutputIndex = 1;

#define REGISTER_KERNEL_TYPED(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      Attention,                                                               \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info, false) {
  disable_fused_self_attention_ =
      sizeof(T) != 2 ||
      ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedSelfAttention, false);

  enable_trt_flash_attention_ =
      sizeof(T) == 2 &&
      !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);

  enable_fused_causal_attention_ =
      sizeof(T) == 2 &&
      ParseEnvironmentVariableWithDefault<bool>(attention::kEnableFusedCausalAttention, false);

#if USE_MEMORY_EFFICIENT_ATTENTION
  disable_memory_efficient_attention_ =
      ParseEnvironmentVariableWithDefault<bool>(attention::kDisableMemoryEfficientAttention, false);
#else
  disable_memory_efficient_attention_ = true;
#endif

#if USE_FLASH_ATTENTION
  disable_flash_attention_ =
      sizeof(T) != 2 ||
      onnxruntime::ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFlashAttention, false);
  min_seq_len_for_flash_attention_packed_qkv_ = ParseEnvironmentVariableWithDefault<int>(
      attention::kMinSeqLenForFlashAttentionPackedQKV,
      attention::kDefaultMinSeqLenForFlashAttentionPackedQKV);
#else
  disable_flash_attention_ = true;
  min_seq_len_for_flash_attention_packed_qkv_ = 0;
#endif
}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(kPastInputIndex);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  parameters.use_tf32 = UseTF32();

  // Use the second dimension from weight for bias to get q_hidden_size when bias is nullptr
  std::vector<int64_t> bias_dims{weights->Shape().GetDims()[1]};
  const TensorShape bias_shape{bias_dims};
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias != nullptr ? bias->Shape() : bias_shape,
                                  mask_index,
                                  past,
                                  relative_position_bias,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock,
                                  past_seq_len));
  assert(parameters.sequence_length == parameters.kv_sequence_length);  // self attention

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      2, parameters.batch_size, parameters.num_heads,
      past_present_share_buffer_ ? parameters.max_sequence_length : parameters.total_sequence_length,
      parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(kPresentOutputIndex, present_shape);

  MHARunner* fused_runner = nullptr;

  // Check whether we can use fused kernel
  const int sm = device_prop.major * 10 + device_prop.minor;
  const bool is_mask_1d_seq_len = parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             (nullptr == relative_position_bias) &&
                             nullptr == past &&
                             nullptr == present &&
                             parameters.hidden_size == parameters.v_hidden_size &&
                             nullptr == mask_index &&
                             onnxruntime::flash::is_supported(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.num_heads);
  // When input is packed QKV format, TensorRT kernel might be faster when sequence length <= 512.
  if (use_flash_attention && parameters.sequence_length < min_seq_len_for_flash_attention_packed_qkv_) {
    use_flash_attention = false;
  }
  // Allocate buffers
  size_t softmax_lse_accum_bytes = 0;
  size_t out_accum_bytes = 0;
  if (use_flash_attention) {
    using namespace std;
    auto [num_splits, slse_accum_bytes, o_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, parameters.kv_sequence_length, parameters.num_heads,
        parameters.head_size, device_prop.multiProcessorCount);
    parameters.num_splits = num_splits;
    softmax_lse_accum_bytes = slse_accum_bytes;
    out_accum_bytes = o_accum_bytes;
  }
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());
#else
  constexpr bool use_flash_attention = false;
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());  // nullptr
  auto out_accum_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());          // nullptr
#endif

  if (!use_flash_attention) {
    if (is_unidirectional_) {  // GPT
      if (enable_fused_causal_attention_) {
        // GPT fused kernels requires left side padding. mask can be:
        //     none (no padding), 1D sequence lengths or 2d mask.
        // Fused kernels don't support different sequence lengths of q and kv, so only apply to the first token
        // where past state is empty.
        bool is_mask_2d_key_padding = parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING;
        bool use_causal_fused_runner = (nullptr == mask_index || is_mask_1d_seq_len || is_mask_2d_key_padding) &&
                                       nullptr == relative_position_bias &&
                                       parameters.past_sequence_length == 0 &&
                                       parameters.hidden_size == parameters.v_hidden_size &&
                                       FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, sequence_length,
                                                                          enable_trt_flash_attention_, true);
        if (use_causal_fused_runner) {
          // Here we assume that num_heads, head_size and is_unidirectional does not change for an Attention node.
          if (nullptr == fused_fp16_runner_.get()) {
            std::call_once(fused_fp16_runner_created_, [&]() {
              fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(num_heads_, parameters.head_size, sm, is_unidirectional_,
                                                                enable_trt_flash_attention_, parameters.scale);
            });
          }

          // Here we assume all causal kernels can be loaded into shared memory. TODO: add a function to check.
          fused_runner = fused_fp16_runner_.get();
        }
      }
    } else {  // BERT
      bool use_fused_runner = !disable_fused_self_attention_ &&
                              (nullptr == mask_index || is_mask_1d_seq_len) &&
                              nullptr == past &&
                              nullptr == present &&
                              nullptr == relative_position_bias &&
                              parameters.hidden_size == parameters.v_hidden_size &&
                              FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, sequence_length,
                                                                 enable_trt_flash_attention_, false);

      if (use_fused_runner) {
        // Here we assume that num_heads, head_size and is_unidirectional does not change for an Attention node.
        if (nullptr == fused_fp16_runner_.get()) {
          std::call_once(fused_fp16_runner_created_, [&]() {
            fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(num_heads_, parameters.head_size, sm, is_unidirectional_,
                                                              enable_trt_flash_attention_, parameters.scale);
          });
        }

        // In case some kernel not loaded due to shared memory limit, we need to double check here.
        const int S = fused_fp16_runner_->getSFromMaxSeqLen(sequence_length);
        if (fused_fp16_runner_->isValid(S)) {
          fused_runner = fused_fp16_runner_.get();
        }
      }
    }
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  bool use_memory_efficient_attention =
      !use_flash_attention &&
      fused_runner == nullptr &&
      !disable_memory_efficient_attention_ &&
      nullptr == past &&
      nullptr == present &&
      (parameters.head_size & 7) == 0 &&
      (parameters.v_head_size & 7) == 0 &&
      (nullptr == mask_index || parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START) &&
      (sizeof(T) == 2 || parameters.sequence_length >= attention::kMinSeqLenForMemoryEfficientAttentionFp32) &&
      has_memory_efficient_attention(sm, sizeof(T) == 2);

  if (use_memory_efficient_attention) {
    bool is_good_for_rpb = relative_position_bias != nullptr && parameters.sequence_length % (4 * sizeof(T)) == 0;
    use_memory_efficient_attention = (nullptr == relative_position_bias || is_good_for_rpb);
  }
#else
  constexpr bool use_memory_efficient_attention = false;
#endif

  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  int m = batch_size * sequence_length;
  int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
  int k = parameters.input_hidden_size;
  IAllocatorUniquePtr<void> gemm_buffer = IAllocator::MakeUniquePtr<void>(allocator, static_cast<size_t>(m * n) * sizeof(T), false, context->GetComputeStream());

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop, UseTF32()));

  constexpr size_t element_size = sizeof(T);
  constexpr bool use_fused_cross_attention = false;
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   parameters.kv_sequence_length,
                                                   parameters.total_sequence_length,
                                                   fused_runner,
                                                   use_flash_attention,
                                                   use_fused_cross_attention,
                                                   use_memory_efficient_attention);
  IAllocatorUniquePtr<void> work_space = IAllocator::MakeUniquePtr<void>(allocator, workSpaceSize, false, context->GetComputeStream());
  ;

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = reinterpret_cast<CudaT*>(gemm_buffer.get());
  if (nullptr != bias) {
    data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  }
  if (nullptr != mask_index) {
    data.mask_index = mask_index->Data<int>();
    data.mask_index_dims = mask_index->Shape().GetDims();
  }
  if (nullptr != past) {
    data.past = reinterpret_cast<const CudaT*>(past->Data<T>());
  }
  if (nullptr != relative_position_bias) {
    data.relative_position_bias = reinterpret_cast<const CudaT*>(relative_position_bias->Data<T>());
  }
  data.has_qkv_workspace = true;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  if (nullptr != present) {
    data.present = reinterpret_cast<CudaT*>(present->MutableData<T>());
  }
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  if (softmax_lse_accum_buffer != nullptr) {
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
  }
  if (out_accum_buffer != nullptr) {
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }

  return QkvToContext<CudaT>(device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

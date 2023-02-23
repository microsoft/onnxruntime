// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/packed_attention.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"

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
PackedAttention<T>::PackedAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int32_t>(num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  if (!info.GetAttrs<int64_t>("qkv_hidden_sizes", qkv_hidden_sizes_).IsOK()) {
    qkv_hidden_sizes_.clear();
  }

  disable_fused_runner_ = sizeof(T) != 2 ||
                          ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedAttention, false);

  enable_trt_flash_attention_ = sizeof(T) == 2 &&
                                !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);

#if USE_FLASH_ATTENTION
  disable_memory_efficient_attention_ = ParseEnvironmentVariableWithDefault<bool>(attention::kDisableMemoryEfficientAttention, false);
#else
  disable_memory_efficient_attention_ = true;
#endif
}

Status PackedAttention<T>::CheckInputs(const TensorShape& input_shape,
                                       const TensorShape& weights_shape,
                                       const TensorShape& bias_shape,
                                       const TensorShape& packing_token_offset_shape,
                                       const TensorShape& cu_seq_len_shape,
                                       const Tensor* relative_position_bias,
                                       PackedAttentionParameters& parameters) const {
  // Abbreviation and Meanings:
  //   T:    token_count
  //   B:    batch_size
  //   S:    sequence_length (input sequence length of query)
  //   N:    num_heads
  //   H:    head size for Q and K, aka q_head_size or v_head_size or qk_head_size
  //   H_v:  v_head_size
  //   D_i:  input hidden size
  //   D:    hidden size for Q and K (D = N * H), aka q_hidden_size or k_hidden_size or qk_hidden_size
  //   D_v:  v_hidden_size = num_heads * v_head_size

  // Input shapes:
  //   input        (Q/K/V)    : (T, D_i)
  //   weights      (Q/K/V)    : (D_i, D + D + D_v)
  //   bias         (Q/K/V)    : (D + D + D_v)
  //   token_offset            : (B, S)
  //   cu_seq_len_shape        : (B + 1)
  //   relative_position_bias            : (B, N, S, T) or NULL

  const auto& input_dims = input_shape.GetDims();
  if (input_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'input' is expected to have 2 dimensions in packing mode, got ",
                           input_dims.size());
  }
  int64_t token_count = input_dims[0];
  int64_t input_hidden_size = input_dims[1];

  const auto& token_offset_dims = packing_token_offset_shape.GetDims();
  if (token_offset_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'packing_token_offset' is expected to have 2 dimensions in packing mode, got ",
                           token_offset_dims.size());
  }

  int64_t batch_size = dims_token_offset[0];
  int64_t sequence_length = dims_token_offset[1];

  const auto& bias_dims = bias_shape.GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }

  const auto& weights_dims = weights_shape.GetDims();
  if (weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'weights' is expected to have 2 dimensions, got ",
                           weights_dims.size());
  }
  if (weights_dims[0] != input_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as dimension 2 of input 0");
  }

  if (bias_dims[0] != weights_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'bias' dimension 0 should have same length as dimension 1 of input 'weights'");
  }

  const auto& cu_seq_len_dims = cu_seq_len_shape.GetDims();
  if (cu_seq_len_dims.size() != 1 || cu_seq_len_dims[0] != batch_size + 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'cumulative_sequence_length' should have 1 dimension with size equal to batch_size + 1");
  }

  int64_t q_hidden_size = bias_dims[0] / static_cast<int64_t>(3);
  int64_t k_hidden_size = q_hidden_size;
  int64_t v_hidden_size = k_hidden_size;
  if (qkv_hidden_sizes_.size() != 0) {
    if (qkv_hidden_sizes_.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "qkv_hidden_sizes attribute should have 3 elements");
    }

    for (size_t i = 0; i < qkv_hidden_sizes_.size(); i++) {
      if (qkv_hidden_sizes_[i] % num_heads_ != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "hidden_size should be divisible by num_heads:", qkv_hidden_sizes_[i]);
      }
    }

    q_hidden_size = qkv_hidden_sizes_[0];
    k_hidden_size = qkv_hidden_sizes_[1];
    v_hidden_size = qkv_hidden_sizes_[2];
  }

  if (q_hidden_size != k_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "qkv_hidden_sizes first element should be same as the second");
  }

  if (bias_dims[0] != q_hidden_size + k_hidden_size + v_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'bias' dimension 0 should have same length as sum of Q/K/V hidden sizes:",
                           " q_hidden_size=", q_hidden_size, " k_hidden_size=", k_hidden_size, " v_hidden_size=",
                           v_hidden_size, "bias_dims[0]=", bias_dims[0]);
  }

  if (relative_position_bias != nullptr) {
    const auto& relative_position_bias_dims = relative_position_bias->Shape().GetDims();

    if (relative_position_bias_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' is expected to have 4 dimensions, got ",
                             relative_position_bias_dims.size());
    }

    if (relative_position_bias_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 0 should be same as batch_size, got ",
                             relative_position_bias_dims[0]);
    }

    if (relative_position_bias_dims[1] != num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 1 should be same as number of heads, got ",
                             relative_position_bias_dims[1]);
    }

    if (relative_position_bias_dims[2] != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 2 should be same as sequence_length, got ",
                             relative_position_bias_dims[2]);
    }

    if (relative_position_bias_dims[3] != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 3 should be same as sequence_length, got ",
                             relative_position_bias_dims[3]);
    }
  }

  parameters.batch_size = static_cast<int>(batch_size);
  parameters.sequence_length = static_cast<int>(sequence_length);
  parameters.input_hidden_size = static_cast<int>(input_hidden_size);
  parameters.hidden_size = static_cast<int>(q_hidden_size);
  parameters.v_hidden_size = static_cast<int>(v_hidden_size);
  parameters.head_size = static_cast<int>(q_hidden_size) / num_heads_;
  parameters.v_head_size = static_cast<int>(v_hidden_size) / num_heads_;
  parameters.num_heads = num_heads_;
  parameters.mask_filter_value = mask_filter_value_;
  parameters.scale = scale_;
  parameters.token_count = static_cast<int32_t>(token_count);

  return Status::OK();
}

template <typename T>
Status PackedAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* packing_token_offset = context->Input<Tensor>(3);
  const Tensor* cumulative_sequence_length = context->Input<Tensor>(4);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);

  auto& device_prop = GetDeviceProp();
  PackedAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  packing_token_offset->Shape(),
                                  cumulative_sequence_length->Shape(),
                                  relative_position_bias,
                                  &parameters));

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape{parameters.token_count, parameters.v_hidden_size};
  Tensor* output = context->Output(0, output_shape);

  MHARunner* fused_runner = nullptr;

  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;
  bool use_fused_runner = !disable_fused_runner_ &&
                          nullptr == relative_position_bias &&
                          parameters.hidden_size == parameters.v_hidden_size &&
                          FusedMHARunnerFP16v2::is_supported(sm, parameters.head_size, sequence_length,
                                                             enable_trt_flash_attention_, false);

  if (use_fused_runner) {
    // Assuming that num_heads and head_size do not change.
    if (nullptr == fused_fp16_runner_.get()) {
      fused_fp16_runner_.reset(new FusedMHARunnerFP16v2(num_heads_, parameters.head_size, sm, false /* causal_mask*/,
                                                        enable_trt_flash_attention_, parameters.scale));
    }

    // In case some kernel not loaded due to shared memory limit, we need to double check here.
    const int S = fused_fp16_runner_->getSFromMaxSeqLen(sequence_length);
    if (fused_fp16_runner_->isValid(S)) {
      fused_runner = fused_fp16_runner_.get();
    }
  }

#if USE_FLASH_ATTENTION
  bool use_memory_efficient_attention = fused_runner == nullptr &&
                                        !disable_memory_efficient_attention_ &&
                                        nullptr == relative_position_bias &&
                                        (sizeof(T) == 2 ||  // sequence length threshold is 0 in FP16
                                         parameters.sequence_length >= attention::kMinSequenceLengthForMemoryEfficientAttentionFp32) &&
                                        has_memory_efficient_attention(sm, sizeof(T) == 2);
#else
  constexpr bool use_memory_efficient_attention = false;
#endif

  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;

  IAllocatorUniquePtr<T> gemm_buffer;
  int m = parameters.token_count;
  int n = parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size;
  int k = parameters.input_hidden_size;
  gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  constexpr size_t element_size = sizeof(T);
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   parameters.kv_sequence_length,
                                                   parameters.total_sequence_length,
                                                   fused_runner,
                                                   use_memory_efficient_attention);
  auto work_space = GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = reinterpret_cast<CudaT*>(gemm_buffer.get());
  data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = nullptr;
  data.key = nullptr;
  data.value = nullptr;
  data.mask_index = (nullptr == mask_index) ? nullptr : mask_index->Data<int>();
  data.mask_index_dims = (nullptr == mask_index) ? gsl::span<const int64_t>() : mask_index->Shape().GetDims();
  data.past = (nullptr == past) ? nullptr : reinterpret_cast<const CudaT*>(past->Data<T>());
  data.relative_position_bias = (nullptr == relative_position_bias) ? nullptr : reinterpret_cast<const CudaT*>(relative_position_bias->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present = (nullptr == present) ? nullptr : reinterpret_cast<CudaT*>(present->MutableData<T>());
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.fused_cross_attention_kernel = nullptr;
  data.use_memory_efficient_attention = use_memory_efficient_attention;

  return QkvToContext<CudaT>(device_prop, cublas, Stream(context), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/packed_attention.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/packed_attention_impl.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      PackedAttention,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      PackedAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
TrtFusedAttention<T>::TrtFusedAttention() {
  disable_fused_runner_ = sizeof(T) != 2 ||
                          ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedSelfAttention, false);

  enable_trt_flash_attention_ = sizeof(T) == 2 &&
                                !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);
}

template <typename T>
MHARunner* TrtFusedAttention<T>::GetFusedRunner(const cudaDeviceProp& device_prop,
                                                const PackedAttentionParameters& parameters) const {
  MHARunner* fused_runner = nullptr;

  bool use_fused_runner = !disable_fused_runner_ &&
                          !parameters.has_relative_position_bias &&
                          parameters.hidden_size == parameters.v_hidden_size;

  if (!use_fused_runner) {
    return fused_runner;
  }

  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;
  bool is_fMHA_supported = FusedMHARunnerFP16v2::is_supported(sm,
                                                              parameters.head_size,
                                                              parameters.sequence_length,
                                                              enable_trt_flash_attention_,
                                                              false /*causal*/);

  if (!is_fMHA_supported) {
    return fused_runner;
  }

  // Assuming that num_heads and head_size do not change.
  if (nullptr == fused_fp16_runner_.get()) {
    fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(parameters.num_heads, parameters.head_size, sm, false /*causal*/,
                                                      enable_trt_flash_attention_, parameters.scale);
  }

  // In case some kernel not loaded due to shared memory limit, we need to double check here.
  const int S = fused_fp16_runner_->getSFromMaxSeqLen(parameters.sequence_length);
  if (fused_fp16_runner_->isValid(S)) {
    fused_runner = fused_fp16_runner_.get();
  }

  return fused_runner;
}

// template class instantiation
template class TrtFusedAttention<float>;
template class TrtFusedAttention<MLFloat16>;

template <typename T>
PackedAttention<T>::PackedAttention(const OpKernelInfo& info) : TrtFusedAttention<T>(), CudaKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int32_t>(num_heads);

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  if (!info.GetAttrs<int64_t>("qkv_hidden_sizes", qkv_hidden_sizes_).IsOK()) {
    qkv_hidden_sizes_.clear();
  }
}

template <typename T>
Status PackedAttention<T>::CheckInputs(const TensorShape& input_shape,
                                       const TensorShape& weights_shape,
                                       const TensorShape& bias_shape,
                                       const TensorShape& token_offset_shape,
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
  //   input:                  : (T, D_i)
  //   weights      (Q/K/V)    : (D_i, D + D + D_v)
  //   bias         (Q/K/V)    : (D + D + D_v)
  //   token_offset            : (B, S)
  //   cu_seq_len_shape        : (B + 1)
  //   relative_position_bias  : (B, N, S, S), (1, N, S, S) or NULL
  const auto& input_dims = input_shape.GetDims();
  if (input_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'input' is expected to have 2 dimensions in packing mode, got ",
                           input_dims.size());
  }
  int64_t token_count = input_dims[0];
  int64_t input_hidden_size = input_dims[1];

  const auto& token_offset_dims = token_offset_shape.GetDims();
  if (token_offset_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'packing_token_offset' is expected to have 2 dimensions in packing mode, got ",
                           token_offset_dims.size());
  }

  int64_t batch_size = token_offset_dims[0];
  int64_t sequence_length = token_offset_dims[1];

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

  const int num_heads = this->GetNumHeads();
  int64_t q_hidden_size = bias_dims[0] / static_cast<int64_t>(3);
  int64_t k_hidden_size = q_hidden_size;
  int64_t v_hidden_size = k_hidden_size;
  if (qkv_hidden_sizes_.size() != 0) {
    if (qkv_hidden_sizes_.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "qkv_hidden_sizes attribute should have 3 elements");
    }

    for (size_t i = 0; i < qkv_hidden_sizes_.size(); i++) {
      if (qkv_hidden_sizes_[i] % num_heads != 0) {
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

  bool broadcast_res_pos_bias = false;
  if (relative_position_bias != nullptr) {
    const auto& relative_position_bias_dims = relative_position_bias->Shape().GetDims();

    if (relative_position_bias_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' is expected to have 4 dimensions, got ",
                             relative_position_bias_dims.size());
    }

    if (relative_position_bias_dims[0] != batch_size && relative_position_bias_dims[0] != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 0 should be same as batch_size or 1, got ",
                             relative_position_bias_dims[0]);
    }
    if (relative_position_bias_dims[0] == 1) {
      broadcast_res_pos_bias = true;
    }

    if (relative_position_bias_dims[1] != num_heads) {
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
  parameters.head_size = static_cast<int>(q_hidden_size) / num_heads;
  parameters.v_head_size = static_cast<int>(v_hidden_size) / num_heads;
  parameters.num_heads = num_heads;
  parameters.scale = this->GetScale();
  parameters.token_count = static_cast<int32_t>(token_count);
  parameters.has_relative_position_bias = nullptr != relative_position_bias;
  parameters.broadcast_res_pos_bias = broadcast_res_pos_bias;

  return Status::OK();
}

template <typename T>
Status PackedAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* token_offset = context->Input<Tensor>(3);
  const Tensor* cumulative_sequence_length = context->Input<Tensor>(4);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);

  PackedAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  token_offset->Shape(),
                                  cumulative_sequence_length->Shape(),
                                  relative_position_bias,
                                  parameters));

  TensorShapeVector output_shape{parameters.token_count, parameters.v_hidden_size};
  Tensor* output = context->Output(0, output_shape);

  auto& device_prop = this->GetDeviceProp();
  MHARunner* fused_runner = this->GetFusedRunner(device_prop, parameters);

  bool use_memory_efficient_attention = false;
#if USE_MEMORY_EFFICIENT_ATTENTION
  if (nullptr == fused_runner) {
    int sm = device_prop.major * 10 + device_prop.minor;
    bool is_good_for_rpb = !parameters.has_relative_position_bias || parameters.sequence_length % (4 * sizeof(T)) == 0;
    use_memory_efficient_attention = is_good_for_rpb &&
                                     sizeof(T) == 2 &&  // only enable for fp16
                                     (parameters.head_size & 7) == 0 &&
                                     (parameters.v_head_size & 7) == 0 &&
                                     has_memory_efficient_attention(sm, sizeof(T) == 2);
  }
#endif

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  IAllocatorUniquePtr<T> gemm_buffer;
  int m = parameters.token_count;
  int n = parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size;
  int k = parameters.input_hidden_size;
  gemm_buffer = this->GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());

  cublasHandle_t cublas = this->GetCublasHandle(context);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  constexpr size_t element_size = sizeof(T);
  constexpr bool no_qkv_workspace = false;  // need workspace to add bias
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   fused_runner,
                                                   false,
                                                   use_memory_efficient_attention,
                                                   no_qkv_workspace);
  auto work_space = this->GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());

  typedef typename ToCudaType<T>::MappedType CudaT;
  PackedAttentionData<CudaT> data;
  data.gemm_buffer = reinterpret_cast<CudaT*>(gemm_buffer.get());
  data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.relative_position_bias = (nullptr == relative_position_bias) ? nullptr : reinterpret_cast<const CudaT*>(relative_position_bias->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.token_offset = token_offset->Data<int32_t>();
  data.cumulative_sequence_length = cumulative_sequence_length->Data<int32_t>();
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.use_memory_efficient_attention = use_memory_efficient_attention;

  return QkvToContext<CudaT>(device_prop, cublas, Stream(context), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

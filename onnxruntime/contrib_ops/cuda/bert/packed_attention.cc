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

PackedAttentionShape MakePackedAttentionShape(const TensorShape& shape) noexcept {
  PackedAttentionShape result;
  const auto& dimensions = shape.GetDims();
  result.rank = dimensions.size();
  const size_t dimensions_to_copy =
      dimensions.size() < result.dimensions.size() ? dimensions.size() : result.dimensions.size();
  for (size_t i = 0; i < dimensions_to_copy; ++i) {
    result.dimensions[i] = dimensions[i];
  }

  return result;
}

Status PackedAttentionWorkspaceStatusToStatus(PackedAttentionWorkspaceStatus status) {
  if (status.IsOK()) {
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, status.message);
}

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
TrtFusedAttention<T>::TrtFusedAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  kernel_options_ = this->GetAttentionKernelOptions();
  disable_fused_runner_ = sizeof(T) != 2 || !kernel_options_->UseTrtFusedAttention();
  enable_trt_flash_attention_ = sizeof(T) == 2 && kernel_options_->UseTrtFlashAttention();
}

template <typename T>
MHARunner* TrtFusedAttention<T>::GetFusedRunner(const cudaDeviceProp& device_prop,
                                                bool has_attention_bias,
                                                const PackedAttentionParameters& parameters) const {
  MHARunner* fused_runner = nullptr;

  bool use_fused_runner = !disable_fused_runner_ &&
                          !has_attention_bias &&
                          parameters.hidden_size == parameters.GetOutputHiddenSize();

  if (!use_fused_runner) {
    return fused_runner;
  }

  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;
  bool is_fMHA_supported = FusedMHARunnerFP16v2::IsSupported(sm,
                                                             parameters.head_size,
                                                             parameters.sequence_length,
                                                             enable_trt_flash_attention_);

  if (!is_fMHA_supported) {
    return fused_runner;
  }

  // Assuming that num_heads and head_size do not change.
  if (nullptr == fused_fp16_runner_.get()) {
    fused_fp16_runner_ = FusedMHARunnerFP16v2::Create(parameters.num_heads, parameters.head_size, sm,
                                                      enable_trt_flash_attention_, parameters.scale);
  }

  // In case some kernel not loaded due to shared memory limit, we need to double check here.
  const int normalized_seq_len = fused_fp16_runner_->NormalizeSequenceLength(parameters.sequence_length);
  if (fused_fp16_runner_->IsValid(normalized_seq_len)) {
    fused_runner = fused_fp16_runner_.get();
  }

  return fused_runner;
}

// template class instantiation
template class TrtFusedAttention<float>;
template class TrtFusedAttention<MLFloat16>;

template <typename T>
PackedAttention<T>::PackedAttention(const OpKernelInfo& info)
    : TrtFusedAttention<T>(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = num_heads;

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
                                       const Tensor* attention_bias,
                                       PackedAttentionParameters& parameters,
                                       PackedAttentionProblem& problem) const {
  PackedAttentionInputShapes inputs;
  inputs.input = MakePackedAttentionShape(input_shape);
  inputs.weights = MakePackedAttentionShape(weights_shape);
  inputs.bias = MakePackedAttentionShape(bias_shape);
  inputs.token_offset = MakePackedAttentionShape(token_offset_shape);
  inputs.cumulative_sequence_length = MakePackedAttentionShape(cu_seq_len_shape);
  inputs.element_size = sizeof(T);
  inputs.num_heads = GetNumHeads();
  inputs.qkv_hidden_sizes_count = qkv_hidden_sizes_.size();
  for (size_t i = 0; i < qkv_hidden_sizes_.size() && i < inputs.qkv_hidden_sizes.size(); ++i) {
    inputs.qkv_hidden_sizes[i] = qkv_hidden_sizes_[i];
  }
  inputs.has_attention_bias = attention_bias != nullptr;
  if (attention_bias != nullptr) {
    inputs.attention_bias = MakePackedAttentionShape(attention_bias->Shape());
  }

  auto problem_result = BuildPackedAttentionProblem(inputs);
  ORT_RETURN_IF_ERROR(PackedAttentionWorkspaceStatusToStatus(problem_result.status));
  problem = problem_result.problem;

  parameters.broadcast_attn_bias_dim_0 = problem.broadcast_attn_bias_dim_0;
  parameters.broadcast_attn_bias_dim_1 = problem.broadcast_attn_bias_dim_1;

  parameters.batch_size = problem.batch_size;
  parameters.sequence_length = problem.sequence_length;
  parameters.input_hidden_size = problem.input_hidden_size;
  parameters.hidden_size = problem.hidden_size;
  parameters.head_size = problem.qk_head_size;
  parameters.v_head_size = problem.v_head_size;
  parameters.num_heads = problem.num_heads;
  parameters.kv_num_heads = problem.num_heads;
  parameters.scale = this->GetScale();
  parameters.token_count = problem.token_count;

  return Status::OK();
}

template <typename T>
Status PackedAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* token_offset = context->Input<Tensor>(3);
  const Tensor* cumulative_sequence_length = context->Input<Tensor>(4);
  const Tensor* attention_bias = context->Input<Tensor>(5);

  PackedAttentionParameters parameters;
  PackedAttentionProblem problem;
  parameters.use_tf32 = this->UseTF32();
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  token_offset->Shape(),
                                  cumulative_sequence_length->Shape(),
                                  attention_bias,
                                  parameters,
                                  problem));

  TensorShapeVector output_shape{parameters.token_count, parameters.GetOutputHiddenSize()};
  Tensor* output = context->Output(0, output_shape);

  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  auto& device_prop = this->GetDeviceProp();
  MHARunner* fused_runner = this->GetFusedRunner(device_prop, attention_bias != nullptr, parameters);

  bool use_memory_efficient_attention = false;
#if USE_MEMORY_EFFICIENT_ATTENTION
  if (nullptr == fused_runner) {
    int sm = device_prop.major * 10 + device_prop.minor;
    use_memory_efficient_attention =
        (attention_bias == nullptr || parameters.sequence_length % (4 * sizeof(T)) == 0) &&
        sizeof(T) == 2 &&  // only enable for fp16
        has_memory_efficient_attention(sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value, parameters.head_size, parameters.v_head_size);
  }
#endif

  if (this->kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_efficient_attention = use_memory_efficient_attention;
    if (fused_runner != nullptr) {
      debug_info.SetTrtFusedKernel(this->enable_trt_flash_attention_, parameters.sequence_length);
    }

    debug_info.Print("PackedAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  problem.backend = fused_runner != nullptr
                        ? PackedAttentionBackend::Trt
                        : (use_memory_efficient_attention
                               ? PackedAttentionBackend::MemoryEfficient
                               : PackedAttentionBackend::Unfused);
  problem.trt_runner_available = fused_runner != nullptr;
  auto workspace_result = GetPackedAttentionWorkspaceRecipe(problem);
  ORT_RETURN_IF_ERROR(PackedAttentionWorkspaceStatusToStatus(workspace_result.status));
  const PackedAttentionWorkspaceRecipe& workspace_recipe = workspace_result.recipe;

  auto gemm_buffer = this->template GetScratchBuffer<void>(
      workspace_recipe.projection_bytes, this->GetComputeStream(context));
  const int m = workspace_recipe.projection_m;
  const int n = workspace_recipe.projection_n;
  const int k = workspace_recipe.projection_k;

  cublasHandle_t cublas = this->GetCublasHandle(context);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrices into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop, this->UseTF32()));

  auto work_space = this->template GetScratchBuffer<void>(
      workspace_recipe.attention_workspace_bytes, this->GetComputeStream(context));

  PackedAttentionData<CudaT> data;
  data.gemm_buffer = reinterpret_cast<CudaT*>(gemm_buffer.get());
  data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.attention_bias = (nullptr == attention_bias) ? nullptr : reinterpret_cast<const CudaT*>(attention_bias->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.token_offset = token_offset->Data<int32_t>();
  data.cumulative_sequence_length = cumulative_sequence_length->Data<int32_t>();
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  data.workspace_recipe = workspace_recipe;

  return QkvToContext<CudaT>(device_prop, cublas, this->Stream(context), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

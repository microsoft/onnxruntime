// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/packed_multihead_attention.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/packed_attention_impl.h"
#include "contrib_ops/cuda/bert/packed_multihead_attention_impl.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

#if !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
#include "contrib_ops/cuda/bert/packed_attention_workspace_estimate.h"
#endif

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      PackedMultiHeadAttention,                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      PackedMultiHeadAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
PackedMultiHeadAttention<T>::PackedMultiHeadAttention(const OpKernelInfo& info)
    : TrtFusedAttention<T>(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = num_heads;

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  disable_flash_attention_ = sizeof(T) != 2 || !this->kernel_options_->UseFlashAttention();

  disable_memory_efficient_attention_ = !this->kernel_options_->UseEfficientAttention();
}

template <typename T>
Status PackedMultiHeadAttention<T>::CheckInputs(const TensorShape& query_shape,
                                                const Tensor* key,
                                                const Tensor* value,
                                                const Tensor* bias,
                                                const TensorShape& token_offset_shape,
                                                const TensorShape& cu_seq_len_shape,
                                                const Tensor* attention_bias,
                                                PackedAttentionParameters& parameters,
                                                PackedMultiHeadAttentionProblem& problem) const {
  PackedMultiHeadAttentionInputShapes inputs;
  inputs.query = MakePackedAttentionShape(query_shape);
  inputs.token_offset = MakePackedAttentionShape(token_offset_shape);
  inputs.cumulative_sequence_length = MakePackedAttentionShape(cu_seq_len_shape);
  inputs.element_size = sizeof(T);
  inputs.num_heads = GetNumHeads();
  inputs.has_key = key != nullptr;
  inputs.has_value = value != nullptr;
  inputs.has_bias = bias != nullptr;
  inputs.has_attention_bias = attention_bias != nullptr;
  if (key != nullptr) {
    inputs.key = MakePackedAttentionShape(key->Shape());
  }
  if (value != nullptr) {
    inputs.value = MakePackedAttentionShape(value->Shape());
  }
  if (bias != nullptr) {
    inputs.bias = MakePackedAttentionShape(bias->Shape());
  }
  if (attention_bias != nullptr) {
    inputs.attention_bias = MakePackedAttentionShape(attention_bias->Shape());
  }

  auto problem_result = BuildPackedMultiHeadAttentionProblem(inputs);
  ORT_RETURN_IF_ERROR(PackedAttentionWorkspaceStatusToStatus(problem_result.status));
  problem = problem_result.problem;

  parameters.broadcast_attn_bias_dim_0 = problem.broadcast_attn_bias_dim_0;
  parameters.broadcast_attn_bias_dim_1 = problem.broadcast_attn_bias_dim_1;

  parameters.batch_size = problem.batch_size;
  parameters.sequence_length = problem.sequence_length;
  parameters.input_hidden_size = -1;  // not applicable
  parameters.hidden_size = problem.hidden_size;
  parameters.head_size = problem.qk_head_size;
  parameters.v_head_size = problem.v_head_size;
  parameters.num_heads = problem.num_heads;
  parameters.kv_num_heads = problem.num_heads;
  parameters.scale = this->GetScale();
  parameters.token_count = problem.token_count;

  return Status::OK();
}

#if !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
// An unavailable adapter estimate is represented by no Level-2 requirements.
template <typename T>
Status PackedMultiHeadAttention<T>::DeclareWorkspaceRequirements(
    gsl::span<const WorkspaceInputShape> input_shapes,
    /*out*/ InlinedVector<WorkspaceRequirement>& requirements) const {
  requirements.clear();

  PackedAttentionWorkspaceEstimateConfig config;
  config.op = PackedAttentionWorkspaceOperator::PackedMultiHeadAttention;
  config.element_size = sizeof(T);
  config.num_heads = num_heads_;

  const auto estimate = EstimatePackedAttentionWorkspace(
      config, input_shapes, this->GetDeviceProp(), *this->kernel_options_);
  if (estimate.has_value()) {
    SetPackedAttentionWorkspaceRequirements(*estimate, requirements);
  }
  return Status::OK();
}
#endif

template <typename T>
Status PackedMultiHeadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* token_offset = context->Input<Tensor>(4);
  const Tensor* cumulative_sequence_length = context->Input<Tensor>(5);
  const Tensor* attention_bias = context->Input<Tensor>(6);

  typedef typename ToCudaType<T>::MappedType CudaT;

  PackedAttentionParameters parameters;
  PackedMultiHeadAttentionProblem problem;
  parameters.use_tf32 = this->UseTF32();
  ORT_RETURN_IF_ERROR(CheckInputs(query->Shape(),
                                  key,
                                  value,
                                  bias,
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

  bool use_flash_attention = false;
#if USE_FLASH_ATTENTION
  if (!disable_flash_attention_) {
    use_flash_attention = nullptr == attention_bias &&
                          parameters.head_size == parameters.v_head_size &&
                          onnxruntime::flash::is_supported<T>(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.num_heads);

    // When input is packed QKV format, TensorRT kernel might be faster when sequence length <= 512.
    if (use_flash_attention && key == nullptr && value == nullptr &&
        parameters.sequence_length < this->kernel_options_->MinSeqLenForFlashAttentionPackedQkv()) {
      use_flash_attention = false;
    }
  }
#endif

  MHARunner* fused_runner = use_flash_attention
                                ? nullptr
                                : this->GetFusedRunner(device_prop, attention_bias != nullptr, parameters);

  bool use_memory_efficient_attention = false;

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (!use_flash_attention && nullptr == fused_runner && !disable_memory_efficient_attention_) {
    int sm = device_prop.major * 10 + device_prop.minor;
    use_memory_efficient_attention =
        (nullptr == attention_bias || parameters.sequence_length % (4 * sizeof(T)) == 0) &&
        (sizeof(T) == 2 || parameters.sequence_length >= this->kernel_options_->MinSeqLenForEfficientAttentionFp32()) &&
        has_memory_efficient_attention(sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value, parameters.head_size, parameters.v_head_size);
  }
#endif

  if (this->kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;
    debug_info.use_efficient_attention = use_memory_efficient_attention;
    if (fused_runner != nullptr) {
      debug_info.SetTrtFusedKernel(this->enable_trt_flash_attention_, parameters.sequence_length);
    }

    debug_info.Print("PackedMultiHeadAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  cublasHandle_t cublas = this->GetCublasHandle(context);

  problem.backend = use_flash_attention
                        ? PackedAttentionBackend::Flash
                        : (fused_runner != nullptr
                               ? PackedAttentionBackend::Trt
                               : (use_memory_efficient_attention
                                      ? PackedAttentionBackend::MemoryEfficient
                                      : PackedAttentionBackend::Unfused));
  problem.trt_runner_available = fused_runner != nullptr;
  auto workspace_result = GetPackedMultiHeadAttentionWorkspaceRecipe(problem);
  ORT_RETURN_IF_ERROR(PackedAttentionWorkspaceStatusToStatus(workspace_result.status));
  const PackedAttentionWorkspaceRecipe& workspace_recipe = workspace_result.recipe;

  auto work_space = this->template GetScratchBuffer<void>(
      workspace_recipe.attention_workspace_bytes, this->GetComputeStream(context));

  PackedMultiHeadAttentionData<CudaT> data;
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = (key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = (value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.bias = (bias == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.attention_bias = (nullptr == attention_bias)
                            ? nullptr
                            : reinterpret_cast<const CudaT*>(attention_bias->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.token_offset = token_offset->Data<int32_t>();
  data.cumulative_sequence_length = cumulative_sequence_length->Data<int32_t>();
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.fused_runner = reinterpret_cast<void*>(fused_runner);
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  data.no_qkv_workspace = workspace_recipe.no_qkv_workspace;
  data.source_qkv_format =
      problem.qkv_format == PackedMultiHeadAttentionQkvFormat::Packed
          ? AttentionQkvFormat::QKV_TN3H
          : AttentionQkvFormat::Q_K_V_TNH;
  data.workspace_recipe = workspace_recipe;

  return QkvToContext<CudaT>(device_prop, cublas, this->Stream(context), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

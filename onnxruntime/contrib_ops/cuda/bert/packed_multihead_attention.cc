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
#include "contrib_ops/cpu/bert/multihead_attention_helper.h"

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
  num_heads_ = static_cast<int32_t>(num_heads);

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
                                                PackedAttentionParameters& parameters) const {
  // Shapes of inputs and output:
  // When Q, K and V are not packed:
  //   Input 'query':                      (token_count, hidden_size)
  //   Input 'key':                        (token_count, hidden_size)
  //   Input 'value':                      (token_count, v_hidden_size)
  // When Q, K and V are packed:
  //   Input 'query':                      (token_count, num_heads, 3, head_size)
  //   Input 'key':                        None
  //   Input 'value':                      None
  // Input 'token_offset':                 (batch_size, sequence_length)
  // Input 'cumulative_sequence_length':   (batch_size + 1)
  // Input 'attention_bias':               (batch_size or 1, num_heads or 1, sequence_length, sequence_length) or None
  // Output 'output':                      (token_count, v_hidden_size)

  const auto& query_dims = query_shape.GetDims();
  if (query_dims.size() != 2 && query_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' is expected to have 2 or 4 dimensions in packing mode, got ",
                           query_dims.size());
  }
  int64_t token_count = query_dims[0];
  int64_t hidden_size = (query_dims.size() == 2) ? query_dims[1] : (query_dims[1] * query_dims[3]);

  const auto& token_offset_dims = token_offset_shape.GetDims();
  if (token_offset_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'token_offset' is expected to have 2 dimensions in packing mode, got ",
                           token_offset_dims.size());
  }

  int64_t batch_size = token_offset_dims[0];
  int64_t sequence_length = token_offset_dims[1];

  int64_t v_hidden_size = hidden_size;
  if (query_dims.size() == 4) {
    if (key != nullptr || value != nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'key' and 'value' is expected to be empty when 'query' has 4 dimensions in packing mode");
    }
  } else {  // query_dims.size() == 2
    if (key == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key' is expected when 'query' has 2 dimensions in packing mode");
    }

    const auto& key_dims = key->Shape().GetDims();
    if (key_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 2 dimension, got ",
                             key_dims.size());
    }
    if (key_dims != query_dims) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' and 'key' is expected to have same shape");
    }

    if (value == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'value' is expected when 'query' has 2 dimensions in packing mode");
    }
    const auto& value_dims = value->Shape().GetDims();
    if (value_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 2 dimensions, got ",
                             value_dims.size());
    }
    if (value_dims[0] != token_count) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 2 dimension 0 should have same length as dimension 0 of input 0");
    }
    v_hidden_size = value_dims[1];
  }

  if (bias != nullptr) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                             bias_dims.size());
    }

    if (bias_dims[0] != hidden_size + hidden_size + v_hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' size is expected to be ",
                             hidden_size + hidden_size + v_hidden_size, ", got ", bias_dims[0]);
    }
  }

  const auto& cu_seq_len_dims = cu_seq_len_shape.GetDims();
  if (cu_seq_len_dims.size() != 1 || cu_seq_len_dims[0] != batch_size + 1) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Input 'cumulative_sequence_length' should have 1 dimension with size equal to batch_size + 1");
  }

  const int num_heads = this->GetNumHeads();

  gsl::span<const int64_t> attention_bias_dims;
  if (attention_bias != nullptr) {
    attention_bias_dims = attention_bias->Shape().GetDims();
    ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckAttentionBias(
        attention_bias_dims, batch_size, num_heads, sequence_length, sequence_length));
  }
  parameters.broadcast_attn_bias_dim_0 = attention_bias_dims.size() > 0 && attention_bias_dims[0] == 1;
  parameters.broadcast_attn_bias_dim_1 = attention_bias_dims.size() > 1 && attention_bias_dims[1] == 1;

  parameters.batch_size = static_cast<int>(batch_size);
  parameters.sequence_length = static_cast<int>(sequence_length);
  parameters.input_hidden_size = -1;  // not applicable
  parameters.hidden_size = static_cast<int>(hidden_size);
  parameters.v_hidden_size = static_cast<int>(v_hidden_size);
  parameters.head_size = static_cast<int>(hidden_size) / num_heads;
  parameters.v_head_size = static_cast<int>(v_hidden_size) / num_heads;
  parameters.num_heads = num_heads;
  parameters.scale = this->GetScale();
  parameters.token_count = static_cast<int32_t>(token_count);

  return Status::OK();
}

template <typename T>
Status PackedMultiHeadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* token_offset = context->Input<Tensor>(4);
  const Tensor* cumulative_sequence_length = context->Input<Tensor>(5);
  const Tensor* attention_bias = context->Input<Tensor>(6);

  PackedAttentionParameters parameters;
  parameters.use_tf32 = this->UseTF32();
  ORT_RETURN_IF_ERROR(CheckInputs(query->Shape(),
                                  key,
                                  value,
                                  bias,
                                  token_offset->Shape(),
                                  cumulative_sequence_length->Shape(),
                                  attention_bias,
                                  parameters));

  TensorShapeVector output_shape{parameters.token_count, parameters.v_hidden_size};
  Tensor* output = context->Output(0, output_shape);

  auto& device_prop = this->GetDeviceProp();

  bool use_flash_attention = false;
#if USE_FLASH_ATTENTION
  if (!disable_flash_attention_) {
    use_flash_attention = nullptr == attention_bias &&
                          parameters.head_size == parameters.v_head_size &&
                          onnxruntime::flash::is_supported(device_prop,
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
        has_memory_efficient_attention(sm, sizeof(T) == 2, parameters.head_size, parameters.v_head_size);
  }
#endif

  if (this->kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;
    debug_info.use_efficient_attention = use_memory_efficient_attention;
    if (fused_runner != nullptr) {
      debug_info.SetTrtFusedKernel(false /*causal*/, this->enable_trt_flash_attention_, parameters.sequence_length);
    }

    debug_info.Print("PackedMultiHeadAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  typedef typename ToCudaType<T>::MappedType CudaT;

  cublasHandle_t cublas = this->GetCublasHandle(context);

  constexpr size_t element_size = sizeof(T);
  // When the source and target format is same (like TN3H => TN3H, or TNH => TNH) and no bias, need not transpose qkv.
  const bool no_qkv_workspace = (fused_runner != nullptr && key == nullptr && bias == nullptr) ||
                                ((use_memory_efficient_attention || use_flash_attention) &&
                                 value != nullptr &&
                                 bias == nullptr);
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   fused_runner,
                                                   use_flash_attention,
                                                   use_memory_efficient_attention,
                                                   no_qkv_workspace);
  auto work_space = this->template GetScratchBuffer<void>(workSpaceSize, context->GetComputeStream());

  typedef typename ToCudaType<T>::MappedType CudaT;
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
  data.no_qkv_workspace = no_qkv_workspace;
  data.source_qkv_format = (key == nullptr) ? AttentionQkvFormat::QKV_TN3H : AttentionQkvFormat::Q_K_V_TNH;

  return QkvToContext<CudaT>(device_prop, cublas, this->Stream(context), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

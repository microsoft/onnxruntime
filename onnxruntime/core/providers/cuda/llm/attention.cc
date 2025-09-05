// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/llm/attention_helper.h"
#include "core/providers/cuda/llm/attention.h"
#include "core/providers/cuda/llm/attention_naive.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::attention_helper;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      Attention,                                                      \
      kOnnxDomain,                                                    \
      23,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
// REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  // kv_num_heads, q_num_head are mandatory for 3D inputs but not used for 4D inputs.
  // The dimension is not yet known. If not specified, the inputs is assumed to be 4D.
  kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  qk_matmul_output_mode_ = info.node().OutputDefs().size() >= 4 && info.node().OutputDefs()[3]->Exists()
                               ? static_cast<QKMatMulOutputMode>(mode)
                               : QKMatMulOutputMode::kNone;
  ORT_ENFORCE(qk_matmul_output_mode_ == QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQKMask ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQKSoftCap ||
                  qk_matmul_output_mode_ == QKMatMulOutputMode::kQKSoftMax,
              "qk_matmul_output_mode must be 0, 1, 2, or 3.");
  // The default scale depends on the input dimensions. It is set to nan to indicate that it should be computed.
  scale_ = info.GetAttrOrDefault<float>("scale", std::numeric_limits<T>::quiet_NaN());
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  softmax_precision_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("softmax_precision", 0));
  ORT_ENFORCE(scale_ > 0 || std::isnan(scale_), "scale must be greater than 0 if specified");
}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* Q = context->Input<Tensor>(0);
  const Tensor* K = context->Input<Tensor>(1);
  const Tensor* V = context->Input<Tensor>(2);
  const Tensor* attn_mask = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(4);
  const Tensor* past_value = context->Input<Tensor>(5);

  AttentionParameters parameters;
  TensorShape y_shape;
  TensorShape present_key_shape;
  TensorShape present_value_shape;
  TensorShape output_qk_shape;

  ORT_ENFORCE(attention_helper::ComputeOutputShapeForAttention(
                  Q,
                  K,
                  V,
                  attn_mask,
                  past_key,
                  past_value,
                  is_causal_,
                  softcap_,
                  softmax_precision_,
                  qk_matmul_output_mode_,
                  kv_num_heads_,
                  q_num_heads_,
                  scale_,
                  parameters,
                  y_shape,
                  present_key_shape,
                  present_value_shape,
                  output_qk_shape)
                  .IsOK(),
              "Output shapes for Attention could not be computed.");

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = parameters.qk_matmul_output_mode == QKMatMulOutputMode::kNone
                          ? nullptr
                          : context->Output(3, output_qk_shape);

#if 0
  // First tentative to following the CPU implementation with the idea behind to have a fallback
  // when contrib ops do not work (for FP8 for example or for QKV output options).
  // This is unfinished.
  NaiveAttention<T> attention_naive_impl;
  cudaStream_t stream = Stream(context);
  return attention_naive_impl.ApplyAttention(context,
                                             stream,
                                             Q->Data<T>(),   // Q
                                             K->Data<T>(),   // K
                                             V->Data<T>(),   // V
                                             attn_mask,      // const Tensor* mask_index,  // mask, nullptr if no mask
                                             past_key,       // past K input tensor (if not using past state)
                                             past_value,     // past V input tensor (if not using past state)
                                             Y,              // first output
                                             present_key,    // present K output tensor (if separating present KV)
                                             present_value,  // present V output tensor (if separating present KV)
                                             output_qk,      // Q*K output tensor (if returning Q*K value)
                                             parameters);    // attention parameters
#else

  onnxruntime::contrib::AttentionParameters cparameters;
  cparameters.batch_size = parameters.batch_size;
  cparameters.sequence_length = parameters.q_sequence_length;
  cparameters.kv_sequence_length = parameters.kv_sequence_length;
  cparameters.past_sequence_length = parameters.past_sequence_length;
  cparameters.total_sequence_length = parameters.total_sequence_length;
  cparameters.max_sequence_length = parameters.total_sequence_length;  // TODO ?
  cparameters.input_hidden_size = parameters.batch_size;
  cparameters.hidden_size = parameters.batch_size;
  cparameters.head_size = parameters.head_size;
  cparameters.v_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
  cparameters.num_heads = parameters.q_num_heads;
  cparameters.head_size = parameters.head_size;
  cparameters.num_splits = 1;  // TODO?
  cparameters.scale = parameters.scale;
  cparameters.do_rotary = false;
  cparameters.is_packed_qkv = false;

  // TODO: Mask 2D does not seem to be supported by the contrib ops.
  cparameters.mask_type = attn_mask == nullptr
                              ? onnxruntime::contrib::AttentionMaskType::MASK_NONE
                              : (attn_mask->Shape().NumDimensions() == 4
                                     ? onnxruntime::contrib::AttentionMaskType::MASK_4D_MEGATRON
                                     : onnxruntime::contrib::AttentionMaskType::MASK_3D_ATTENTION);
  cparameters.qkv_format = Q->Shape().NumDimensions() == 4
                               ? onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BSNH   // for non-packed qkv, not permuted, used by memory efficient attention or MultiHeadAttention
                               : onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;  // for non-packed qkv, permuted

  /*
  // TODO: investigate these parameters.
  int num_splits;      // number of splits for splitkv
  int rotary_dim = 0;  // rotary embedding dimension
  int beam_width;
  bool is_unidirectional;
  bool past_present_share_buffer;
  bool is_packed_qkv = false;  // whether qkv is packed
  bool broadcast_attn_bias_dim_0;
  bool broadcast_attn_bias_dim_1;
  float mask_filter_value;
  bool use_tf32;
  */

  // Calls the contrib ops implementation.
  // The main issue is to retrieve the expected QK outputs. Attention(23) has many unsupported options.
  typedef typename ToCudaType<T>::MappedType CudaT;
  onnxruntime::contrib::cuda::AttentionData<CudaT> data;

  // T* gemm_buffer = nullptr;
  // const T* bias = nullptr;
  // int* seqlens_k_total = nullptr;

  data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
  data.value = reinterpret_cast<const CudaT*>(V->Data<T>());

  // const int* mask_index = nullptr;
  // gsl::span<const int64_t> mask_index_dims;
  // const T* past = nullptr;
  data.past_key = reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = reinterpret_cast<const CudaT*>(past_value->Data<T>());
  // const int32_t* cache_indirection = nullptr;
  // const T* attention_bias = nullptr;

  // bool has_qkv_workspace = false;
  // T* workspace = nullptr;

  data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  // T* present = nullptr;
  data.present_key = present_key != nullptr ? reinterpret_cast<CudaT*>(present_key->MutableData<T>()) : nullptr;
  data.present_value = present_value != nullptr ? reinterpret_cast<CudaT*>(present_value->MutableData<T>()) : nullptr;
  data.output_qk = output_qk != nullptr ? reinterpret_cast<CudaT*>(output_qk->MutableData<T>()) : nullptr;

  // void* fused_runner = nullptr;
  // const void* fused_cross_attention_kernel = nullptr;=
  // bool use_flash_attention = false;
  // bool use_memory_efficient_attention = false;
  // bool use_decoder_masked_multihead_attention = false;
  // const int32_t* cumulated_sequence_length_q_cache = nullptr;
  // const int32_t* cumulated_sequence_length_kv_cache = nullptr;

  // Intermediate data
  // T* q = nullptr;
  // T* k = nullptr;
  // T* v = nullptr;
  // T* scratch = nullptr;
  data.qkv_format = cparameters.qkv_format;

  // Flash buffers
  // T* softmax_lse = nullptr;
  // T* softmax_lse_accum = nullptr;
  // T* out_accum = nullptr;

  // Flash Atttention and Lean Attention
  // int num_splits;

  // Lean Attention
  // bool use_lean_attention = false;
  // size_t workspace_bytes = 0;
  // bool allow_debug_info = false;

  // For MultiHeadAttention only.
  // data.kernel_type = AttentionKernelType::AttentionKernel_Default;
  // AllocatorPtr allocator = nullptr;

  /*
  const bool no_qkv_workspace = NoQkvWorkspace(parameters, data);
  size_t workspace_bytes = GetAttentionWorkspaceSize(sizeof(T),
                                                     parameters.batch_size,
                                                     parameters.num_heads,
                                                     parameters.head_size,
                                                     parameters.v_head_size,
                                                     parameters.sequence_length,
                                                     parameters.kv_sequence_length,
                                                     parameters.total_sequence_length,
                                                     fused_runner,
                                                     use_flash_attention,
                                                     use_lean_attention,
                                                     use_fused_cross_attention,
                                                     use_memory_efficient_attention,
                                                     use_cudnn_sdpa,
                                                     no_qkv_workspace);
  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  data.has_qkv_workspace = !no_qkv_workspace;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.workspace_bytes = workspace_bytes;
  */
  // QK type = T type
  typedef typename ToCudaType<T>::MappedType CudaQK;
  cublasHandle_t cublas = GetCublasHandle(context);
  cudnnHandle_t cudnn = GetCudnnHandle(context);
  auto& device_prop = GetDeviceProp();
  return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaQK>(
      device_prop, cublas, cudnn, context->GetComputeStream(), cparameters, data);

#endif
}

}  // namespace cuda
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/llm/attention_helper.h"
#include "core/providers/cuda/llm/attention.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"

using namespace onnxruntime::cuda;

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
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  // kv_num_heads, q_num_head are mandatory for 3D inputs but not used for 4D inputs.
  // The dimension is not yet known. If not specified, the inputs is assumed to be 4D.
  kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  qk_matmul_output_mode_ = info.node().OutputDefs().size() >= 4 && info.node().OutputDefs()[3]->Exists()
                               ? static_cast<attention_helper::QKMatMulOutputMode>(mode)
                               : attention_helper::QKMatMulOutputMode::kNone;
  ORT_ENFORCE(qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKMask ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKSoftCap ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKSoftMax,
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

  attention_helper::AttentionParameters parameters;
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
  Tensor* output_qk = context->Output(3, output_qk_shape);

  // To reuse the existing attention-cuda implementation in contrib ops,
  // map the parameters to contribop_parameters (MHA).
  onnxruntime::contrib::AttentionParameters contribop_parameters;

  // QKV format: Determine based on input dimensions
  // 3D inputs (B, S, D): Q_K_V_BSNH - will be transposed by PrepareQkv to BNSH
  // transpose_output is true for 3D inputs, false for 4D inputs
  if (!parameters.transpose_output) {
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;
    contribop_parameters.is_output_bnsh = true;
  } else {
    // 3D inputs in BSNH format (will be transposed)
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BSNH;
    contribop_parameters.is_output_bnsh = false;
  }

  typedef typename ToCudaType<T>::MappedType CudaT;

  // Check if this is Group Query Attention (GQA)
  const bool is_gqa = parameters.kv_num_heads != parameters.q_num_heads;

  if (is_gqa) {
    // Use GQA path
    // GQA only supports float16 and bfloat16 types
    if (std::is_same<T, float>::value) {
      ORT_THROW("GQA in Attention op (CUDA) does not support float32. Please use float16 or bfloat16.");
    }
    // For now, GQA doesn't support qk_matmul_output_mode other than kNone
    if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone) {
      ORT_THROW("qk_matmul_output_mode is not supported yet in GQA path of Attention op (CUDA).");
    }
    // GQA doesn't support softmax_precision yet
    if (parameters.softmax_precision != 0) {
      ORT_THROW("softmax_precision is not supported yet in GQA path of Attention op (CUDA).");
    }
    // causal attention is required for GQA
    if (!parameters.is_causal) {
      ORT_THROW("Non-causal attention is not supported yet in GQA path of Attention op (CUDA).");
    }
    // Bridge parameters to GroupQueryAttentionParameters
    onnxruntime::contrib::GroupQueryAttentionParameters gqa_parameters;
    gqa_parameters.batch_size = parameters.batch_size;
    gqa_parameters.sequence_length = parameters.q_sequence_length;
    gqa_parameters.seqlen_past_kv_cache = parameters.past_sequence_length;
    gqa_parameters.seqlen_present_kv_cache = parameters.total_sequence_length;
    gqa_parameters.total_sequence_length = parameters.total_sequence_length;
    gqa_parameters.kv_sequence_length = parameters.kv_sequence_length;
    gqa_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
    gqa_parameters.num_heads = parameters.q_num_heads;
    gqa_parameters.head_size = parameters.head_size;
    gqa_parameters.v_head_size = parameters.v_head_size;
    gqa_parameters.kv_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
    gqa_parameters.kv_num_heads = parameters.kv_num_heads;
    gqa_parameters.scale = parameters.scale;
    gqa_parameters.softcap = parameters.softcap;
    gqa_parameters.qkv_format = contribop_parameters.qkv_format;

    // Unset or set to default values for GQA-specific fields
    gqa_parameters.rotary_dim = 0;                     // New Attention op doesn't use rotary embeddings directly
    gqa_parameters.is_unidirectional = true;           // GQA requires causal attention
    gqa_parameters.past_present_share_buffer = false;  // New Attention op doesn't share buffer (unfused path)
    gqa_parameters.is_packed_qkv = false;              // New Attention op has separate Q, K, V inputs
    gqa_parameters.is_subsequent_prompt = false;       // Unfused path
    gqa_parameters.is_first_prompt = parameters.past_sequence_length == 0;
    gqa_parameters.do_rotary = false;  // New Attention op doesn't use rotary embeddings
    gqa_parameters.rotary_interleaved = false;
    gqa_parameters.use_smooth_softmax = false;
    gqa_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;
    gqa_parameters.past_kv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;
    gqa_parameters.local_window_size = -1;  // No local window for standard attention
    gqa_parameters.zeros_count = 0;
    gqa_parameters.zero_ptr = nullptr;
    gqa_parameters.num_splits = 1;  // No splits for unfused path

    // Construct GroupQueryAttentionData
    onnxruntime::contrib::cuda::GroupQueryAttentionData<CudaT> gqa_data;

    // Set input pointers
    gqa_data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
    gqa_data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
    gqa_data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
    gqa_data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
    gqa_data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());

    // Set output pointers
    gqa_data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());
    gqa_data.present_key = (present_key == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
    gqa_data.present_value = (present_value == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());

    // GQA only supports masking, not additive bias.
    // For bool mask, we need to convert it to sequence lengths.
    IAllocatorUniquePtr<int> seqlens_k_buffer;
    IAllocatorUniquePtr<int> seq_lens_buffer;
    if (attn_mask != nullptr && attn_mask->IsDataType<bool>()) {
      // Allocate buffer for seqlens_k (total_sequence_length - 1)
      seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, context->GetComputeStream());
      int* seqlens_k = seqlens_k_buffer.get();

      const bool* b_mask = attn_mask->Data<bool>();

      for (int b = 0; b < parameters.batch_size; ++b) {
        const bool* row = b_mask + b * parameters.total_sequence_length;
        int seq_len = 0;

        // Find the actual sequence length by looking for the last valid (true) position
        // Mask convention per Attention spec: true = valid (should participate), false = masked out
        for (int i = parameters.total_sequence_length - 1; i >= 0; --i) {
          if (row[i]) {
            seq_len = i + 1;
            break;
          }
        }
        // seqlens_k is total_sequence_length - 1 for historical reasons (matching GroupQueryAttention convention)
        seqlens_k[b] = seq_len - 1;
      }

      // Process seqlens_k to compute past_seq_lens, total_seq_lens, and padded_seq_lens
      // This follows the same pattern as GroupQueryAttention
      seq_lens_buffer = GetScratchBuffer<int>(3 * parameters.batch_size, context->GetComputeStream());
      auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
      gqa_data.past_seq_lens = seq_lens_buffer.get();
      gqa_data.total_seq_lens = seq_lens_buffer.get() + parameters.batch_size;
      gqa_data.padded_seq_lens = gqa_data.total_seq_lens + parameters.batch_size;

      auto& device_prop = GetDeviceProp();
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchGetSequenceLengths(
          seqlens_k,
          gqa_data.past_seq_lens,
          gqa_data.total_seq_lens,
          gqa_data.padded_seq_lens,
          parameters.batch_size,
          parameters.q_sequence_length,
          gqa_parameters.is_first_prompt,
          cuda_stream,
          device_prop.maxThreadsPerBlock));
    } else if (attn_mask != nullptr) {
      ORT_THROW("Non-boolean attn_mask is not supported yet in GQA path of Attention op (CUDA).");
    } else {
      // No mask provided - set sequence length pointers to nullptr
      gqa_data.total_seq_lens = nullptr;
      gqa_data.past_seq_lens = nullptr;
      gqa_data.padded_seq_lens = nullptr;
    }

    // Set GQA-specific fields
    gqa_data.cos_cache = nullptr;  // No rotary embeddings
    gqa_data.sin_cache = nullptr;
    gqa_data.head_sink = nullptr;
    gqa_data.position_ids = nullptr;

    // Use unfused attention path only
    gqa_data.use_flash_attention = false;
    gqa_data.use_memory_efficient_attention = false;
    gqa_data.use_flash_attention_fast_decode = false;

    // Flash buffers (not used in unfused path)
    gqa_data.softmax_lse = nullptr;
    gqa_data.softmax_lse_accum = nullptr;
    gqa_data.out_accum = nullptr;

    // Memory efficient buffers (not used)
    gqa_data.fmha_buffer = nullptr;
    gqa_data.qkv_buffer = nullptr;
    gqa_data.k = nullptr;
    gqa_data.v = nullptr;

#ifndef NDEBUG
    // Initialize debug tracking fields
    gqa_data.unpacked_qkv_buffer_size = 0;
    gqa_data.rotary_buffer_size = 0;
    gqa_data.position_ids_buffer_size = 0;
    gqa_data.unpacked_qkv_max_used = 0;
    gqa_data.rotary_max_used = 0;
    gqa_data.position_ids_max_used = 0;
#endif

    // Call GQA kernel (unfused path)
    auto& device_prop = GetDeviceProp();
    cublasHandle_t cublas = GetCublasHandle(context);

    return onnxruntime::contrib::cuda::QkvToContext<CudaT>(
        device_prop, cublas, context->GetComputeStream(), gqa_parameters, gqa_data);
  }

  // MHA path (kv_num_heads == q_num_heads)
  contribop_parameters.batch_size = parameters.batch_size;
  contribop_parameters.sequence_length = parameters.q_sequence_length;
  contribop_parameters.kv_sequence_length = parameters.kv_sequence_length;
  contribop_parameters.past_sequence_length = parameters.past_sequence_length;
  contribop_parameters.total_sequence_length = parameters.total_sequence_length;
  // max_sequence_length: For non-buffer-sharing case, this equals total_sequence_length (the present KV cache size)
  contribop_parameters.max_sequence_length = parameters.total_sequence_length;
  contribop_parameters.input_hidden_size = 0;  // Not applicable - new Attention op takes pre-projected Q/K/V
  contribop_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
  contribop_parameters.head_size = parameters.head_size;
  contribop_parameters.v_head_size = parameters.v_head_size;
  contribop_parameters.v_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
  contribop_parameters.num_heads = parameters.q_num_heads;
  contribop_parameters.rotary_dim = 0;
  contribop_parameters.num_splits = 1;
  contribop_parameters.beam_width = 1;
  contribop_parameters.is_unidirectional = parameters.is_causal;
  contribop_parameters.past_present_share_buffer = false;  // New Attention op doesn't share buffer
  contribop_parameters.is_packed_qkv = false;
  contribop_parameters.do_rotary = false;

  // The new Attention op uses attn_mask as attention_bias (additive bias), not as key_padding_mask
  // So mask_type should always be MASK_NONE since we don't have a separate padding mask input
  contribop_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;

  // Determine broadcast flags for attention_bias (if it exists)
  // Note: The new Attention op uses attn_mask as attention_bias
  // The attention_bias should be broadcastable to (batch_size, kv_num_heads, q_sequence_length, total_sequence_length)
  // attn_mask can be 2D, 3D, or 4D. Broadcasting aligns from the right (trailing dimensions).
  if (attn_mask != nullptr) {
    // TODO(titaiwang, xadupre): attn_mask bool is not supported yet
    if (attn_mask->IsDataType<bool>()) {
      ORT_THROW("Boolean attn_mask is not supported yet in Attention op (CUDA).");
    }

    size_t attn_mask_dims_size = attn_mask->Shape().NumDimensions();
    auto attn_mask_dims = attn_mask->Shape().GetDims();
    // For 2D mask (q_seq_len, total_seq_len): both batch and heads dimensions need broadcasting
    // For 3D mask (X, q_seq_len, total_seq_len): batch needs broadcasting if X==1, heads always needs broadcasting
    // For 4D mask (B, H, q_seq_len, total_seq_len): check if B==1 and H==1

    if (attn_mask_dims_size == 2) {
      // 2D mask: both dimensions need broadcasting
      contribop_parameters.broadcast_attn_bias_dim_0 = true;
      contribop_parameters.broadcast_attn_bias_dim_1 = true;
    } else if (attn_mask_dims_size == 3) {
      // 3D mask: dim 0 broadcasts if it's 1, dim 1 (heads) always broadcasts
      contribop_parameters.broadcast_attn_bias_dim_0 = attn_mask_dims[0] == 1;
      contribop_parameters.broadcast_attn_bias_dim_1 = true;
    } else {
      // 4D mask: check both dim 0 and dim 1 explicitly
      contribop_parameters.broadcast_attn_bias_dim_0 = attn_mask_dims[0] == 1;
      contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[1] == 1;
    }
  } else {
    contribop_parameters.broadcast_attn_bias_dim_0 = false;
    contribop_parameters.broadcast_attn_bias_dim_1 = false;
  }

  contribop_parameters.mask_filter_value = -10000.0f;
  contribop_parameters.scale = parameters.scale;
  contribop_parameters.use_tf32 = UseTF32();
  // TODO(titaiwang, xadupre): qk_matmul_output_mode only supports kNone and kQK for now
  if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone &&
      qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kQK) {
    ORT_THROW("qk_matmul_output_mode other than -1 (None) and 0 (QK) is not supported yet in Attention op (CUDA).");
  }
  // TODO(titaiwang, xadupre): softcap and softmax_precision are not used yet
  if (parameters.softcap != 0.0f) {
    ORT_THROW("softcap is not supported yet in Attention op (CUDA).");
  }
  if (parameters.softmax_precision != 0) {
    ORT_THROW("softmax_precision is not supported yet in Attention op (CUDA).");
  }

  // Construct AttentionData to pass to QkvToContext
  onnxruntime::contrib::cuda::AttentionData<CudaT> data;

  // Set input pointers
  data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
  data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
  data.mask_index = nullptr;  // New Attention op doesn't have key_padding_mask
  data.mask_index_dims = gsl::span<const int64_t>();
  data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());

  // Set output pointers
  data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  data.present_key = (present_key == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (present_value == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  if (nullptr != output_qk) {
    data.output_qk = reinterpret_cast<CudaT*>(output_qk->MutableData<T>());
  }

  // Set additional fields
  data.bias = nullptr;  // New Attention op doesn't have bias
  if (nullptr != attn_mask) {
    data.attention_bias = reinterpret_cast<const CudaT*>(attn_mask->Data<T>());
  }
  data.qkv_format = contribop_parameters.qkv_format;

  // TODO: Determine which kernel to use (Flash Attention, Memory Efficient Attention, etc.)
  // For now, set flags to false and let QkvToContext use the unfused path
  data.use_flash_attention = false;
  data.use_memory_efficient_attention = false;
  data.fused_runner = nullptr;
  data.fused_cross_attention_kernel = nullptr;

  // Allocate workspace for Q, K, V processing and scratch buffer
  const bool no_qkv_workspace = onnxruntime::contrib::cuda::NoQkvWorkspace(contribop_parameters, data);
  size_t workspace_bytes = onnxruntime::contrib::cuda::GetAttentionWorkspaceSize(
      sizeof(T),
      contribop_parameters.batch_size,
      contribop_parameters.num_heads,
      contribop_parameters.head_size,
      contribop_parameters.v_head_size,
      contribop_parameters.sequence_length,
      contribop_parameters.kv_sequence_length,
      contribop_parameters.total_sequence_length,
      nullptr,  // fused_runner
      false,    // use_flash_attention
      false,    // use_lean_attention
      false,    // use_fused_cross_attention
      false,    // use_memory_efficient_attention
      false,    // use_cudnn_flash_attention
      no_qkv_workspace);
  auto work_space = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  data.has_qkv_workspace = !no_qkv_workspace;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.workspace_bytes = workspace_bytes;

  // Call QkvToContext to perform the attention computation
  auto& device_prop = GetDeviceProp();
  cublasHandle_t cublas = GetCublasHandle(context);
  cudnnHandle_t cudnn = GetCudnnHandle(context);

  // QkvToContext takes two template parameters: T for computation type, QK for output_qk type
  // For now, both are the same type (CudaT)
  return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaT>(
      device_prop, cublas, cudnn, context->GetComputeStream(), contribop_parameters, data);
}
}  // namespace cuda
}  // namespace onnxruntime

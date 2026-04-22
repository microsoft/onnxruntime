// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/llm/attention.h"
#include "core/providers/cpu/llm/attention_helper.h"
#include "core/providers/cuda/llm/attention.h"
#include "core/providers/cuda/llm/attention_mask_impl.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "core/providers/cuda/cuda_type_conversion.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

namespace llm_attention_detail {

template <typename NodeType>
bool HasOutput(const NodeType& node, size_t output_index) {
  if constexpr (requires(const NodeType& candidate) { candidate.OutputCount(); candidate.OutputExists(output_index); }) {
    return node.OutputCount() > output_index && node.OutputExists(output_index);
  } else {
    return node.OutputDefs().size() > output_index && node.OutputDefs()[output_index]->Exists();
  }
}

}  // namespace llm_attention_detail

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                            \
      Attention,                                                      \
      kOnnxDomain,                                                    \
      23,                                                             \
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

#define REGISTER_KERNEL_TYPED_24(T)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      Attention,                                                      \
      kOnnxDomain,                                                    \
      24,                                                             \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED_24(float)
REGISTER_KERNEL_TYPED_24(MLFloat16)
REGISTER_KERNEL_TYPED_24(BFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  const auto& node = info.node();
  qk_matmul_output_mode_ = llm_attention_detail::HasOutput(node, 3)
                               ? static_cast<attention_helper::QKMatMulOutputMode>(mode)
                               : attention_helper::QKMatMulOutputMode::kNone;
  ORT_ENFORCE(qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKMask ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKSoftCap ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQKSoftMax,
              "qk_matmul_output_mode must be one of: kNone(-1), kQK(0), kQKMask(1), kQKSoftCap(2), kQKSoftMax(3).");
  scale_ = info.GetAttrOrDefault<float>("scale", std::numeric_limits<T>::quiet_NaN());
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  ORT_ENFORCE(softcap_ >= 0.0f, "softcap must be non-negative");
  softmax_precision_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("softmax_precision", 0));
  // Valid softmax_precision values are TensorProto data types: 0 (not set), 1 (FLOAT), 10 (FLOAT16), 16 (BFLOAT16)
  // DOUBLE (11) is excluded — CUDA computes softmax in FP32 and cannot satisfy FP64 precision.
  ORT_ENFORCE(softmax_precision_ == 0 || softmax_precision_ == 1 || softmax_precision_ == 10 ||
                  softmax_precision_ == 16,
              "softmax_precision must be a valid TensorProto data type (0, 1, 10, or 16).");
  ORT_ENFORCE(scale_ > 0 || std::isnan(scale_), "scale must be greater than 0 if specified");

  const auto* kernel_options = this->GetAttentionKernelOptions();
  disable_flash_attention_ = std::is_same<T, float>::value || !kernel_options->UseFlashAttention();
  disable_memory_efficient_attention_ = !kernel_options->UseEfficientAttention();
}

// ============================================================================
// Transpose helpers: eliminate repeated if-constexpr type-switch blocks.
// T is the ORT type (MLFloat16, BFloat16, float). The helpers map T to the
// corresponding CUDA type via ToCudaType<T>::MappedType and forward to the
// overloaded Transpose functions in contrib_ops.
// ============================================================================

template <typename T>
static Status TransposeBNSHtoBSNH(int batch_size, int sequence_length,
                                  int num_heads, int head_size,
                                  const void* input, void* output,
                                  cudaStream_t stream, int max_threads_per_block) {
  using CudaT = typename ToCudaType<T>::MappedType;
  return onnxruntime::contrib::cuda::Transpose_BNSH_to_BSNH(
      batch_size, sequence_length, num_heads, head_size,
      reinterpret_cast<const CudaT*>(input),
      reinterpret_cast<CudaT*>(output),
      stream, max_threads_per_block);
}

template <typename T>
static Status TransposeBSNHtoBNSH(int batch_size, int sequence_length,
                                  int num_heads, int head_size,
                                  const void* input, void* output,
                                  cudaStream_t stream, int max_threads_per_block) {
  using CudaT = typename ToCudaType<T>::MappedType;
  return onnxruntime::contrib::cuda::Transpose_BSNH_to_BNSH(
      batch_size, sequence_length, num_heads, head_size,
      reinterpret_cast<const CudaT*>(input),
      reinterpret_cast<CudaT*>(output),
      stream, max_threads_per_block);
}

// ============================================================================
// ConvertAttnMaskToBias: shared helper for mask→additive bias conversion.
// Used by the MEA path to convert masks before the CUTLASS kernel call.
// Converts bool masks to additive bias (true→0, false→mask_filter_value),
// passes float masks through directly, and sets broadcast flags from mask shape.
// ============================================================================
template <typename T>
Status Attention<T>::ConvertAttnMaskToBias(
    OpKernelContext* context,
    const Tensor* attn_mask,
    cudaStream_t cuda_stream,
    int max_threads_per_block,
    IAllocatorUniquePtr<void>& converted_mask_buffer,
    const void*& attn_bias_data,
    bool& broadcast_bias_dim_0,
    bool& broadcast_bias_dim_1) const {
  if (attn_mask->IsDataType<bool>()) {
    using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
    int64_t num_elements = attn_mask->Shape().Size();
    converted_mask_buffer = GetScratchBuffer<void>(
        num_elements * sizeof(NativeCudaT), GetComputeStream(context));
    float mask_filter_value = static_cast<float>(std::numeric_limits<T>::lowest());
    ORT_RETURN_IF_ERROR(LaunchConvertBoolMaskToAttentionBias<NativeCudaT>(
        attn_mask->Data<bool>(),
        reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
        num_elements, mask_filter_value, cuda_stream,
        max_threads_per_block));
    attn_bias_data = converted_mask_buffer.get();
  } else {
    attn_bias_data = attn_mask->Data<T>();
  }

  size_t mask_dims = attn_mask->Shape().NumDimensions();
  auto dims = attn_mask->Shape().GetDims();
  if (mask_dims == 2) {
    broadcast_bias_dim_0 = true;
    broadcast_bias_dim_1 = true;
  } else if (mask_dims == 3) {
    broadcast_bias_dim_0 = true;
    broadcast_bias_dim_1 = dims[0] == 1;
  } else {
    broadcast_bias_dim_0 = dims[0] == 1;
    broadcast_bias_dim_1 = dims[1] == 1;
  }
  return Status::OK();
}

// ============================================================================
// RunFlashAttention: Direct flash attention kernel call
// ============================================================================
//
// Flash Attention dispatch paths:
//   Path 1: nonpad_kv_seqlen (opset 24 external cache) -> mha_fwd_kvcache
//   Path 2: past_key + past_value (internal cache decode) -> mha_fwd_kvcache
//           - No mask support (attn_mask rejected at eligibility)
//           - 4D BNSH: transposes Q/K/V to BSNH before kernel
//   Path 3: no past, no mask (prompt) -> mha_fwd
//   Eligibility: fp16/bf16, head_size==v_head_size, no output_qk, attn_mask==nullptr
//   Note: softcap is passed to the Flash kernel natively. softmax_precision is
//   inherently satisfied (Flash accumulates softmax in FP32).
//
// PERFORMANCE NOTE: ONNX Attention's internal-cache decode path (past_key/past_value)
// is ~15-30% slower than contrib GQA's decode path for grouped-query attention workloads.
// When using external KV cache via TensorScatter + nonpad_kv_seqlen (opset 24), the
// copy overhead (point 1) is eliminated. The remaining ~5-15% gap is from the missing
// XQA kernel (point 2).
//
// The internal-cache overhead comes from:
//
// 1. No past_present_share_buffer: The ONNX Attention spec requires past_key/value
//    shape = (B, H, past_seq, head_size) and present_key/value shape =
//    (B, H, total_seq, head_size) where total_seq = past_seq + kv_seq.
//    Since past and present have different shapes, they cannot share the same buffer.
//    Contrib GQA allows past and present to be the same tensor (in-place append),
//    eliminating the concat copy overhead. ONNX Attention uses LaunchConcatNewToPastKV
//    to fuse past copy + new token append in one kernel (no memset or strided copy).
//    This overhead does NOT apply to the external-cache path (TensorScatter +
//    nonpad_kv_seqlen), which bypasses past/present entirely.
//
// 2. No XQA kernel: GQA's specialized XQA decode kernel (xqa_loader.h) requires
//    past_present_share_buffer to function. Since ONNX Attention cannot share buffers
//    (see point 1), XQA is fundamentally incompatible with this op's spec design.
//    This accounts for the remaining ~5-15% gap even on the external-cache path.
//
// 3. These are spec-level limitations, not implementation gaps. For production LLM
//    inference, the external-cache path (TensorScatter + nonpad_kv_seqlen) is
//    recommended and achieves near-parity with contrib GQA performance.
//
template <typename T>
Status Attention<T>::RunFlashAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    const attention_helper::AttentionParameters& parameters) const {
#if USE_FLASH_ATTENTION
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = Stream(context);
  const bool is_bf16 = std::is_same<T, BFloat16>::value;
  const bool is_bsnh = parameters.transpose_output;  // 3D inputs → BSNH

  // --- Common buffer allocation ---
  size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(
      parameters.q_sequence_length, parameters.batch_size, parameters.q_num_heads);

  auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
      onnxruntime::flash::get_num_splits_and_buffer_sizes(
          parameters.batch_size, parameters.q_sequence_length,
          parameters.total_sequence_length, parameters.q_num_heads,
          parameters.head_size, device_prop.multiProcessorCount);

  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, GetComputeStream(context));
  auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, GetComputeStream(context));
  auto out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, GetComputeStream(context));

  if (softmax_lse_accum_bytes > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(softmax_lse_accum_buffer.get(), 0,
                                         softmax_lse_accum_bytes, cuda_stream));
  }
  if (out_accum_bytes > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(out_accum_buffer.get(), 0,
                                         out_accum_bytes, cuda_stream));
  }

  // --- Transpose Q from BNSH to BSNH (flash always expects Q as BSNH) ---
  const void* q_data = Q->Data<T>();
  IAllocatorUniquePtr<void> q_bsnh_buffer;
  if (!is_bsnh) {
    size_t q_bytes = sizeof(T) * parameters.batch_size * parameters.q_sequence_length *
                     parameters.q_num_heads * parameters.head_size;
    q_bsnh_buffer = GetScratchBuffer<void>(q_bytes, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(TransposeBNSHtoBSNH<T>(
        parameters.batch_size, parameters.q_sequence_length,
        parameters.q_num_heads, parameters.head_size,
        Q->Data<T>(), q_bsnh_buffer.get(),
        cuda_stream, device_prop.maxThreadsPerBlock));
    q_data = q_bsnh_buffer.get();
  }

  // Flash outputs BSNH. If Y expects BNSH, write to scratch then transpose.
  void* out_data = Y->MutableData<T>();
  IAllocatorUniquePtr<void> out_bsnh_buffer;
  if (!is_bsnh) {
    size_t out_bytes = sizeof(T) * parameters.batch_size * parameters.q_sequence_length *
                       parameters.q_num_heads * parameters.v_head_size;
    out_bsnh_buffer = GetScratchBuffer<void>(out_bytes, GetComputeStream(context));
    out_data = out_bsnh_buffer.get();
  }

  bool present_kv_already_populated = false;

  // --- Path 1: nonpad_kv_seqlen (opset 24 external KV cache) ---
  if (nonpad_kv_seqlen != nullptr) {
    ORT_ENFORCE(parameters.past_sequence_length == 0,
                "RunFlashAttention with nonpad_kv_seqlen requires K/V to be the full cache "
                "(past_sequence_length must be 0, got ",
                parameters.past_sequence_length, ").");

    // seqlens_k_buffer lifetime: allocated via BFC arena, remains valid for all kernel
    // launches on the same CUDA stream until the IAllocatorUniquePtr goes out of scope.
    auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
        nonpad_kv_seqlen->Data<int64_t>(),
        seqlens_k_buffer.get(),
        parameters.batch_size,
        parameters.total_sequence_length,
        cuda_stream,
        device_prop.maxThreadsPerBlock));

    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
        device_prop, cuda_stream,
        const_cast<void*>(q_data),
        const_cast<void*>(static_cast<const void*>(K->Data<T>())),
        const_cast<void*>(static_cast<const void*>(V->Data<T>())),
        /*k=*/nullptr, /*v=*/nullptr,
        out_data,
        softmax_lse_buffer.get(),
        const_cast<void*>(static_cast<const void*>(seqlens_k_buffer.get())),
        /*rotary_cos=*/nullptr, /*rotary_sin=*/nullptr,
        /*cache_batch_idx=*/nullptr, /*leftpad_k=*/nullptr,
        /*head_sink=*/nullptr, /*block_table=*/nullptr,
        parameters.batch_size, parameters.q_num_heads, parameters.kv_num_heads,
        parameters.head_size,
        parameters.q_sequence_length, parameters.kv_sequence_length,
        /*seqlen_k_new=*/0, /*rotary_dim=*/0,
        parameters.scale, parameters.softcap,
        parameters.is_causal, is_bf16, /*use_smooth_softmax=*/false,
        /*past_bsnh=*/is_bsnh,
        static_cast<int>(num_splits),
        softmax_lse_accum_buffer.get(), out_accum_buffer.get(),
        /*local_window_size=*/-1, /*is_rotary_interleaved=*/false,
        /*is_packed_qkv=*/false));
  }
  // --- Path 2: Decode with past KV cache ---
  else if (past_key != nullptr) {
    ORT_ENFORCE(past_value != nullptr, "past_key requires past_value.");
    ORT_ENFORCE(present_key != nullptr && present_value != nullptr,
                "present_key/value outputs are required when past_key is provided.");

    // TODO(titaiwang): Consolidate preprocessing (RoPE, mask conversion, KV cache concat) into a
    // single fused kernel like GQA's LaunchUnpackRoPEAppend. Current decode path uses 4-6 kernel
    // launches; a fused approach would reduce to ~2, saving ~21μs launch overhead and ~256KB
    // intermediate buffer traffic per decode step.

    // Concat past + new KV directly into present buffers using a single fused kernel.
    // This replaces the old pattern of memset + strided cudaMemcpy2DAsync + Flash's
    // internal Append_KV, eliminating redundant memory writes per decode step (proportional to B×H×total_seq×head_size).
    // LaunchConcatNewToPastKV reads past (BNSH) and new (BSNH), writes present (BNSH).
    // OrtToCudaType maps BFloat16 → __nv_bfloat16 (native HW arithmetic on SM80+),
    // consistent with GQA's early native-type conversion pattern.
    using NativeCudaT = typename OrtToCudaType<T>::type;

    // Step 1: Compute per-batch past sequence lengths for the concat kernel.
    // The concat kernel needs past_seq_lens to know where past data ends and new begins.
    // attn_mask is always nullptr here (Flash rejects attn_mask), so use uniform past_seq.
    auto past_seqlens_buffer = GetScratchBuffer<int>(parameters.batch_size, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchFillInt32(past_seqlens_buffer.get(), parameters.past_sequence_length,
                                        parameters.batch_size, cuda_stream,
                                        device_prop.maxThreadsPerBlock));

    // Step 2: Transpose K/V to BSNH if input is 4D BNSH (concat kernel reads new as BSNH).
    const T* k_new_bsnh = K->Data<T>();
    const T* v_new_bsnh = V->Data<T>();
    IAllocatorUniquePtr<void> k_bsnh_buffer;
    IAllocatorUniquePtr<void> v_bsnh_buffer;
    if (!is_bsnh) {
      size_t k_bytes = sizeof(T) * parameters.batch_size * parameters.kv_sequence_length *
                       parameters.kv_num_heads * parameters.head_size;
      size_t v_bytes = sizeof(T) * parameters.batch_size * parameters.kv_sequence_length *
                       parameters.kv_num_heads * parameters.v_head_size;
      k_bsnh_buffer = GetScratchBuffer<void>(k_bytes, GetComputeStream(context));
      v_bsnh_buffer = GetScratchBuffer<void>(v_bytes, GetComputeStream(context));
      ORT_RETURN_IF_ERROR(TransposeBNSHtoBSNH<T>(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          K->Data<T>(), k_bsnh_buffer.get(),
          cuda_stream, device_prop.maxThreadsPerBlock));
      ORT_RETURN_IF_ERROR(TransposeBNSHtoBSNH<T>(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          V->Data<T>(), v_bsnh_buffer.get(),
          cuda_stream, device_prop.maxThreadsPerBlock));
      k_new_bsnh = static_cast<const T*>(k_bsnh_buffer.get());
      v_new_bsnh = static_cast<const T*>(v_bsnh_buffer.get());
    }

    // Step 3: Fused concat: past_key + new_key → present_key (and same for values).
    // One kernel copies past data from [0, past_seq) and new data from BSNH layout
    // into present buffer at [past_seq, past_seq + kv_seq), all in BNSH.
    // Note: is_bsnh=false means past/present cache layout is BNSH. New tokens
    // (k_new_bsnh/v_new_bsnh) are always read as BSNH by the kernel (hardcoded strides).
    // past_seqlens is uniform (no mask) so every position in the present buffer is written.
    ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchConcatNewToPastKV<NativeCudaT>(
        parameters.batch_size,
        parameters.kv_num_heads,
        parameters.head_size,
        parameters.kv_sequence_length,
        parameters.past_sequence_length,
        parameters.total_sequence_length,
        /*is_bsnh=*/false,
        past_seqlens_buffer.get(),
        /*total_seq_lens=*/nullptr,
        reinterpret_cast<const NativeCudaT*>(past_key->Data<T>()),
        reinterpret_cast<const NativeCudaT*>(past_value->Data<T>()),
        reinterpret_cast<const NativeCudaT*>(k_new_bsnh),
        reinterpret_cast<const NativeCudaT*>(v_new_bsnh),
        reinterpret_cast<NativeCudaT*>(present_key->MutableData<T>()),
        reinterpret_cast<NativeCudaT*>(present_value->MutableData<T>()),
        cuda_stream,
        device_prop.maxThreadsPerBlock,
        /*past_only=*/false));

    // Step 4: Compute total seqlens for mha_fwd_kvcache.
    // With k_new=nullptr, the kernel treats seqlens_k as the total valid token count
    // (not pre-append count), so we need past + new.
    // attn_mask is always nullptr here (Flash rejects attn_mask), so use uniform seqlens.
    auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchFillInt32(
        seqlens_k_buffer.get(),
        parameters.past_sequence_length + parameters.kv_sequence_length,
        parameters.batch_size, cuda_stream,
        device_prop.maxThreadsPerBlock));

    // Step 5: Flash attention on pre-populated cache.
    // k_new=nullptr tells mha_fwd_kvcache to skip its internal Append_KV — the cache
    // is already fully populated by LaunchConcatNewToPastKV above.
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd_kvcache(
        device_prop, cuda_stream,
        const_cast<void*>(q_data),
        static_cast<void*>(present_key->MutableData<T>()),
        static_cast<void*>(present_value->MutableData<T>()),
        /*k_new=*/nullptr, /*v_new=*/nullptr,
        out_data,
        softmax_lse_buffer.get(),
        static_cast<void*>(seqlens_k_buffer.get()),
        /*rotary_cos=*/nullptr, /*rotary_sin=*/nullptr,
        /*cache_batch_idx=*/nullptr, /*leftpad_k=*/nullptr,
        /*head_sink=*/nullptr, /*block_table=*/nullptr,
        parameters.batch_size, parameters.q_num_heads, parameters.kv_num_heads,
        parameters.head_size,
        parameters.q_sequence_length, parameters.total_sequence_length,
        /*seqlen_k_new=*/0, /*rotary_dim=*/0,
        parameters.scale, parameters.softcap,
        parameters.is_causal, is_bf16, /*use_smooth_softmax=*/false,
        /*past_bsnh=*/false,  // present cache is BNSH
        static_cast<int>(num_splits),
        softmax_lse_accum_buffer.get(), out_accum_buffer.get(),
        /*local_window_size=*/-1, /*is_rotary_interleaved=*/false,
        /*is_packed_qkv=*/false));

    present_kv_already_populated = true;
  }
  // --- Path 3: Prompt flash (no past, no mask) ---
  // Note: prompt with bool mask is handled by MEA (flash_eligible excludes it).
  else {
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
        device_prop, cuda_stream,
        const_cast<void*>(q_data),
        const_cast<void*>(static_cast<const void*>(K->Data<T>())),
        const_cast<void*>(static_cast<const void*>(V->Data<T>())),
        out_data,
        softmax_lse_buffer.get(),
        parameters.batch_size, parameters.q_num_heads, parameters.kv_num_heads,
        parameters.head_size,
        parameters.q_sequence_length, parameters.kv_sequence_length,
        parameters.scale, parameters.softcap,
        parameters.is_causal, is_bf16, /*use_smooth_softmax=*/false,
        static_cast<int>(num_splits),
        softmax_lse_accum_buffer.get(), out_accum_buffer.get(),
        is_bsnh));
  }

  // --- Transpose output BSNH → BNSH if input was 4D (BNSH) ---
  if (!is_bsnh && out_bsnh_buffer != nullptr) {
    ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(
        parameters.batch_size, parameters.q_sequence_length,
        parameters.q_num_heads, parameters.v_head_size,
        out_bsnh_buffer.get(), Y->MutableData<T>(),
        cuda_stream, device_prop.maxThreadsPerBlock));
  }

  // --- Populate present_key/value (BNSH) from K/V (BSNH) ---
  // Skip for decode path where mha_fwd_kvcache already populated present buffers.
  if (!present_kv_already_populated) {
    if (present_key != nullptr && is_bsnh) {
      ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.head_size,
          K->Data<T>(), present_key->MutableData<T>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if (present_key != nullptr && !is_bsnh) {
      // 4D BNSH prompt: K is already BNSH, just D2D copy to present
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          present_key->MutableData<T>(), K->Data<T>(),
          K->SizeInBytes(), cudaMemcpyDeviceToDevice, cuda_stream));
    }
    if (present_value != nullptr && is_bsnh) {
      ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(
          parameters.batch_size, parameters.kv_sequence_length,
          parameters.kv_num_heads, parameters.v_head_size,
          V->Data<T>(), present_value->MutableData<T>(),
          cuda_stream, device_prop.maxThreadsPerBlock));
    } else if (present_value != nullptr && !is_bsnh) {
      // 4D BNSH prompt: V is already BNSH, just D2D copy to present
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          present_value->MutableData<T>(), V->Data<T>(),
          V->SizeInBytes(), cudaMemcpyDeviceToDevice, cuda_stream));
    }
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(Q);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(V);
  ORT_UNUSED_PARAMETER(past_key);
  ORT_UNUSED_PARAMETER(past_value);
  ORT_UNUSED_PARAMETER(nonpad_kv_seqlen);
  ORT_UNUSED_PARAMETER(Y);
  ORT_UNUSED_PARAMETER(present_key);
  ORT_UNUSED_PARAMETER(present_value);
  ORT_UNUSED_PARAMETER(parameters);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Flash attention is not available in this build.");
#endif
}

// ============================================================================
// RunMemoryEfficientAttention: Direct memory-efficient attention kernel call
// ============================================================================
//
// Memory Efficient Attention (cutlass FMHA) dispatch paths:
//   Path 1: nonpad_kv_seqlen (opset 24 external cache) -> has_custom_right_padding mode
//   Path 2: no past, with mask (prompt) -> standard MEA with additive bias
//   Path 3: no past, no mask (prompt) -> standard MEA
//   Eligibility: see has_memory_efficient_attention() (SM50+/53+/80+ by dtype,
//                head_size <= 1024), plus: no output_qk, no past_key (decode excluded),
//                bias stride alignment.
//   Note: softcap is forwarded to the MEA kernel via p.softcap. softmax_precision
//   is inherently satisfied (cutlass FMHA accumulates softmax in FP32).
//
template <typename T>
Status Attention<T>::RunMemoryEfficientAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    const attention_helper::AttentionParameters& parameters) const {
#if USE_MEMORY_EFFICIENT_ATTENTION
  ORT_UNUSED_PARAMETER(past_key);
  ORT_UNUSED_PARAMETER(past_value);
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = Stream(context);
  const bool is_bsnh = parameters.transpose_output;
  const int sm = device_prop.major * 10 + device_prop.minor;

  // Q/K/V pointers — MEA expects BSNH format for Q
  const void* q_data = Q->Data<T>();
  const void* k_data = K->Data<T>();
  const void* v_data = V->Data<T>();

  // --- Transpose Q from BNSH to BSNH if 4D input ---
  IAllocatorUniquePtr<void> q_bsnh_buffer;
  if (!is_bsnh) {
    size_t q_bytes = sizeof(T) * parameters.batch_size * parameters.q_sequence_length *
                     parameters.q_num_heads * parameters.head_size;
    q_bsnh_buffer = GetScratchBuffer<void>(q_bytes, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(TransposeBNSHtoBSNH<T>(
        parameters.batch_size, parameters.q_sequence_length,
        parameters.q_num_heads, parameters.head_size,
        Q->Data<T>(), q_bsnh_buffer.get(),
        cuda_stream, device_prop.maxThreadsPerBlock));
    q_data = q_bsnh_buffer.get();
  }

  // MEA output is BSNH. If Y expects BNSH, write to scratch then transpose.
  void* out_data = Y->MutableData<T>();
  IAllocatorUniquePtr<void> out_bsnh_buffer;
  if (!is_bsnh) {
    size_t out_bytes = sizeof(T) * parameters.batch_size * parameters.q_sequence_length *
                       parameters.q_num_heads * parameters.v_head_size;
    out_bsnh_buffer = GetScratchBuffer<void>(out_bytes, GetComputeStream(context));
    out_data = out_bsnh_buffer.get();
  }

  // GQA head expansion: MEA requires matching num_heads for Q/K/V.
  // When q_num_heads != kv_num_heads, expand K/V via LaunchUngroup.
  const bool is_gqa = parameters.q_num_heads != parameters.kv_num_heads;
  IAllocatorUniquePtr<void> k_expand_buffer;
  IAllocatorUniquePtr<void> v_expand_buffer;

  if (is_gqa) {
    // GQA+MEA only works with fp16/bf16 (LaunchUngroup lacks fp32 template instantiation
    // in group_query_attention_impl.cu).
    // Use if constexpr to avoid instantiating LaunchUngroup<float>.
    if constexpr (std::is_same_v<T, float>) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "GQA with Memory Efficient Attention requires fp16 or bf16, not fp32.");
    } else {
      ORT_ENFORCE(parameters.head_size == parameters.v_head_size,
                  "GQA with MEA requires head_size == v_head_size for LaunchUngroup.");
      ORT_ENFORCE(parameters.head_size % 4 == 0,
                  "GQA with MEA requires head_size divisible by 4 for LaunchUngroup (float2 access).");
      const size_t expanded_kv_elements = static_cast<size_t>(parameters.batch_size) *
                                          static_cast<size_t>(parameters.total_sequence_length) *
                                          static_cast<size_t>(parameters.q_num_heads) *
                                          static_cast<size_t>(parameters.head_size);
      k_expand_buffer = GetScratchBuffer<void>(expanded_kv_elements * sizeof(T), GetComputeStream(context));
      v_expand_buffer = GetScratchBuffer<void>(expanded_kv_elements * sizeof(T), GetComputeStream(context));

      onnxruntime::contrib::GroupQueryAttentionParameters ungroup_params = {};
      ungroup_params.batch_size = parameters.batch_size;
      ungroup_params.num_heads = parameters.q_num_heads;
      ungroup_params.kv_num_heads = parameters.kv_num_heads;
      ungroup_params.head_size = parameters.head_size;

      using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchUngroup<NativeCudaT>(
          ungroup_params,
          reinterpret_cast<float2*>(k_expand_buffer.get()),
          reinterpret_cast<float2*>(v_expand_buffer.get()),
          reinterpret_cast<const float2*>(k_data),
          reinterpret_cast<const float2*>(v_data),
          parameters.total_sequence_length,
          parameters.total_sequence_length,
          is_bsnh,
          cuda_stream,
          device_prop.maxThreadsPerBlock));

      k_data = k_expand_buffer.get();
      v_data = v_expand_buffer.get();
    }
  }

  // Note: MEA with past_key/value is handled by the unfused fallback.
  // The cascade in ComputeInternal ensures past_key == nullptr when we reach here.

  // Handle attention mask → attention_bias conversion
  IAllocatorUniquePtr<void> converted_mask_buffer;
  const void* attn_bias_data = nullptr;
  bool broadcast_bias_dim_0 = false;
  bool broadcast_bias_dim_1 = false;

  if (nonpad_kv_seqlen != nullptr) {
    // Convert nonpad_kv_seqlen to seqlens_k for custom right padding.
    // MEA expects actual token count (not count-1), so use FlashSeqlensK variant.
    auto seqlens_k_buffer = GetScratchBuffer<int>(parameters.batch_size, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
        nonpad_kv_seqlen->Data<int64_t>(),
        seqlens_k_buffer.get(),
        parameters.batch_size,
        parameters.total_sequence_length,
        cuda_stream,
        device_prop.maxThreadsPerBlock));

    // When attn_mask is also provided, convert it to additive attn_bias so MEA
    // applies both custom right padding (seqlens_k) and the attention mask (attn_bias).
    if (attn_mask != nullptr) {
      ORT_RETURN_IF_ERROR(ConvertAttnMaskToBias(context, attn_mask, cuda_stream,
                                                device_prop.maxThreadsPerBlock,
                                                converted_mask_buffer, attn_bias_data,
                                                broadcast_bias_dim_0, broadcast_bias_dim_1));
    }

    onnxruntime::contrib::cuda::MemoryEfficientAttentionParams p;
    p.sm = sm;
    p.is_half = std::is_same<T, MLFloat16>::value;
    p.is_bf16 = std::is_same<T, BFloat16>::value;
    p.is_kv_bsnh = is_bsnh;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.q_num_heads;
    p.sequence_length = parameters.q_sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.max_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_causal;
    p.scale = parameters.scale;
    p.softcap = parameters.softcap;
    p.seqlen_k_ptr = seqlens_k_buffer.get();
    p.has_custom_right_padding = true;
    p.broadcast_attn_bias_dim_0 = broadcast_bias_dim_0;
    p.broadcast_attn_bias_dim_1 = broadcast_bias_dim_1;
    p.query = q_data;
    p.key = k_data;
    p.value = v_data;
    p.attn_bias = attn_bias_data;
    p.stream = cuda_stream;
    p.output = out_data;

    IAllocatorUniquePtr<void> workspace_buffer;
    if (onnxruntime::contrib::cuda::MemoryEfficientAttentionParams::need_workspace(
            parameters.v_head_size, sizeof(T) == sizeof(float))) {
      size_t workspace_bytes = sizeof(float) * parameters.batch_size * parameters.q_sequence_length *
                               parameters.q_num_heads * parameters.v_head_size;
      workspace_buffer = GetScratchBuffer<void>(workspace_bytes, GetComputeStream(context));
      p.workspace = workspace_buffer.get();
    } else {
      p.workspace = nullptr;
    }
    onnxruntime::contrib::cuda::run_memory_efficient_attention(p);

    // On the MEA (CUTLASS) path (used for both MHA and GQA when nonpad_kv_seqlen is provided),
    // zero out output for fully-masked batches to produce zeros (matching Flash behavior).
    // CUTLASS epilogue computes 1/s_prime where s_prime=0 for seqlens_k=0, producing NaN.
    {
      using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
      int64_t elements_per_batch = static_cast<int64_t>(parameters.q_sequence_length) *
                                   parameters.q_num_heads * parameters.v_head_size;
      ORT_RETURN_IF_ERROR(LaunchZeroOutputForFullyMaskedBatches<CudaT>(
          reinterpret_cast<CudaT*>(out_data),
          seqlens_k_buffer.get(),
          parameters.batch_size,
          elements_per_batch,
          cuda_stream,
          device_prop.maxThreadsPerBlock));
    }
  }
  // Standard MEA path: float attention bias, bool mask (converted to bias), or no mask.
  // Bool masks are converted to additive attention bias (true→0, false→mask_filter_value)
  // which correctly handles all-false masks (uniform softmax weights) unlike the
  // custom_right_padding seqlens approach which would produce NaN.
  else {
    if (attn_mask != nullptr) {
      ORT_RETURN_IF_ERROR(ConvertAttnMaskToBias(context, attn_mask, cuda_stream,
                                                device_prop.maxThreadsPerBlock,
                                                converted_mask_buffer, attn_bias_data,
                                                broadcast_bias_dim_0, broadcast_bias_dim_1));
    }

    onnxruntime::contrib::cuda::MemoryEfficientAttentionParams p;
    p.sm = sm;
    p.is_half = std::is_same<T, MLFloat16>::value;
    p.is_bf16 = std::is_same<T, BFloat16>::value;
    p.is_kv_bsnh = is_bsnh;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.q_num_heads;
    p.sequence_length = parameters.q_sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.max_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_causal;
    p.scale = parameters.scale;
    p.softcap = parameters.softcap;
    p.broadcast_attn_bias_dim_0 = broadcast_bias_dim_0;
    p.broadcast_attn_bias_dim_1 = broadcast_bias_dim_1;
    p.query = q_data;
    p.key = k_data;
    p.value = v_data;
    p.attn_bias = attn_bias_data;
    p.stream = cuda_stream;
    p.output = out_data;

    IAllocatorUniquePtr<void> workspace_buffer;
    if (onnxruntime::contrib::cuda::MemoryEfficientAttentionParams::need_workspace(
            parameters.v_head_size, sizeof(T) == sizeof(float))) {
      size_t workspace_bytes = sizeof(float) * parameters.batch_size * parameters.q_sequence_length *
                               parameters.q_num_heads * parameters.v_head_size;
      workspace_buffer = GetScratchBuffer<void>(workspace_bytes, GetComputeStream(context));
      p.workspace = workspace_buffer.get();
    } else {
      p.workspace = nullptr;
    }
    onnxruntime::contrib::cuda::run_memory_efficient_attention(p);
  }

  // --- Transpose output BSNH → BNSH if input was 4D (BNSH) ---
  if (!is_bsnh && out_bsnh_buffer != nullptr) {
    ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(
        parameters.batch_size, parameters.q_sequence_length,
        parameters.q_num_heads, parameters.v_head_size,
        out_bsnh_buffer.get(), Y->MutableData<T>(),
        cuda_stream, device_prop.maxThreadsPerBlock));
  }

  // Populate present_key/present_value (BNSH) if requested
  if (present_key != nullptr && is_bsnh) {
    ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(
        parameters.batch_size, parameters.kv_sequence_length,
        parameters.kv_num_heads, parameters.head_size,
        K->Data<T>(), present_key->MutableData<T>(),
        cuda_stream, device_prop.maxThreadsPerBlock));
  } else if (present_key != nullptr && !is_bsnh) {
    // 4D BNSH prompt: K is already BNSH, just D2D copy to present
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        present_key->MutableData<T>(), K->Data<T>(),
        K->SizeInBytes(), cudaMemcpyDeviceToDevice, cuda_stream));
  }
  if (present_value != nullptr && is_bsnh) {
    ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(
        parameters.batch_size, parameters.kv_sequence_length,
        parameters.kv_num_heads, parameters.v_head_size,
        V->Data<T>(), present_value->MutableData<T>(),
        cuda_stream, device_prop.maxThreadsPerBlock));
  } else if (present_value != nullptr && !is_bsnh) {
    // 4D BNSH prompt: V is already BNSH, just D2D copy to present
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        present_value->MutableData<T>(), V->Data<T>(),
        V->SizeInBytes(), cudaMemcpyDeviceToDevice, cuda_stream));
  }

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(Q);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(V);
  ORT_UNUSED_PARAMETER(attn_mask);
  ORT_UNUSED_PARAMETER(past_key);
  ORT_UNUSED_PARAMETER(past_value);
  ORT_UNUSED_PARAMETER(nonpad_kv_seqlen);
  ORT_UNUSED_PARAMETER(Y);
  ORT_UNUSED_PARAMETER(present_key);
  ORT_UNUSED_PARAMETER(present_value);
  ORT_UNUSED_PARAMETER(parameters);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Memory efficient attention is not available in this build.");
#endif
}

// ============================================================================
// RunUnfusedAttention: Delegates to MHA's QkvToContext (unfused GEMM+softmax+GEMM)
// ============================================================================
//
// Unfused Attention dispatch paths:
//   Universal fallback via MHA's QkvToContext.
//   Path 1: nonpad_kv_seqlen only -> converts to attention_bias [B, q_seq, total_seq]
//   Path 2: nonpad_kv_seqlen + attn_mask -> composes both into attention_bias [B, q_seq, total_seq]
//           (nonpad bias + mask bias added element-wise with cyclic broadcasting)
//   Path 3: all other cases -> passes mask/bias directly
//   Supports: all dtypes (fp16/bf16/fp32), all mask types (bool/float/none), all head sizes
//   Not supported: softcap (rejected at fallback), output_qk modes beyond kNone/kQK
//   Limitation: MHA only (q_num_heads must equal kv_num_heads)
//
template <typename T>
Status Attention<T>::RunUnfusedAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    Tensor* output_qk,
    const attention_helper::AttentionParameters& parameters) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  // OrtToCudaType maps BFloat16 → __nv_bfloat16 (native HW type), matching kernel instantiations.
  using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = Stream(context);
  auto ort_stream = GetOrtStream(context);

  // Bridge to contrib::AttentionParameters for the MHA unfused path
  onnxruntime::contrib::AttentionParameters contribop_parameters;

  if (!parameters.transpose_output) {
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BNSH;
    contribop_parameters.is_output_bnsh = true;
  } else {
    contribop_parameters.qkv_format = onnxruntime::contrib::AttentionQkvFormat::Q_K_V_BSNH;
    contribop_parameters.is_output_bnsh = false;
  }

  contribop_parameters.batch_size = parameters.batch_size;
  contribop_parameters.sequence_length = parameters.q_sequence_length;
  contribop_parameters.kv_sequence_length = parameters.kv_sequence_length;
  contribop_parameters.past_sequence_length = parameters.past_sequence_length;
  contribop_parameters.total_sequence_length = parameters.total_sequence_length;
  contribop_parameters.max_sequence_length = parameters.total_sequence_length;
  contribop_parameters.input_hidden_size = 0;
  contribop_parameters.hidden_size = parameters.q_num_heads * parameters.head_size;
  contribop_parameters.head_size = parameters.head_size;
  contribop_parameters.v_head_size = parameters.v_head_size;
  contribop_parameters.v_hidden_size = parameters.kv_num_heads * parameters.v_head_size;
  contribop_parameters.num_heads = parameters.q_num_heads;
  contribop_parameters.rotary_dim = 0;
  contribop_parameters.num_splits = 1;
  contribop_parameters.beam_width = 1;
  contribop_parameters.is_unidirectional = parameters.is_causal;
  contribop_parameters.past_present_share_buffer = false;
  contribop_parameters.is_packed_qkv = false;
  contribop_parameters.do_rotary = false;
  contribop_parameters.mask_type = onnxruntime::contrib::AttentionMaskType::MASK_NONE;
  contribop_parameters.mask_filter_value = static_cast<float>(std::numeric_limits<T>::lowest());
  contribop_parameters.scale = parameters.scale;
  contribop_parameters.use_tf32 = UseTF32();

  // Determine broadcast flags for attention_bias
  if (attn_mask != nullptr) {
    size_t attn_mask_dims_size = attn_mask->Shape().NumDimensions();
    auto attn_mask_dims = attn_mask->Shape().GetDims();
    if (attn_mask_dims_size == 2) {
      contribop_parameters.broadcast_attn_bias_dim_0 = true;
      contribop_parameters.broadcast_attn_bias_dim_1 = true;
    } else if (attn_mask_dims_size == 3) {
      contribop_parameters.broadcast_attn_bias_dim_0 = true;
      contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[0] == 1;
    } else {
      contribop_parameters.broadcast_attn_bias_dim_0 = attn_mask_dims[0] == 1;
      contribop_parameters.broadcast_attn_bias_dim_1 = attn_mask_dims[1] == 1;
    }
  } else {
    contribop_parameters.broadcast_attn_bias_dim_0 = false;
    contribop_parameters.broadcast_attn_bias_dim_1 = false;
  }

  // Construct AttentionData
  onnxruntime::contrib::cuda::AttentionData<CudaT> data;
  data.query = reinterpret_cast<const CudaT*>(Q->Data<T>());
  data.key = reinterpret_cast<const CudaT*>(K->Data<T>());
  data.value = reinterpret_cast<const CudaT*>(V->Data<T>());
  data.mask_index = nullptr;
  data.mask_index_dims = gsl::span<const int64_t>();
  data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.output = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  data.present_key = (present_key == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (present_value == nullptr) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  if (output_qk != nullptr) {
    data.output_qk = reinterpret_cast<CudaT*>(output_qk->MutableData<T>());
  }
  data.bias = nullptr;

  // Handle attention mask / nonpad_kv_seqlen → attention_bias
  IAllocatorUniquePtr<void> converted_mask_buffer;
  IAllocatorUniquePtr<void> mask_bias_buffer;  // temp buffer for mask→bias when composing
  if (nonpad_kv_seqlen != nullptr) {
    // Convert nonpad_kv_seqlen to additive attention bias: [B, q_seq, total_seq]
    int64_t bias_elements = static_cast<int64_t>(parameters.batch_size) *
                            parameters.q_sequence_length *
                            parameters.total_sequence_length;
    converted_mask_buffer = GetScratchBuffer<void>(bias_elements * sizeof(NativeCudaT), GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToAttentionBias<NativeCudaT>(
        nonpad_kv_seqlen->Data<int64_t>(),
        reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
        parameters.batch_size,
        parameters.q_sequence_length,
        parameters.total_sequence_length,
        contribop_parameters.mask_filter_value,
        cuda_stream,
        device_prop.maxThreadsPerBlock));

    // When attn_mask is also present, compose it into the nonpad bias additively.
    // The nonpad bias is [B, q, t]; the mask is added with cyclic broadcasting
    // (e.g. a 2D [q, t] mask repeats over the batch dimension).
    // Only 2D masks and 4D masks with head_dim=1 are supported — per-head masks
    // (3D [H,q,t] or 4D [B,H>1,q,t]) cannot be composed into a [B,q,t] buffer.
    if (attn_mask != nullptr) {
      const auto& mask_shape = attn_mask->Shape();
      int mask_dims = static_cast<int>(mask_shape.NumDimensions());
      ORT_ENFORCE(mask_dims == 2 || (mask_dims == 4 && mask_shape[1] == 1),
                  "nonpad_kv_seqlen + attn_mask composition in unfused path only supports "
                  "2D masks [q, t] and 4D masks with head_dim=1 [B, 1, q, t]. "
                  "Got mask shape: ",
                  mask_shape);

      int64_t mask_elements = mask_shape.Size();
      const NativeCudaT* mask_bias_ptr = nullptr;

      if (attn_mask->IsDataType<bool>()) {
        // Convert bool mask to additive bias in a temp buffer, then add in-place.
        mask_bias_buffer = GetScratchBuffer<void>(mask_elements * sizeof(NativeCudaT), GetComputeStream(context));
        ORT_RETURN_IF_ERROR(LaunchConvertBoolMaskToAttentionBias<NativeCudaT>(
            attn_mask->Data<bool>(),
            reinterpret_cast<NativeCudaT*>(mask_bias_buffer.get()),
            mask_elements,
            contribop_parameters.mask_filter_value,
            cuda_stream,
            device_prop.maxThreadsPerBlock));
        mask_bias_ptr = reinterpret_cast<const NativeCudaT*>(mask_bias_buffer.get());
      } else {
        // Float mask is already in additive bias format.
        mask_bias_ptr = reinterpret_cast<const NativeCudaT*>(attn_mask->Data<T>());
      }

      // Add mask bias into nonpad bias with cyclic broadcasting.
      // 2D mask [q, t]: mask_elements = q*t, repeats for each batch → correct.
      // 4D mask [B, 1, q, t]: mask_elements = B*q*t = bias_elements → direct add.
      ORT_RETURN_IF_ERROR(LaunchAddBiasInPlace<NativeCudaT>(
          reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
          mask_bias_ptr,
          bias_elements,
          mask_elements,
          cuda_stream,
          device_prop.maxThreadsPerBlock));
    }

    data.attention_bias = reinterpret_cast<const CudaT*>(converted_mask_buffer.get());
    // Composed bias is [B, q_seq, total_seq] → broadcasts over heads but not batch.
    contribop_parameters.broadcast_attn_bias_dim_0 = false;
    contribop_parameters.broadcast_attn_bias_dim_1 = true;
  } else if (attn_mask != nullptr) {
    if (attn_mask->IsDataType<bool>()) {
      int64_t num_elements = attn_mask->Shape().Size();
      converted_mask_buffer = GetScratchBuffer<void>(num_elements * sizeof(NativeCudaT), GetComputeStream(context));
      ORT_RETURN_IF_ERROR(LaunchConvertBoolMaskToAttentionBias<NativeCudaT>(
          attn_mask->Data<bool>(),
          reinterpret_cast<NativeCudaT*>(converted_mask_buffer.get()),
          num_elements,
          contribop_parameters.mask_filter_value,
          cuda_stream,
          device_prop.maxThreadsPerBlock));
      data.attention_bias = reinterpret_cast<const CudaT*>(converted_mask_buffer.get());
    } else {
      data.attention_bias = reinterpret_cast<const CudaT*>(attn_mask->Data<T>());
    }
  }

  data.qkv_format = contribop_parameters.qkv_format;
  data.use_flash_attention = false;
  data.use_memory_efficient_attention = false;
  data.fused_runner = nullptr;
  data.fused_cross_attention_kernel = nullptr;
  data.kernel_type = onnxruntime::contrib::AttentionKernelType::AttentionKernel_Unfused;

  // Allocate workspace
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
      nullptr, false, false, false, false, false,
      no_qkv_workspace);
  auto work_space = GetScratchBuffer<void>(workspace_bytes, GetComputeStream(context));

  data.has_qkv_workspace = !no_qkv_workspace;
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.workspace_bytes = workspace_bytes;

  cublasHandle_t cublas = GetCublasHandle(context);
  cudnnHandle_t cudnn = GetCudnnHandle(context);

  // Note: unfused attention produces valid finite output (mean-of-V via uniform softmax)
  // for fully-masked batches, so ZeroOutput is not needed here. Only MEA requires
  // ZeroOutput to prevent NaN from the CUTLASS epilogue's 1/s_prime division.
  return onnxruntime::contrib::cuda::QkvToContext<CudaT, CudaT>(
      device_prop, cublas, cudnn, ort_stream.get(), contribop_parameters, data);
}

// ============================================================================
// ComputeInternal: Dispatch to appropriate attention kernel
// ============================================================================
// MHA path (q_num_heads == kv_num_heads): uses direct kernel dispatch cascade
//   flash → memory efficient → unfused
// GQA path (q_num_heads != kv_num_heads): uses flash (handles GQA natively) or MEA
//   (with head expansion via LaunchUngroup). Unfused fallback not yet supported for GQA.
// ============================================================================
template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* Q = context->Input<Tensor>(0);
  const Tensor* K = context->Input<Tensor>(1);
  const Tensor* V = context->Input<Tensor>(2);
  const Tensor* attn_mask = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(4);
  const Tensor* past_value = context->Input<Tensor>(5);
  const Tensor* nonpad_kv_seqlen = context->Input<Tensor>(6);  // optional, Opset 24

  // When both nonpad_kv_seqlen and attn_mask are provided, Flash Attention cannot handle
  // the combination (no bias parameter). Route to MEA or Unfused which support composition.
  if (nonpad_kv_seqlen != nullptr && attn_mask != nullptr) {
    LOGS_DEFAULT(VERBOSE) << "Both nonpad_kv_seqlen and attn_mask provided. "
                          << "Flash Attention does not support this combination; "
                          << "falling back to Memory Efficient Attention or Unfused path.";
  }

  attention_helper::AttentionParameters parameters;
  TensorShape y_shape;
  TensorShape present_key_shape;
  TensorShape present_value_shape;
  TensorShape output_qk_shape;

  ORT_ENFORCE(attention_helper::ComputeOutputShapeForAttention(
                  Q, K, V, attn_mask, past_key, past_value, nonpad_kv_seqlen,
                  is_causal_, softcap_, softmax_precision_,
                  qk_matmul_output_mode_, kv_num_heads_, q_num_heads_, scale_,
                  parameters, y_shape, present_key_shape, present_value_shape, output_qk_shape,
                  true /* skip_nonpad_data_validation: data is on GPU */)
                  .IsOK(),
              "Output shapes for Attention could not be computed.");

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = context->Output(3, output_qk_shape);

  const bool is_gqa = parameters.kv_num_heads != parameters.q_num_heads;

  // === KERNEL SELECTION CASCADE ===
  // Priority: flash attention > memory efficient attention > unfused attention
  //
  // 4D BNSH handling per kernel:
  //   Flash: strictly requires BSNH — Q is transposed BNSH→BSNH before calling mha_fwd*.
  //          K/V passed as BNSH to mha_fwd_kvcache (it handles both layouts).
  //   MEA:   accepts both BSNH and BNSH natively via is_kv_bsnh flag. Q transposed to BSNH.
  //   Unfused: accepts both via QkvToContext's qkv_format (Q_K_V_BSNH or Q_K_V_BNSH).
  //
  // nonpad_kv_seqlen + attn_mask routing:
  //   Flash: cannot handle this combo (no bias param when seqlens_k is used) → excluded.
  //   MEA:   supports both (custom_right_padding for seqlens + additive attn_bias for mask).
  //   Unfused: nonpad → attention_bias; mask composed additively when both present.
#if USE_FLASH_ATTENTION || USE_MEMORY_EFFICIENT_ATTENTION
  const bool has_output_qk = (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone);
#endif

  // softmax_precision: All CUDA backends (Flash, MEA, Unfused) compute softmax in
  // FP32 internally (Flash/MEA via tile-based FP32 accumulators, Unfused via FP32
  // softmax kernel). softmax_precision=1 (FP32) is inherently satisfied;
  // softmax_precision=0 (default) is also fine since higher precision is always
  // acceptable per the ONNX spec.

#if USE_FLASH_ATTENTION
  {
    auto& device_prop = GetDeviceProp();
    bool flash_eligible =
        !disable_flash_attention_ &&
        !std::is_same<T, float>::value &&
        onnxruntime::flash::is_supported<T>(device_prop, parameters.head_size,
                                            parameters.q_num_heads, parameters.kv_num_heads) &&
        parameters.head_size == parameters.v_head_size &&
        !has_output_qk &&
        // Flash does not support attention masks (no bias parameter in mha_fwd/mha_fwd_kvcache).
        // Bool attn_mask + past_key is rejected because Flash uses paged KV cache semantics
        // that produce spec-divergent present_kv layout for partial masks (e.g. [T,T,T,F]).
        // Unfused handles bool+past_key spec-correctly via standard ConcatPastToPresent.
        // TODO(titaiwang): GQA + bool attn_mask + past_key currently has no runner (Flash
        // rejected here, unfused doesn't support GQA, MEA blocked by past_key != nullptr).
        // Once PR #27851 merges (MEA supports past_key), this gap will be covered.
        attn_mask == nullptr;

    if (flash_eligible) {
      return RunFlashAttention(context, Q, K, V, past_key, past_value,
                               nonpad_kv_seqlen, Y, present_key, present_value, parameters);
    }
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  {
    auto& device_prop = GetDeviceProp();
    int sm = device_prop.major * 10 + device_prop.minor;
    bool mea_eligible =
        !disable_memory_efficient_attention_ &&
        onnxruntime::contrib::cuda::has_memory_efficient_attention(
            sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value,
            parameters.head_size, parameters.v_head_size) &&
        !has_output_qk &&
        past_key == nullptr;

    // Cutlass FMHA requires bias strides to satisfy minimum alignment even in the
    // "unaligned" kernel path. When an attention mask is present (with or without
    // nonpad_kv_seqlen), it becomes an additive bias with bias_strideM =
    // total_sequence_length. Skip MEA if this stride can't satisfy the kernel's
    // minimum alignment requirement.
    if (mea_eligible && attn_mask != nullptr) {
      // NOTE: CUTLASS uses kMinimumAlignment = 4 (elements, not bytes) for the bias
      // pointer in its epilogue. total_sequence_length is the bias row stride in elements,
      // so we check alignment in element count. The contrib_ops convention (4 * sizeof(T))
      // conflates bytes with elements; we use the correct value of 4 elements here.
      // Note: on SM50/53 (Maxwell), CUTLASS kMinimumAlignment=1, so this is stricter than
      // necessary — cases with odd total_sequence_length that previously used MEA on those
      // GPUs will now fall to unfused. This is acceptable for these very old architectures.
      constexpr int min_bias_align = 4;
      if (parameters.total_sequence_length % min_bias_align != 0) {
        mea_eligible = false;
      }
    }

    if (mea_eligible) {
      return RunMemoryEfficientAttention(context, Q, K, V, attn_mask, past_key, past_value,
                                         nonpad_kv_seqlen, Y, present_key, present_value, parameters);
    }
  }
#endif

  // Fallback: unfused attention
  // Softcap is not implemented in the unfused path — it requires Flash or MEA.
  if (parameters.softcap > 0.0f) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "softcap requires flash attention or memory efficient attention, "
                           "but neither is eligible for this configuration. Check dtype (fp16/bf16 required for Flash), "
                           "head_size constraints, and past_key compatibility.");
  }

  // TODO(titaiwang): Support additional output_qk modes beyond kNone and kQK.
  // Currently only unfused handles output_qk, and only kNone/kQK modes.
  if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone &&
      qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kQK) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "qk_matmul_output_mode other than kNone and kQK is not supported yet "
                           "in Attention op (CUDA).");
  }

  if (is_gqa) {
    // TODO(titaiwang): Support GQA in unfused attention path for fp32/old-GPU fallback.
    // Currently blocked because QkvToContext allocates K/V workspace assuming
    // num_heads == kv_num_heads. GQA needs a head expansion step (ExpandKVHeads kernel)
    // to replicate kv_num_heads -> q_num_heads before unfused can process.
    // Requires ~160 lines. See issue #27516.
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "ONNX Attention with GQA (q_num_heads != kv_num_heads) is not supported by the "
                           "unfused runner. Flash requires fp16/bf16, SM>=80, and attn_mask==nullptr; MEA "
                           "requires past_key==nullptr. See PR #27851 for MEA past_key support.");
  }

  return RunUnfusedAttention(context, Q, K, V, attn_mask, past_key, past_value,
                             nonpad_kv_seqlen, Y, present_key, present_value, output_qk, parameters);
}

}  // namespace cuda
}  // namespace onnxruntime

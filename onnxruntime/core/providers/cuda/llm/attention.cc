// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/llm/attention.h"
#include "core/providers/cpu/llm/attention_helper.h"
#include "core/providers/cuda/llm/attention.h"
#include "core/providers/cuda/llm/attention_mask_impl.h"
// attention_impl.h provides Transpose_BNSH_to_BSNH / Transpose_BSNH_to_BNSH used
// by the transpose helpers.
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/unfused_attention.h"
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
    // CUTLASS online softmax multiplies attention scores by kLog2e (≈1.4427).
    // For float/bf16, |lowest() × kLog2e| > FLT_MAX, overflowing to -inf and
    // causing s_prime=0 → NaN for fully-masked batches. Cap to prevent this.
    // See kCutlassSafeMaskFilterValue in memory_efficient_attention.h for details.
    float mask_filter_value = std::max(static_cast<float>(std::numeric_limits<T>::lowest()),
                                       ::onnxruntime::contrib::cuda::kCutlassSafeMaskFilterValue);
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
//           - 4D BNSH: transposes Q to BSNH; new K/V to BSNH for concat (cache stays BNSH)
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

    // TODO(titaiwang): Consolidate preprocessing (transpose, KV cache concat) into a
    // single fused kernel like GQA's LaunchUnpackRoPEAppend. Current decode path uses 4-6 kernel
    // launches; a fused approach would reduce to ~2, saving launch overhead and intermediate
    // buffer traffic per decode step.

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

  // --- Populate present_key/value (BNSH) from K/V (BSNH or BNSH) ---
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
//   Path 1: Decode with past KV cache -> LaunchConcatNewToPastKV then standard MEA
//   Path 2: nonpad_kv_seqlen (opset 24 external cache) -> has_custom_right_padding mode
//   Path 3: Prompt with mask -> standard MEA with additive bias
//   Path 4: Prompt without mask -> standard MEA
//   Eligibility: see has_memory_efficient_attention() (SM50+/53+/80+ by dtype,
//                head_size <= 1024, head_size divisible by 8), plus: no output_qk, bias stride alignment.
//   Note: softcap is forwarded to the MEA kernel via p.softcap. CUTLASS applies
//   softcap before bias (fused in kernel tiles), matching ONNX spec ordering
//   (onnx/onnx#7865): QK → softcap → mask/bias → softmax. softmax_precision
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

  bool present_kv_already_populated = false;
  // Track the effective layout of k_data/v_data. Initially matches input layout,
  // but changes to BNSH (false) after decode concat into present buffers.
  bool kv_is_bsnh = is_bsnh;

  // Scratch buffers for decode concat output when present_key/value are optional.
  // Declared at function scope so they outlive the decode block (k_data/v_data may point here).
  IAllocatorUniquePtr<void> present_k_scratch;
  IAllocatorUniquePtr<void> present_v_scratch;

  // --- Decode path: concat past + new K/V → present buffers (BNSH) ---
  // nonpad_kv_seqlen and past_key are mutually exclusive (enforced at validation),
  // so the decode path only needs the internal-cache (past_key/present_key) flow.
  if (past_key != nullptr) {
    ORT_RETURN_IF_NOT(past_value != nullptr, "past_key requires past_value.");
    ORT_RETURN_IF_NOT(nonpad_kv_seqlen == nullptr,
                      "nonpad_kv_seqlen and past_key are mutually exclusive (internal vs external cache).");
    ORT_RETURN_IF_NOT(parameters.head_size == parameters.v_head_size,
                      "MEA decode (past_key) requires head_size == v_head_size for LaunchConcatNewToPastKV.");

    using NativeCudaT = typename OrtToCudaType<T>::type;

    // Allocate scratch buffers for concat output when present_key/value are not requested.
    // The concat kernel needs a destination buffer regardless of whether the caller wants present outputs.
    T* present_k_data = nullptr;
    T* present_v_data = nullptr;

    size_t present_k_bytes = sizeof(T) * parameters.batch_size * parameters.kv_num_heads *
                             parameters.total_sequence_length * parameters.head_size;
    size_t present_v_bytes = sizeof(T) * parameters.batch_size * parameters.kv_num_heads *
                             parameters.total_sequence_length * parameters.v_head_size;

    if (present_key != nullptr) {
      present_k_data = present_key->MutableData<T>();
    } else {
      present_k_scratch = GetScratchBuffer<void>(present_k_bytes, GetComputeStream(context));
      present_k_data = static_cast<T*>(present_k_scratch.get());
    }
    if (present_value != nullptr) {
      present_v_data = present_value->MutableData<T>();
    } else {
      present_v_scratch = GetScratchBuffer<void>(present_v_bytes, GetComputeStream(context));
      present_v_data = static_cast<T*>(present_v_scratch.get());
    }

    // Step 1: Uniform past sequence lengths for the concat kernel.
    // ONNX past_key has shape [B, H, past_seq, head_size] — all batches share
    // the same past_seq dimension. Bool masks do NOT change where tokens are stored;
    // they change which tokens are attended to (via additive bias, handled below).
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
    // No memset needed: uniform past_seq_lens means every position in the present
    // buffer is written by the concat kernel. Padding positions in past_key are copied
    // as-is; the attention mask (additive bias) handles correctness at the attention level.
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
        reinterpret_cast<NativeCudaT*>(present_k_data),
        reinterpret_cast<NativeCudaT*>(present_v_data),
        cuda_stream,
        device_prop.maxThreadsPerBlock,
        /*past_only=*/false));

    // Point MEA's K/V inputs at the concatenated buffers (BNSH).
    k_data = present_k_data;
    v_data = present_v_data;
    kv_is_bsnh = false;
    present_kv_already_populated = true;
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
          kv_is_bsnh,
          cuda_stream,
          device_prop.maxThreadsPerBlock));

      k_data = k_expand_buffer.get();
      v_data = v_expand_buffer.get();
    }
  }

  // Note: When past_key is present (decode), k_data/v_data already point to present
  // buffers (BNSH) after LaunchConcatNewToPastKV above, so MEA sees the full cache.

  // Handle attention mask → attention_bias conversion
  IAllocatorUniquePtr<void> converted_mask_buffer;
  const void* attn_bias_data = nullptr;
  bool broadcast_bias_dim_0 = false;
  bool broadcast_bias_dim_1 = false;

  if (nonpad_kv_seqlen != nullptr) {
    // Convert nonpad_kv_seqlen to seqlens_k for custom right padding.
    // MEA expects seqlens_k as actual token count, so use FlashSeqlensK variant
    // (which converts int64→int32 without subtracting 1).
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
    p.is_kv_bsnh = kv_is_bsnh;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.q_num_heads;
    p.sequence_length = parameters.q_sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.max_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_causal;
    // ONNX spec: is_causal means upper-left alignment in the full attention matrix.
    // When past_sequence_length == 0 and S_q != S_kv (cross-attention without KV cache),
    // queries start at absolute position 0, so causal mask is upper-left.
    // When past_sequence_length > 0 (decode with KV cache), queries start at position
    // past_seq, so causal mask is effectively lower-right on the [S_q x total_kv] sub-matrix.
    // NOTE: For external KV cache (TensorScatter), nonpad_kv_seqlen provides per-batch
    // actual lengths and seqlens_k handles the masking — the causal_from_top_left flag
    // is only consulted when params.causal is true, so it's correct here.
    p.causal_from_top_left = (parameters.past_sequence_length == 0);
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
    // zero out output for fully-masked batches to prevent NaN.
    // CUTLASS epilogue computes 1/s_prime where s_prime=0 for seqlens_k=0, producing NaN.
    // TODO(titaiwang): ZeroOutputForFullyMaskedBatches outputs zeros for fully-masked
    // batches (seqlens_k=0), which diverges from CPU/Unfused behavior (uniform mean of V).
    // For cross-EP consistency, replace with LaunchMeanOfVForFullyMaskedBatches that
    // computes mean(V[b,n,:,h]) for each masked batch. See issue #27516.
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
  // Bool masks are converted to additive attention bias (true→0, false→mask_filter_value).
  // For fully-masked batches (all-false bool mask), ConvertAttnMaskToBias uses a capped
  // mask_filter_value (-1e+30) that stays finite through CUTLASS's kLog2e multiplication,
  // producing correct uniform softmax → mean(V) output.
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
    p.is_kv_bsnh = kv_is_bsnh;
    p.batch_size = parameters.batch_size;
    p.num_heads = parameters.q_num_heads;
    p.sequence_length = parameters.q_sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.max_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_causal;
    // ONNX spec: is_causal means upper-left alignment in the full attention matrix.
    // When past_sequence_length == 0 and S_q != S_kv (cross-attention without KV cache),
    // queries start at absolute position 0, so causal mask is upper-left.
    // When past_sequence_length > 0 (decode with KV cache), queries start at position
    // past_seq, so causal mask is effectively lower-right on the [S_q x total_kv] sub-matrix.
    p.causal_from_top_left = (parameters.past_sequence_length == 0);
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

  // Populate present_key/present_value (BNSH) if requested.
  // Skip for decode path where LaunchConcatNewToPastKV already populated present buffers.
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
// RunUnfusedAttention: Unified unfused path for both MHA and GQA
// ============================================================================
//
// Routes to LaunchUnfusedAttention from contrib_ops/cuda/bert/unfused_attention.h.
//
// Handles:
//   - MHA as a degenerate case (group_size=1, no head expansion needed).
//   - GQA natively (no K/V head replication; reshape-Q trick inside kernel).
//   - fp16/bf16 with large head_size via FP32 QK scratch (fixes issue #28195:
//     unfused attention producing NaN when head_dim > 256 at scale=1.0).
//   - Different Q/K sequence lengths, past_key+past_value, nonpad_kv_seqlen.
//   - attn_mask (bool/float, 2D/3D/4D), causal, softcap.
//
// Not supported (returns NOT_IMPLEMENTED upstream):
//   - qk_matmul_output_mode beyond kNone/kQK (kQKMask, kQKSoftCap, kQKSoftMax).
// ============================================================================
template <typename T>
Status Attention<T>::RunUnfusedAttention(
    OpKernelContext* context,
    const Tensor* Q, const Tensor* K, const Tensor* V,
    const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
    const Tensor* nonpad_kv_seqlen,
    Tensor* Y, Tensor* present_key, Tensor* present_value,
    Tensor* output_qk,
    const attention_helper::AttentionParameters& parameters) const {
  using NativeCudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  auto& device_prop = GetDeviceProp();
  auto cuda_stream = Stream(context);
  const bool is_bsnh = parameters.transpose_output;
  const int B = parameters.batch_size;
  const int S_q = parameters.q_sequence_length;
  const int N_q = parameters.q_num_heads;
  const int N_kv = parameters.kv_num_heads;
  const int H = parameters.head_size;
  const int H_v = parameters.v_head_size;
  const int total_kv = parameters.total_sequence_length;
  const int max_threads = device_prop.maxThreadsPerBlock;

  // -------- Build BNSH Q (transpose if input was BSNH) ------------------------
  const NativeCudaT* q_bnsh = nullptr;
  IAllocatorUniquePtr<void> q_bnsh_buffer;
  if (is_bsnh) {
    const size_t q_bytes = SafeInt<size_t>(B) * S_q * N_q * H * sizeof(T);
    q_bnsh_buffer = GetScratchBuffer<void>(q_bytes, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(B, S_q, N_q, H,
                                               Q->Data<T>(), q_bnsh_buffer.get(),
                                               cuda_stream, max_threads));
    q_bnsh = reinterpret_cast<const NativeCudaT*>(q_bnsh_buffer.get());
  } else {
    q_bnsh = reinterpret_cast<const NativeCudaT*>(Q->Data<T>());
  }

  // -------- Build BNSH K/V cache of length total_kv --------------------------
  // Three cases:
  //   (a) nonpad_kv_seqlen: K/V are the full cache (kv_seq == total_kv).
  //   (b) past_key + new K/V: concat via LaunchConcatNewToPastKV into present buffers.
  //   (c) no past: K/V are the new tokens only (total_kv == kv_sequence_length).
  // In cases (a) and (c) the cache is contiguous in the input tensors (subject
  // to a BSNH->BNSH transpose). Case (b) writes into present_key/present_value.
  const NativeCudaT* k_cache = nullptr;
  const NativeCudaT* v_cache = nullptr;
  IAllocatorUniquePtr<void> k_bnsh_buffer;
  IAllocatorUniquePtr<void> v_bnsh_buffer;
  bool present_already_populated = false;

  if (past_key != nullptr) {
    ORT_ENFORCE(past_value != nullptr, "past_key requires past_value.");
    ORT_ENFORCE(present_key != nullptr && present_value != nullptr,
                "present_key/value outputs are required when past_key is provided.");
    auto past_seqlens_buffer = GetScratchBuffer<int>(B, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchFillInt32(past_seqlens_buffer.get(),
                                        parameters.past_sequence_length, B,
                                        cuda_stream, max_threads));

    // New K/V must be BSNH for the concat kernel; transpose if 4D BNSH input.
    const T* k_new_bsnh = K->Data<T>();
    const T* v_new_bsnh = V->Data<T>();
    if (!is_bsnh) {
      const size_t kn_bytes = SafeInt<size_t>(B) * parameters.kv_sequence_length * N_kv * H * sizeof(T);
      const size_t vn_bytes = SafeInt<size_t>(B) * parameters.kv_sequence_length * N_kv * H_v * sizeof(T);
      k_bnsh_buffer = GetScratchBuffer<void>(kn_bytes, GetComputeStream(context));
      v_bnsh_buffer = GetScratchBuffer<void>(vn_bytes, GetComputeStream(context));
      ORT_RETURN_IF_ERROR(TransposeBNSHtoBSNH<T>(B, parameters.kv_sequence_length, N_kv, H,
                                                 K->Data<T>(), k_bnsh_buffer.get(),
                                                 cuda_stream, max_threads));
      ORT_RETURN_IF_ERROR(TransposeBNSHtoBSNH<T>(B, parameters.kv_sequence_length, N_kv, H_v,
                                                 V->Data<T>(), v_bnsh_buffer.get(),
                                                 cuda_stream, max_threads));
      k_new_bsnh = static_cast<const T*>(k_bnsh_buffer.get());
      v_new_bsnh = static_cast<const T*>(v_bnsh_buffer.get());
    }

    if (H == H_v) {
      // K and V have the same head_size -- single concat call handles both.
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchConcatNewToPastKV<NativeCudaT>(
          B, N_kv, H, parameters.kv_sequence_length, parameters.past_sequence_length, total_kv,
          /*is_bsnh=*/false,
          past_seqlens_buffer.get(), /*total_seq_lens=*/nullptr,
          reinterpret_cast<const NativeCudaT*>(past_key->Data<T>()),
          reinterpret_cast<const NativeCudaT*>(past_value->Data<T>()),
          reinterpret_cast<const NativeCudaT*>(k_new_bsnh),
          reinterpret_cast<const NativeCudaT*>(v_new_bsnh),
          reinterpret_cast<NativeCudaT*>(present_key->MutableData<T>()),
          reinterpret_cast<NativeCudaT*>(present_value->MutableData<T>()),
          cuda_stream, max_threads, /*past_only=*/false));
    } else {
      // H != H_v: LaunchConcatNewToPastKV uses a single head_size for both K and V
      // (grid Z=0 for K, Z=1 for V with the same block dims). We must call it
      // twice with different head_size values -- once for K (head_size=H) and once
      // for V (head_size=H_v). Each call duplicates K data into V params (or vice
      // versa) so both Z indices write to the same buffer harmlessly.
      //
      // Trade-off: each call does 2× GPU work (both Z slices execute). This is
      // acceptable because H!=H_v decode through MEA is rare, and modifying the
      // shared kernel (contrib_ops/cuda/bert/attention_kv_cache.cu) to support
      // nullptr outputs or K-only/V-only modes would risk breaking GQA callers.
      auto* pk = reinterpret_cast<const NativeCudaT*>(past_key->Data<T>());
      auto* pv = reinterpret_cast<const NativeCudaT*>(past_value->Data<T>());
      auto* nk = reinterpret_cast<const NativeCudaT*>(k_new_bsnh);
      auto* nv = reinterpret_cast<const NativeCudaT*>(v_new_bsnh);
      auto* out_k = reinterpret_cast<NativeCudaT*>(present_key->MutableData<T>());
      auto* out_v = reinterpret_cast<NativeCudaT*>(present_value->MutableData<T>());
      // Concat K with head_size=H (V params duplicate K data -- harmless)
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchConcatNewToPastKV<NativeCudaT>(
          B, N_kv, H, parameters.kv_sequence_length, parameters.past_sequence_length, total_kv,
          /*is_bsnh=*/false,
          past_seqlens_buffer.get(), /*total_seq_lens=*/nullptr,
          pk, pk, nk, nk, out_k, out_k,
          cuda_stream, max_threads, /*past_only=*/false));
      // Concat V with head_size=H_v (K params duplicate V data -- harmless)
      ORT_RETURN_IF_ERROR(onnxruntime::contrib::cuda::LaunchConcatNewToPastKV<NativeCudaT>(
          B, N_kv, H_v, parameters.kv_sequence_length, parameters.past_sequence_length, total_kv,
          /*is_bsnh=*/false,
          past_seqlens_buffer.get(), /*total_seq_lens=*/nullptr,
          pv, pv, nv, nv, out_v, out_v,
          cuda_stream, max_threads, /*past_only=*/false));
    }
    k_cache = reinterpret_cast<const NativeCudaT*>(present_key->MutableData<T>());
    v_cache = reinterpret_cast<const NativeCudaT*>(present_value->MutableData<T>());
    present_already_populated = true;
  } else if (is_bsnh) {
    // BSNH K/V -> BNSH. total_kv == kv_sequence_length (no past).
    // When present_key/present_value outputs exist, transpose directly into them
    // to avoid a redundant copy later.
    if (present_key != nullptr && present_value != nullptr) {
      ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(B, total_kv, N_kv, H,
                                                 K->Data<T>(), present_key->MutableData<T>(),
                                                 cuda_stream, max_threads));
      ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(B, total_kv, N_kv, H_v,
                                                 V->Data<T>(), present_value->MutableData<T>(),
                                                 cuda_stream, max_threads));
      k_cache = reinterpret_cast<const NativeCudaT*>(present_key->Data<T>());
      v_cache = reinterpret_cast<const NativeCudaT*>(present_value->Data<T>());
      present_already_populated = true;
    } else {
      const size_t k_bytes = SafeInt<size_t>(B) * total_kv * N_kv * H * sizeof(T);
      const size_t v_bytes = SafeInt<size_t>(B) * total_kv * N_kv * H_v * sizeof(T);
      k_bnsh_buffer = GetScratchBuffer<void>(k_bytes, GetComputeStream(context));
      v_bnsh_buffer = GetScratchBuffer<void>(v_bytes, GetComputeStream(context));
      ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(B, total_kv, N_kv, H,
                                                 K->Data<T>(), k_bnsh_buffer.get(),
                                                 cuda_stream, max_threads));
      ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(B, total_kv, N_kv, H_v,
                                                 V->Data<T>(), v_bnsh_buffer.get(),
                                                 cuda_stream, max_threads));
      k_cache = reinterpret_cast<const NativeCudaT*>(k_bnsh_buffer.get());
      v_cache = reinterpret_cast<const NativeCudaT*>(v_bnsh_buffer.get());
    }
  } else {
    // 4D BNSH input, no past: use directly.
    k_cache = reinterpret_cast<const NativeCudaT*>(K->Data<T>());
    v_cache = reinterpret_cast<const NativeCudaT*>(V->Data<T>());
  }

  // -------- Build per-batch seqlens (for nonpad_kv_seqlen) --------------------
  const int* seqlens_k_ptr = nullptr;
  IAllocatorUniquePtr<int> seqlens_k_buffer;
  if (nonpad_kv_seqlen != nullptr) {
    seqlens_k_buffer = GetScratchBuffer<int>(B, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
        nonpad_kv_seqlen->Data<int64_t>(), seqlens_k_buffer.get(),
        B, total_kv, cuda_stream, max_threads));
    seqlens_k_ptr = seqlens_k_buffer.get();
  }

  // -------- Build attn_bias from attn_mask ------------------------------------
  IAllocatorUniquePtr<void> mask_bias_buffer;
  const NativeCudaT* attn_bias_data = nullptr;
  bool bcast0 = false, bcast1 = false;
  if (attn_mask != nullptr) {
    const void* bias_void = nullptr;
    ORT_RETURN_IF_ERROR(ConvertAttnMaskToBias(context, attn_mask, cuda_stream, max_threads,
                                              mask_bias_buffer, bias_void, bcast0, bcast1));
    attn_bias_data = reinterpret_cast<const NativeCudaT*>(bias_void);
  }

  // -------- Allocate output BNSH scratch (if 3D BSNH output needed) ----------
  NativeCudaT* out_bnsh = reinterpret_cast<NativeCudaT*>(Y->MutableData<T>());
  IAllocatorUniquePtr<void> out_bnsh_buffer;
  if (is_bsnh) {
    const size_t out_bytes = SafeInt<size_t>(B) * S_q * N_q * H_v * sizeof(T);
    out_bnsh_buffer = GetScratchBuffer<void>(out_bytes, GetComputeStream(context));
    out_bnsh = reinterpret_cast<NativeCudaT*>(out_bnsh_buffer.get());
  }

  // -------- Allocate kernel workspace -----------------------------------------
  const size_t ws_bytes = onnxruntime::contrib::cuda::GetUnfusedAttentionWorkspaceSize(
      B, N_q, S_q, total_kv);
  auto ws_buffer = GetScratchBuffer<void>(ws_bytes, GetComputeStream(context));

  // -------- Call the kernel ---------------------------------------------------
  onnxruntime::contrib::cuda::UnfusedAttentionParams p;
  p.batch_size = B;
  p.num_heads = N_q;
  p.kv_num_heads = N_kv;
  p.head_size = H;
  p.v_head_size = H_v;
  p.q_sequence_length = S_q;
  p.total_kv_length = total_kv;
  p.max_kv_length = total_kv;  // ONNX Attention caches are packed (no shared buffer).
  p.broadcast_attn_bias_dim_0 = bcast0;
  p.broadcast_attn_bias_dim_1 = bcast1;
  p.is_causal = parameters.is_causal;
  p.local_window_size = -1;  // ONNX Attention (opset 23/24) does not expose sliding window.
  p.past_kv_length = parameters.past_sequence_length;
  p.scale = parameters.scale;
  p.softcap = parameters.softcap;
  p.seqlens_k = seqlens_k_ptr;

  NativeCudaT* output_qk_data = (output_qk != nullptr)
                                    ? reinterpret_cast<NativeCudaT*>(output_qk->MutableData<T>())
                                    : nullptr;

  ORT_RETURN_IF_ERROR((onnxruntime::contrib::cuda::LaunchUnfusedAttention<NativeCudaT>(
      device_prop, GetCublasHandle(context), cuda_stream,
      p, q_bnsh, k_cache, v_cache, attn_bias_data, out_bnsh, ws_buffer.get(),
      output_qk_data)));

  // -------- Transpose output BNSH -> BSNH if input was 3D --------------------
  if (is_bsnh && out_bnsh_buffer != nullptr) {
    ORT_RETURN_IF_ERROR(TransposeBNSHtoBSNH<T>(B, S_q, N_q, H_v,
                                               out_bnsh_buffer.get(), Y->MutableData<T>(),
                                               cuda_stream, max_threads));
  }

  // -------- Populate present_key/present_value if requested ------------------
  if (!present_already_populated) {
    if (present_key != nullptr) {
      if (is_bsnh) {
        ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(B, parameters.kv_sequence_length, N_kv, H,
                                                   K->Data<T>(), present_key->MutableData<T>(),
                                                   cuda_stream, max_threads));
      } else {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
            present_key->MutableData<T>(), K->Data<T>(),
            K->SizeInBytes(), cudaMemcpyDeviceToDevice, cuda_stream));
      }
    }
    if (present_value != nullptr) {
      if (is_bsnh) {
        ORT_RETURN_IF_ERROR(TransposeBSNHtoBNSH<T>(B, parameters.kv_sequence_length, N_kv, H_v,
                                                   V->Data<T>(), present_value->MutableData<T>(),
                                                   cuda_stream, max_threads));
      } else {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
            present_value->MutableData<T>(), V->Data<T>(),
            V->SizeInBytes(), cudaMemcpyDeviceToDevice, cuda_stream));
      }
    }
  }

  return Status::OK();
}

// ============================================================================
// ComputeInternal: Dispatch to appropriate attention kernel
// ============================================================================
// Dispatch cascade: Flash → MEA (Memory Efficient) → Unified Unfused Attention.
// The unified unfused kernel handles both MHA (num_heads == kv_num_heads) and
// GQA (num_heads != kv_num_heads) via a reshape-Q trick (no K/V head replication).
// MEA uses head expansion via LaunchUngroup (fp16/bf16 only) for GQA.
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
  //   Unfused: accepts both BSNH and BNSH (transposes if needed).
  //
  // nonpad_kv_seqlen + attn_mask routing:
  //   Flash: cannot handle this combo (no bias param when seqlens_k is used) → excluded.
  //   MEA:   supports both (custom_right_padding for seqlens + additive attn_bias for mask).
  //   Unfused: nonpad → seqlens_k; mask → attention_bias; both handled independently in softmax kernel.
#if USE_FLASH_ATTENTION || USE_MEMORY_EFFICIENT_ATTENTION
  const bool has_output_qk = (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone);
#endif

  // softmax_precision: All CUDA backends (Flash, MEA, Unfused) compute softmax in
  // FP32 internally (Flash/MEA via tile-based FP32 accumulators, Unfused via FP32
  // softmax kernel). softmax_precision=1 (FP32) is inherently satisfied;
  // softmax_precision=0 (default) is also fine since higher precision is always
  // acceptable per the ONNX spec.

#if USE_FLASH_ATTENTION
  // Flash Attention uses lower-right (bottom-right) causal alignment with no option for
  // upper-left. The ONNX spec requires upper-left alignment when there is no past context:
  // query[0] attends only to key[0]. The difference only manifests when S_q != S_kv
  // (cross-attention shape) with no past. Skip Flash for this case; MEA handles it correctly
  // via the causal_from_top_left flag, and Unified Unfused uses past_kv_length=0.
  const bool causal_cross_no_past = parameters.is_causal &&
                                    parameters.q_sequence_length != parameters.total_sequence_length &&
                                    parameters.past_sequence_length == 0;
  {
    auto& device_prop = GetDeviceProp();
    bool flash_eligible =
        !disable_flash_attention_ &&
        !std::is_same<T, float>::value &&
        onnxruntime::flash::is_supported<T>(device_prop, parameters.head_size,
                                            parameters.q_num_heads, parameters.kv_num_heads) &&
        parameters.head_size == parameters.v_head_size &&
        !has_output_qk &&
        !causal_cross_no_past &&
        // Flash does not support attention masks — reject when attn_mask is present.
        attn_mask == nullptr;

    if (flash_eligible) {
      LOGS_DEFAULT(VERBOSE) << "ONNX Attention: using Flash Attention"
                            << " (batch=" << parameters.batch_size
                            << ", q_seq=" << parameters.q_sequence_length
                            << ", total_seq=" << parameters.total_sequence_length
                            << ", past=" << (past_key != nullptr ? "yes" : "no") << ")";
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
        // MEA decode requires head_size == v_head_size for LaunchConcatNewToPastKV
        // (single head_size parameter). Fall back to unfused when they differ.
        (past_key == nullptr || parameters.head_size == parameters.v_head_size) &&
        // GQA+MEA requires LaunchUngroup which only has fp16/bf16 instantiations.
        // FP32 GQA must fall through to the unfused path.
        !(is_gqa && std::is_same<T, float>::value);

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
      LOGS_DEFAULT(VERBOSE) << "ONNX Attention: using Memory Efficient Attention"
                            << " (batch=" << parameters.batch_size
                            << ", q_seq=" << parameters.q_sequence_length
                            << ", total_seq=" << parameters.total_sequence_length
                            << ", past=" << (past_key != nullptr ? "yes" : "no")
                            << ", mask=" << (attn_mask != nullptr ? "yes" : "no") << ")";
      return RunMemoryEfficientAttention(context, Q, K, V, attn_mask, past_key, past_value,
                                         nonpad_kv_seqlen, Y, present_key, present_value, parameters);
    }
  }
#endif

  // Fallback: unified unfused attention
  // Routes ALL cases to LaunchUnfusedAttention, which handles:
  //   - GQA natively (reshape-Q trick inside kernel, no K/V head replication)
  //   - MHA as a degenerate case (group_size=1)
  //   - fp16/bf16 with large head_size via FP32 QK scratch
  //   - softcap, attn_mask, causal, past_key+past_value, nonpad_kv_seqlen
  //   - output_qk (kQK mode: scale * Q @ K^T, before softcap/mask/softmax)
  //   - past_key with H != H_v (separate concat calls for K and V)

  // Guard: unified kernel only supports kNone and kQK output modes.
  // Other modes (kQKMask, kQKSoftCap, kQKSoftMax) expect QK values captured at
  // different pipeline stages that the unified kernel does not implement.
  if (qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kNone &&
      qk_matmul_output_mode_ != attention_helper::QKMatMulOutputMode::kQK) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Only kNone and kQK output modes are supported in unified unfused attention. Mode: ",
                           static_cast<int>(qk_matmul_output_mode_));
  }

  LOGS_DEFAULT(VERBOSE) << "Attention: using unified unfused path (is_gqa=" << is_gqa
                        << ", head_size=" << parameters.head_size
                        << ", softcap=" << parameters.softcap << ")";
  return RunUnfusedAttention(context, Q, K, V, attn_mask, past_key, past_value,
                             nonpad_kv_seqlen, Y, present_key, present_value,
                             output_qk, parameters);
}

}  // namespace cuda
}  // namespace onnxruntime

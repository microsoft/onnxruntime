// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_MEMORY_EFFICIENT_ATTENTION

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "41_fused_multi_head_attention/kernel_forward.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename AttentionKernel, int kQueriesPerBlock>
struct RightPaddingBatchHook {
  using scalar_t = typename AttentionKernel::scalar_t;
  using accum_t = typename AttentionKernel::accum_t;
  using lse_scalar_t = typename AttentionKernel::lse_scalar_t;
  using output_t = typename AttentionKernel::output_t;
  using output_accum_t = typename AttentionKernel::output_accum_t;

  static constexpr bool kSupportsDropout = AttentionKernel::kSupportsDropout;
  static constexpr bool kSupportsBias = AttentionKernel::kSupportsBias;
  static constexpr int kKeysPerBlock = AttentionKernel::kKeysPerBlock;
  static constexpr bool kIsAligned = AttentionKernel::kIsAligned;
  static constexpr bool kSingleValueIteration = AttentionKernel::kSingleValueIteration;
  static constexpr int32_t kAlignLSE = AttentionKernel::kAlignLSE;  // block size of backward
  static constexpr bool kPreloadV = AttentionKernel::kPreloadV;
  static constexpr bool kKeepOutputInRF = AttentionKernel::kKeepOutputInRF;
  static constexpr bool kNeedsOutputAccumulatorBuffer = AttentionKernel::kNeedsOutputAccumulatorBuffer;

  template <typename Params>
  static CUTLASS_DEVICE bool AdvanceToBlockForGQA(Params& p) {
    auto batch_id = blockIdx.z;
    auto head_id = blockIdx.y;
    auto query_start = blockIdx.x * kQueriesPerBlock;

    auto lse_dim = ceil_div((int32_t)(p.num_queries), kAlignLSE) * kAlignLSE;

    // Advance to current batch - in case of different sequence lengths
    if (p.seqlen_k_ptr) {
      p.num_keys = p.seqlen_k_ptr[batch_id];
    }

    if (query_start >= p.num_queries) {
      return false;
    }

    // Advance to the current batch / head / query_start
    p.query_ptr += batch_id * p.q_strideB + query_start * p.q_strideM + head_id * p.q_strideH;
    p.key_ptr += batch_id * p.k_strideB + head_id * p.k_strideH;
    p.value_ptr += batch_id * p.v_strideB + head_id * p.v_strideH;
    p.output_ptr += int64_t(batch_id * p.num_queries) * p.o_strideM + int64_t(query_start) * p.o_strideM + head_id * p.head_dim_value;

    if (kSupportsBias && p.attn_bias_ptr != nullptr) {
      p.attn_bias_ptr += (batch_id * p.bias_strideB) + (head_id * p.bias_strideH);
    }
    if (p.output_accum_ptr != nullptr) {
      p.output_accum_ptr += int64_t(batch_id * p.num_queries) * (p.head_dim_value * p.num_heads) +
                            int64_t(query_start) * (p.head_dim_value * p.num_heads) +
                            head_id * p.head_dim_value;
    } else {
      // Accumulate directly in the destination buffer (eg for f32)
      p.output_accum_ptr = (accum_t*)(p.output_ptr);
    }

    if (p.logsumexp_ptr != nullptr) {
      // lse[batch_id, head_id, query_start]
      p.logsumexp_ptr +=
          batch_id * lse_dim * p.num_heads + head_id * lse_dim + query_start;
    }

    // Custom masking
    // if (p.causal_diagonal_ptr) {
    //   p.causal_diagonal_offset = p.causal_diagonal_ptr[batch_id];
    // }
    if (p.custom_mask_type == AttentionKernel::CausalFromBottomRight) {
      p.causal_diagonal_offset += p.num_keys - p.num_queries;
    }
    if (p.custom_mask_type == AttentionKernel::CausalFromTopLeft ||
        p.custom_mask_type == AttentionKernel::CausalFromBottomRight) {
      // the bottom row of the current block is query_start + kQueriesPerBlock
      // the last active key is then query_start + causal_diagonal_offset +
      // kQueriesPerBlock so num_keys is the min between actual num_keys and
      // this to avoid extra computations
      p.num_keys = cutlass::fast_min(
          int32_t(query_start + p.causal_diagonal_offset + kQueriesPerBlock),
          p.num_keys);
    }

    p.num_queries -= query_start;
    p.num_batches = 0;  // no longer used after

    // If num_queries == 1, and there is only one key head we're wasting
    // 15/16th of tensor core compute In that case :
    //  - we only launch kernels for head_id % kQueriesPerBlock == 0
    //  - we iterate over heads instead of queries (strideM = strideH)
    if (p.num_queries == 1 && p.k_strideH == 0 && p.v_strideH == 0) {
      if (head_id % kQueriesPerBlock != 0)
        return false;
      p.q_strideM = p.q_strideH;
      p.num_queries = p.num_heads;
      p.num_heads = 1;  // unused but here for intent
      // remove causal since n_query = 1
      // otherwise, offset would change with head !
      p.custom_mask_type = AttentionKernel::NoCustomMask;
      p.o_strideM = p.head_dim_value;
    }

    // Make sure the compiler knows these variables are the same on all
    // the threads of the warp.
    p.query_ptr = warp_uniform(p.query_ptr);
    p.key_ptr = warp_uniform(p.key_ptr);
    p.value_ptr = warp_uniform(p.value_ptr);
    if (kSupportsBias) {
      p.attn_bias_ptr = warp_uniform(p.attn_bias_ptr);
    }
    p.output_ptr = warp_uniform(p.output_ptr);
    p.output_accum_ptr = warp_uniform(p.output_accum_ptr);
    p.logsumexp_ptr = warp_uniform(p.logsumexp_ptr);
    p.num_queries = warp_uniform(p.num_queries);
    p.num_keys = warp_uniform(p.num_keys);
    p.num_heads = warp_uniform(p.num_heads);
    p.head_dim = warp_uniform(p.head_dim);
    p.head_dim_value = warp_uniform(p.head_dim_value);
    p.o_strideM = warp_uniform(p.o_strideM);
    p.custom_mask_type = warp_uniform(p.custom_mask_type);
    return true;
  }
};

template <typename AK, int kQueriesPerBlock>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched_impl_right_padding(typename AK::Params p) {
  if (!RightPaddingBatchHook<AK, kQueriesPerBlock>::AdvanceToBlockForGQA(p)) {
    return;
  }
  AK::attention_kernel(p);
}

template <typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block, int kMaxK>
void LaunchCutlassFmha(const MemoryEfficientAttentionParams& params) {
  using Attention = AttentionKernel<T, ArchTag, is_aligned, queries_per_block, keys_per_block,
                                    kMaxK, false, true>;
  typename Attention::Params p;
  {  // set parameters
    p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query));
    p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key));
    p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value));
    p.attn_bias_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.attn_bias));
    p.seqstart_q_ptr = params.seqstart_q_ptr;
    p.seqstart_k_ptr = params.seqstart_k_ptr;
    p.seqlen_k_ptr = params.seqlen_k_ptr;

    p.logsumexp_ptr = nullptr;  // [num_heads, num_queries] for backward or nullptr for forward
    p.output_ptr = reinterpret_cast<T*>(params.output);
    if (Attention::kNeedsOutputAccumulatorBuffer) {
      using Acc = typename Attention::accum_t;
      // workspace size: batch_size * sequence_length * num_heads * v_head_size * sizeof(float)
      ORT_ENFORCE(params.workspace != nullptr, "Need output accumulator buffer but no workspace provided");
      p.output_accum_ptr = reinterpret_cast<Acc*>(params.workspace);
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.num_heads = params.num_heads;
    p.num_batches = params.batch_size;
    p.head_dim = params.qk_head_size;
    p.head_dim_value = params.v_head_size;

    p.scale = params.scale;

    // When params.cu_seqlens_q is provided, num_queries is max_seq_q and num_keys will be set inside the kernel
    p.num_queries = params.sequence_length;
    p.num_keys = params.kv_sequence_length;

    if (params.causal) {
      p.custom_mask_type = Attention::CausalFromBottomRight;
    }

    // We use max_sequence_length to calculate KV stride
    if (params.is_kv_bsnh) {
      // Input Q, K, V format is BxSxNxH, output is BxSxNxH
      p.q_strideH = params.qk_head_size;
      p.k_strideH = params.qk_head_size;
      p.v_strideH = params.v_head_size;
      p.bias_strideH = nullptr == params.attn_bias ? 0 : p.num_queries * p.num_keys;

      p.q_strideM = params.num_heads * params.qk_head_size;
      p.k_strideM = params.num_heads * params.qk_head_size;
      p.v_strideM = params.num_heads * params.v_head_size;
      p.o_strideM = params.num_heads * params.v_head_size;
      p.bias_strideM = nullptr == params.attn_bias ? 0 : p.num_keys;

      p.q_strideB = static_cast<int64_t>(p.q_strideM) * params.sequence_length;
      p.k_strideB = static_cast<int64_t>(p.k_strideM) * params.max_sequence_length;
      p.v_strideB = static_cast<int64_t>(p.v_strideM) * params.max_sequence_length;
      p.bias_strideB = params.is_attn_bias_batched ? static_cast<int64_t>(p.bias_strideH) * params.num_heads : 0;
    } else {
      // Input K, V format is BxNxSxH, Input Q is BxSxNxH, output is BxSxNxH
      p.q_strideH = params.qk_head_size;
      p.k_strideH = params.max_sequence_length * params.qk_head_size;
      p.v_strideH = params.max_sequence_length * params.v_head_size;
      p.bias_strideH = nullptr == params.attn_bias ? 0 : p.num_queries * p.num_keys;

      p.q_strideM = params.num_heads * params.qk_head_size;
      p.k_strideM = params.qk_head_size;
      p.v_strideM = params.v_head_size;
      p.o_strideM = params.num_heads * params.v_head_size;
      p.bias_strideM = nullptr == params.attn_bias ? 0 : p.num_keys;

      p.q_strideB = params.num_heads * params.qk_head_size * params.sequence_length;
      p.k_strideB = params.num_heads * params.qk_head_size * params.max_sequence_length;
      p.v_strideB = params.num_heads * params.v_head_size * params.max_sequence_length;
      p.bias_strideB = params.is_attn_bias_batched ? static_cast<int64_t>(p.bias_strideH) * params.num_heads : 0;
    }
  }

  auto kernel_fn = attention_kernel_batched_impl<Attention>;
  if (params.has_custom_right_padding) {
    kernel_fn = attention_kernel_batched_impl_right_padding<Attention, queries_per_block>;
  }

  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    ORT_ENFORCE(params.sm >= 70, "This kernel requires too much shared memory on this machine!");
    static bool once = [&]() {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      return true;
    }();
  }

  ORT_ENFORCE(Attention::check_supported(p));
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, params.stream>>>(p);
}

template <typename T, typename ArchTag, int queries_per_block, int keys_per_block, int kMaxK>
void DispatchIsAligned(const MemoryEfficientAttentionParams& params) {
  using AlignedAK = AttentionKernel<T, ArchTag, true, queries_per_block, keys_per_block, kMaxK, false, true>;
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 6287)
#endif
  // Run a more efficient kernel with `isAligned=True` when memory is correctly aligned.
  bool is_aligned = params.qk_head_size % AlignedAK::kAlignmentQ == 0 &&
                    params.qk_head_size % AlignedAK::kAlignmentK == 0 &&
                    params.v_head_size % AlignedAK::kAlignmentV == 0;
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
  DISPATCH_BOOL(is_aligned, kIsAligned, ([&]() {
                  LaunchCutlassFmha<T, ArchTag, kIsAligned, queries_per_block, keys_per_block, kMaxK>(params);
                }));
}

template <typename T, typename ArchTag>
void DispatchBlockSize(const MemoryEfficientAttentionParams& params) {
  if (params.v_head_size <= 64) {
    DispatchIsAligned<T, ArchTag, 64, 64, 64>(params);
  } else if (params.v_head_size <= 128) {
    DispatchIsAligned<T, ArchTag, 32, 128, 128>(params);
  } else {
    DispatchIsAligned<T, ArchTag, 32, 128, 65536>(params);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif  // USE_MEMORY_EFFICIENT_ATTENTION

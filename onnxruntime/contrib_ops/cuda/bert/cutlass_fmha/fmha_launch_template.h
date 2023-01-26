// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_FLASH_ATTENTION

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/kernel_forward.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block, bool single_value_iteration>
void LaunchCutlassFmha(const MemoryEfficientAttentionParams& params) {
  using Attention = AttentionKernel<T, ArchTag, is_aligned, queries_per_block, keys_per_block, single_value_iteration>;
  typename Attention::Params p;
  {  // set parameters
    p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query));
    p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key));
    p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value));
    p.cu_seqlens_q_ptr = params.cu_seqlens_q;
    p.cu_seqlens_k_ptr = params.cu_seqlens_k;

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

    // When params.cu_seqlens_q is provided, num_queries is max_seq_q and num_keys will be set inside the kernel
    p.num_queries = params.sequence_length;
    p.num_keys = params.kv_sequence_length;

    p.causal = params.causal;

    // Input format is BxSxNxH, output is BxSxNxH
    p.q_strideH = params.qk_head_size;
    p.k_strideH = params.qk_head_size;
    p.v_strideH = params.v_head_size;
    p.o_strideH = params.v_head_size;

    p.q_strideM = params.num_heads * params.qk_head_size;
    p.k_strideM = params.num_heads * params.qk_head_size;
    p.v_strideM = params.num_heads * params.v_head_size;

    p.q_strideB = static_cast<int64_t>(p.q_strideM) * params.sequence_length;
    p.k_strideB = static_cast<int64_t>(p.k_strideM) * params.kv_sequence_length;
    p.v_strideB = static_cast<int64_t>(p.v_strideM) * params.kv_sequence_length;
    p.o_strideB = static_cast<int64_t>(params.num_heads) * params.v_head_size * params.sequence_length;

    p.causal = params.causal;
  }

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
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

template <typename T, typename ArchTag, int queries_per_block, int keys_per_block, bool single_value_iteration>
void DispatchIsAligned(const MemoryEfficientAttentionParams& params) {
  using AlignedAK = AttentionKernel<T, ArchTag, true, queries_per_block, keys_per_block, single_value_iteration>;

  // Run a more efficient kernel with `isAligned=True` when memory is correctly aligned.
  bool is_aligned = params.qk_head_size % AlignedAK::kAlignmentQ == 0 &&
                    params.qk_head_size % AlignedAK::kAlignmentK == 0 &&
                    params.v_head_size % AlignedAK::kAlignmentV == 0;

  DISPATCH_BOOL(is_aligned, kIsAligned, ([&]() {
                  LaunchCutlassFmha<T, ArchTag, kIsAligned, queries_per_block, keys_per_block, single_value_iteration>(params);
                }));
}

template <typename T, typename ArchTag>
void DispatchBlockSize(const MemoryEfficientAttentionParams& params) {
  if (params.v_head_size <= 64) {
    DispatchIsAligned<T, ArchTag, 64, 64, true>(params);
  } else if (params.v_head_size <= 128) {
    DispatchIsAligned<T, ArchTag, 32, 128, true>(params);
  } else {
    DispatchIsAligned<T, ArchTag, 32, 128, false>(params);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif  // USE_FLASH_ATTENTION

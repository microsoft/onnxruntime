// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Template for the paged-KV XQA kernel instantiation. Mirrors xqa_impl_gen.cuh but binds the
// paged (PAGED_KV_CACHE_LAYOUT == 1 / vLLM-style) entry point of launchMHA.
//
// Expected macros:
//   NAMESPACE_NAME: name of the namespace (e.g. grp8_int8_paged)
//   GRP_SIZE:       integer value for HEAD_GRP_SIZE

namespace NAMESPACE_NAME {
// Undefine dependent guard to allow header re-processing
#undef MHA_H_DEPENDENT

#define HEAD_GRP_SIZE GRP_SIZE

// See xqa_impl_gen.cuh for why the SM80 guard is written this way.
#undef XQA_HAS_SM80_TARGET
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#define XQA_HAS_SM80_TARGET 1
#endif
#elif defined(HAS_SM80_OR_LATER) || !defined(__CUDACC__)
#define XQA_HAS_SM80_TARGET 1
#endif

#ifdef XQA_HAS_SM80_TARGET
#include "mha_impl.cuh"
#endif

#undef HEAD_GRP_SIZE

template <typename T>
inline Status Launch(
    [[maybe_unused]] const cudaDeviceProp& device_prop,
    [[maybe_unused]] cudaStream_t stream,
    [[maybe_unused]] const void* query,
    [[maybe_unused]] const void* key_cache,
    [[maybe_unused]] const void* value_cache,
    [[maybe_unused]] void* output,
    [[maybe_unused]] const int* page_table,
    [[maybe_unused]] const int batch_size,
    [[maybe_unused]] const int num_heads,
    [[maybe_unused]] const int kv_num_heads,
    [[maybe_unused]] const int head_size,
    [[maybe_unused]] const int max_pages_per_seq,
    [[maybe_unused]] const float scale,
    [[maybe_unused]] const int local_window_size,
    [[maybe_unused]] const int* past_seq_lens,
    [[maybe_unused]] const float* attention_sinks,
    [[maybe_unused]] const float* k_cache_scale,
    [[maybe_unused]] const float* v_cache_scale,
    [[maybe_unused]] void* workspace,
    [[maybe_unused]] size_t workspace_size) {
#ifdef XQA_HAS_SM80_TARGET
  const InputHead* q_ptr = reinterpret_cast<const InputHead*>(query);
  GMemCacheHead* k_ptr = reinterpret_cast<GMemCacheHead*>(const_cast<void*>(key_cache));
  GMemCacheHead* v_ptr = reinterpret_cast<GMemCacheHead*>(const_cast<void*>(value_cache));
  OutputHead* out_ptr = reinterpret_cast<OutputHead*>(output);

  // maxSeqLen must be a whole number of pages: launchMHA derives maxNbPagesPerSeq (the page-table
  // row stride) as exactDiv(maxSeqLen, tokensPerPage).
  const uint32_t max_seq_len = static_cast<uint32_t>(max_pages_per_seq) * tokensPerPage;

  uint32_t* semaphores = nullptr;
  void* scratch = nullptr;

  if (workspace != nullptr) {
    uint32_t nbSeq = static_cast<uint32_t>(batch_size * kv_num_heads);
    size_t semaphore_size = nbSeq * sizeof(uint32_t);
    size_t padded_sem_size = roundUp<size_t>(semaphore_size, 128);

    uint32_t nbSubSeqPerSeq = computeNbSubSeqPerSeqMHA(
        device_prop,
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(kv_num_heads),
        max_seq_len);
    size_t required_scratch_size = NAMESPACE_NAME::GetScratchSize(nbSeq, nbSubSeqPerSeq);
    size_t total_required = padded_sem_size + required_scratch_size;

    if (workspace_size < total_required) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Paged XQA workspace size is too small. Expected at least ",
                             total_required, ", but got ", workspace_size);
    }
    semaphores = reinterpret_cast<uint32_t*>(workspace);
    scratch = reinterpret_cast<char*>(workspace) + padded_sem_size;

    cudaMemsetAsync(semaphores, 0, semaphore_size, stream);
  }

#if SLIDING_WINDOW
  // See xqa_impl_gen.cuh: -1 (global) maps to a window >= max_seq_len so no masking work is done.
  uint32_t const sliding_win_size = (local_window_size > 0)
                                        ? static_cast<uint32_t>(local_window_size)
                                        : max_seq_len;
#endif

  launchMHA(
      device_prop,
      static_cast<uint32_t>(kv_num_heads),
#if SLIDING_WINDOW
      sliding_win_size,
#endif
      scale,
      out_ptr,
      q_ptr,
      attention_sinks,
      k_ptr,
      v_ptr,
      reinterpret_cast<const KVCachePageIndex*>(page_table),
      max_seq_len,
      reinterpret_cast<const uint32_t*>(past_seq_lens),
      static_cast<uint32_t>(batch_size),
      k_cache_scale,
      v_cache_scale,
      semaphores,
      scratch,
      stream);
  return Status::OK();
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA is only supported on Ampere (SM80) or newer GPUs.");
#endif
}

#ifndef GENERATE_CUBIN
// See xqa_impl_gen.cuh::GetSmemSize.
inline size_t GetSmemSize() {
#ifdef XQA_HAS_SM80_TARGET
  uint32_t size = 0;
  if (cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize)) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return static_cast<size_t>(size);
#else
  return 0;
#endif
}
#endif  // GENERATE_CUBIN

#undef XQA_HAS_SM80_TARGET
}  // namespace NAMESPACE_NAME

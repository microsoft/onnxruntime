// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Template for XQA Kernel Implementation
// Expected macros:
// NAMESPACE_NAME: Name of the namespace (e.g., grp8)
// GRP_SIZE: Integer value for HEAD_GRP_SIZE

namespace NAMESPACE_NAME {
// Undefine dependent guard to allow header re-processing
#undef MHA_H_DEPENDENT

// Define macro for mha_impl.cuh (which includes mha.h)
// We assume mha.h's dependent part relies on this macro
#define HEAD_GRP_SIZE GRP_SIZE

// Include implementation (re-compiles kernel for this group size)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
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
    [[maybe_unused]] const int batch_size,
    [[maybe_unused]] const int num_heads,
    [[maybe_unused]] const int kv_num_heads,
    [[maybe_unused]] const int head_size,
    [[maybe_unused]] const int max_seq_len,
    [[maybe_unused]] const float scale,
    [[maybe_unused]] const bool is_bsnh,
    [[maybe_unused]] const int* past_seq_lens,
    [[maybe_unused]] const float* kv_cache_scale,
    [[maybe_unused]] void* workspace,
    [[maybe_unused]] size_t workspace_size) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const InputHead* q_ptr = reinterpret_cast<const InputHead*>(query);
  GMemKVCacheHead* k_ptr = reinterpret_cast<GMemKVCacheHead*>(const_cast<void*>(key_cache));
  GMemKVCacheHead* v_ptr = reinterpret_cast<GMemKVCacheHead*>(const_cast<void*>(value_cache));
  OutputHead* out_ptr = reinterpret_cast<OutputHead*>(output);

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
        static_cast<uint32_t>(max_seq_len));
    size_t required_scratch_size = NAMESPACE_NAME::GetScratchSize(nbSeq, nbSubSeqPerSeq);
    size_t total_required = padded_sem_size + required_scratch_size;

    if (workspace_size < total_required) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA workspace size is too small. Expected at least ", total_required, ", but got ", workspace_size);
    }
    semaphores = reinterpret_cast<uint32_t*>(workspace);
    scratch = reinterpret_cast<char*>(workspace) + padded_sem_size;

    // Initialize semaphores to 0
    cudaMemsetAsync(semaphores, 0, semaphore_size, stream);
  }

  launchMHA(
      device_prop,
      static_cast<uint32_t>(kv_num_heads),
      scale,
      out_ptr,
      q_ptr,
      nullptr,  // attentionSinks
      k_ptr,
      v_ptr,
      is_bsnh,
      static_cast<uint32_t>(max_seq_len),
      reinterpret_cast<const uint32_t*>(past_seq_lens),
      static_cast<uint32_t>(batch_size),
      kv_cache_scale,  // Pass kv_cache_scale for INT8 dequantization
      semaphores,      // semaphores
      scratch,         // scratch
      stream);
  return Status::OK();
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA is only supported on Ampere (SM80) or newer GPUs.");
#endif
}
}  // namespace NAMESPACE_NAME

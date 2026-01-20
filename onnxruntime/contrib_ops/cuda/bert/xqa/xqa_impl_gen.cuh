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
#include "mha_impl.cuh"

#undef HEAD_GRP_SIZE

inline size_t GetScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int kv_num_heads,
    int max_seq_len) {
  uint32_t nbSubSeqPerSeq = computeNbSubSeqPerSeqMHA(
      device_prop,
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(kv_num_heads),
      static_cast<uint32_t>(max_seq_len));
  uint32_t nbSeq = static_cast<uint32_t>(batch_size * kv_num_heads);

  size_t semaphore_size = nbSeq * sizeof(uint32_t);
  size_t scratch_size = NAMESPACE_NAME::GetScratchSize(nbSeq, nbSubSeqPerSeq);

  // Return total size with alignment padding
  return roundUp<size_t>(semaphore_size, 128) + scratch_size;
}

template <typename T>
inline Status Launch(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int actual_seq_len,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* seq_lens,
    const float* kv_cache_scale,
    void* workspace,
    size_t workspace_size) {
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

    if (workspace_size < padded_sem_size) {
      // Or assert/return error? For now assume caller passed correct size.
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
      reinterpret_cast<const uint32_t*>(seq_lens),
      static_cast<uint32_t>(batch_size),
      kv_cache_scale,  // Pass kv_cache_scale for INT8 dequantization
      semaphores,      // semaphores
      scratch,         // scratch
      stream);
  return Status::OK();
}
}  // namespace NAMESPACE_NAME

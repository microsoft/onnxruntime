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

// XQA kernels require SM80+ (Ampere or newer). We need a guard that works correctly
// during both host and device compilation passes:
//   - Device pass: __CUDA_ARCH__ is defined, check it directly.
//   - Host pass: rely on HAS_SM80_OR_LATER from cmake/external/cuda_configuration.cmake.
//     If any SM80+ arch is enabled, the host stub must be emitted.
//   - Non-nvcc parsers usually won't see the CMake-provided define, so keep editor parsing
//     intact by taking the fallback branch when __CUDACC__ is not defined.
// Using only !defined(__CUDA_ARCH__) here would be WRONG: it always evaluates true during
// the host pass, causing the kernel to be declared even when no SM80+ device code exists.
// CUDA 13+ then fails to generate a host stub, producing C2129 / LNK2001.
#undef XQA_HAS_SM80_TARGET
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#define XQA_HAS_SM80_TARGET 1
#endif
#elif defined(HAS_SM80_OR_LATER) || !defined(__CUDACC__)
#define XQA_HAS_SM80_TARGET 1
#endif

// Include implementation (re-compiles kernel for this group size)
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
    [[maybe_unused]] const int batch_size,
    [[maybe_unused]] const int num_heads,
    [[maybe_unused]] const int kv_num_heads,
    [[maybe_unused]] const int head_size,
    [[maybe_unused]] const int max_seq_len,
    [[maybe_unused]] const float scale,
    [[maybe_unused]] const bool is_bsnh,
    [[maybe_unused]] const int* past_seq_lens,
    [[maybe_unused]] const float* attention_sinks,
    [[maybe_unused]] const float* kv_cache_scale,
    [[maybe_unused]] void* workspace,
    [[maybe_unused]] size_t workspace_size,
    // No default: every caller must thread local_window_size through explicitly so a future
    // caller that forgets it fails to compile instead of silently running global attention.
    [[maybe_unused]] const int local_window_size) {
#ifdef XQA_HAS_SM80_TARGET
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

#if SLIDING_WINDOW
  // SLIDING_WINDOW is #defined to 1 only by the fp16/bf16/int8/fp8 xqa_loader_*_impl.cuh headers
  // that include this file; it defaults to 0 in defines.h, so this block is dead (and
  // local_window_size is unused) in any translation unit that does not opt in.
  // ORT local_window_size semantics: -1 => global attention; >0 => each query attends to the
  // last local_window_size tokens (including the current one). XQA's slidingWinSize uses the
  // same "last N tokens incl. current" definition, so pass it through directly. For global
  // attention, use max_seq_len so the kernel's runtime guard (cacheSeqLen > slidingWinSize) is
  // never taken and no masking work is performed (numerically identical to the global kernel).
  uint32_t const sliding_win_size = (local_window_size > 0)
                                        ? static_cast<uint32_t>(local_window_size)
                                        : static_cast<uint32_t>(max_seq_len);
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

#ifndef GENERATE_CUBIN
// Host helper: dynamic shared-memory size (bytes) this kernel instantiation requests at launch,
// read from the device `smemSize` symbol. Because it is read from the loaded module, the value
// reflects the ACTUAL kernel that will run -- including a kernel JIT-compiled from PTX for a
// different SM, whose shared-memory layout (and therefore smemSize) was fixed at PTX-generation
// time. The GQA dispatcher uses this to avoid selecting XQA on devices whose per-block opt-in
// shared memory is smaller than the kernel needs, which would otherwise fail at launch with
// cudaErrorInvalidValue (e.g. consumer Blackwell sm_120 running a kernel JIT'd from sm_90 PTX,
// where the Hopper layout needs ~140 KB but sm_120 allows only ~99 KB). Returns 0 if the value
// cannot be queried.
inline size_t GetSmemSize() {
#ifdef XQA_HAS_SM80_TARGET
  uint32_t size = 0;
  if (cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize)) != cudaSuccess) {
    (void)cudaGetLastError();  // clear any sticky error so it does not leak to the next CUDA call
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

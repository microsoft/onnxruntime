// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Per-group-size instantiation of the speculative-decoding (multi-token) XQA kernel over a contiguous
// (non-paged) KV cache. Mirrors xqa_impl_gen.cuh with SPEC_DEC enabled.
//
// Expected macros:
//   NAMESPACE_NAME: namespace for this instantiation (e.g. grp16_bf16_spec_dec)
//   GRP_SIZE:       integer value for HEAD_GRP_SIZE

namespace NAMESPACE_NAME {
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

#ifdef XQA_HAS_SM80_TARGET
// Causal mask for uniform query length: bit j of row s is set when query token s may attend to new token j.
__global__ void XqaSpecDecCausalMaskKernel(uint32_t* __restrict__ mask, const uint32_t rows, const uint32_t q_len,
                                           const uint32_t words_per_row) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * words_per_row) {
    return;
  }
  const uint32_t token = (i / words_per_row) % q_len;
  const int32_t allowed_bits = static_cast<int32_t>(token + 1) - static_cast<int32_t>((i % words_per_row) * 32);
  // Avoid min/max clamping here; see PagedXqaSpecDecMaskKernel for the ptxas miscompile it triggers.
  uint32_t value;
  if (allowed_bits >= 32) {
    value = ~uint32_t{0};
  } else if (allowed_bits <= 0) {
    value = 0u;
  } else {
    value = (uint32_t{1} << allowed_bits) - 1u;
  }
  mask[i] = value;
}

inline uint32_t TokenBlocksPerGroup(int sequence_length) {
  return divUp(static_cast<uint32_t>(sequence_length * GRP_SIZE), static_cast<uint32_t>(M_TILESIZE));
}

inline size_t MaskBytes(int batch_size, int sequence_length) {
  const size_t words = static_cast<size_t>(batch_size) * sequence_length * divUp(static_cast<uint32_t>(sequence_length), 32u);
  return roundUp<size_t>(words * sizeof(uint32_t), 128);
}

inline size_t SemaphoreBytes(int batch_size, int kv_num_heads, int sequence_length) {
  const size_t nb_seq = static_cast<size_t>(batch_size) * kv_num_heads * TokenBlocksPerGroup(sequence_length);
  return roundUp<size_t>(nb_seq * sizeof(uint32_t), 128);
}
#endif

inline size_t GetWorkspaceSize([[maybe_unused]] const cudaDeviceProp& device_prop,
                               [[maybe_unused]] int batch_size,
                               [[maybe_unused]] int kv_num_heads,
                               [[maybe_unused]] int sequence_length,
                               [[maybe_unused]] int max_seq_len) {
#ifdef XQA_HAS_SM80_TARGET
  const uint32_t token_blocks = TokenBlocksPerGroup(sequence_length);
  const uint32_t nb_seq = static_cast<uint32_t>(batch_size * kv_num_heads) * token_blocks;
  // Must match launchMHA under XQA_SPEC_DEC_SPLIT_PER_TOKEN_BLOCK.
  const uint32_t nb_sub_seq_per_seq = computeNbSubSeqPerSeqMHA(
      device_prop, static_cast<uint32_t>(batch_size), static_cast<uint32_t>(kv_num_heads) * token_blocks,
      static_cast<uint32_t>(max_seq_len));
  return MaskBytes(batch_size, sequence_length) + SemaphoreBytes(batch_size, kv_num_heads, sequence_length) +
         GetScratchSize(nb_seq, nb_sub_seq_per_seq);
#else
  return 0;
#endif
}

template <typename T>
inline Status Launch(
    [[maybe_unused]] const cudaDeviceProp& device_prop,
    [[maybe_unused]] cudaStream_t stream,
    [[maybe_unused]] const void* query,
    [[maybe_unused]] const void* key_cache,
    [[maybe_unused]] const void* value_cache,
    [[maybe_unused]] void* output,
    [[maybe_unused]] const int batch_size,
    [[maybe_unused]] const int sequence_length,
    [[maybe_unused]] const int kv_num_heads,
    [[maybe_unused]] const int max_seq_len,
    [[maybe_unused]] const float scale,
    [[maybe_unused]] const int local_window_size,
    [[maybe_unused]] const bool is_bsnh,
    [[maybe_unused]] const int* past_seq_lens,
    [[maybe_unused]] void* workspace,
    [[maybe_unused]] size_t workspace_size) {
#ifdef XQA_HAS_SM80_TARGET
  const size_t required = GetWorkspaceSize(device_prop, batch_size, kv_num_heads, sequence_length, max_seq_len);
  if (workspace == nullptr || workspace_size < required) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Speculative XQA workspace is too small. Expected at least ",
                           required, ", but got ", workspace_size);
  }
  char* base = reinterpret_cast<char*>(workspace);
  uint32_t* mask = reinterpret_cast<uint32_t*>(base);
  const size_t mask_bytes = MaskBytes(batch_size, sequence_length);
  uint32_t* semaphores = reinterpret_cast<uint32_t*>(base + mask_bytes);
  const size_t semaphore_bytes = SemaphoreBytes(batch_size, kv_num_heads, sequence_length);
  void* scratch = base + mask_bytes + semaphore_bytes;

  const uint32_t words_per_row = divUp(static_cast<uint32_t>(sequence_length), 32u);
  const uint32_t mask_rows = static_cast<uint32_t>(batch_size * sequence_length);
  constexpr uint32_t kThreads = 128;
  XqaSpecDecCausalMaskKernel<<<divUp(mask_rows * words_per_row, kThreads), kThreads, 0, stream>>>(
      mask, mask_rows, static_cast<uint32_t>(sequence_length), words_per_row);
  // Cleared with a kernel: inside a CUDA graph a memset node adds more dependency latency than a kernel node.
  onnxruntime::cuda::Fill<int32_t>(stream, reinterpret_cast<int32_t*>(semaphores), 0,
                                   static_cast<int64_t>(semaphore_bytes / sizeof(int32_t)));

  // -1 (global attention) maps to a window >= max_seq_len so no masking work is done.
  const uint32_t sliding_win_size = (local_window_size > 0) ? static_cast<uint32_t>(local_window_size)
                                                            : static_cast<uint32_t>(max_seq_len);
  // qCuSeqLens == nullptr selects the uniform query length; the mask has one row per query token.
  const SpecDecParams spec_dec_params{static_cast<uint32_t>(sequence_length), nullptr, mask};

  launchMHA(
      device_prop,
      static_cast<uint32_t>(kv_num_heads),
      sliding_win_size,
      scale,
      reinterpret_cast<OutputHead*>(output),
      reinterpret_cast<const InputHead*>(query),
      /*attentionSinks=*/nullptr,
      reinterpret_cast<GMemKVCacheHead*>(const_cast<void*>(key_cache)),
      reinterpret_cast<GMemKVCacheHead*>(const_cast<void*>(value_cache)),
      is_bsnh,
      static_cast<uint32_t>(max_seq_len),
      reinterpret_cast<const uint32_t*>(past_seq_lens),
      static_cast<uint32_t>(batch_size),
      /*kCacheScale=*/nullptr,
      /*vCacheScale=*/nullptr,
      spec_dec_params,
      semaphores,
      scratch,
      stream);
  return CUDA_CALL(cudaGetLastError());
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA is only supported on Ampere (SM80) or newer GPUs.");
#endif
}

#ifndef GENERATE_CUBIN
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

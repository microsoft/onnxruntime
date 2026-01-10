/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "contrib_ops/cuda/bert/lean_attention/static_switch.h"
#include "contrib_ops/cuda/bert/lean_attention/flash.h"
#include "contrib_ops/cuda/bert/lean_attention/lean_fwd_kernel.h"

namespace onnxruntime {
namespace lean {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
  template <typename Kernel_traits, __VA_ARGS__>     \
  __global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(lean_fwd_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K, int kMaxSplits, bool Append_KV) {
#if defined(ARCH_SUPPORTS_FLASH)
  lean::lean_compute_attn<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, kMaxSplits, Append_KV>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

template <typename Kernel_traits>
void run_lean_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
  static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
  constexpr size_t smem_size = Kernel_traits::kSmemSize;
  dim3 grid(1, 1, params.lean_griddimz);
  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
      EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        MAXSPLIT_SWITCH(params.num_splits, [&] {
          BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV_Const, [&] {
            auto kernel = &lean_fwd_kernel < Kernel_traits, Is_causal, IsEvenMNConst && IsEvenKConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, kMaxSplits, Append_KV_Const > ;
            if (2 * smem_size >= 48 * 1024) {
              cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * smem_size);
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
          });
        });
      });
    });
  });
}

template <typename T, int Headdim>
void run_mha_fwd_lean_dispatch(Flash_fwd_params& params, cudaStream_t stream) {
  // This should be modified according to optimal lean tile size
  constexpr static int kBlockM = Headdim <= 64 ? 64 : (Headdim <= 128 ? 64 : 64);
  constexpr static int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
  run_lean_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>>(params, stream);
}

}  // namespace lean
}  // namespace onnxruntime
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#pragma once

#include "contrib_ops/cuda/bert/flash_attention/static_switch.h"
#include "contrib_ops/cuda/bert/flash_attention/flash.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_kernel.h"

namespace onnxruntime {
namespace flash {

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
  flash::compute_attn<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Return_softmax>(params);
}

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV>
__global__ void flash_fwd_splitkv_kernel(Flash_fwd_params params) {
  flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Split, Append_KV>(params);
}

template <typename Kernel_traits, int Log_max_splits, bool Is_even_K>
__global__ void flash_fwd_splitkv_combine_kernel(Flash_fwd_params params) {
  static_assert(Log_max_splits >= 1);
  flash::combine_attn_seqk_parallel<Kernel_traits, Log_max_splits, Is_even_K>(params);
}

template <typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr size_t smem_size = Kernel_traits::kSmemSize;

  // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
  // https://github.com/kokkos/kokkos-kernels/issues/349
  // https://github.com/HazyResearch/flash-attention/issues/21

  const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.b, params.h);
  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;
  BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
      // Will only return softmax if dropout, to reduce compilation time.
      auto kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, IsEvenMNConst, IsEvenKConst, false>;
      // auto kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, IsEvenMNConst, true, ReturnSoftmaxConst>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        // ORT_ENFORCE(cudaFuncSetAttribute(
        //     kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      }
      // int ctas_per_sm;
      // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
      //  printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
      kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
  });
}

template <typename Kernel_traits>
void run_flash_splitkv_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
  static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
  constexpr size_t smem_size = Kernel_traits::kSmemSize;
  const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
      BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(params.num_splits > 1, Split, [&] {
          BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
            // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
            // printf("About to launch, Split = %d, Append_KV = %d, knew_ptr = %p\n", Split, Append_KV, params.knew_ptr);
            auto kernel = &flash_fwd_splitkv_kernel < Kernel_traits, Is_causal, IsEvenMNConst && !Append_KV, IsEvenKConst, Split, Append_KV > ;
            // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, true, Split, Append_KV>;
            // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, IsEvenKConst>;
            if (smem_size >= 48 * 1024) {
              cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
          });
        });
      });
    });
  });
  if (params.num_splits > 1) {
    dim3 grid_combine((params.b * params.h * params.seqlen_q + 16 - 1) / 16);
    BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
      if (params.num_splits <= 2) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 4) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 8) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 16) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 32) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 64) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 128) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      }
    });
  }
}

template <typename T, int Headdim>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params& params, cudaStream_t stream) {
  bool is_sm8x = params.dprops->major == 8 && params.dprops->minor > 0;
  constexpr int kBlockM = 64;  // Fixed for all head dimensions
  if (!is_sm8x) {              // A100, H100
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    constexpr int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 160 ? 128 : 64);
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>>(params, stream);
  } else {  // Only 99KB of smem, so we have to set kBlockN smaller for Headdim 160 and above
    constexpr int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>>(params, stream);
  }
}

template <typename T>
void run_mha_fwd_hdim32(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr int Headdim = 32;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(params, stream);
  });
}

template <typename T>
void run_mha_fwd_hdim64(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr int Headdim = 64;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
    // Using block size (64 x 256) is 27% slower for seqlen=2k
    // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_causal>(params, stream);
  });
}

template <typename T>
void run_mha_fwd_hdim96(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr int Headdim = 96;
  const bool is_sm8x = params.dprops->major == 8 && params.dprops->minor > 0;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    if (is_sm8x) {
      if constexpr (!Is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(params, stream);
      } else {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
      }
    } else {
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(params, stream);
    }
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_causal>(params, stream);
    // These two are always slower
    // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
  });
}

template <typename T>
void run_mha_fwd_hdim128(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr int Headdim = 128;
  const bool is_sm8x = params.dprops->major == 8 && params.dprops->minor > 0;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
    if (is_sm8x) {
      if constexpr (!Is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_causal>(params, stream);
      } else {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
      }
    } else {
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(params, stream);
    }
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_causal>(params, stream);
    // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
    // 1st ones are good for H100, A100
    // 2nd one is good for A6000 bc we get slightly better occupancy
  });
}

template <typename T>
void run_mha_fwd_hdim160(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr int Headdim = 160;
  const bool is_sm8x = params.dprops->major == 8 && params.dprops->minor > 0;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    // For A100, H100, 128 x 32 is the fastest.
    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    // and 128 x 64 with 8 warps is the fastest for non-causal.
    if (is_sm8x) {
      if constexpr (!Is_causal) {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
      } else {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
      }
    } else {
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_causal>(params, stream);
    }
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, true, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
  });
}

template <typename T>
void run_mha_fwd_hdim192(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr int Headdim = 192;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
  });
}

template <typename T>
void run_mha_fwd_hdim224(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr size_t Headdim = 224;
  constexpr size_t threshold = 2 * Headdim * (128 + 2 * 64);
  size_t max_smem_per_block = params.dprops->sharedMemPerBlockOptin;
  //  printf("max_smem_per_block = %d\n", max_smem_per_block);
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    if (max_smem_per_block >= threshold) {  // 112 KB
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
    } else {
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
    }
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_causal>(params, stream);
    // We can't do 128 x 32 with 8 warps because with headdim 224, kBlockKSmem = 32.
    // If we have N = 32, there are only 1024 elements to load at once, where each load
    // is 8 elements. This means we can only use 128 threads and not 256 threads.
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_causal>(params, stream);
  });
}

template <typename T>
void run_mha_fwd_hdim256(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr size_t Headdim = 256;
  constexpr size_t min_threshold = 2 * Headdim * (128 + 2 * 64);
  constexpr size_t max_threshold = 4 * Headdim * (64 + 2 * 64);
  size_t max_smem_per_sm = params.dprops->sharedMemPerMultiprocessor;
  size_t max_smem_per_block = params.dprops->sharedMemPerBlockOptin;
  //  printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    // For A100, we want to run with 128 x 64 (128KB smem).
    // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
    if (max_smem_per_block >= min_threshold && max_smem_per_sm < max_threshold) {
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
    } else {
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
    }
    // 64 KB
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_causal>(params, stream);
    // 96 KB
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_causal>(params, stream);
  });
}

}  // namespace flash
}  // namespace onnxruntime

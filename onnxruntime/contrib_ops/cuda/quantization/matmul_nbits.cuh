// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// __launch_bounds__ minimum-blocks hint for the 4-bit single-row (M=1) GEMV kernel MatMulFloatInt4Kernel
// in matmul_4bits.cu. That kernel is latency-bound pure weight streaming: with no occupancy hint it uses
// ~40 registers/thread, which on a 64-warp/SM part caps it to 6 of 8 blocks (~75% occupancy), leaving too
// few resident warps to hide the global-load latency.
//
// NOTE: this hint is deliberately NOT applied to the 8-bit M=1 kernel (matmul_8bits.cu,
// MatMulFloat8bKernelM1). Measured on A100 (sm_80) the same minBlocks=8 raises that kernel's occupancy
// (39->32 regs, 75%->100%) but REGRESSES latency by ~2-11% (e.g. K=12288 N=4096 M=1: 68.75->76.06us),
// because the 8-bit path streams 2x the weight bytes (more DRAM-bound) and already has enough warps to
// hide latency, so cutting registers only adds recompute and DRAM/L2 contention. Kept here so a future
// N-bit M=1 kernel can opt in after its own A/B, but it is currently consumed only by the 4-bit kernel.
//
// The minimum-blocks argument is a COMPILE-TIME register-capping hint; it cannot be chosen from a runtime
// device query. ORT compiles a per-architecture fatbin, so we select it via __CUDA_ARCH__ to request full
// occupancy (maxWarpsPerSM / 8 warps-per-block) on each target WITHOUT over-capping registers (which would
// spill to local memory and regress):
//   * 64 warps/SM (sm_50/52/53 Maxwell, sm_60/61/62 Pascal, sm_70/72 Volta, sm_80 A100, sm_90 Hopper,
//     sm_100/103 datacenter Blackwell) -> 8 blocks. This is where the hint matters: it lowers registers
//     40 -> 32 and lifts occupancy to 100% (measured ~13% faster on A100). 8 * 256 = 2048 = per-SM max.
//   * 32 warps/SM (sm_75 Turing, RTX 20xx) -> 4 blocks.
//   * 48 warps/SM (sm_86/87 client Ampere RTX 30xx / Orin, sm_89 Ada RTX 40xx, sm_120 Blackwell consumer
//     RTX 50xx) -> 6 blocks. NOTE: Ada (CC 8.9) doubled the FP32 datapaths but -- like client Ampere
//     (CC 8.6) -- keeps only 48 resident warps / 1536 threads per SM (NOT 64). Verified with ptxas: under
//     __launch_bounds__(256, 8) sm_89 clamps to 6 blocks (39-40 regs), whereas the 2048-thread sm_80 is
//     cut to 32. At ~40 registers these parts already reach full occupancy, so 6 is a safe no-op instead
//     of the harmful 32-register cap a blanket 8 forces.
// The default (else) of 6 is the safe under-estimate for any unlisted/future arch: it never forces a
// register cut below what full occupancy needs, so at worst it leaves a little occupancy on the table.
#if !defined(__CUDA_ARCH__)
constexpr int kMatMulNBitsM1MinBlocksPerSM = 8;                               // host compilation pass; not used for device codegen
#elif __CUDA_ARCH__ == 500 || __CUDA_ARCH__ == 520 || __CUDA_ARCH__ == 530 || /* Maxwell (GTX 9xx)    */ \
    __CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 610 || __CUDA_ARCH__ == 620 ||   /* Pascal  (GTX 10xx)   */ \
    __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 720 ||                           /* Volta                */ \
    __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 ||                           /* A100 / Hopper        */ \
    __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 ||                         /* datacenter Blackwell */ \
    __CUDA_ARCH__ == 1100                                                     /* Jetson 5000          */
constexpr int kMatMulNBitsM1MinBlocksPerSM = 8;  // 64 warps/SM
#elif __CUDA_ARCH__ == 750
constexpr int kMatMulNBitsM1MinBlocksPerSM = 4;  // 32 warps/SM (Turing / RTX 20xx)
#else
constexpr int kMatMulNBitsM1MinBlocksPerSM = 6;  // 48 warps/SM (client Ampere RTX 30xx, Ada RTX 40xx, RTX 50xx)
#endif

template <class T>
bool TryMatMul4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    const T* bias_data,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

template <class T>
bool TryMatMul8Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

template <class T>
bool TryMatMulNBits(
    int bits,
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    const T* bias_data,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream) {
  if (bits == 8) {
    if (bias_data != nullptr) {
      return false;
    }
    return TryMatMul8Bits<T>(output, a_data, b_data_quant, scales_data, zero_points,
                             m, n, k, block_size, shared_mem_per_block, stream);
  }

  if (bits == 4) {
    return TryMatMul4Bits<T>(output, a_data, b_data_quant, scales_data, zero_points, bias_data,
                             m, n, k, block_size, shared_mem_per_block, stream);
  }

  return false;
}

// Adds a per-column bias of shape [n] to the output of shape [m, n] (row-major).
// Used as a fallback when the fused bias GEMV specialization does not apply.
template <class T>
void LaunchMatMulNBitsBiasAdd(
    T* output,
    const T* bias_data,
    int m,
    int n,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

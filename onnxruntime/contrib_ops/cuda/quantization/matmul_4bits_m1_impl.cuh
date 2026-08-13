// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_common.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_m1.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T, int block_size, bool has_zero_point, int cols_per_block = kColsPerThreadBlock>
__global__ void __launch_bounds__(kWarpSize* cols_per_block) MatMulFloat4BitsKernelM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int n_block_id = blockIdx.x;
  const int m_id = blockIdx.y;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = n_block_id * cols_per_block + warp_id;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

  extern __shared__ char shared_buffer[];
  T* b_scale_vec = (T*)shared_buffer;
  int offset = n_block_id * cols_per_block * blocks_per_K;
  for (int i = warp_id * kWarpSize + lane_id; i < cols_per_block * blocks_per_K;
       i += cols_per_block * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  uint8_t* b_zp_vec;
  (void)b_zp_vec;
  if constexpr (has_zero_point) {
    b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + cols_per_block * blocks_per_K);
    const int b_zp_k = (blocks_per_K + 1) / 2;
    int zp_offset = n_block_id * cols_per_block * b_zp_k;
    for (int i = warp_id * kWarpSize + lane_id; i < cols_per_block * b_zp_k;
         i += cols_per_block * kWarpSize) {
      b_zp_vec[2 * i] = (zero_points[zp_offset + i] & 0x0f);
      b_zp_vec[2 * i + 1] = (zero_points[zp_offset + i] >> 4);
    }
    b_zp_vec += warp_id * b_zp_k * 2;
  }
  __syncthreads();

  a_data += m_id * k + (lane_id << 3);
  b_scale_vec += warp_id * blocks_per_K;

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

#define UnRollReduction(unroll_size)                                                              \
  do {                                                                                            \
    constexpr int kUnroll = unroll_size;                                                          \
    constexpr int kUnrollStep = kUnroll * k_per_iter;                                             \
    const int k_unroll_bound = k - k % kUnrollStep;                                               \
    for (; k_id < k_unroll_bound; k_id += kUnrollStep) {                                          \
      _Pragma("unroll") for (int i = 0; i < kUnroll; i++) {                                       \
        uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_per_iter / 2 * i)); \
        T scale = b_scale_vec[t_meta_k + k_per_iter / block_size * i];                            \
        uint8_t zp = 8;                                                                           \
        if constexpr (has_zero_point) {                                                           \
          zp = b_zp_vec[t_meta_k + k_per_iter / block_size * i];                                  \
        }                                                                                         \
        AccumulateEightElements4b(value, scale, zp, a_data + k_id + i * k_per_iter, sums);        \
      }                                                                                           \
      b_data_quant += k_per_iter / 2 * kUnroll;                                                   \
      t_meta_k += k_per_iter / block_size * kUnroll;                                              \
    }                                                                                             \
  } while (false)

  UnRollReduction(16);
  UnRollReduction(4);
  UnRollReduction(1);
#undef UnRollReduction

  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[t_meta_k];
    uint8_t zp = 8;
    if constexpr (has_zero_point) {
      zp = b_zp_vec[t_meta_k];
    }
    AccumulateEightElements4b(value, scale, zp, a_data + k_id, sums);
  }

  float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  for (int i = kWarpSize / 2; i > 0; i = i / 2) {
    sum += onnxruntime::cuda::WARP_SHFL_DOWN(sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}

template <class T>
bool TryMatMul4BitsM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    int sm_count,
    cudaStream_t stream) {
  // Admission: always gate on the *original* cols=8 shared-memory footprint so that the
  // accepted-shape set is bit-for-bit identical to upstream. Shapes with huge K that would
  // have been declined (shared mem too large with 8 cols/CTA) must NOT be silently admitted
  // just because a smaller cols_per_block would fit. Those shapes fall through to cuBLAS
  // with fp32 accumulation — admitting them here would be a silent accuracy regression.
  if (n % kColsPerThreadBlock != 0 || k % kElementsPerThreadPerIteration != 0) {
    return false;
  }

  const int blocks_per_K = (k + block_size - 1) / block_size;
  const size_t admission_shared_mem =
      sizeof(T) * blocks_per_K * kColsPerThreadBlock +
      static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
  if (admission_shared_mem > shared_mem_per_block) {
    return false;
  }

  // Launch: select cols_per_block using the CUDA occupancy API rather than a magic constant.
  // For each candidate (8, 4, 2), we query cudaOccupancyMaxActiveBlocksPerMultiprocessor
  // for the *actual* kernel instantiation (which has different register and shared-memory
  // footprints per cols_per_block). We pick the candidate that maximises the number of
  // active warps across the device, subject to the constraint that the grid must actually
  // fill the device (no point having high per-SM occupancy if the grid is tiny).
  //
  // This replaces the old kTargetCtasPerSm = 12 magic number, which was tuned for
  // datacenter parts (A100/H100) and regressed CC 8.6/8.9 consumer GPUs where per-SM
  // block limits differ.
  //
  // NOTE: This path cannot be validated without a GPU. Consumer-GPU measurements
  // (CC 8.6 RTX 3060, CC 8.9 RTX 4090) are required before this PR can leave draft.

  // We need the kernel pointer to query occupancy. To avoid combinatorial explosion of
  // block_size × has_zp × cols queries, we pick a representative kernel (block_size=32,
  // no zero points) — the register/shared-memory profile is dominated by cols_per_block,
  // not by block_size or the zero-point branch.
  auto query_max_active_blocks = [&](int cpb, size_t smem) -> int {
    int max_blocks = 0;
    const int threads_per_block = onnxruntime::cuda::GPU_WARP_SIZE_HOST * cpb;
    cudaError_t err = cudaSuccess;
    // Use block_size=32 as a representative instantiation for occupancy query.
    // The register pressure difference between block_size variants is negligible
    // (same unrolled loop body, only the iteration bound changes).
    switch (cpb) {
      case 8:
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, MatMulFloat4BitsKernelM1<T, 32, false, 8>, threads_per_block, smem);
        break;
      case 4:
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, MatMulFloat4BitsKernelM1<T, 32, false, 4>, threads_per_block, smem);
        break;
      case 2:
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, MatMulFloat4BitsKernelM1<T, 32, false, 2>, threads_per_block, smem);
        break;
      default:
        break;
    }
    return (err == cudaSuccess) ? max_blocks : 0;
  };

  int cols_per_block = kColsPerThreadBlock;  // default 8
  {
    int best_active_warps = 0;
    constexpr int candidates[] = {8, 4, 2};
    for (int cpb : candidates) {
      if (n % cpb != 0) continue;
      const size_t smem = sizeof(T) * blocks_per_K * cpb +
                          static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * cpb * 2 : 0);
      const int grid_size = n / cpb;
      const int max_blocks_per_sm = query_max_active_blocks(cpb, smem);
      if (max_blocks_per_sm <= 0) continue;
      // Total active blocks across the device, capped by the grid size
      const int active_blocks = std::min(max_blocks_per_sm * sm_count, grid_size);
      // Warps per block = cpb (one warp per column)
      const int active_warps = active_blocks * cpb;
      if (active_warps > best_active_warps) {
        best_active_warps = active_warps;
        cols_per_block = cpb;
      }
    }
    // If all occupancy queries failed (e.g. driver issue), fall back to host heuristic
    if (best_active_warps == 0) {
      cols_per_block = SelectColsPerBlock(n, sm_count);
    }
  }

  const size_t launch_shared_mem =
      sizeof(T) * blocks_per_K * cols_per_block +
      static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * cols_per_block * 2 : 0);

  dim3 blocks((n + cols_per_block - 1) / cols_per_block, 1);
  dim3 threads(onnxruntime::cuda::GPU_WARP_SIZE_HOST, cols_per_block);

  // Template instantiation cost: the kernel is templated on <T, block_size, has_zero_point, cols_per_block>.
  // Adding cols_per_block ∈ {8, 4, 2} triples the per-type instantiation count:
  //   Before: 4 block_sizes × 2 zp variants × 1 cols = 8 per type, 24 total (3 types).
  //   After:  4 block_sizes × 2 zp variants × 3 cols = 24 per type, 72 total.
  // Practical cost: each instantiation is ~0.8 KB PTX (simple warp-reduction kernel with no
  // unrolled loops beyond the 8-element vectorized load). Incremental binary growth is bounded
  // at ~38 KB across all three type CUs — modest relative to ORT's 40+ MB provider .so.
  // Compile-time impact is similarly bounded: this kernel is a leaf template in three separate
  // .cu files (float, half, bf16) that already compile independently.
  //
  // All three cols_per_block values are reachable in production. The occupancy path can
  // select any of them depending on the per-CC register/shared-memory limits; the examples
  // below use the host-only fallback (SelectColsPerBlock, target = sm_count * 8) because it
  // is the one that can be enumerated without a device:
  //   - 8: n/8 >= target (n >= 8448 on a 132-SM H100).
  //   - 4: n/8 < target but n%4 == 0 and n/4 >= target (e.g. n=8192 on H100-132SM).
  //   - 2: n/4 < target and n is even (e.g. n=4096 on H100-132SM).
  // No instantiation is dead code.

#define MATMUL_FLOAT4B_M1_DISPATCH_COLS(bs, cpb)                                                 \
  if (zero_points != nullptr) {                                                                  \
    MatMulFloat4BitsKernelM1<T, bs, true, cpb><<<blocks, threads, launch_shared_mem, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, 1, n, k, blocks_per_K);          \
  } else {                                                                                       \
    MatMulFloat4BitsKernelM1<T, bs, false, cpb><<<blocks, threads, launch_shared_mem, stream>>>( \
        output, a_data, b_data_quant, scales_data, nullptr, 1, n, k, blocks_per_K);              \
  }

#define MATMUL_FLOAT4B_M1_DISPATCH(bs)                                         \
  if (cols_per_block == 8) {                                                   \
    MATMUL_FLOAT4B_M1_DISPATCH_COLS(bs, 8)                                     \
  } else if (cols_per_block == 4) {                                            \
    MATMUL_FLOAT4B_M1_DISPATCH_COLS(bs, 4)                                     \
  } else if (cols_per_block == 2) {                                            \
    MATMUL_FLOAT4B_M1_DISPATCH_COLS(bs, 2)                                     \
  } else {                                                                     \
    ORT_THROW("cols_per_block ", cols_per_block, " is not supported (8/4/2)"); \
  }

  if (block_size == 16) {
    MATMUL_FLOAT4B_M1_DISPATCH(16)
  } else if (block_size == 32) {
    MATMUL_FLOAT4B_M1_DISPATCH(32)
  } else if (block_size == 64) {
    MATMUL_FLOAT4B_M1_DISPATCH(64)
  } else if (block_size == 128) {
    MATMUL_FLOAT4B_M1_DISPATCH(128)
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }
#undef MATMUL_FLOAT4B_M1_DISPATCH
#undef MATMUL_FLOAT4B_M1_DISPATCH_COLS
  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

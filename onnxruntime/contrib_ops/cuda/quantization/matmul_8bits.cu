// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_nbits.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kColsPerThreadBlock = 8;
constexpr int kElementsPerThreadPerIteration = 8;
constexpr int kWarpSize = GPU_WARP_SIZE;  // Typically 32
constexpr uint8_t kDefaultZeroPoint = 128;
constexpr int kKernelAlgo = 0; // Choices: 0 (unroll), 1 (simple loop), 2 (block size iteration)
constexpr bool kUseCUB = true;

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __ushort2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  // Extract 8 uint8_t values from the 64-bit input.
  uint8_t q[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    q[i] = (values_quant >> (i * 8)) & 0xFF;
  }

  // Convert pairs to half2 (0,4 1,5 2,6 3,7 interleaved order).
  half2 element04 = __halves2half2(__ushort2half_rn(q[0]), __ushort2half_rn(q[4]));
  half2 element15 = __halves2half2(__ushort2half_rn(q[1]), __ushort2half_rn(q[5]));
  half2 element26 = __halves2half2(__ushort2half_rn(q[2]), __ushort2half_rn(q[6]));
  half2 element37 = __halves2half2(__ushort2half_rn(q[3]), __ushort2half_rn(q[7]));

  half2 v0 = element04 * scale_half2 + zp_adjust2;
  half2 v1 = element15 * scale_half2 + zp_adjust2;
  half2 v2 = element26 * scale_half2 + zp_adjust2;
  half2 v3 = element37 * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_a.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_a.w))));
}

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, float scale, uint8_t zp, const float* a, float* sums) {
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xFF) * scale + zp_adjust;
  float v1 = float((values_quant >> 8) & 0xFF) * scale + zp_adjust;
  float v2 = float((values_quant >> 16) & 0xFF) * scale + zp_adjust;
  float v3 = float((values_quant >> 24) & 0xFF) * scale + zp_adjust;
  float v4 = float((values_quant >> 32) & 0xFF) * scale + zp_adjust;
  float v5 = float((values_quant >> 40) & 0xFF) * scale + zp_adjust;
  float v6 = float((values_quant >> 48) & 0xFF) * scale + zp_adjust;
  float v7 = float((values_quant >> 56) & 0xFF) * scale + zp_adjust;

  sums[0] += v0 * a_vec_0.x;
  sums[1] += v1 * a_vec_0.y;
  sums[2] += v2 * a_vec_0.z;
  sums[3] += v3 * a_vec_0.w;
  sums[4] += v4 * a_vec_1.x;
  sums[5] += v5 * a_vec_1.y;
  sums[6] += v6 * a_vec_1.z;
  sums[7] += v7 * a_vec_1.w;
}

// kernel for 8bits quantized gemv, i.e., computing A(1, K) x B(K, N)
// B(K, N) is quantized with 8bits and block_size bs and stored as [N, K/bs, bs]
// kColsPerThreadBlock (C) = 8 is the number of columns each thread block computes
// kElementsPerThreadPerIteration (E) = 8 is the number of elements each thread computes in one iteration
// Constraints: N % C == 0, K % E == 0
// The thread block size is (kWarpSize, C) and grid size is (N / C, 1)
// Each thread block computes [1, K] x [C, K/bs, bs],
//     i.e., computing kColsPerThreadBlock per block and a warp reduce (1, K) x (K)
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloat8bKernel(
    T* output,
    const T* a_data,              // Base pointer for A[1, K]
    const uint8_t* b_data_quant,  // Base pointer for B [N, K/bs, bs]
    const T* scales_data,         // Base pointer for scales [N, K/bs]
    const uint8_t* zero_points,   // Base pointer for ZPs [N, K/bs]
    int m,                        // Should be 1
    int n,                        // Constraint: N % C == 0, where C=8
    int k,                        // Constraint: K % E == 0, where E=8
    int blocks_per_K) {           // blocks_per_K = K/bs

  const int n_block_id = blockIdx.x;                            // Block column index in the range of [0, N / C)
  const int m_id = blockIdx.y;                                  // Block row index (0)

  const int lane_id = threadIdx.x;                              // Thread index in warp (0..31)
  const int warp_id = threadIdx.y;                              // Warp index 0..7 in the range of [0, C-1)
  const int n_block_head = n_block_id * kColsPerThreadBlock;    // Head index of block C in the range of [0, N)
  const int n_id = n_block_head + warp_id;                      // Global output column index this warp computes

  // Ensure n_id does not go out of bounds if n is not perfectly divisible by kColsPerThreadBlock
  // Although TryMatMul8Bits checks for n % kColsPerThreadBlock == 0, it's safer.
  if (n_id >= n) return;

  extern __shared__ char shared_buffer[];

  // Load scales to shared_buffer.
  T* b_scale_vec_shared = (T*)shared_buffer;
  const int scale_zp_offset_block = n_block_head * blocks_per_K; // Offset for the block's columns
  for (int i = threadIdx.y * kWarpSize + threadIdx.x; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
      int offset = scale_zp_offset_block + i;
      if (offset < n * blocks_per_K) {
          b_scale_vec_shared[i] = scales_data[offset];
    }
  }

  // When there is zero points, load zero points and append to shared_buffer.
  [[maybe_unused]] uint8_t* b_zp_vec_shared = nullptr;
  [[maybe_unused]] const uint8_t* b_zp_vec_thread = nullptr; // Thread's ZP pointer
  if constexpr (has_zero_point) {
    b_zp_vec_shared = reinterpret_cast<uint8_t*>(b_scale_vec_shared + kColsPerThreadBlock * blocks_per_K);
    for (int i = threadIdx.y * kWarpSize + threadIdx.x; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
      int offset = scale_zp_offset_block + i;
      if (offset < n * blocks_per_K) {
        b_zp_vec_shared[i] = zero_points[offset];
      }
    }
     b_zp_vec_thread = b_zp_vec_shared + warp_id * blocks_per_K;
  }

  __syncthreads(); // Ensure scales and ZPs are loaded


  // Point a_data to the start of the row (m_id is 0) and offset by lane's portion.
  const int lane_offset = lane_id * kElementsPerThreadPerIteration;
  a_data += lane_offset;

  // Pointer of the specific N column (n_id) in B of layout [N, blocks_per_K, block_size].
  const uint8_t* b_base_ptr_n = b_data_quant + n_id * blocks_per_K * block_size;

 // Pointer to the start of scales for the specific column (n_id) this warp handles.
 const T* b_scale_vec_thread = b_scale_vec_shared + warp_id * blocks_per_K;

  T sums[kElementsPerThreadPerIteration] = {static_cast<T>(0.0f)};  // Initialize sums to zero

  if constexpr (kKernelAlgo == 0 || kKernelAlgo == 1) {
    // Note that k_per_iter (typical value is 256) is multiple of block_size (typical value is 16, 32, 64, 128, or 256).
    constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

    int k_id = 0;
    // Adjust b_data_quant pointer for the specific N column and the thread's lane.
    const uint8_t* b_data_quant_thread = b_base_ptr_n + lane_offset;

    if constexpr (kKernelAlgo == 0) { // Algorithm 0: Unrolling
      // Calculate the block index within the K dimension for the start of this thread's work.
      int k_start_block = lane_offset / block_size;

#define UnRollReduction(unroll_size)                                                                            \
  do {                                                                                                          \
    constexpr int kUnroll = unroll_size;                                                                        \
    constexpr int kElementsPerUnrollIter = k_per_iter * kUnroll; /* Elements processed per outer loop iter */   \
    for (; k_id + kElementsPerUnrollIter <= k; k_id += kElementsPerUnrollIter) {                                \
      _Pragma("unroll") for (int i = 0; i < kUnroll; ++i) {                                                     \
        /* Calculate pointer for this inner iteration's quantized data */                                       \
        /* Base pointer `b_data_quant_thread` is already offset by n_id and lane_id */                          \
        /* Offset further by k_id progression and inner loop step 'i' */                                        \
        /* Each full warp iteration consumes k_per_iter = 256 elements = 256 bytes */                           \
        const uint8_t* current_b_ptr = b_data_quant_thread + k_id + i * k_per_iter;                             \
        /* Load 8 uint8 into uint64_t safely to avoid alignment issues*/                                        \
        const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);                               \
        uint32_t val_low = ptr32[0];                                                                            \
        uint32_t val_high = ptr32[1];                                                                           \
        uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;                                     \
        /* Calculate index into scale/zp for this inner iteration */                                            \
        /* Here assumes k_per_iter % block_size == 0, so the unroll kernel can support up to block_size=256  */ \
        int current_meta_k = k_start_block + (k_id / block_size) + i * (k_per_iter / block_size);               \
        T scale = b_scale_vec_thread[current_meta_k];                                                           \
        uint8_t zp = kDefaultZeroPoint;                                                                         \
        if constexpr (has_zero_point) {                                                                         \
          zp = b_zp_vec_thread[current_meta_k];                                                                 \
        }                                                                                                       \
        AccumulateEightElements8b(value, scale, zp, a_data + k_id + i * k_per_iter, sums);                      \
      }                                                                                                         \
    }                                                                                                           \
  } while (false)

      UnRollReduction(16);
      UnRollReduction(4);
      UnRollReduction(1);

#undef UnRollReduction
    } else {  // Algorithm 1: Unrolling is disabled, use a simpler loop structure
      for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
        const uint8_t* current_b_ptr = b_data_quant_thread + k_id;

        // Load 8 uint8 into uint64_t safely to avoid alignment issues.
        const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);
        uint32_t val_low = ptr32[0];
        uint32_t val_high = ptr32[1];
        uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;

        int current_meta_k = (lane_offset + k_id) / block_size;  // Recalculate meta index based on absolute K pos
        T scale = b_scale_vec_thread[current_meta_k];

        uint8_t zp;
        if constexpr (has_zero_point) {
          zp = b_zp_vec_thread[current_meta_k];
        } else {
          zp = kDefaultZeroPoint;
        }

        AccumulateEightElements8b(value, scale, zp, a_data + k_id, sums);
      }
    }

    // Handle the tail elements (less than k_per_iter) if any
    if (lane_offset + k_id < k) {
      const uint8_t* current_b_ptr = b_data_quant_thread + k_id;

      // Load 8 uint8 into uint64_t safely to avoid alignment issues.
      const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);
      uint32_t val_low = ptr32[0];
      uint32_t val_high = ptr32[1];
      uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;

      int current_meta_k = (lane_offset + k_id)  / block_size;

      T scale = b_scale_vec_thread[current_meta_k];
      uint8_t zp;
      if constexpr (has_zero_point) {
        zp = b_zp_vec_thread[current_meta_k];
      } else {
        zp = kDefaultZeroPoint;
      }
      AccumulateEightElements8b(value, scale, zp, a_data + k_id, sums);
    }
  } else {  // Algorithm 2: block size iteration.
    // We need to handle the remainder elements *within* the last block_size segment if K is not a multiple of block_size
    // Each thread `lane_id` is responsible for elements `k` such that `k % kWarpSize == lane_id`.
    // More accurately, thread `lane_id` handles `a_data` indices `lane_id*8 + 0..7`, `lane_id*8 + k_per_iter + 0..7`, etc.

    for (int block_k_idx = 0; block_k_idx < blocks_per_K; ++block_k_idx) {
      int k_start_block = block_k_idx * block_size;
      int k_end_block = k_start_block + block_size;

      // Get scale/zp for this block
      T scale = b_scale_vec_thread[block_k_idx];
      uint8_t zp;
      if constexpr (has_zero_point) {
        zp = b_zp_vec_thread[block_k_idx];
      } else {
        zp = kDefaultZeroPoint;
      }

      // Iterate within the block, assigning 8 elements per thread per step
      // Each thread `lane_id` handles elements k_start_block + lane_id*8, k_start_block + lane_id*8 + 1, ..., k_start_block + lane_id*8 + 7
      // We only proceed if these elements are within the valid K range and within the current block.
      int current_k_base = k_start_block + lane_offset;  // Base K index for this thread in this block

      if (current_k_base < k_end_block && current_k_base < k) {
        int elements_to_process = min(kElementsPerThreadPerIteration, k - current_k_base);
        elements_to_process = min(elements_to_process, k_end_block - current_k_base);

        // Check if we have a full 8 elements (uint64_t) to load
        if (elements_to_process == kElementsPerThreadPerIteration) {
          const uint8_t* current_b_ptr = b_base_ptr_n + current_k_base;

          const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);
          uint32_t val_low = ptr32[0];
          uint32_t val_high = ptr32[1];
          uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;

          AccumulateEightElements8b(value, scale, zp, a_data + k_start_block, sums);
        } else {
          // Handle partial tail elements (less than 8) is not needed since TryMatMul8Bits ensures k is a multiple of 8.
        }
      }
    }
  }

  if constexpr (!kUseCUB) {
    float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
    // warp reduction
    for (int i = kWarpSize / 2; i > 0; i = i / 2) {
      sum += WARP_SHFL_DOWN(sum, i);
    }

    if (lane_id == 0) {
      output[m_id * n + n_id] = sum;
    }
  } else {
    // The reduction needs to sum the 8 partial sums within each thread first.
    T total_sum_thread = static_cast<T>(0.0f);
#pragma unroll
    for (int i = 0; i < kElementsPerThreadPerIteration; ++i) {
      total_sum_thread += sums[i];
    }

    // Use CUB for efficient warp reduction
    using BlockReduce = cub::WarpReduce<T>;
    __shared__ typename BlockReduce::TempStorage temp_storage[kColsPerThreadBlock];
    total_sum_thread = BlockReduce(temp_storage[warp_id]).Sum(total_sum_thread);

    if (lane_id == 0) {
      output[m_id * n + n_id] = total_sum_thread; // m_id is 0
    }
  }
}

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
    cudaStream_t stream) {
  if (n % kColsPerThreadBlock != 0 || k % kElementsPerThreadPerIteration != 0 || m > 1) {
    return false;
  }
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);
  dim3 threads(GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  int blocks_per_K = (k + block_size - 1) / block_size;
  size_t shared_mem_size = (sizeof(T) + (zero_points != nullptr ? sizeof(uint8_t) : 0)) * blocks_per_K * kColsPerThreadBlock;
  if constexpr (kUseCUB) {
    shared_mem_size += static_cast<size_t>(kColsPerThreadBlock) * sizeof(typename cub::WarpReduce<T>::TempStorage);
  }
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }


#define MatMulFloat8bKernelDispatch(bs)                                                 \
  if (nullptr != zero_points) {                                                         \
    MatMulFloat8bKernel<T, bs, true><<<blocks, threads, shared_mem_size, stream>>>(     \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K); \
  } else {                                                                              \
    MatMulFloat8bKernel<T, bs, false><<<blocks, threads, shared_mem_size, stream>>>(    \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K); \
  }

  if (16 == block_size) {
    MatMulFloat8bKernelDispatch(16);
  } else if (32 == block_size) {
    MatMulFloat8bKernelDispatch(32);
  } else if (64 == block_size) {
    MatMulFloat8bKernelDispatch(64);
  } else if (128 == block_size) {
    MatMulFloat8bKernelDispatch(128);
  } else if (256 == block_size) {
    MatMulFloat8bKernelDispatch(256);
  } else {
    return false;  // Fall back to generic kernel for unsupported block size.
  }

#undef MatMulFloat8bKernelDispatch

  return true;
}

template bool TryMatMul8Bits<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

template bool TryMatMul8Bits<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

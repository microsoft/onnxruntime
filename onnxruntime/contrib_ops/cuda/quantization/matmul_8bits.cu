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
constexpr int kKernelAlgo = 0;  // Choose algorithm here: 0 (unroll), 1 (simple loop), 2 (block size iteration)
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

  float zp_adjust = -scale * float(zp);

  // Extract and dequantize 8 float values
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t q_val = (values_quant >> (i * 8)) & 0xFF;
    v[i] = float(q_val) * scale + zp_adjust;
  }

  // Accumulate using fmaf for potentially better precision/performance
  sums[0] = fmaf(v[0], a_vec_0.x, sums[0]);
  sums[1] = fmaf(v[1], a_vec_0.y, sums[1]);
  sums[2] = fmaf(v[2], a_vec_0.z, sums[2]);
  sums[3] = fmaf(v[3], a_vec_0.w, sums[3]);
  sums[4] = fmaf(v[4], a_vec_1.x, sums[4]);
  sums[5] = fmaf(v[5], a_vec_1.y, sums[5]);
  sums[6] = fmaf(v[6], a_vec_1.z, sums[6]);
  sums[7] = fmaf(v[7], a_vec_1.w, sums[7]);
}

// kernel for 8bits quantized GEMM, i.e., computing C(M, N) = A(M, K) x B(K, N)
// B(K, N) is quantized with 8bits and block_size bs and stored as [N, K/bs, bs]
// kColsPerThreadBlock (C) = 8 is the number of columns each thread block computes per row of A
// kElementsPerThreadPerIteration (E) = 8 is the number of elements each thread computes in one iteration along K
// Constraints: N % C == 0, K % E == 0
// The thread block size is (kWarpSize, C) = (32, 8)
// Grid size is (Ceil(N / C), M)
// Each thread block computes kColsPerThreadBlock columns for a specific row m_id
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloat8bKernel(
    T* output,                    // Base pointer for Output C [M, N]
    const T* a_data,              // Base pointer for A [M, K]
    const uint8_t* b_data_quant,  // Base pointer for B [N, K/bs, bs]
    const T* scales_data,         // Base pointer for scales [N, K/bs]
    const uint8_t* zero_points,   // Base pointer for ZPs [N, K/bs]
    int m,                        // Number of rows in A and Output C
    int n,                        // Number of columns in B and Output C (Constraint: N % C == 0)
    int k,                        // Number of columns in A / rows in B (Constraint: K % E == 0)
    int blocks_per_K) {           // blocks_per_K = K/bs

  const int n_block_id = blockIdx.x;  // Block column index in the range of [0, Ceil(N / C))
  const int m_id = blockIdx.y;        // Block row index (identifies the row of A and C) [0, M)

  // Check if this block is needed for the M dimension
  if (m_id >= m) return;

  const int lane_id = threadIdx.x;                            // Thread index in warp (0..31)
  const int warp_id = threadIdx.y;                            // Warp index 0..7 in the range of [0, C-1)
  const int n_block_head = n_block_id * kColsPerThreadBlock;  // Head column index for this block [0, N)
  const int n_id = n_block_head + warp_id;                    // Global output column index this warp computes

  // Ensure n_id does not go out of bounds (already checked by TryMatMul8Bits, but safer)
  if (n_id >= n) return;

  extern __shared__ char shared_buffer[];

  // Load scales to shared_buffer
  T* b_scale_vec_shared = (T*)shared_buffer;
  for (int i = threadIdx.y * kWarpSize + threadIdx.x; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    // Boundary check needed if N is not perfectly divisible by kColsPerThreadBlock * blocks_per_K,
    // though the N constraint N % C == 0 helps simplify this for scales/ZPs.
    int current_n = n_block_head + (i / blocks_per_K);
    int current_k_block = i % blocks_per_K;
    if (current_n < n) {  // Check if the column is valid
      b_scale_vec_shared[i] = scales_data[current_n * blocks_per_K + current_k_block];
    }
  }

  // Load zero points if they exist (logic remains the same, depends on n_block_id)
  [[maybe_unused]] uint8_t* b_zp_vec_shared = nullptr;
  [[maybe_unused]] const uint8_t* b_zp_vec_thread = nullptr;  // Thread's ZP pointer
  if constexpr (has_zero_point) {
    b_zp_vec_shared = reinterpret_cast<uint8_t*>(b_scale_vec_shared + kColsPerThreadBlock * blocks_per_K);
    for (int i = threadIdx.y * kWarpSize + threadIdx.x; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
      int current_n = n_block_head + (i / blocks_per_K);
      int current_k_block = i % blocks_per_K;
      if (current_n < n) {  // Check if the column is valid
        b_zp_vec_shared[i] = zero_points[current_n * blocks_per_K + current_k_block];
      }
    }
    b_zp_vec_thread = b_zp_vec_shared + warp_id * blocks_per_K;
  }

  __syncthreads();  // Ensure scales and ZPs are loaded

  // Point a_data to the correct row based on m_id
  const T* a_row_data = a_data + static_cast<int64_t>(m_id) * k;

  // Each thread calculates its part of the dot product along K.
  // Point to the start of the elements this thread is responsible for in the current row of A.
  const int lane_offset = lane_id * kElementsPerThreadPerIteration;
  const T* a_thread_data_base = a_row_data + lane_offset;  // Base pointer for this thread in row m_id

  // Pointer to the start of B data for the specific column n_id this warp handles.
  // Layout of B is [N, K/bs, bs].
  const uint8_t* b_base_ptr_n = b_data_quant + static_cast<int64_t>(n_id) * blocks_per_K * block_size;

  // Pointer to the start of scales for the specific column (n_id) this warp handles (from shared mem).
  const T* b_scale_vec_thread = b_scale_vec_shared + warp_id * blocks_per_K;

  T sums[kElementsPerThreadPerIteration] = {static_cast<T>(0.0f)};  // Initialize sums to zero

  if constexpr (kKernelAlgo == 0 || kKernelAlgo == 1) {
    // Note that k_per_iter (typical value is 256) is multiple of block_size (typical value is 16, 32, 64, 128, or 256).
    constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

    int k_id = 0;
    // Pointer to B data for this thread's starting element in K, for column n_id.
    // B layout: [N, K/bs, bs]. Access is effectively [n_id, k_block, k_within_block]
    // Pointer for thread should start at its `lane_offset` within the K dimension for column `n_id`.
    const uint8_t* b_data_quant_thread = b_base_ptr_n + lane_offset;

    if constexpr (kKernelAlgo == 0) {                // Algorithm 0: Unrolling
      int k_start_block = lane_offset / block_size;  // Block index in K for thread start

#define UnRollReduction(unroll_size)                                                                   \
  do {                                                                                                 \
    constexpr int kUnroll = unroll_size;                                                               \
    constexpr int kElementsPerUnrollIter = k_per_iter * kUnroll;                                       \
    for (; k_id + kElementsPerUnrollIter <= k; k_id += kElementsPerUnrollIter) {                       \
      _Pragma("unroll") for (int i = 0; i < kUnroll; ++i) {                                            \
        const uint8_t* current_b_ptr = b_data_quant_thread + k_id + i * k_per_iter;                    \
        const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);                      \
        uint32_t val_low = ptr32[0];                                                                   \
        uint32_t val_high = ptr32[1];                                                                  \
        uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;                            \
        /* Requires k_per_iter % block_size == 0 */                                                    \
        int current_meta_k = k_start_block + (k_id / block_size) + i * (k_per_iter / block_size);      \
        T scale = b_scale_vec_thread[current_meta_k];                                                  \
        uint8_t zp = kDefaultZeroPoint;                                                                \
        if constexpr (has_zero_point) {                                                                \
          zp = b_zp_vec_thread[current_meta_k];                                                        \
        }                                                                                              \
        /* Pass pointer to A for the current k segment */                                              \
        AccumulateEightElements8b(value, scale, zp, a_thread_data_base + k_id + i * k_per_iter, sums); \
      }                                                                                                \
    }                                                                                                  \
  } while (false)

      UnRollReduction(16);
      UnRollReduction(4);
      UnRollReduction(1);

#undef UnRollReduction
    } else {  // Algorithm 1: Simple loop
      for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
        const uint8_t* current_b_ptr = b_data_quant_thread + k_id;
        const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);
        uint32_t val_low = ptr32[0];
        uint32_t val_high = ptr32[1];
        uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;

        int current_meta_k = (lane_offset + k_id) / block_size;
        T scale = b_scale_vec_thread[current_meta_k];
        uint8_t zp = kDefaultZeroPoint;
        if constexpr (has_zero_point) {
          zp = b_zp_vec_thread[current_meta_k];
        }
        /* Pass pointer to A for the current k segment */
        AccumulateEightElements8b(value, scale, zp, a_thread_data_base + k_id, sums);
      }
    }

    // Handle the tail elements (less than k_per_iter) if k is not multiple of k_per_iter
    // Since k % kElementsPerThreadPerIteration == 0 is enforced, the tail is simpler.
    // Each thread processes its remaining elements if its start offset is < k.
    if (lane_offset + k_id < k) {  // Check if this thread has any elements left
                                   // Check if the *start* k for this final iteration is valid
      const uint8_t* current_b_ptr = b_data_quant_thread + k_id;
      const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);
      uint32_t val_low = ptr32[0];
      uint32_t val_high = ptr32[1];
      uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;

      int current_meta_k = (lane_offset + k_id) / block_size;
      T scale = b_scale_vec_thread[current_meta_k];
      uint8_t zp = kDefaultZeroPoint;
      if constexpr (has_zero_point) {
        zp = b_zp_vec_thread[current_meta_k];
      }
      /* Pass pointer to A for the current k segment */
      AccumulateEightElements8b(value, scale, zp, a_thread_data_base + k_id, sums);
    }
  } else {  // Algorithm 2: block size iteration.
    for (int block_k_idx = 0; block_k_idx < blocks_per_K; ++block_k_idx) {
      int k_start_block = block_k_idx * block_size;
      // B data pointer for the start of this block, for column n_id
      const uint8_t* b_block_ptr_n = b_base_ptr_n + k_start_block;

      // Get scale/zp for this block (already loaded for the warp)
      T scale = b_scale_vec_thread[block_k_idx];
      uint8_t zp = kDefaultZeroPoint;
      if constexpr (has_zero_point) {
        zp = b_zp_vec_thread[block_k_idx];
      }

      // Each thread `lane_id` handles elements starting at `lane_offset` within the K dimension.
      // Calculate the base K index for this thread *within the current block*.
      int k_base_in_block = k_start_block + lane_offset;

      // Check if the *start* of the 8 elements this thread would process is within the current block
      // AND within the bounds of K dimension.
      if (k_base_in_block < k_start_block + block_size && k_base_in_block < k) {
        // Since K % 8 == 0 is enforced, we don't need partial element handling.
        // We are guaranteed to process 8 elements if the start is valid.
        const uint8_t* current_b_ptr = b_block_ptr_n + lane_offset;  // Offset within the block

        const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);
        uint32_t val_low = ptr32[0];
        uint32_t val_high = ptr32[1];
        uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;

        // Pointer to A data corresponding to this thread's elements within this block
        const T* current_a_ptr = a_row_data + k_base_in_block;  // Use a_row_data + absolute k index

        AccumulateEightElements8b(value, scale, zp, current_a_ptr, sums);
      }
    }
  }

  // Sum the 8 partial sums within each thread first.
  T total_sum_thread = static_cast<T>(0.0f);
#pragma unroll
  for (int i = 0; i < kElementsPerThreadPerIteration; ++i) {
    total_sum_thread += sums[i];
  }

  if constexpr (!kUseCUB) {
    for (int i = kWarpSize / 2; i > 0; i = i / 2) {
      total_sum_thread += __shfl_down_sync(0xFFFFFFFF, total_sum_thread, i);
    }

    if (lane_id == 0) {
      // Calculate output index: output[m_id, n_id]
      output[static_cast<int64_t>(m_id) * n + n_id] = total_sum_thread;
    }
  } else {
    // Use CUB for efficient warp reduction
    using BlockReduce = cub::WarpReduce<T>;

    // Shared memory for CUB reduction storage (one per warp)
    __shared__ typename BlockReduce::TempStorage temp_storage[kColsPerThreadBlock];
    total_sum_thread = BlockReduce(temp_storage[warp_id]).Sum(total_sum_thread);

    if (lane_id == 0) {
      // Write the final result for the element C[m_id, n_id]
      output[static_cast<int64_t>(m_id) * n + n_id] = total_sum_thread;
    }
  }
}

template <class T>
bool TryMatMul8Bits(
    T* output,                    // Output C [M, N]
    const T* a_data,              // Input A [M, K]
    const uint8_t* b_data_quant,  // Input B Quantized [N, K/bs, bs]
    const T* scales_data,         // Scales [N, K/bs]
    const uint8_t* zero_points,   // Zero Points [N, K/bs] (can be nullptr)
    int m,                        // Rows of A and C (M >= 1)
    int n,                        // Columns of B and C
    int k,                        // Columns of A / Rows of B
    int block_size,               // Quantization block size for B
    size_t shared_mem_per_block,  // Available shared memory
    cudaStream_t stream) {
  // Constraints Check
  // N must be a multiple of kColsPerThreadBlock (8) for warps to align with columns.
  // K must be a multiple of kElementsPerThreadPerIteration (8) for full uint64_t reads/processing per thread iter.
  if (n % kColsPerThreadBlock != 0 || k % kElementsPerThreadPerIteration != 0) {
    return false;
  }

  // Also check if block_size is valid and K is a multiple of block_size if required by kernel logic assumptions
  // (The provided code seems to handle K not being multiple of block_size via blocks_per_K,
  // but algo 0 assumes k_per_iter % block_size == 0 for simple indexing).
  // Ensure k_per_iter (256) is multiple of block_size for algo 0.
  if constexpr (kKernelAlgo == 0) {
    constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;
    if (k_per_iter % block_size != 0) {
      // Algorithm 0 with the current scale/zp indexing requires this.
      // Could add more complex indexing if needed, or fallback to other algos.
      return false;
    }
  }

  // Grid and Thread Block Configuration
  dim3 threads(kWarpSize, kColsPerThreadBlock);  // (32, 8)
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);

  int blocks_per_K = (k + block_size - 1) / block_size;  // K / block_size rounded up
  // Shared memory needed for scales and optionally zero points for the columns handled by the block
  size_t shared_mem_size = (sizeof(T) + (zero_points != nullptr ? sizeof(uint8_t) : 0)) * blocks_per_K * kColsPerThreadBlock;

  // Add shared memory for CUB reduction storage if used
  if constexpr (kUseCUB) {
    shared_mem_size += static_cast<size_t>(kColsPerThreadBlock) * sizeof(typename cub::WarpReduce<T>::TempStorage);
  }

  // Check if required shared memory exceeds limits
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

  // Macro simplifies dispatching for different block sizes and presence of zero_points
#define MatMulFloat8bKernelDispatch(bs)                                                             \
  if (nullptr != zero_points) {                                                                     \
    /* Launch kernel with zero points */                                                            \
    MatMulFloat8bKernel<T, bs, true><<<blocks, threads, shared_mem_size, stream>>>(                 \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);             \
  } else {                                                                                          \
    /* Launch kernel without zero points */                                                         \
    MatMulFloat8bKernel<T, bs, false><<<blocks, threads, shared_mem_size, stream>>>(                \
        output, a_data, b_data_quant, scales_data, nullptr /*zero_points*/, m, n, k, blocks_per_K); \
  }

  // Dispatch based on block_size value
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
    // Unsupported block size.
    return false;
  }

#undef MatMulFloat8bKernelDispatch

  return true;
}

// Template instantiations
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

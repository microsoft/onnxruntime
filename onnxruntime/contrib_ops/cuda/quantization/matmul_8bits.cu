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
constexpr bool kUseFloatInPartialSum = false;  // Use float to accumulate partial sum of 8 half elements in a thread. Default is false like 4 bits kernel.

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  // --- Dequantization Setup ---
  // Convert scale and zero point to half format suitable for half2 operations
  half2 scale_h2 = __half2half2(scale);  // Broadcast scale to half2
  half zp_h = __ushort2half_rn(zp);      // Convert uint8 zp to half
  half2 zp_h2 = __half2half2(zp_h);      // Broadcast zp to half2

  // --- Extract 8 uint8_t values from the 64-bit input ---
  uint8_t q[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    q[i] = (values_quant >> (i * 8)) & 0xFF;
  }

  // --- Dequantize 8 values into 4 half2 vectors: b_vec = (q - zp) * scale ---
  // Convert uint8 q values to half2 vectors {q0,q1}, {q2,q3}, {q4,q5}, {q6,q7}
  half2 q_01 = __halves2half2(__ushort2half_rn(q[0]), __ushort2half_rn(q[1]));
  half2 q_23 = __halves2half2(__ushort2half_rn(q[2]), __ushort2half_rn(q[3]));
  half2 q_45 = __halves2half2(__ushort2half_rn(q[4]), __ushort2half_rn(q[5]));
  half2 q_67 = __halves2half2(__ushort2half_rn(q[6]), __ushort2half_rn(q[7]));

  // Calculate q - zp
  half2 diff_01 = __hsub2(q_01, zp_h2);
  half2 diff_23 = __hsub2(q_23, zp_h2);
  half2 diff_45 = __hsub2(q_45, zp_h2);
  half2 diff_67 = __hsub2(q_67, zp_h2);

  // Calculate b_vec = (q - zp) * scale
  half2 b_vec0 = __hmul2(diff_01, scale_h2);  // {b0, b1}
  half2 b_vec1 = __hmul2(diff_23, scale_h2);  // {b2, b3}
  half2 b_vec2 = __hmul2(diff_45, scale_h2);  // {b4, b5}
  half2 b_vec3 = __hmul2(diff_67, scale_h2);  // {b6, b7}

  // --- Load Input A (8 half values as 4 half2 vectors) ---
  // Directly cast 'a' pointer to read half2 vectors.
  // This assumes 'a' is properly aligned for half2 reads.
  const half2* a_half2 = reinterpret_cast<const half2*>(a);
  half2 a_vec0 = a_half2[0];  // {a0, a1}
  half2 a_vec1 = a_half2[1];  // {a2, a3}
  half2 a_vec2 = a_half2[2];  // {a4, a5}
  half2 a_vec3 = a_half2[3];  // {a6, a7}

  // --- Accumulate: sums += a * b_vec using half2 FMA ---
  // Cast sums pointer to half2* for vectorized accumulation.
  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = __hfma2(a_vec0, b_vec0, sums_half2[0]);  // {s0+=a0*b0, s1+=a1*b1}
  sums_half2[1] = __hfma2(a_vec1, b_vec1, sums_half2[1]);  // {s2+=a2*b2, s3+=a3*b3}
  sums_half2[2] = __hfma2(a_vec2, b_vec2, sums_half2[2]);  // {s4+=a4*b4, s5+=a5*b5}
  sums_half2[3] = __hfma2(a_vec3, b_vec3, sums_half2[3]);  // {s6+=a6*b6, s7+=a7*b7}
}

// --- Keep Original Float Version ---
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
      b_scale_vec_shared[i] = scales_data[static_cast<int64_t>(current_n) * blocks_per_K + current_k_block];
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
        b_zp_vec_shared[i] = zero_points[static_cast<int64_t>(current_n) * blocks_per_K + current_k_block];
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
        /* Assume alignment allows uint64_t load */                                                    \
        uint64_t value = *reinterpret_cast<const uint64_t*>(current_b_ptr);                            \
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
        uint64_t value = *reinterpret_cast<const uint64_t*>(current_b_ptr);

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
      const uint8_t* current_b_ptr = b_data_quant_thread + k_id;
      uint64_t value = *reinterpret_cast<const uint64_t*>(current_b_ptr);

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
      int k_offset_in_block = lane_offset;  // Offset relative to block start

      // Check if the *start* of the 8 elements this thread would process is within the current block
      // AND within the bounds of K dimension.
      // Since K % 8 == 0 and thread handles 8 elements, if the thread's work starts within K,
      // it finishes within K. We only need to check if the *block* has enough elements
      // for this thread's starting offset.
      if (k_offset_in_block < block_size && k_start_block + k_offset_in_block < k) {
        // Calculate absolute K index for A pointer
        int k_abs_idx = k_start_block + k_offset_in_block;

        // Since K % 8 == 0 is enforced, we don't need partial element handling.
        // We are guaranteed to process 8 elements if the start is valid.
        const uint8_t* current_b_ptr = b_block_ptr_n + k_offset_in_block;  // Offset within the block
        uint64_t value = *reinterpret_cast<const uint64_t*>(current_b_ptr);

        // Pointer to A data corresponding to this thread's elements within this block
        const T* current_a_ptr = a_row_data + k_abs_idx;  // Use a_row_data + absolute k index

        AccumulateEightElements8b(value, scale, zp, current_a_ptr, sums);
      }
    }
  }

  // Sum the 8 partial sums within each thread first.
  float total_sum_thread = 0.0f;

  if constexpr (std::is_same_v<T, half>) {
    if constexpr (kUseFloatInPartialSum) {
// Convert 8 elements to float, then accumulate.
#pragma unroll
      for (int i = 0; i < kElementsPerThreadPerIteration; ++i) {
        total_sum_thread += __half2float(sums[i]);
      }
    } else {
      // Accumulate 8 elements in half, then convert sum to float once.
      T temp_sum = static_cast<T>(0.0f);
#pragma unroll
      for (int i = 0; i < kElementsPerThreadPerIteration; ++i) {
        temp_sum += sums[i];
      }
      total_sum_thread = __half2float(temp_sum);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kElementsPerThreadPerIteration; ++i) {
      total_sum_thread += sums[i];
    }
  }

  if constexpr (!kUseCUB) {
    for (int i = kWarpSize / 2; i > 0; i = i / 2) {
      total_sum_thread += __shfl_down_sync(0xFFFFFFFF, total_sum_thread, i);
    }

    if (lane_id == 0) {
      // Calculate output index: output[m_id, n_id]
      output[static_cast<int64_t>(m_id) * n + n_id] = static_cast<T>(total_sum_thread);
    }
  } else {
    // Use CUB for efficient warp reduction
    using BlockReduce = cub::WarpReduce<float>;

    // Shared memory for CUB reduction storage (one per warp)
    __shared__ typename BlockReduce::TempStorage temp_storage[kColsPerThreadBlock];
    total_sum_thread = BlockReduce(temp_storage[warp_id]).Sum(total_sum_thread);

    if (lane_id == 0) {
      // Write the final result for the element C[m_id, n_id]
      output[static_cast<int64_t>(m_id) * n + n_id] = static_cast<T>(total_sum_thread);
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

  // Ensure k_per_iter is multiple of block_size for algo 0.
  if constexpr (kKernelAlgo == 0) {
    constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;
    if (k_per_iter % block_size != 0) {
      return false;
    }
  }

  if constexpr (kKernelAlgo == 1 || kKernelAlgo == 2) {
    if (k % block_size != 0) {
      // The indexing `(lane_offset + k_id) / block_size` in Algo 1 and the block iteration
      // in Algo 2 rely on K being compatible with block_size for correct scale/zp lookup.
      // While blocks_per_K handles rounding up, the core loops assume alignment.
      // If K is not multiple of block_size, the last block is partial, potentially
      // causing issues with scale/zp indexing.
      // Let's enforce K % block_size == 0 for simplicity/correctness guarantee here.
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
    shared_mem_size += static_cast<size_t>(kColsPerThreadBlock) * sizeof(typename cub::WarpReduce<float>::TempStorage);
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

  // Here we do not use cudaGetLastError() to check kernel launch errors. That will be done later.
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

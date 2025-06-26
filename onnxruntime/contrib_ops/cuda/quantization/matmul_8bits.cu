// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// --- Kernel Configuration Constants ---
constexpr int kColsPerThreadBlock = 8;             // Number of columns (N dimension) processed per thread block
constexpr int kElementsPerThreadPerIteration = 8;  // Number of elements (K dimension) processed per thread per iteration
constexpr int kWarpSize = GPU_WARP_SIZE;           // Typically 32
constexpr uint8_t kDefaultZeroPoint = 128;         // Default zero point if not provided

// --- Device Function: Accumulate 8 Elements (half precision) ---
// Dequantizes 8 uint8_t values and accumulates the result with 8 half values from A into float sums.
// sums_f += A_h * dequant(B_quant)
__device__ __forceinline__ void AccumulateEightElements8b(
    uint64_t values_quant,  // 8 packed uint8_t values from B
    half scale,             // Dequantization scale for this block
    uint8_t zp,             // Dequantization zero point for this block
    const half* a,          // Pointer to 8 half values from A
    float* sums_f) {        // Pointer to 8 partial sums (float)

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
  // --- Dequantization Setup ---
  half2 scale_h2 = __half2half2(scale);  // Broadcast scale
  half zp_h = __ushort2half_rn(zp);      // Convert zp to half
  half2 zp_h2 = __half2half2(zp_h);      // Broadcast zp_h

  // --- Extract 8 uint8_t values ---
  uint8_t q[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    q[i] = (values_quant >> (i * 8)) & 0xFF;
  }

  // --- Dequantize 8 values into 4 half2 vectors: b_vec = (q - zp) * scale ---
  half2 q_01 = __halves2half2(__ushort2half_rn(q[0]), __ushort2half_rn(q[1]));
  half2 q_23 = __halves2half2(__ushort2half_rn(q[2]), __ushort2half_rn(q[3]));
  half2 q_45 = __halves2half2(__ushort2half_rn(q[4]), __ushort2half_rn(q[5]));
  half2 q_67 = __halves2half2(__ushort2half_rn(q[6]), __ushort2half_rn(q[7]));

  half2 diff_01 = __hsub2(q_01, zp_h2);
  half2 diff_23 = __hsub2(q_23, zp_h2);
  half2 diff_45 = __hsub2(q_45, zp_h2);
  half2 diff_67 = __hsub2(q_67, zp_h2);

  half2 b_vec0 = __hmul2(diff_01, scale_h2);  // {b0, b1}
  half2 b_vec1 = __hmul2(diff_23, scale_h2);  // {b2, b3}
  half2 b_vec2 = __hmul2(diff_45, scale_h2);  // {b4, b5}
  half2 b_vec3 = __hmul2(diff_67, scale_h2);  // {b6, b7}

  // --- Load Input A (8 half values as 4 half2 vectors) ---
  const half2* a_half2 = reinterpret_cast<const half2*>(a);
  half2 a_vec0 = a_half2[0];  // {a0, a1}
  half2 a_vec1 = a_half2[1];  // {a2, a3}
  half2 a_vec2 = a_half2[2];  // {a4, a5}
  half2 a_vec3 = a_half2[3];  // {a6, a7}

  // Convert half2 inputs to float2 for fmaf operations on sums_f
  float2 a_vec0_f = __half22float2(a_vec0);
  float2 a_vec1_f = __half22float2(a_vec1);
  float2 a_vec2_f = __half22float2(a_vec2);
  float2 a_vec3_f = __half22float2(a_vec3);

  float2 b_vec0_f = __half22float2(b_vec0);
  float2 b_vec1_f = __half22float2(b_vec1);
  float2 b_vec2_f = __half22float2(b_vec2);
  float2 b_vec3_f = __half22float2(b_vec3);

  sums_f[0] = fmaf(a_vec0_f.x, b_vec0_f.x, sums_f[0]);
  sums_f[1] = fmaf(a_vec0_f.y, b_vec0_f.y, sums_f[1]);
  sums_f[2] = fmaf(a_vec1_f.x, b_vec1_f.x, sums_f[2]);
  sums_f[3] = fmaf(a_vec1_f.y, b_vec1_f.y, sums_f[3]);
  sums_f[4] = fmaf(a_vec2_f.x, b_vec2_f.x, sums_f[4]);
  sums_f[5] = fmaf(a_vec2_f.y, b_vec2_f.y, sums_f[5]);
  sums_f[6] = fmaf(a_vec3_f.x, b_vec3_f.x, sums_f[6]);
  sums_f[7] = fmaf(a_vec3_f.y, b_vec3_f.y, sums_f[7]);

#else  // older GPUs of compute capability < 5.3, which lacks native half support.
  float scale_f = __half2float(scale);
  float zp_f = static_cast<float>(zp);

  float b_dequant[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t q = (values_quant >> (i * 8)) & 0xFF;
    b_dequant[i] = (static_cast<float>(q) - zp_f) * scale_f;
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float a_f = __half2float(a[i]);
    sums_f[i] = fmaf(a_f, b_dequant[i], sums_f[i]);
  }
#endif
}

// --- Device Function: Accumulate 8 Elements (float precision) ---
// Dequantizes 8 uint8_t values and accumulates the result with 8 float values from A.
// sums_f += A_f * dequant(B_quant)
__device__ __forceinline__ void AccumulateEightElements8b(
    uint64_t values_quant,  // 8 packed uint8_t values from B
    float scale,            // Dequantization scale for this block
    uint8_t zp,             // Dequantization zero point for this block
    const float* a,         // Pointer to 8 float values from A
    float* sums_f) {        // Pointer to 8 partial sums (float)

  // Load A using float4 for potentially better memory bandwidth
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  // Precompute scale * (-zp) adjustment
  float zp_adjust = -scale * float(zp);

  // Extract, dequantize, and accumulate 8 float values
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t q_val = (values_quant >> (i * 8)) & 0xFF;
    // Dequantize: float(q_val) * scale - scale * float(zp) = float(q_val) * scale + zp_adjust
    v[i] = float(q_val) * scale + zp_adjust;
  }

  // Accumulate using fmaf (fused multiply-add)
  sums_f[0] = fmaf(v[0], a_vec_0.x, sums_f[0]);
  sums_f[1] = fmaf(v[1], a_vec_0.y, sums_f[1]);
  sums_f[2] = fmaf(v[2], a_vec_0.z, sums_f[2]);
  sums_f[3] = fmaf(v[3], a_vec_0.w, sums_f[3]);
  sums_f[4] = fmaf(v[4], a_vec_1.x, sums_f[4]);
  sums_f[5] = fmaf(v[5], a_vec_1.y, sums_f[5]);
  sums_f[6] = fmaf(v[6], a_vec_1.z, sums_f[6]);
  sums_f[7] = fmaf(v[7], a_vec_1.w, sums_f[7]);
}


// --- Device Function: Accumulate 8 Elements (bfloat16 precision) ---
// Dequantizes 8 uint8_t values and accumulates the result with 8 nv_bfloat16 values from A.
// sums_f += A_bf16 * dequant(B_quant)
__device__ __forceinline__ void AccumulateEightElements8b(
    uint64_t values_quant,   // 8 packed uint8_t values from B
    nv_bfloat16 scale,     // Dequantization scale for this block
    uint8_t zp,              // Dequantization zero point for this block
    const nv_bfloat16* a,  // Pointer to 8 nv_bfloat16 values from A
    float* sums_f) {         // Pointer to 8 partial sums (float)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float scale_f = __bfloat162float(scale);
  float zp_f = static_cast<float>(zp);

  float zp_adjust = -scale_f * zp_f;

  float a_f[8];
  float b_dequant_f[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    a_f[i] = __bfloat162float(a[i]);
    uint8_t q_val = (values_quant >> (i * 8)) & 0xFF;
    b_dequant_f[i] = static_cast<float>(q_val) * scale_f + zp_adjust;
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sums_f[i] = fmaf(a_f[i], b_dequant_f[i], sums_f[i]);
  }
#endif
}

// --- CUDA Kernel: MatMulFloat8bKernel (Optimized for m=1) ---
// Computes C(1, N) = A(1, K) x B(K, N)
// B(K, N) is quantized with 8 bits and block_size bs, stored as [N, K/bs, bs]
//
// Template Parameters:
//   T: Data type for A and C (float, half, or nv_bfloat16)
//   block_size: Quantization block size for B
//   has_zero_point: Boolean indicating if zero points are provided
//
// Grid size: (Ceil(N / kColsPerThreadBlock), 1)
// Block size: (kWarpSize, kColsPerThreadBlock) = (32, 8)
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloat8bKernelM1(
    T* output,                    // Output C [1, N] (effectively [N])
    const T* a_data,              // Input A [1, K] (effectively [K])
    const uint8_t* b_data_quant,  // Quantized Input B [N, K/bs, bs]
    const T* scales_data,         // Scales [N, K/bs]
    const uint8_t* zero_points,   // Zero Points [N, K/bs] (optional)
    int n,                        // Columns in B and C (Constraint: N % kColsPerThreadBlock == 0)
    int k,                        // Columns in A / Rows in B (Constraint: K % kElementsPerThreadPerIteration == 0)
    int blocks_per_K) {           // K / block_size (rounded up)

  // --- Thread Indexing ---
  const int n_block_id = blockIdx.x;  // Block column index [0, Ceil(N / kColsPerThreadBlock))

  const int lane_id = threadIdx.x;  // Thread index in warp (0..31)
  const int warp_id = threadIdx.y;  // Warp index in block (0..kColsPerThreadBlock-1)

  // Calculate the starting column index (n_id) this warp is responsible for
  const int n_block_head = n_block_id * kColsPerThreadBlock;
  const int n_id = n_block_head + warp_id;  // Global output column index for this warp

  // Boundary check for the column index (safety check, though N % kColsPerThreadBlock==0 is enforced)
  if (n_id >= n) return;

  // --- Shared Memory Allocation ---
  extern __shared__ char shared_buffer[];
  // Shared memory for scales
  T* b_scale_vec_shared = reinterpret_cast<T*>(shared_buffer);
  // Shared memory for zero points (if used) immediately after scales
  [[maybe_unused]] uint8_t* b_zp_vec_shared = nullptr;  // Initialize to avoid unused warning
  if constexpr (has_zero_point) {
    b_zp_vec_shared = reinterpret_cast<uint8_t*>(b_scale_vec_shared + kColsPerThreadBlock * blocks_per_K);
  }

  // --- Load Scales and Zero Points into Shared Memory ---
  // Each thread loads a portion of the scales/ZPs for the columns handled by this block
  for (int i = threadIdx.y * kWarpSize + threadIdx.x;  // Linear thread index within the block
       i < kColsPerThreadBlock * blocks_per_K;         // Total elements to load for the block
       i += kColsPerThreadBlock * kWarpSize) {         // Stride by total threads in block
    int current_n_offset = i / blocks_per_K;           // Column offset within the block [0, kColsPerThreadBlock-1]
    int current_k_block = i % blocks_per_K;            // K block index [0, blocks_per_K-1]
    int current_n = n_block_head + current_n_offset;   // Global N index

    if (current_n < n) {  // Boundary check for N
      // Calculate global index into scales/ZPs: N * blocks_per_K + k_block
      int64_t scale_zp_idx = static_cast<int64_t>(current_n) * blocks_per_K + current_k_block;
      // Load scale
      b_scale_vec_shared[i] = scales_data[scale_zp_idx];
      // Load zero point if applicable
      if constexpr (has_zero_point) {
        b_zp_vec_shared[i] = zero_points[scale_zp_idx];
      }
    }
  }

  __syncthreads();  // Ensure all scales and ZPs are loaded before proceeding

  // --- Pointers Setup ---
  // A data pointer (since m=1, no row offset needed)
  const T* a_row_data = a_data;

  // Each thread calculates its part of the dot product along K.
  // Point to the start of the elements this thread is responsible for in A.
  const int lane_offset = lane_id * kElementsPerThreadPerIteration;  // Offset in K for this thread
  const T* a_thread_data_base = a_row_data + lane_offset;            // Base pointer in A for this thread

  // Base pointer to B data for the specific column n_id this warp handles.
  // Layout of B is [N, K/bs, bs].
  const uint8_t* b_base_ptr_n = b_data_quant + static_cast<int64_t>(n_id) * blocks_per_K * block_size;

  // Pointer to the start of scales for column n_id (from shared memory)
  const T* b_scale_vec_thread = b_scale_vec_shared + warp_id * blocks_per_K;

  // Pointer to the start of zero points for column n_id (from shared memory, if used)
  [[maybe_unused]] const uint8_t* b_zp_vec_thread = nullptr;  // Initialize to avoid unused warning
  if constexpr (has_zero_point) {
    b_zp_vec_thread = b_zp_vec_shared + warp_id * blocks_per_K;
  }

  // --- Accumulation ---
  // Initialize partial sums for this thread to zero. Always accumulate in float for precision.
  float sums[kElementsPerThreadPerIteration] = {0.0f};

  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;  // Elements processed per warp per iteration (e.g., 32*8 = 256)
  int k_id = 0;                                                           // Current position along the K dimension

  // Pointer to B data for this thread's starting element in K, for column n_id.
  const uint8_t* b_data_quant_thread = b_base_ptr_n + lane_offset;

  for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
    const uint8_t* current_b_ptr = b_data_quant_thread + k_id;
    uint64_t value = *reinterpret_cast<const uint64_t*>(current_b_ptr);

    int current_meta_k = (lane_offset + k_id) / block_size;
    T scale = b_scale_vec_thread[current_meta_k];
    uint8_t zp = kDefaultZeroPoint;
    if constexpr (has_zero_point) {
      zp = b_zp_vec_thread[current_meta_k];
    }

    AccumulateEightElements8b(value, scale, zp, a_thread_data_base + k_id, sums);
  }

  // Handle the tail elements along K dimension for this thread.
  // This loop handles the final iteration if k is not a multiple of k_per_iter.
  // Since K % kElementsPerThreadPerIteration == 0 is enforced, each thread
  // processes a full set of kElementsPerThreadPerIteration if it has work left.
  if (lane_offset + k_id < k) {  // Check if this thread has remaining elements
    const uint8_t* current_b_ptr = b_data_quant_thread + k_id;
    uint64_t value = *reinterpret_cast<const uint64_t*>(current_b_ptr);

    // Calculate k_block index for the tail part
    int current_meta_k = (lane_offset + k_id) / block_size;
    T scale = b_scale_vec_thread[current_meta_k];
    uint8_t zp = kDefaultZeroPoint;
    if constexpr (has_zero_point) {
      zp = b_zp_vec_thread[current_meta_k];
    }
    // Pointer to A data for the tail part
    const T* current_a_ptr = a_thread_data_base + k_id;
    // Perform dequantization and accumulation
    AccumulateEightElements8b(value, scale, zp, current_a_ptr, sums);
  }

  // --- Intra-Thread Reduction ---
  // Sum the kElementsPerThreadPerIteration partial sums within each thread.
  // Always accumulate in float to avoid precision loss.
  float total_sum_thread = 0.0f;

#pragma unroll
  for (int i = 0; i < kElementsPerThreadPerIteration; ++i) {
    total_sum_thread += sums[i];
  }

  // --- Inter-Thread Reduction (Warp Level) ---
  // Use CUB for efficient and robust warp reduction
  using BlockReduce = cub::WarpReduce<float>;
  // Allocate shared memory for CUB temporary storage (one per warp)
  __shared__ typename BlockReduce::TempStorage temp_storage[kColsPerThreadBlock];

  // Perform warp-level sum reduction.
  total_sum_thread = BlockReduce(temp_storage[warp_id]).Sum(total_sum_thread);

  // Lane 0 of each warp writes the final reduced sum to global memory
  if (lane_id == 0) {
    // Write result (cast back to T)
    output[n_id] = static_cast<T>(total_sum_thread);
  }
}

// --- Host Function: TryMatMul8Bits (Optimized for m=1) ---
// Launches the MatMulFloat8bKernelM1 kernel if constraints are met.
// Enforces m == 1.
template <class T>
bool TryMatMul8Bits(
    T* output,                    // Output C [1, N]
    const T* a_data,              // Input A [1, K]
    const uint8_t* b_data_quant,  // Input B Quantized [N, K/bs, bs]
    const T* scales_data,         // Scales [N, K/bs]
    const uint8_t* zero_points,   // Zero Points [N, K/bs] (can be nullptr)
    int m,                        // Rows of A and C (MUST be 1)
    int n,                        // Columns of B and C
    int k,                        // Columns of A / Rows of B
    int block_size,               // Quantization block size for B
    size_t shared_mem_per_block,  // Available shared memory per block
    cudaStream_t stream) {
  // Constraints Check
  // m must be 1 (since this kernel is optimized for m=1)
  // N must be a multiple of kColsPerThreadBlock (8) for warps to align with columns.
  // K must be a multiple of kElementsPerThreadPerIteration (8) for full uint64_t reads/processing.
  if (m != 1 || n % kColsPerThreadBlock != 0 || k % kElementsPerThreadPerIteration != 0) {
    return false;
  }

  // Ensure k_per_iter (kWarpSize * kElementsPerThreadPerIteration) is multiple of block_size.
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;
  if (k_per_iter % block_size != 0) {
    // This constraint is needed for the scale/zp indexing calculation within the unrolled loop.
    return false;
  }

  // K must be a multiple of block_size for correct scale/zp lookup within blocks.
  // While blocks_per_K handles rounding up, the kernel logic assumes full blocks for indexing.
  if (k % block_size != 0) {
    return false;
  }

  // --- Grid and Thread Block Configuration ---
  dim3 threads(kWarpSize, kColsPerThreadBlock);  // Block dimensions (32, 8)
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);

  // Calculate K / block_size (no rounding needed due to k % block_size == 0 check)
  int blocks_per_K = k / block_size;

  // --- Shared Memory Calculation ---
  // Memory for scales + optional zero points for the columns handled by the block
  size_t scale_zp_shared_mem = (sizeof(T) + (zero_points != nullptr ? sizeof(uint8_t) : 0)) *
                               static_cast<size_t>(blocks_per_K) * kColsPerThreadBlock;

  size_t total_shared_mem = scale_zp_shared_mem;

  // Add shared memory for CUB reduction storage if used
  total_shared_mem += static_cast<size_t>(kColsPerThreadBlock) * sizeof(typename cub::WarpReduce<float>::TempStorage);

  // Check if required shared memory exceeds device limits for the block
  if (total_shared_mem > shared_mem_per_block) {
    return false;
  }

  // --- Kernel Launch ---
  // Macro to simplify dispatching based on block size and presence of zero_points
#define MatMulFloat8bKernelM1Dispatch(bs)                                                        \
  if (nullptr != zero_points) {                                                                  \
    /* Launch kernel with zero points */                                                         \
    MatMulFloat8bKernelM1<T, bs, true><<<blocks, threads, total_shared_mem, stream>>>(           \
        output, a_data, b_data_quant, scales_data, zero_points, n, k, blocks_per_K);             \
  } else {                                                                                       \
    /* Launch kernel without zero points (passing nullptr) */                                    \
    MatMulFloat8bKernelM1<T, bs, false><<<blocks, threads, total_shared_mem, stream>>>(          \
        output, a_data, b_data_quant, scales_data, nullptr /*zero_points*/, n, k, blocks_per_K); \
  }

  // Dispatch based on the provided block_size value
  // Note: Only block sizes compatible with k_per_iter % block_size == 0 and k % block_size == 0 will pass checks.
  if (16 == block_size) {
    MatMulFloat8bKernelM1Dispatch(16);
  } else if (32 == block_size) {
    MatMulFloat8bKernelM1Dispatch(32);
  } else if (64 == block_size) {
    MatMulFloat8bKernelM1Dispatch(64);
  } else if (128 == block_size) {
    MatMulFloat8bKernelM1Dispatch(128);
  } else if (256 == block_size) {
    MatMulFloat8bKernelM1Dispatch(256);
  } else {
    // Unsupported block size
    return false;
  }

#undef MatMulFloat8bKernelM1Dispatch

  // Kernel launch succeeded (error checking, e.g., cudaGetLastError(), should be done by the caller)
  return true;
}

// --- Template Instantiations ---
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

// Add template instantiation for nv_bfloat16
template bool TryMatMul8Bits<nv_bfloat16>(
    nv_bfloat16* output,
    const nv_bfloat16* a_data,
    const uint8_t* b_data_quant,
    const nv_bfloat16* scales_data,
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

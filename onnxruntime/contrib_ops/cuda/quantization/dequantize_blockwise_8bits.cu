// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <type_traits>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "dequantize_blockwise.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Processes 4 elements (since each is 8 bits, 4 fit in uint32_t)
__device__ __forceinline__ void DequantizeFourElements(uint32_t values_quant, half scale, half zp, half* output) {
  half2 scale_half2 = {scale, scale};
  // Formula: val = (quant - zp) * scale = quant * scale - zp * scale
  half zp_adjust = -scale * zp;
  half2 zp_adjust2 = {zp_adjust, zp_adjust};

  alignas(16) half2 results[2];  // Store 4 half values

  // Extract 4 uint8_t values from uint32_t
  half v0 = __ushort2half_rn(static_cast<uint8_t>(values_quant & 0xFF));
  half v1 = __ushort2half_rn(static_cast<uint8_t>((values_quant >> 8) & 0xFF));
  results[0] = __halves2half2(v0, v1) * scale_half2 + zp_adjust2;

  half v2 = __ushort2half_rn(static_cast<uint8_t>((values_quant >> 16) & 0xFF));
  half v3 = __ushort2half_rn(static_cast<uint8_t>((values_quant >> 24) & 0xFF));
  results[1] = __halves2half2(v2, v3) * scale_half2 + zp_adjust2;

  // Write 4 half values (equivalent to float2)
  *(reinterpret_cast<float2*>(output)) = *(reinterpret_cast<float2*>(results));
}

// Processes 4 elements (since each is 8 bits, 4 fit in uint32_t) for bfloat16
__device__ __forceinline__ void DequantizeFourElements(uint32_t values_quant, __nv_bfloat16 scale, __nv_bfloat16 zp, __nv_bfloat16* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat162 scale_bf162 = {scale, scale};
  // Formula: val = (quant - zp) * scale = quant * scale - zp * scale
  __nv_bfloat16 zp_adjust = -scale * zp;
  __nv_bfloat162 zp_adjust2 = {zp_adjust, zp_adjust};

  alignas(16) __nv_bfloat162 results[2];  // Store 4 bfloat16 values

  // Extract 4 uint8_t values from uint32_t
  __nv_bfloat16 v0 = __uint2bfloat16_rn(static_cast<uint8_t>(values_quant & 0xFF));
  __nv_bfloat16 v1 = __uint2bfloat16_rn(static_cast<uint8_t>((values_quant >> 8) & 0xFF));
  results[0] = __halves2bfloat162(v0, v1) * scale_bf162 + zp_adjust2;

  __nv_bfloat16 v2 = __uint2bfloat16_rn(static_cast<uint8_t>((values_quant >> 16) & 0xFF));
  __nv_bfloat16 v3 = __uint2bfloat16_rn(static_cast<uint8_t>((values_quant >> 24) & 0xFF));
  results[1] = __halves2bfloat162(v2, v3) * scale_bf162 + zp_adjust2;

  // Write 4 bfloat16 values
  *(reinterpret_cast<__nv_bfloat162*>(output)) = results[0];
  *(reinterpret_cast<__nv_bfloat162*>(output + 2)) = results[1];
#endif
}

// Processes 4 elements (since each is 8 bits, 4 fit in uint32_t)
__device__ __forceinline__ void DequantizeFourElements(uint32_t values_quant, float scale, float zp, float* output) {
  // Assuming ZP is symmetric or already adjusted if needed. Standard formula: val = (quant - zp) * scale = quant * scale - zp * scale
  float zp_adjust = -scale * zp;

  // Extract 4 uint8_t values from uint32_t
  output[0] = float(values_quant & 0xFF) * scale + zp_adjust;
  output[1] = float((values_quant >> 8) & 0xFF) * scale + zp_adjust;
  output[2] = float((values_quant >> 16) & 0xFF) * scale + zp_adjust;
  output[3] = float((values_quant >> 24) & 0xFF) * scale + zp_adjust;
}

// REVIEW: Deprecate reorder_idx (Recommend to reorder scales and zero points during model conversion instead of using reorder_idx).
// Reorder index is a 1D array of size [K] to support desc_act used in GPTQ quantization.
// However, it impacts inference performance of the kernel since it is not optimized for coalescing memory access.
template <class T>
__global__ void Dequantize8BitsKernelReOrder(
    T* output,
    const uint8_t* quant_data,
    const T* scale_data,
    const uint8_t* zero_points,  // Assuming uint8_t zero points for reorder case
    const int32_t* reorder_idx,
    int block_size,
    int groups_per_K,
    int groups_per_threadblock,
    int total_groups) {
  constexpr int element_per_thread = 4;  // Process 4 elements (uint8_t) per thread using uint32_t load
  int group_id = blockIdx.x * groups_per_threadblock + ((threadIdx.x * element_per_thread) / block_size);
  if (group_id >= total_groups) {
    return;
  }

  // element_offset corresponds to the start of the 4 elements processed by this thread iteration
  int element_offset = group_id * block_size + ((threadIdx.x * element_per_thread) & (block_size - 1));

  T* output_i = output + element_offset;

  // shape of scales and zero_points is [N, groups_per_K]. Compute the 2D indices below.
  int n_idx = group_id / groups_per_K;
  int kb_idx = group_id % groups_per_K;

  // Read 4 uint8_t values packed into a uint32_t
  uint32_t quant_value = *(reinterpret_cast<const uint32_t*>(quant_data + element_offset));

  // Adjust reorder index pointer to the start of the 4 indices for this thread iteration
  const int32_t* g_idx = reorder_idx + kb_idx * block_size + ((threadIdx.x * element_per_thread) & (block_size - 1));

  for (int i = 0; i < element_per_thread; i++) {
    // Typical value of g_idx is in the range of [0, groups_per_K) for reordering groups.
    // No range check here so it might have out-of-bound access if the reorder_idx is not valid.
    int32_t rid = g_idx[i];
    ptrdiff_t scale_zp_offset = n_idx * groups_per_K + rid;
    T scale = *(scale_data + scale_zp_offset);

    uint8_t zp = 128;  // Default zero point
    if (zero_points) {
      zp = zero_points[scale_zp_offset];
    }

    // Extract the i-th uint8_t value
    uint8_t q_val = (quant_value >> (8 * i)) & 0xFF;

    if constexpr (std::is_same_v<T, half>) {
      T zp_T = __ushort2half_rn(zp);
      T zp_adjust = -scale * zp_T;
      output_i[i] = __ushort2half_rn(q_val) * scale + zp_adjust;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      T zp_T = __uint2bfloat16_rn(zp);
      T zp_adjust = -scale * zp_T;
      output_i[i] = __uint2bfloat16_rn(q_val) * scale + zp_adjust;
    } else {
      T zp_T = static_cast<T>(zp);
      T zp_adjust = -scale * zp_T;
      output_i[i] = static_cast<T>(q_val) * scale + zp_adjust;
    }
  }
}

template <class T, typename ZeroT = uint8_t>
__global__ void Dequantize8BitsKernel(
    T* output,
    const uint8_t* quant_data,
    const T* scale_data,
    const ZeroT* zero_points,
    int block_size,
    int groups_per_threadblock,
    int total_groups) {
  constexpr int element_per_thread = 4;  // Process 4 elements (uint8_t) per thread using uint32_t load
  int block_id = blockIdx.x * groups_per_threadblock + ((threadIdx.x * element_per_thread) / block_size);
  if (block_id >= total_groups) {
    return;
  }

  // element_offset corresponds to the start of the 4 elements processed by this thread iteration
  int element_offset = block_id * block_size + ((threadIdx.x * element_per_thread) & (block_size - 1));

  // Read 4 uint8_t values packed into a uint32_t
  uint32_t quant_value = *(reinterpret_cast<const uint32_t*>(quant_data + element_offset));
  T scale = *(scale_data + block_id);  // One scale per block

  T zero_point_value;
  if constexpr (std::is_same_v<ZeroT, uint8_t>) {
    // Assuming one uint8_t zero point per block. Default 128 for uint8 asymmetric.
    uint8_t zp = 128;
    if (zero_points) {
      zp = zero_points[block_id];  // Direct lookup, no packing
    }
    // Convert uint8_t zp to T (float/half/bfloat16)
    if constexpr (std::is_same_v<T, half>) {
      zero_point_value = __uint2half_rn(zp);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      zero_point_value = __uint2bfloat16_rn(zp);
    } else {
      zero_point_value = static_cast<T>(zp);
    }
  } else {  // ZeroT is T (float, half, or bfloat16)
    // Default 0 for float/half/bfloat16 zero point
    zero_point_value = zero_points ? *(zero_points + block_id) : static_cast<T>(0.0f);
  }

  output = output + element_offset;  // Point output to the start of the 4 elements
  DequantizeFourElements(quant_value, scale, zero_point_value, output);
}

template <class T, typename ZeroT>
Status Dequantize8Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const ZeroT* zero_points,    // Shape: [N, K_blocks] or [N * K_blocks]
    const int32_t* reorder_idx,  // If provided, ZeroT is expected to be uint8_t
    int k,                       // Original dimension before padding
    int n,                       // Other dimension
    int block_size,
    cudaStream_t stream) {
  ORT_ENFORCE(k % block_size == 0, "k must be a multiple of block_size");  // K shall be padded to multiple of block_size.

  constexpr int element_per_thread = 4;
  int groups_per_K = k / block_size;
  int total_groups = n * groups_per_K;  // Total number of blocks

  assert(block_size <= GridDim::maxThreadsPerBlock * element_per_thread);
  int groups_per_threadblock = GridDim::maxThreadsPerBlock * element_per_thread / block_size;
  int groups_per_grid = CeilDiv(total_groups, groups_per_threadblock);

  dim3 grid_dim(groups_per_grid);
  dim3 block_dim(GridDim::maxThreadsPerBlock);

  DUMP_TENSOR_INIT();
  if (!reorder_idx || std::is_same_v<ZeroT, T>) {
    DUMP_STRING("Launch standard kernel for Dequantize8Bits");
    Dequantize8BitsKernel<T, ZeroT><<<grid_dim, block_dim, 0, stream>>>(
        output,
        quant_data,
        scales_data,
        zero_points,
        block_size,
        groups_per_threadblock,
        total_groups);
  } else {
    if constexpr (std::is_same_v<ZeroT, uint8_t>) {
      DUMP_STRING("Launch reorder kernel for Dequantize8Bits");
      Dequantize8BitsKernelReOrder<T><<<grid_dim, block_dim, 0, stream>>>(
          output,
          quant_data,
          scales_data,
          (const uint8_t*)zero_points,
          reorder_idx,
          block_size,
          groups_per_K,
          groups_per_threadblock,
          total_groups);
    } else {
      return Status(::onnxruntime::common::ONNXRUNTIME, ::onnxruntime::common::INVALID_ARGUMENT,
                    "Reorder kernel currently expects uint8_t zero points.");
    }
  }

  return CUDA_CALL(cudaGetLastError());  // Check for launch errors
}

// Template instantiations for 8-bit
template Status Dequantize8Bits<float, uint8_t>(
    float* output,
    const uint8_t* quant_data,
    const float* scales_data,
    const uint8_t* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize8Bits<half, uint8_t>(
    half* output,
    const uint8_t* quant_data,
    const half* scales_data,
    const uint8_t* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize8Bits<__nv_bfloat16, uint8_t>(
    __nv_bfloat16* output,
    const uint8_t* quant_data,
    const __nv_bfloat16* scales_data,
    const uint8_t* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize8Bits<float, float>(
    float* output,
    const uint8_t* quant_data,
    const float* scales_data,
    const float* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize8Bits<half, half>(
    half* output,
    const uint8_t* quant_data,
    const half* scales_data,
    const half* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize8Bits<__nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* output,
    const uint8_t* quant_data,
    const __nv_bfloat16* scales_data,
    const __nv_bfloat16* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

// Generic dequantization kernel for 8 bits
template <
    typename ElementT,
    int32_t block_size,
    int32_t qbits,
    bool Columnwise>
__global__ void dequantizeThread8b(ElementT* dst,
                                   const uint8_t* weights,  // Quantized data (uint8_t)
                                   const ElementT* scales,
                                   const uint8_t* zero_points,  // Assuming uint8_t zero points
                                   int rows,
                                   int columns,
                                   int thread_row_blocks) {  // Number of thread blocks along row dimension

  using QuantBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::QuantBlk;
  using ThreadBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::ThreadBlk;

  static_assert(qbits == 8, "Only 8b block quantization is supported by this kernel specialization!");

  const auto thread_idx_global = blockIdx.x * blockDim.x + threadIdx.x;

  // Total blocks along row dim for scales/zp
  const auto total_row_blks = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

  // Total blocks along col dim for scales/zp
  // const auto total_col_blks = (columns + QuantBlk::kColumn - 1) / QuantBlk::kColumn;

  // Total number of blocks to process
  // const auto total_quant_blocks = total_row_blks * total_col_blks;

  // Iterate over the quantization blocks assigned to this thread
  // Each thread might process multiple QuantBlks
  // This loop structure assumes 1D grid/block launch. A 2D launch might map threads differently.
  const auto block_idx = thread_idx_global;  // Assuming 1 thread processes 1 ThreadBlk here

  // Calculate row and column block indices for this thread
  // Map 1D block_idx back to 2D block indices (row_blk, col_blk)
  const auto r_blk_idx_thread = static_cast<int32_t>(block_idx % thread_row_blocks);  // Thread block index along rows
  const auto c_blk_idx_thread = static_cast<int32_t>(block_idx / thread_row_blocks);  // Thread block index along columns

  // Calculate starting row and column for this thread's work item (ThreadBlk)
  int32_t r_start = r_blk_idx_thread * ThreadBlk::kRow;
  int32_t c_start = c_blk_idx_thread * ThreadBlk::kColumn;

  // Check if this thread is out of bounds for the overall work
  if (c_start >= columns) {
    return;
  }

  // Determine the actual end row/column considering matrix boundaries
  int32_t r_end = std::min(r_start + ThreadBlk::kRow, rows);
  int32_t c_end = std::min(c_start + ThreadBlk::kColumn, columns);

  // Process elements within the assigned ThreadBlk
  for (int32_t c = c_start; c < c_end; ++c) {
    // Calculate the block index for scale/zp lookup based on the current column 'c'
    const auto scale_zp_col_blk_idx = c / QuantBlk::kColumn;

    // Calculate base pointer for this column in the quantized weights matrix
    // Assuming weights stored column-major: shape [rows, columns] -> layout [columns, rows]
    // Each element is uint8_t.
    // const uint8_t* q_col_ptr = weights + static_cast<size_t>(scale_zp_col_blk_idx) * rows;

    for (int32_t r = r_start; r < r_end; ++r) {
      // Calculate the block index for scale/zp lookup based on current row 'r'
      const auto scale_zp_row_blk_idx = r / QuantBlk::kRow;
      const auto scale_zp_flat_idx = scale_zp_col_blk_idx * total_row_blks + scale_zp_row_blk_idx;

      // Get scale and zero point for this block
      const ElementT scale = scales[scale_zp_flat_idx];
      const uint8_t zp_uint8 = (zero_points == nullptr) ? 128 : zero_points[scale_zp_flat_idx];

      // Get the quantized value (uint8_t)
      // Assuming weights are stored col-major for block quantization (e.g. [cols, rows/block_size, block_size])
      // Row-major logical layout for weights access: index = c * rows + r
      const size_t q_val_idx = static_cast<size_t>(c) * rows + r;
      const uint8_t q_val = weights[q_val_idx];

      // Dequantize
      if constexpr (std::is_same<ElementT, half>::value) {
        const half zp_half = __uint2half_rn(zp_uint8);
        const half adjust = -scale * zp_half;
        dst[q_val_idx] = __uint2half_rn(q_val) * scale + adjust;
      } else if constexpr (std::is_same<ElementT, __nv_bfloat16>::value) {
        const __nv_bfloat16 zp_bf16 = __uint2bfloat16_rn(zp_uint8);
        const __nv_bfloat16 adjust = -scale * zp_bf16;
        dst[q_val_idx] = __uint2bfloat16_rn(q_val) * scale + adjust;
      } else {  // Float
        static_assert(std::is_same<ElementT, float>::value, "Only float, half and bfloat16 are supported!");
        const float zp_float = static_cast<float>(zp_uint8);
        const float adjust = -scale * zp_float;
        dst[q_val_idx] = static_cast<float>(q_val) * scale + adjust;
      }
    }
  }
}

// Launcher function for the generic 8-bit kernel
template <
    typename ElementT,
    int32_t block_size,
    int32_t qbits,
    bool Columnwise>
static void dequantize8b_generic(ElementT* dst, const uint8_t* weights, const ElementT* scales,
                                 const uint8_t* zero_points, int32_t rows, int32_t columns,
                                 cudaStream_t stream) {
  using ThreadBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::ThreadBlk;

  const auto thread_row_blocks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
  const auto thread_col_blocks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
  const auto thread_total_blocks = thread_row_blocks * thread_col_blocks;

  const auto grids = (thread_total_blocks + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
  dequantizeThread8b<ElementT, block_size, qbits, Columnwise><<<grids, GridDim::maxThreadsPerBlock, 0, stream>>>(
      dst,
      weights,
      scales,
      zero_points,
      rows,
      columns,
      thread_row_blocks);
}

template <typename T>
Status
DequantizeBlockwise8b(
    T* dst,
    const uint8_t* src,  // Quantized uint8_t data
    const T* scales,
    const uint8_t* zero_points,  // Assuming uint8_t zero points
    int block_size,
    bool columnwise,  // Orientation of elements within a block
    int rows,
    int columns,
    cudaStream_t stream) {
  // Use the generic launcher, passing qbits=8
  switch (block_size) {
    case 16:
      if (columnwise) {
        dequantize8b_generic<T, 16, 8, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize8b_generic<T, 16, 8, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 32:
      if (columnwise) {
        dequantize8b_generic<T, 32, 8, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize8b_generic<T, 32, 8, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 64:
      if (columnwise) {
        dequantize8b_generic<T, 64, 8, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize8b_generic<T, 64, 8, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 128:
      if (columnwise) {
        dequantize8b_generic<T, 128, 8, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize8b_generic<T, 128, 8, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 256:
      if (columnwise) {
        dequantize8b_generic<T, 256, 8, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize8b_generic<T, 256, 8, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    default:
      // Only block size 16, 32, 64, 128, 256 are supported.
      return Status(::onnxruntime::common::ONNXRUNTIME, ::onnxruntime::common::FAIL,
                    "Unsupported block size for 8b blockwise quantization.");
  }
}

// Template instantiations for 8-bit blockwise
template Status DequantizeBlockwise8b<float>(
    float* dst,
    const uint8_t* src,
    const float* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

template Status DequantizeBlockwise8b<half>(
    half* dst,
    const uint8_t* src,
    const half* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

template Status DequantizeBlockwise8b<__nv_bfloat16>(
    __nv_bfloat16* dst,
    const uint8_t* src,
    const __nv_bfloat16* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

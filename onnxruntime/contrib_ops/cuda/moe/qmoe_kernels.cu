
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/common/narrow.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/cub.cuh"
#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cfloat>

namespace onnxruntime {
namespace contrib {
namespace cuda {

int Compute1DGridSize(int num_elements, int block_size) {
  return (num_elements + block_size - 1) / block_size;
}

constexpr float kTopKNormalizeEpsilon = 1e-6f;

__device__ __forceinline__ float SoftmaxScale(float logit, float max_val, float inv_sum) {
  return (inv_sum > 0.0f) ? expf(logit - max_val) * inv_sum : 0.0f;
}

__device__ __forceinline__ float SafeInvSum(float sum) {
  return (sum > 0.0f) ? (1.0f / sum) : 0.0f;
}

__device__ __forceinline__ float TopKNormalizeDenom(bool normalize_scales, float scale_sum) {
  return (normalize_scales && scale_sum > kTopKNormalizeEpsilon) ? scale_sum : 1.0f;
}

__device__ __forceinline__ float WarpReduceMax(float value) {
  constexpr int kWarpSize = onnxruntime::cuda::topk::kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFF, value, offset));
  }
  return value;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
  constexpr int kWarpSize = onnxruntime::cuda::topk::kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }
  return value;
}

template <typename BlockReduce>
__device__ __forceinline__ float BlockReduceMax(float value, typename BlockReduce::TempStorage& temp_storage) {
#if CUDART_VERSION >= 12090
  return BlockReduce(temp_storage).Reduce(value, ::cuda::maximum());
#else
  return BlockReduce(temp_storage).Reduce(value, cub::Max());
#endif
}

template <typename BlockReduce>
__device__ __forceinline__ float BlockReduceSum(float value, typename BlockReduce::TempStorage& temp_storage) {
#if CUDART_VERSION >= 12090
  return BlockReduce(temp_storage).Reduce(value, ::cuda::std::plus());
#else
  return BlockReduce(temp_storage).Reduce(value, cub::Sum());
#endif
}

template <typename T>
__global__ void SoftmaxTopKKernel(const T* logits, float* topk_scales, int* topk_indices,
                                  int num_rows, int num_experts, int k, bool normalize_scales) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  const T* row_logits = logits + row * num_experts;
  float* row_scales = topk_scales + row * k;
  int* row_indices = topk_indices + row * k;

  // 1. Find max for numerical stability
  float max_val = onnxruntime::cuda::topk::kNegativeInfinity;
  for (int i = 0; i < num_experts; ++i) {
    float val = static_cast<float>(row_logits[i]);
    if (val > max_val) max_val = val;
  }

  // 2. Compute exp sum
  float sum_exp = 0.0f;
  for (int i = 0; i < num_experts; ++i) {
    sum_exp += expf(static_cast<float>(row_logits[i]) - max_val);
  }
  const float inv_sum = SafeInvSum(sum_exp);

  // 3. Compute Softmax and find TopK
  // For small k, we can do a simple selection.
  // Note: This is efficient only for small k and small num_experts.

  // We can compute softmax values on the fly or store them.
  // Given we need topK, let's just compute all softmax values then pick top K.
  // (Optimization: use a heap or similar if K is small and N is large)

  for (int i = 0; i < k; ++i) {
    row_scales[i] = -FLT_MAX;
    row_indices[i] = -1;
  }

  for (int i = 0; i < num_experts; ++i) {
    float prob = SoftmaxScale(static_cast<float>(row_logits[i]), max_val, inv_sum);

    // Insert into top-k logic
    // Simple insertion sort for very small k (e.g. k=2)
    for (int j = 0; j < k; ++j) {
      if (prob > row_scales[j]) {
        // Shift current values down
        for (int m = k - 1; m > j; --m) {
          row_scales[m] = row_scales[m - 1];
          row_indices[m] = row_indices[m - 1];
        }
        row_scales[j] = prob;
        row_indices[j] = i;
        break;
      }
    }
  }

  // 4. Normalize if requested
  if (normalize_scales) {
    float scale_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      scale_sum += row_scales[i];
    }
    if (scale_sum > kTopKNormalizeEpsilon) {
      for (int i = 0; i < k; ++i) {
        row_scales[i] /= scale_sum;
      }
    }
  }
}

// Block-per-row softmax + top-k using a CUB block sort. Each block sorts one
// row's logits (descending) and reads the first k. A full sort of 256 logits is
// ~2.5x faster than k rounds of block-argmax on this size (benchmarked), and is
// the layout onnxruntime-genai's top-k benchmarks also recommend (CUB block
// merge) for sort sizes up to ~1024. The capacity (kBlockSize*kItemsPerThread)
// must be >= num_experts; padding lanes carry (-inf, INT_MAX) so valid -inf
// expert scores sort ahead of padding. Tie-breaking matches the scalar kernel
// (lower expert index first) via the same packed stable sort key used by the
// warp merge path.

template <typename T, int kBlockSize, int kItemsPerThread>
__global__ void SoftmaxTopKMergeKernel(const T* logits, float* topk_scales, int* topk_indices,
                                       int num_rows, int num_experts, int k, bool normalize_scales) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;

  const T* row_logits = logits + static_cast<size_t>(row) * num_experts;
  const int tid = threadIdx.x;

  using BlockMergeSort = cub::BlockMergeSort<uint64_t, kBlockSize, kItemsPerThread, cub::NullType>;
  using BlockReduce = cub::BlockReduce<float, kBlockSize>;
  __shared__ union {
    typename BlockMergeSort::TempStorage merge;
    typename BlockReduce::TempStorage reduce;
  } temp;
  __shared__ float s_topk[64];  // k <= 64
  __shared__ float s_max;
  __shared__ float s_sum;

  // Load this thread's packed (logit, expert index) keys in a blocked
  // arrangement: thread t owns indices [t*ipt, t*ipt+ipt).
  uint64_t keys[kItemsPerThread];
  float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
#pragma unroll
  for (int j = 0; j < kItemsPerThread; ++j) {
    const int idx = tid * kItemsPerThread + j;
    const float logit = (idx < num_experts) ? static_cast<float>(row_logits[idx])
                                            : onnxruntime::cuda::topk::kNegativeInfinity;
    const int index = (idx < num_experts) ? idx : INT_MAX;
    keys[j] = onnxruntime::cuda::topk::PackStableSortKey(logit, index);
    local_max = fmaxf(local_max, logit);
  }

  // Softmax denominator over all experts (needed when normalize_scales is false;
  // when true it cancels in the top-k normalization but is still correct).
  const float block_max = BlockReduceMax<BlockReduce>(local_max, temp.reduce);
  if (tid == 0) s_max = block_max;
  // Single barrier: publishes s_max to all threads and also separates the two
  // BlockReduce uses that share temp.reduce.
  __syncthreads();
  const float max_val = s_max;

  float local_sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kItemsPerThread; ++j) {
    const int idx = tid * kItemsPerThread + j;
    if (idx < num_experts) {
      local_sum += expf(onnxruntime::cuda::topk::UnpackStableSortScore(keys[j]) - max_val);
    }
  }
  const float block_sum = BlockReduceSum<BlockReduce>(local_sum, temp.reduce);
  if (tid == 0) s_sum = block_sum;
  // Single barrier: publishes s_sum and separates temp.reduce from temp.merge.
  __syncthreads();
  const float inv_sum = SafeInvSum(s_sum);

  // Sort packed (logit, index) keys descending. Result stays in a blocked
  // layout, so sorted rank r lives in thread (r / ipt), item (r % ipt). Sort()
  // leaves the sorted keys in each thread's registers and temp.merge is not
  // reused afterwards, so no barrier is needed here; the shared s_topk writes
  // below are published by the barrier that follows them.
  BlockMergeSort(temp.merge).Sort(keys, onnxruntime::cuda::topk::Greater<uint64_t>());

#pragma unroll
  for (int j = 0; j < kItemsPerThread; ++j) {
    const int rank = tid * kItemsPerThread + j;
    if (rank < k) {
      const uint64_t key = keys[j];
      topk_indices[static_cast<size_t>(row) * k + rank] =
          onnxruntime::cuda::topk::UnpackStableSortIndex(key);
      s_topk[rank] = SoftmaxScale(onnxruntime::cuda::topk::UnpackStableSortScore(key), max_val, inv_sum);
    }
  }
  __syncthreads();

  if (tid == 0) {
    if (normalize_scales) {
      float scale_sum = 0.0f;
      for (int i = 0; i < k; ++i) scale_sum += s_topk[i];
      const float denom = TopKNormalizeDenom(normalize_scales, scale_sum);
      for (int i = 0; i < k; ++i) topk_scales[static_cast<size_t>(row) * k + i] = s_topk[i] / denom;
    } else {
      for (int i = 0; i < k; ++i) topk_scales[static_cast<size_t>(row) * k + i] = s_topk[i];
    }
  }
}

// Warp-bitonic softmax + top-k for num_experts <= 32. Each warp handles one
// row, with lane `l` owning expert `l`. The whole softmax reduction and the
// sort are done with warp shuffles (no shared memory). This is the fastest path
// for tiny expert counts per the onnxruntime-genai top-k benchmark. Tie-breaking
// (equal scores prefer the lower expert index) matches SoftmaxTopKMergeKernel.
template <typename T, int kWarpsPerBlock>
__global__ void SoftmaxTopKWarpBitonicKernel(const T* logits, float* topk_scales, int* topk_indices,
                                             int num_rows, int num_experts, int k, bool normalize_scales) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * kWarpsPerBlock + threadIdx.y;
  if (row >= num_rows) return;

  const T* row_logits = logits + static_cast<size_t>(row) * num_experts;
  const float logit = (lane < num_experts) ? static_cast<float>(row_logits[lane])
                                           : onnxruntime::cuda::topk::kNegativeInfinity;

  const float max_val = WarpReduceMax(logit);

  // Warp-wide exp sum (softmax denominator over all experts).
  const float sum_exp = WarpReduceSum((lane < num_experts) ? expf(logit - max_val) : 0.0f);
  const float inv_sum = SafeInvSum(sum_exp);

  // Sort (logit, expert index) descending; sorting by logit is equivalent to
  // sorting by softmax probability since the mapping is monotonic.
  float score = logit;
  int index = (lane < num_experts) ? lane : INT_MAX;
  onnxruntime::cuda::topk::WarpBitonicSortDescending(score, index);

  // Lane r now holds the rank-r element. Compute the top-k probabilities.
  float prob = (lane < k) ? SoftmaxScale(score, max_val, inv_sum) : 0.0f;

  if (normalize_scales) {
    prob /= TopKNormalizeDenom(normalize_scales, WarpReduceSum(prob));
  }

  if (lane < k) {
    topk_scales[static_cast<size_t>(row) * k + lane] = prob;
    topk_indices[static_cast<size_t>(row) * k + lane] = index;
  }
}

// Warp CUB merge sort softmax + top-k for num_experts <= kBufferSize (64). One
// warp (32 threads) per block sorts a row's logits held in shared memory. This
// is the genai-recommended path for sort sizes in (32, 64]. Tie-breaking
// matches SoftmaxTopKMergeKernel via a packed stable sort key.
template <typename T, int kBufferSize>
__global__ void SoftmaxTopKWarpMergeKernel(const T* logits, float* topk_scales, int* topk_indices,
                                           int num_rows, int num_experts, int k, bool normalize_scales) {
  constexpr int kWarpSize = onnxruntime::cuda::topk::kWarpSize;
  using WarpMergeSorter = onnxruntime::cuda::topk::WarpMergeSorter<kBufferSize>;

  const int row = blockIdx.x;
  if (row >= num_rows) return;
  const int lane = threadIdx.x;

  __shared__ float s_scores[kBufferSize];
  __shared__ int s_indices[kBufferSize];
  __shared__ typename WarpMergeSorter::TempStorage temp_storage;

  const T* row_logits = logits + static_cast<size_t>(row) * num_experts;

  // Load logits into shared memory and compute the warp-wide max.
  float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
  for (int i = lane; i < kBufferSize; i += kWarpSize) {
    const float v = (i < num_experts) ? static_cast<float>(row_logits[i])
                                      : onnxruntime::cuda::topk::kNegativeInfinity;
    s_scores[i] = v;
    s_indices[i] = (i < num_experts) ? i : INT_MAX;
    local_max = fmaxf(local_max, v);
  }
  const float max_val = WarpReduceMax(local_max);

  // Warp-wide exp sum over all experts.
  float local_sum = 0.0f;
  for (int i = lane; i < num_experts; i += kWarpSize) {
    local_sum += expf(s_scores[i] - max_val);
  }
  const float inv_sum = SafeInvSum(WarpReduceSum(local_sum));

  __syncwarp();
  WarpMergeSorter::Sort(s_scores, s_indices, temp_storage, num_experts);
  __syncwarp();

  // s_scores[r]/s_indices[r] now hold the rank-r logit/expert index.
  float scale_sum = 0.0f;
  if (normalize_scales) {
    for (int i = lane; i < k; i += kWarpSize) {
      scale_sum += SoftmaxScale(s_scores[i], max_val, inv_sum);
    }
    scale_sum = WarpReduceSum(scale_sum);
  }
  const float denom = TopKNormalizeDenom(normalize_scales, scale_sum);

  for (int i = lane; i < k; i += kWarpSize) {
    const float prob = SoftmaxScale(s_scores[i], max_val, inv_sum);
    topk_scales[static_cast<size_t>(row) * k + i] = normalize_scales ? (prob / denom) : prob;
    topk_indices[static_cast<size_t>(row) * k + i] = s_indices[i];
  }
}

template <typename T>
void DispatchSoftmaxTopK(const T* logits, float* topk_scales, int* topk_indices,
                         int num_rows, int num_experts, int k, bool normalize_scales,
                         cudaStream_t stream) {
  ORT_ENFORCE(k > 0 && k <= num_experts,
              "SoftmaxTopK requires 0 < k <= num_experts, got k=", k,
              " and num_experts=", num_experts);

  // Block-per-row CUB merge sort is the fastest path for the common decode case
  // (one block fully sorts a row). Pick the smallest capacity that covers
  // num_experts. k must fit in s_topk (<= 64).
  const dim3 grid(static_cast<unsigned>(num_rows));
  if (k <= 64 && num_experts <= 1024) {
    // Tiny expert counts: a single warp sorts a row entirely in registers via
    // an in-register bitonic sort (no shared memory). Multiple warps per block
    // process multiple rows for better occupancy.
    if (num_experts <= onnxruntime::cuda::topk::kWarpBitonicMaxSize) {
      constexpr int kWarpsPerBlock = 8;
      const dim3 block(static_cast<unsigned>(onnxruntime::cuda::topk::kWarpSize), kWarpsPerBlock);
      const dim3 bitonic_grid(static_cast<unsigned>((num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock));
      SoftmaxTopKWarpBitonicKernel<T, kWarpsPerBlock><<<bitonic_grid, block, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else if (num_experts <= onnxruntime::cuda::topk::kWarpMergeMaxSize) {
      // Single warp per row sorts up to 64 logits in shared memory (CUB warp
      // merge sort), the genai-recommended path for sort sizes in (32, 64].
      SoftmaxTopKWarpMergeKernel<T, 64><<<grid, onnxruntime::cuda::topk::kWarpSize, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else if (num_experts <= 128) {
      SoftmaxTopKMergeKernel<T, 128, 1><<<grid, 128, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else if (num_experts <= 256) {
      SoftmaxTopKMergeKernel<T, 128, 2><<<grid, 128, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else if (num_experts <= 512) {
      SoftmaxTopKMergeKernel<T, 128, 4><<<grid, 128, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    } else /*if (num_experts <= 1024)*/ {
      SoftmaxTopKMergeKernel<T, 256, 4><<<grid, 256, 0, stream>>>(
          logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
      return;
    }
  } else {
    // Fall back to the simple one-thread-per-row kernel.
    const int block = 256;
    const int grid_1d = Compute1DGridSize(num_rows, block);
    SoftmaxTopKKernel<T><<<grid_1d, block, 0, stream>>>(
        logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
  }
}

void LaunchSoftmaxTopK(
    const float* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream) {
  DispatchSoftmaxTopK<float>(logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales, stream);
}

void LaunchSoftmaxTopK(
    const half* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream) {
  DispatchSoftmaxTopK<half>(logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales, stream);
}

void LaunchSoftmaxTopK(
    const __nv_bfloat16* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream) {
  DispatchSoftmaxTopK<__nv_bfloat16>(logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales, stream);
}

template <typename T>
__global__ void QMoEPrePackZPKernel(const uint8_t* zp, const T* scales, T* out, int num_elements, float offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float s = static_cast<float>(scales[idx]);
    float z = static_cast<float>(zp[idx]);
    // Compute bias = (offset - zp) * scale
    // If offset = 0, bias = -zp * scale
    // If offset = 128 (e.g. for uint8 -> int8 shift), bias = (128 - zp) * scale
    out[idx] = static_cast<T>((offset - z) * s);
  }
}

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<float><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, 0.0f);
}

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<half><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, 0.0f);
}

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, 0.0f);
}

template <typename T>
__global__ void QMoEPrePackPacked4BitZPKernel(const uint8_t* packed_zp, const T* scales, T* out, int num_elements, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float s = static_cast<float>(scales[idx]);

    // 4-bit unpacking with stride N
    // row = idx / N; col = idx % N;
    // byte_row = row >> 1; nibble = row & 1;
    // byte_idx = byte_row * N + col;

    int row = idx / N;
    int col = idx % N;
    int byte_idx = (row >> 1) * N + col;

    uint8_t packed_byte = packed_zp[byte_idx];
    uint8_t val = (packed_byte >> ((row & 1) << 2)) & 0x0F;
    float z = static_cast<float>(val);

    // Bias calculation for Cutlass dequantizer: (8.0 - ZP) * Scale
    // Cutlass dequantizer uses formula: (q - 8) * scale + bias
    // We want: (q - zp) * scale
    // (q - 8) * scale + bias = q*scale - 8*scale + bias
    // q*scale - zp*scale = q*scale - zp*scale
    // So: -8*scale + bias = -zp*scale => bias = (8 - zp) * scale
    out[idx] = static_cast<T>((8.0f - z) * s);
  }
}

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const float* scales,
    float* output,
    int num_elements,
    int N,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackPacked4BitZPKernel<float><<<grid, block, 0, stream>>>(packed_zp, scales, output, num_elements, N);
}

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const half* scales,
    half* output,
    int num_elements,
    int N,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackPacked4BitZPKernel<half><<<grid, block, 0, stream>>>(packed_zp, scales, output, num_elements, N);
}

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    int N,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackPacked4BitZPKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(packed_zp, scales, output, num_elements, N);
}

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    float offset,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<float><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, offset);
}

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    float offset,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<half><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, offset);
}

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    float offset,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEPrePackZPKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(zp, scales, output, num_elements, offset);
}

// Batched 4-bit packed ZP scaled bias kernel.
// Grid: (ceil(n/16), ceil(k_blocks/16), experts)
// Each thread computes one output element: output[e][out_row][out_col]
// where out_row in [0, k_blocks), out_col in [0, n).
// ZP layout: [experts, n, packed_k_blocks] with packed_k_blocks = (k_blocks+1)/2
// Scale/Output layout: [experts, k_blocks, n]
template <typename T>
__global__ void QMoEScaledZP4BitBatchedKernel(
    const uint8_t* packed_zp,
    const T* transposed_scale,
    T* scaled_zero_point,
    int n, int k_blocks,
    float default_zero_point) {
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int expert = blockIdx.z;

  if (out_col < n && out_row < k_blocks) {
    int packed_k_blocks = (k_blocks + 1) / 2;
    int64_t expert_zp_offset = static_cast<int64_t>(expert) * n * packed_k_blocks;
    int64_t expert_scale_offset = static_cast<int64_t>(expert) * k_blocks * n;

    // ZP is [n, packed_k_blocks] per expert; in_row = out_col, in_col = out_row
    int in_row = out_col;
    int in_col = out_row;
    int64_t packed_zp_offset = expert_zp_offset + static_cast<int64_t>(in_row) * packed_k_blocks + in_col / 2;
    uint8_t packed_byte = packed_zp[packed_zp_offset];
    float zero_point_val = static_cast<float>((in_col & 0x01) ? (packed_byte >> 4) : (packed_byte & 0x0f));

    int64_t output_offset = expert_scale_offset + static_cast<int64_t>(out_row) * n + out_col;
    T scale_val = transposed_scale[output_offset];
    float result = static_cast<float>(scale_val) * (-zero_point_val + default_zero_point);
    scaled_zero_point[output_offset] = static_cast<T>(result);
  }
}

void LaunchQMoEScaledZP4BitBatched(
    const uint8_t* packed_zp,
    const half* transposed_scale,
    half* scaled_zero_point,
    int experts, int n, int k_blocks,
    float default_zero_point,
    cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
      (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
      (k_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE,
      experts);
  QMoEScaledZP4BitBatchedKernel<half><<<gridDim, blockDim, 0, stream>>>(
      packed_zp, transposed_scale, scaled_zero_point, n, k_blocks, default_zero_point);
}

void LaunchQMoEScaledZP4BitBatched(
    const uint8_t* packed_zp,
    const __nv_bfloat16* transposed_scale,
    __nv_bfloat16* scaled_zero_point,
    int experts, int n, int k_blocks,
    float default_zero_point,
    cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
      (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
      (k_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE,
      experts);
  QMoEScaledZP4BitBatchedKernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(
      packed_zp, transposed_scale, scaled_zero_point, n, k_blocks, default_zero_point);
}

__global__ void QMoEShiftWeightsKernel(const uint8_t* input, uint8_t* output, int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = input[idx] ^ 0x80;
  }
}

void LaunchQMoEShiftWeights(
    const uint8_t* input,
    uint8_t* output,
    int num_elements,
    cudaStream_t stream) {
  int block = 256;
  int grid = Compute1DGridSize(num_elements, block);
  QMoEShiftWeightsKernel<<<grid, block, 0, stream>>>(input, output, num_elements);
}

// ====================== Sparse Mixer Kernel ===============================
// Ported from old/moe_kernel.cu

static constexpr int WARP_SIZE = 32;

template <typename T, int TPB, int NUM_EXPERTS>
__launch_bounds__(TPB) __global__
    void sparse_mixer_top2(const T* inputs, float* output, int* indices, int* source_rows, const float jitter_eps) {
  static constexpr int K = 2;

  using cub_kvp = cub::KeyValuePair<int, T>;
  using KVBlockReduce = cub::BlockReduce<cub_kvp, TPB>;

  __shared__ float result_kvp_value[K];
  __shared__ typename KVBlockReduce::TempStorage kvTmpStorage;

  cub_kvp thread_kvp;
  // cub::ArgMax arg_max; // Use default ArgMax

  // Manually define ArgMax functor if not available or to ensure behavior
  struct ArgMax {
    __device__ __forceinline__ cub_kvp operator()(const cub_kvp& a, const cub_kvp& b) const {
      return (b.value > a.value) ? b : a;
    }
  } arg_max;

  int num_rows = gridDim.x;
  const int block_row = blockIdx.x;

  const int thread_row_offset = blockIdx.x * NUM_EXPERTS;

  float factor[K];
  bool logits_mask[K];

#pragma unroll
  for (int k_idx = 0; k_idx < K; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-FLT_MAX);

    cub_kvp inp_kvp;
#pragma unroll
    for (int expert = threadIdx.x; expert < NUM_EXPERTS; expert += TPB) {
      const int idx = thread_row_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[K * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp = KVBlockReduce(kvTmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = K * block_row + k_idx;
      result_kvp_value[k_idx] = (float)result_kvp.value;
      indices[idx] = result_kvp.key;
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();

#pragma unroll
    for (int expert = threadIdx.x; expert < NUM_EXPERTS; expert += TPB) {
      const int idx = thread_row_offset + expert;
      factor[k_idx] = max(abs((float)inputs[idx]), result_kvp_value[k_idx]);
      logits_mask[k_idx] = (result_kvp_value[k_idx] - (float)inputs[idx]) > (2 * jitter_eps * factor[k_idx]);
      if (k_idx == 1 && expert == indices[K * block_row]) {
        logits_mask[1] = true;
      }
    }
  }

#pragma unroll
  for (int k_idx = 0; k_idx < K; ++k_idx) {
    float row_sum(0);

#pragma unroll
    for (int ii = threadIdx.x; ii < NUM_EXPERTS; ii += TPB) {
      const int idx = thread_row_offset + ii;
      row_sum += logits_mask[k_idx] ? 0 : exp((static_cast<float>(inputs[idx]) - result_kvp_value[k_idx]));
    }

#pragma unroll
    for (int mask = NUM_EXPERTS / 2; mask > 0; mask /= 2) {
      row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, NUM_EXPERTS);
    }

    const float normalizing_factor = 1.f / row_sum;

    const int idx = K * block_row + k_idx;
    if (threadIdx.x == indices[idx]) {
      const int input_idx = thread_row_offset + threadIdx.x;
      output[idx] = logits_mask[k_idx] ? 0
                                       : exp((static_cast<float>(inputs[input_idx]) - result_kvp_value[k_idx])) *
                                             normalizing_factor;
    }
  }
}

template <typename T>
void LaunchSparseMixerTop2Impl(
    const T* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int TPB = WARP_SIZE * WARPS_PER_TB;
  static constexpr float jitter_eps = 0.01f;

  switch (num_experts) {
    case 8: {
      sparse_mixer_top2<T, TPB, 8><<<num_rows, TPB, 0, stream>>>(input, output, indices, source_rows, jitter_eps);
      break;
    }
    case 16: {
      sparse_mixer_top2<T, TPB, 16><<<num_rows, TPB, 0, stream>>>(input, output, indices, source_rows, jitter_eps);
      break;
    }
    // Replicate logic for other sizes if needed, or fallback/throw
    default: {
      ORT_THROW("Sparse mixer only supports 8 or 16 experts, got ", num_experts);
    }
  }
}

void LaunchSparseMixerTop2(
    const float* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream) {
  LaunchSparseMixerTop2Impl<float>(input, output, indices, source_rows, num_rows, num_experts, stream);
}

void LaunchSparseMixerTop2(
    const half* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream) {
  LaunchSparseMixerTop2Impl<half>(input, output, indices, source_rows, num_rows, num_experts, stream);
}

void LaunchSparseMixerTop2(
    const __nv_bfloat16* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream) {
  LaunchSparseMixerTop2Impl<__nv_bfloat16>(input, output, indices, source_rows, num_rows, num_experts, stream);
}

template <typename T>
__global__ void QMoETranspose2DKernel(const T* input, T* output, int num_elements_per_batch, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int batch = blockIdx.z;

  if (col < cols && row < rows) {
    int in_idx = batch * num_elements_per_batch + row * cols + col;
    int out_idx = batch * num_elements_per_batch + col * rows + row;
    output[out_idx] = input[in_idx];
  }
}

void LaunchQMoETranspose2D(
    const float* input,
    float* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, batch_size);
  QMoETranspose2DKernel<float><<<grid, block, 0, stream>>>(input, output, rows * cols, rows, cols);
}

void LaunchQMoETranspose2D(
    const half* input,
    half* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, batch_size);
  QMoETranspose2DKernel<half><<<grid, block, 0, stream>>>(input, output, rows * cols, rows, cols);
}

void LaunchQMoETranspose2D(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, batch_size);
  QMoETranspose2DKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(input, output, rows * cols, rows, cols);
}

void LaunchQMoETranspose2D(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream) {
  dim3 block(32, 32);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, batch_size);
  QMoETranspose2DKernel<uint8_t><<<grid, block, 0, stream>>>(input, output, rows * cols, rows, cols);
}

__device__ __forceinline__ int64_t QMoEBlockScaleInterleaveOffset(
    int batch, int row, int col, int rows_padded, int cols_padded) {
  int64_t num_k_tiles = (cols_padded + 3) / 4;
  int64_t m_tile_idx = row / 128;
  int64_t k_tile_idx = col / 4;
  int64_t tile_offset = ((m_tile_idx * num_k_tiles) + k_tile_idx) * 512;
  int64_t intra_tile_offset = (row % 32) * 16 + ((row % 128) / 32) * 4 + (col % 4);
  int64_t batch_stride = ((rows_padded + 127) / 128) * num_k_tiles * 512;
  return static_cast<int64_t>(batch) * batch_stride + tile_offset + intra_tile_offset;
}

__global__ void QMoEBlockScaleInterleaveKernel(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    int rows_padded,
    int cols_padded) {
  for (int row = blockIdx.x; row < rows_padded; row += gridDim.x) {
    for (int batch = 0; batch < batch_size; ++batch) {
      for (int col = threadIdx.x; col < cols_padded; col += blockDim.x) {
        uint8_t scale = 0;
        if (row < rows && col < cols) {
          scale = input[static_cast<int64_t>(batch) * rows * cols + row * cols + col];
        }
        output[QMoEBlockScaleInterleaveOffset(batch, row, col, rows_padded, cols_padded)] = scale;
      }
    }
  }
}

void LaunchQMoEBlockScaleInterleave(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    int rows_padded,
    int cols_padded,
    int multi_processor_count,
    cudaStream_t stream) {
  dim3 block(std::min(cols_padded, 1024));
  int num_blocks_per_sm = std::max(1, 4096 / static_cast<int>(block.x));
  dim3 grid(std::min(rows_padded, multi_processor_count * num_blocks_per_sm));
  QMoEBlockScaleInterleaveKernel<<<grid, block, 0, stream>>>(
      input, output, batch_size, rows, cols, rows_padded, cols_padded);
}

__device__ __forceinline__ float DecodeFp4E2M1(uint8_t code) {
  constexpr float kValues[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float value = kValues[code & 0x7];
  return (code & 0x8) ? -value : value;
}

__device__ __forceinline__ float DecodeUE8M0(uint8_t code) {
  return code == 0 ? 0.0f : exp2f(static_cast<int>(code) - 127);
}

template <typename T>
__global__ void QMoEDequantizeFp4WeightsKernel(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int num_experts,
    int n,
    int k) {
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int64_t expert_stride = static_cast<int64_t>(n) * k;
  int expert = static_cast<int>(index / expert_stride);
  int64_t offset = index - static_cast<int64_t>(expert) * expert_stride;
  int row = static_cast<int>(offset / k);
  int col = static_cast<int>(offset - static_cast<int64_t>(row) * k);

  int packed_n = n / 2;
  uint8_t packed = packed_weights[(static_cast<int64_t>(expert) * k + col) * packed_n + row / 2];
  uint8_t fp4_code = (row & 1) == 0 ? (packed & 0x0F) : (packed >> 4);

  int scale_k = k / 32;
  uint8_t scale_code = block_scales[(static_cast<int64_t>(expert) * n + row) * scale_k + col / 32];
  float value = DecodeFp4E2M1(fp4_code) * DecodeUE8M0(scale_code) * global_scales[expert];
  output[index] = static_cast<T>(value);
}

template <typename T>
void LaunchQMoEDequantizeFp4WeightsImpl(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    T* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  constexpr int block = 256;
  int grid = onnxruntime::narrow<int>((total + block - 1) / block);
  QMoEDequantizeFp4WeightsKernel<<<grid, block, 0, stream>>>(
      packed_weights, block_scales, global_scales, output, num_experts, n, k);
}

void LaunchQMoEDequantizeFp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  LaunchQMoEDequantizeFp4WeightsImpl(packed_weights, block_scales, global_scales, output, num_experts, n, k, stream);
}

void LaunchQMoEDequantizeFp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  LaunchQMoEDequantizeFp4WeightsImpl(packed_weights, block_scales, global_scales, output, num_experts, n, k, stream);
}

__device__ __forceinline__ float DecodeFloat8E4M3FN(uint8_t code) {
  // ONNX float8e4m3fn has no infinities. The only NaN payloads are 0x7F/0xFF;
  // finite values, including the max finite code 0x7E, use the normal E4M3 formula.
  const int sign = code & 0x80;
  const int exponent = (code >> 3) & 0x0F;
  const int mantissa = code & 0x07;

  if ((code & 0x7F) == 0) {
    return sign ? -0.0f : 0.0f;
  }
  if (exponent == 0x0F && mantissa == 0x07) {
    return __int_as_float(0x7fffffff);
  }

  float value = 0.0f;
  if (exponent == 0) {
    value = ldexpf(static_cast<float>(mantissa), -9);
  } else {
    value = ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
  }
  return sign ? -value : value;
}

template <typename T>
__global__ void QMoEDequantizeFp8WeightsKernel(
    const uint8_t* weights,
    const float* global_scales,
    T* output,
    int num_experts,
    int n,
    int k) {
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  int64_t expert_stride = static_cast<int64_t>(n) * k;
  int expert = static_cast<int>(index / expert_stride);
  float value = DecodeFloat8E4M3FN(weights[index]) * global_scales[expert];
  output[index] = static_cast<T>(value);
}

template <typename T>
void LaunchQMoEDequantizeFp8WeightsImpl(
    const uint8_t* weights,
    const float* global_scales,
    T* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  int64_t total = static_cast<int64_t>(num_experts) * n * k;
  constexpr int block = 256;
  int grid = onnxruntime::narrow<int>((total + block - 1) / block);
  QMoEDequantizeFp8WeightsKernel<<<grid, block, 0, stream>>>(
      weights, global_scales, output, num_experts, n, k);
}

void LaunchQMoEDequantizeFp8Weights(
    const uint8_t* weights,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  LaunchQMoEDequantizeFp8WeightsImpl(weights, global_scales, output, num_experts, n, k, stream);
}

void LaunchQMoEDequantizeFp8Weights(
    const uint8_t* weights,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream) {
  LaunchQMoEDequantizeFp8WeightsImpl(weights, global_scales, output, num_experts, n, k, stream);
}

// Repack column-major FP4 packed weights to row-major layout.
// Input: [experts, k, n/2] packed col-major (each byte holds 2 values along n).
// Output: [experts, n, k/2] packed row-major (each byte holds 2 values along k).
// One thread per output byte.
__global__ void QMoERepackFP4ColToRowKernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int experts,
    int64_t k,
    int64_t n) {
  const int64_t k_half = k / 2;
  const int64_t n_half = n / 2;
  const int64_t out_expert_stride = n * k_half;
  const int64_t in_expert_stride = k * n_half;
  const int64_t total = static_cast<int64_t>(experts) * out_expert_stride;

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int64_t expert = idx / out_expert_stride;
  int64_t rem = idx - expert * out_expert_stride;
  int64_t row = rem / k_half;       // output row index [0, n)
  int64_t col_byte = rem % k_half;  // output byte index [0, k/2)

  int64_t col_even = col_byte * 2;
  int64_t col_odd = col_even + 1;

  // Source byte addresses: src[expert][col][row/2]
  int64_t in_base = expert * in_expert_stride;
  uint8_t src_even = input[in_base + col_even * n_half + row / 2];
  uint8_t src_odd = input[in_base + col_odd * n_half + row / 2];

  // Extract nibble based on row parity
  uint8_t low_code = (row % 2 == 0) ? (src_even & 0x0F) : ((src_even >> 4) & 0x0F);
  uint8_t high_code = (row % 2 == 0) ? (src_odd & 0x0F) : ((src_odd >> 4) & 0x0F);

  output[idx] = low_code | static_cast<uint8_t>(high_code << 4);
}

void LaunchQMoERepackFP4ColToRow(
    const uint8_t* input,
    uint8_t* output,
    int experts,
    int64_t k,
    int64_t n,
    cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(experts) * n * (k / 2);
  constexpr int kThreads = 256;
  int blocks = onnxruntime::narrow<int>((total + kThreads - 1) / kThreads);
  QMoERepackFP4ColToRowKernel<<<blocks, kThreads, 0, stream>>>(
      input, output, experts, k, n);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

namespace onnxruntime::llm::kernels {

template <typename T>
__global__ void BatchedTransposeKernel(const T* __restrict__ input, T* __restrict__ output, int batch, int rows, int cols) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t matrix_size = static_cast<int64_t>(rows) * cols;
  int64_t total_size = static_cast<int64_t>(batch) * matrix_size;

  if (idx < total_size) {
    int64_t b = idx / matrix_size;
    int64_t rem = idx % matrix_size;
    int r = rem / cols;
    int c = rem % cols;

    int64_t out_idx = b * matrix_size + static_cast<int64_t>(c) * rows + r;
    output[out_idx] = input[idx];
  }
}

void LaunchBatchedTranspose(cudaStream_t stream, const void* input, void* output, int batch, int rows, int cols, int element_size) {
  int64_t total_elements = static_cast<int64_t>(batch) * rows * cols;
  int threads = 256;
  int blocks = onnxruntime::narrow<int>((total_elements + threads - 1) / threads);

  if (element_size == 1) {
    BatchedTransposeKernel<uint8_t><<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output), batch, rows, cols);
  } else if (element_size == 2) {
    BatchedTransposeKernel<uint16_t><<<blocks, threads, 0, stream>>>(static_cast<const uint16_t*>(input), static_cast<uint16_t*>(output), batch, rows, cols);
  } else if (element_size == 4) {
    BatchedTransposeKernel<uint32_t><<<blocks, threads, 0, stream>>>(static_cast<const uint32_t*>(input), static_cast<uint32_t*>(output), batch, rows, cols);
  } else {
    ORT_THROW("LaunchBatchedTranspose: unsupported element_size ", element_size,
              " (supported: 1, 2, 4)");
  }
}

}  // namespace onnxruntime::llm::kernels

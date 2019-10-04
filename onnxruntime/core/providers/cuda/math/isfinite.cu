#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "isfinite.h"

namespace onnxruntime {
namespace cuda {

template<typename T>
__device__ __forceinline__ bool _IsFiniteScalar(const T value) {
  return isfinite(value);
}

template<>
__device__ __forceinline__ bool _IsFiniteScalar(const half value) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return !__hisinf(value) && !__hisnan(value);
#else
  return isfinite(float(value));
#endif
}

template <typename TSrc>
__global__ void _IsFinite(const TSrc* input, bool* output, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = _IsFiniteScalar(input[id]);
}

template <typename TSrc>
void IsFinite(const TSrc* input, bool* output, size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _IsFinite<<<blocksPerGrid, GridDim::maxThreadsPerBlock>>>(input, output, N);
}

#define SPECIALIZE_ISFINITE_IMPL(T) \
template void IsFinite(const T* input, bool* output, size_t count);

SPECIALIZE_ISFINITE_IMPL(half)
SPECIALIZE_ISFINITE_IMPL(float)
SPECIALIZE_ISFINITE_IMPL(double)

// Have one block to process one chunk.
template <typename TSrc>
__global__ void _IsFinite(
    const ChunkGroup<TSrc> chunks,
    bool* output) {
  const int block_idx = blockIdx.x;
  const int tensor_idx = chunks.block_index_to_tensor_index[block_idx];
  const int tensor_size = chunks.tensor_sizes[tensor_idx];
  const TSrc* tensor_ptr = chunks.tensor_ptrs[tensor_idx];
  const int chunk_start_idx = chunks.block_index_to_chunk_start_index[block_idx];
  // chunk_size is chunks.chunk_size if the loaded chunk is full. Otherwise (this
  // chunk is the last one in the source tensor), the actual size is determined
  // by the bound of the source tensor.
  const int chunk_size = min(tensor_size, chunk_start_idx + chunks.chunk_size) - chunk_start_idx;

  const TSrc* chunk_ptr = tensor_ptr + chunk_start_idx;
  bool result = true;
#pragma unroll(4)
  for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
      result &= _IsFiniteScalar(chunk_ptr[i]);
  }

  if (!result) {
    *output = false;
  }
}

template <typename TSrc>
void IsAllFinite(const ChunkGroup<TSrc> chunks, bool* output) {
  const int block_count = std::min(MAX_BLOCK_COUNT, chunks.chunk_count);
  // One thread loads PARALLEL_LOADS elements.
  const int thread_count = (chunks.chunk_size + PARALLEL_LOADS - 1) / PARALLEL_LOADS;
  // One warp contains 32 threads.
  const int warp_count = (thread_count + WARP_THREAD_COUNT - 1) / WARP_THREAD_COUNT;
  // Thread count should be greater than warp count * 32 and smaller than a pre-specified threshold.
  const int block_thread_count = std::min(MAX_BLOCK_THREAD_COUNT, warp_count * WARP_THREAD_COUNT);
  _IsFinite<<<block_count, block_thread_count, 0>>>(chunks, output);
}

#define SPECIALIZE_ISALLFINITE_IMPL(T) \
template void IsAllFinite(const ChunkGroup<T> chunks, bool* output);

SPECIALIZE_ISALLFINITE_IMPL(half)
SPECIALIZE_ISALLFINITE_IMPL(float)
SPECIALIZE_ISALLFINITE_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
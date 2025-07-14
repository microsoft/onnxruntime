#pragma once

#define ENABLE_FLASH_DEBUG 1

#ifdef ENABLE_FLASH_DEBUG
#include <cuda_runtime.h>
#include <cstdio>

namespace onnxruntime {
namespace flash {

extern __device__ volatile int flash_debug_block_sync;

__device__ __forceinline__ int get_linear_block_id() {
  dim3 gridDim_ = gridDim;
  return blockIdx.x + blockIdx.y * gridDim_.x + blockIdx.z * gridDim_.x * gridDim_.y;
}

// Block sync helper (no actual work here)
__device__ __forceinline__ void flash_debug_block_sync_wait(int my_block_id) {
  if (threadIdx.x == 0) {
    while (atomicAdd((int*)&flash_debug_block_sync, 0) != my_block_id) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(1000);
#endif
    }
    printf("FLASH block %d (%d,%d,%d) executing\n", my_block_id, blockIdx.x, blockIdx.y, blockIdx.z);
  }
  __syncthreads();
}

__device__ __forceinline__ void flash_debug_block_sync_advance() {
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd((int*)&flash_debug_block_sync, 1);
  }
}

// Macro for debug-synchronized block execution
#define FLASH_DEBUG_SYNC_RUN(block_id_expr, work_block) \
  do {                                                  \
    int __block_id = block_id_expr;                     \
    flash_debug_block_sync_wait(__block_id);            \
    work_block                                          \
    flash_debug_block_sync_advance();                   \
  } while (0)

#define FLASH_DEBUG_BLOCK_SYNC_BEGIN      \
  int __block_id = get_linear_block_id(); \
  flash_debug_block_sync_wait(__block_id);

#define FLASH_DEBUG_BLOCK_SYNC_END \
  flash_debug_block_sync_advance();

template <typename T>
__device__ __forceinline__ void print_1d_tensor_values(const T& tensor) {
  printf("Tensor size: %d:\n", (int)tensor.size());
  // This loop flattens the tensor for printing.
  // T is the type, decltype(tensor.size()) is the size type.
  for (int i = 0; i < tensor.size(); ++i) {
    // The `(T)tensor(i)` cast is important to ensure the correct type is printed.
    // Use `printf` instead of `std::cout` in CUDA device code.
    printf("%f ", (float)tensor(i));
  }
  printf("\n");
}

template <typename T>
__device__ __forceinline__ void print_2d_tensor_values(const T& tensor) {
  printf("Tensor size: %d x %d):\n", (int)size<0>(tensor), (int)size<1>(tensor));
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      printf("%f ", (float)tensor(mi, ni));
    }
    printf("\n");
  }
}

}  // namespace flash
}  // namespace onnxruntime
#else
#define FLASH_DEBUG_BLOCK_SYNC_BEGIN
#define FLASH_DEBUG_BLOCK_SYNC_END
#endif

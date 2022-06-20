/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

#include "orttraining/training_ops/cuda/nn/dropout_grad_impl.h"

#include <algorithm>
#include "core/providers/cuda/cu_inc/bitmask.cuh"

namespace onnxruntime {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

template <typename T, bool UseBitmask>
__global__ void DropoutGradientKernel(const int64_t N, const fast_divmod fdm_bits_per_element, const T* dY_data,
                                      const void* mask_data, const float scale, T* dX_data) {
  CUDA_LONG id = (blockDim.x * blockIdx.x + threadIdx.x) * kNumUnroll;
  bool masks[kNumUnroll];
  if (UseBitmask && id < N) {
    GetMasks<kNumUnroll>(id, fdm_bits_per_element, reinterpret_cast<const BitmaskElementType*>(mask_data), masks);
  }

#pragma unroll
  for (int i = 0; i < kNumUnroll; ++i) {
    CUDA_LONG li = id + i;
    if (li < N) {
      bool mask = UseBitmask ? masks[i] : reinterpret_cast<const bool*>(mask_data)[li];
      dX_data[li] = static_cast<T>(static_cast<float>(dY_data[li]) * mask * scale);
    }
  }
}

template <typename T, bool UseBitmask>
__global__ void DropoutGradientVectorizedKernel(const int64_t N, const fast_divmod fdm_bits_per_element,
                                                const T* dY_data, const void* mask_data, const float scale,
                                                T* dX_data) {
  // using vectorized data load/store approach when N % 4 == 0
  // since this is typical case for input shape size
  using LoadT = aligned_vector<T, kNumUnroll>;
  using MaskLoadT = aligned_vector<bool, kNumUnroll>;

  CUDA_LONG id = (blockDim.x * blockIdx.x + threadIdx.x) * kNumUnroll;

  if (id < N) {
    // vectorized load into storage
    T src[kNumUnroll];
    LoadT* value1 = reinterpret_cast<LoadT*>(&src);
    *value1 = *reinterpret_cast<const LoadT*>(&dY_data[id]);

    bool masks[kNumUnroll];
    if (UseBitmask) {
      GetMasks<kNumUnroll>(id, fdm_bits_per_element, reinterpret_cast<const BitmaskElementType*>(mask_data), masks);
    } else {
      MaskLoadT* value2 = reinterpret_cast<MaskLoadT*>(&masks);
      *value2 = *reinterpret_cast<const MaskLoadT*>(&reinterpret_cast<const bool*>(mask_data)[id]);
    }

    T r[kNumUnroll];

// actual computation
#pragma unroll
    for (int ii = 0; ii < kNumUnroll; ++ii) {
      r[ii] = static_cast<T>(static_cast<float>(src[ii]) * masks[ii] * scale);
    }

    // Vectorized writes for dX_data
    *(reinterpret_cast<LoadT*>(&dX_data[id])) = *reinterpret_cast<LoadT*>(&r[0]);
  }
}

#define LAUNCH_DROPOUT_GRAD_KERNEL(FuncName, UseBitmask) \
  FuncName<T, UseBitmask>                                \
      <<<blocksPerGrid, kBlockSize, 0, stream>>>(N, fdm_bits_per_element, dY_data, mask_data, scale, dX_data)

#define HANDLE_DROPOUT_GRAD_USE_BITMASK(FuncName) \
  if (use_bitmask) {                              \
    LAUNCH_DROPOUT_GRAD_KERNEL(FuncName, true);   \
  } else {                                        \
    LAUNCH_DROPOUT_GRAD_KERNEL(FuncName, false);  \
  }

template <typename T>
void DropoutGradientKernelImpl(cudaStream_t stream, const int64_t N, const T* dY_data, const void* mask_data,
                               const float ratio, T* dX_data, bool use_bitmask) {
  if (ratio == 0.0f) {
    if (dY_data != dX_data) {
      CUDA_CALL_THROW(cudaMemcpyAsync(dX_data, dY_data, N * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    }
  } else {
    const float scale = 1.f / (1.f - ratio);
    const int blocksPerGrid = static_cast<int>(CeilDiv(N, kBlockSize * kNumUnroll));
    fast_divmod fdm_bits_per_element(kNumBitsPerBitmaskElement);
    if (N % kNumUnroll != 0) {
      HANDLE_DROPOUT_GRAD_USE_BITMASK(DropoutGradientKernel);
    } else {
      HANDLE_DROPOUT_GRAD_USE_BITMASK(DropoutGradientVectorizedKernel);
    }
  }
}

#undef HANDLE_DROPOUT_GRAD_USE_BITMASK
#undef LAUNCH_DROPOUT_GRAD_KERNEL

#define SPECIALIZED_DROPOUT_GRAD_IMPL(T)                                                             \
  template void DropoutGradientKernelImpl<T>(cudaStream_t stream, const int64_t N, const T* dY_data, \
                                             const void* mask_data, const float scale, T* dX_data, bool use_bitmask);

SPECIALIZED_DROPOUT_GRAD_IMPL(float)
SPECIALIZED_DROPOUT_GRAD_IMPL(double)
SPECIALIZED_DROPOUT_GRAD_IMPL(half)
SPECIALIZED_DROPOUT_GRAD_IMPL(BFloat16)

#undef SPECIALIZED_DROPOUT_GRAD_IMPL

}  // namespace cuda
}  // namespace onnxruntime

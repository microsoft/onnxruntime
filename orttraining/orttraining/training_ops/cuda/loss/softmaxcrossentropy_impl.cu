// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {
#ifdef USE_ROCM
constexpr int kElementsPerThread = 2;
constexpr int kThreadsPerBlock = 512;
#else
constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
#endif
}  // namespace

template <typename T>
__global__ void _SoftMaxCrossEntropy(const T* log_prob_data, const T* label_data, T normalize_factor, T* output_data,
                                     CUDA_LONG N) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      value[i] = -log_prob_data[id] * label_data[id] / normalize_factor;
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T>
void SoftMaxCrossEntropyImpl(cudaStream_t stream, const T* log_prob, const T* label, size_t normalize_factor,
                             T* output_data, size_t count) {
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  int blocksPerGrid = CeilDiv(N, kElementsPerThread * kThreadsPerBlock);
  T nf = static_cast<T>(normalize_factor);
  _SoftMaxCrossEntropy<T><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(log_prob, label, nf, output_data, N);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyImpl(T)                                                  \
  template void SoftMaxCrossEntropyImpl(cudaStream_t stream, const T* log_prob, const T* label, \
                                        size_t normalize_factor, T* output_data, size_t count)

SPECIALIZED_IMPL_SoftMaxEntropyImpl(float);

template <typename T>
__global__ void _SoftMaxCrossEntropyGrad(const T* dY, const T* log_prob, const T* label, T normalize_factor,
                                         T* output_data, CUDA_LONG N) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      value[i] = (_Exp(log_prob[id]) - label[id]) * (*dY) / normalize_factor;
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T>
void SoftMaxCrossEntropyGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const T* label,
                                 size_t normalize_factor, T* output_data, size_t count) {
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  int blocksPerGrid = CeilDiv(N, kElementsPerThread * kThreadsPerBlock);
  T nf = static_cast<T>(normalize_factor);
  _SoftMaxCrossEntropyGrad<T><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(dY, log_prob, label, nf, output_data, N);
}

#define SPECIALIZED_IMPL_SoftMaxEntropyGradImpl(T)                                                               \
  template void SoftMaxCrossEntropyGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const T* label, \
                                            size_t normalize_factor, T* output_data, size_t count)

SPECIALIZED_IMPL_SoftMaxEntropyGradImpl(float);

template <typename T, typename Tin, bool IsWeighted>
__global__ void _SparseSoftmaxCrossEntropy(const T* log_prob_data, const Tin* label_data, const T* weight_data,
                                           const T* normalize_factor_data, T* output_data, Tin D, CUDA_LONG N) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      if (*normalize_factor_data == T(0.f)) {
        value[i] = T(0.f);
      } else {
        CUDA_KERNEL_ASSERT(label_data[id] >= 0 && label_data[id] < D);
        value[i] = -log_prob_data[id * D + label_data[id]] * (IsWeighted ? weight_data[id] : T(1.f)) /
                   (*normalize_factor_data);
      }
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyImpl(cudaStream_t stream, const T* log_prob, const Tin* label, const T* weight,
                                   const T* normalize_factor, T* output_data, size_t count, size_t label_depth) {
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  int blocksPerGrid = CeilDiv(N, kElementsPerThread * kThreadsPerBlock);
  Tin D = static_cast<Tin>(label_depth);
  if (weight) {
    _SparseSoftmaxCrossEntropy<T, Tin, true>
        <<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(log_prob, label, weight, normalize_factor, output_data, D, N);
  } else {
    _SparseSoftmaxCrossEntropy<T, Tin, false>
        <<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(log_prob, label, weight, normalize_factor, output_data, D, N);
  }
}

#define SPECIALIZED_IMPL_SparseSoftMaxEntropyImpl(T, Tin)                                                 \
  template void SparseSoftmaxCrossEntropyImpl(cudaStream_t stream, const T* log_prob, const Tin* label,   \
                                              const T* weight, const T* normalize_factor, T* output_data, \
                                              size_t count, size_t label_depth)

SPECIALIZED_IMPL_SparseSoftMaxEntropyImpl(float, int32_t);
SPECIALIZED_IMPL_SparseSoftMaxEntropyImpl(float, int64_t);

template <typename T, typename Tin, bool IsWeighted>
__global__ void _SparseSoftmaxCrossEntropyGrad(const T* dY, const T* log_prob, const Tin* label, const T* weight,
                                               const T* normalize_factor, T* output_data, fast_divmod D_fdm,
                                               CUDA_LONG N) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      if (*normalize_factor == T(0.f)) {
        value[i] = T(0.f);
      } else {
        int row, d;
        D_fdm.divmod(id, row, d);
        value[i] = (*dY) * (IsWeighted ? weight[row] : T(1.f)) * (_Exp(log_prob[id]) - (T)(d == label[row])) /
                   (*normalize_factor);
      }
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const Tin* label,
                                       const T* weight, const T* normalize_factor, T* output_data, size_t count,
                                       size_t label_depth) {
  CUDA_LONG N = static_cast<CUDA_LONG>(count * label_depth);
  int blocksPerGrid = CeilDiv(N, kElementsPerThread * kThreadsPerBlock);
  fast_divmod D_fdm(static_cast<int>(label_depth));
  if (weight) {
    _SparseSoftmaxCrossEntropyGrad<T, Tin, true><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
        dY, log_prob, label, weight, normalize_factor, output_data, D_fdm, N);
  } else {
    _SparseSoftmaxCrossEntropyGrad<T, Tin, false><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
        dY, log_prob, label, weight, normalize_factor, output_data, D_fdm, N);
  }
}

#define SPECIALIZED_IMPL_SparseSoftMaxEntropyGradImpl(T, Tin)                                                   \
  template void SparseSoftmaxCrossEntropyGradImpl(cudaStream_t stream, const T* dY, const T* log_prob,          \
                                                  const Tin* label, const T* weight, const T* normalize_factor, \
                                                  T* output_data, size_t count, size_t label_depth)

SPECIALIZED_IMPL_SparseSoftMaxEntropyGradImpl(float, int32_t);
SPECIALIZED_IMPL_SparseSoftMaxEntropyGradImpl(float, int64_t);

}  // namespace cuda
}  // namespace onnxruntime

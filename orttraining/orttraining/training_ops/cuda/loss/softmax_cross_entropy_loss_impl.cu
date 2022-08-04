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

template <typename T, typename Tin, bool IsWeighted>
__global__ void _ComputeWeightsSoftmaxCrossEntropy(T* weight_data_nd, const Tin* label_data, const T* weight_data,
                                                   CUDA_LONG N_D, Tin C, Tin ignore_index) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];
  bool mask[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N_D) {
      mask[i] = label_data[id] != ignore_index;
      if (mask[i]) {
        CUDA_KERNEL_ASSERT(label_data[id] >= 0 && label_data[id] < C);
        value[i] = IsWeighted ? weight_data[label_data[id]] : T(1.f);
      }
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N_D) {
      if (mask[i]) weight_data_nd[id] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T, typename Tin>
void ComputeWeightsSoftmaxCrossEntropyImpl(cudaStream_t stream, const Tin* label, const T* weight, size_t count,
                                           size_t label_depth, int64_t ignore_index, T* weight_data_nd) {
  CUDA_LONG N_D = static_cast<CUDA_LONG>(count);
  int blocksPerGrid = CeilDiv(N_D, kElementsPerThread * kThreadsPerBlock);
  Tin C = static_cast<Tin>(label_depth);
  Tin II = static_cast<Tin>(ignore_index);
  if (weight) {
    _ComputeWeightsSoftmaxCrossEntropy<T, Tin, true>
        <<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(weight_data_nd, label, weight, N_D, C, II);
  } else {
    _ComputeWeightsSoftmaxCrossEntropy<T, Tin, false>
        <<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(weight_data_nd, label, weight, N_D, C, II);
  }
}

template <typename T, typename TAcc, typename Tin>
__global__ void _WeightedSoftmaxCrossEntropyLoss(const T* log_prob_data, const Tin* label_data, const T* weight_data,
                                                 const TAcc* normalize_factor_data, T* output_data, CUDA_LONG N_D,
                                                 Tin C, Tin ignore_index) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N_D) {
      if (label_data[id] == ignore_index) {
        value[i] = T(0.f);
      } else {
        CUDA_KERNEL_ASSERT(label_data[id] >= 0 && label_data[id] < C);
        value[i] = static_cast<T>(static_cast<TAcc>(-log_prob_data[id * C + label_data[id]] * weight_data[id]) /
                                  (*normalize_factor_data));
      }
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N_D) {
      output_data[id] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T, typename TAcc, typename Tin>
void SoftmaxCrossEntropyLossImpl(cudaStream_t stream, const T* log_prob, const Tin* label, const T* weight,
                                 const TAcc* normalize_factor, size_t count, size_t label_depth, int64_t ignore_index,
                                 T* output_data) {
  CUDA_LONG N_D = static_cast<CUDA_LONG>(count);
  int blocksPerGrid = CeilDiv(N_D, kElementsPerThread * kThreadsPerBlock);
  Tin C = static_cast<Tin>(label_depth);
  Tin II = static_cast<Tin>(ignore_index);
  _WeightedSoftmaxCrossEntropyLoss<T, TAcc, Tin><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
      log_prob, label, weight, normalize_factor, output_data, N_D, C, II);
}

#define INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(T, TAcc, Tin)                                                          \
  template void SoftmaxCrossEntropyLossImpl(cudaStream_t stream, const T* log_prob, const Tin* label, const T* weight, \
                                            const TAcc* normalize_factor, size_t count, size_t label_depth,            \
                                            int64_t ignore_index, T* output_data)

INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(float, float, int32_t);
INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(float, float, int64_t);
INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(half, float, int64_t);
INSTANTIATE_IMPL_SoftMaxEntropyLossImpl(BFloat16, float, int64_t);

template <typename T, typename TAcc, typename Tin, bool IsReductionNone>
__global__ void _WeightedSoftmaxCrossEntropyLossGrad(const T* dY, const T* log_prob, const Tin* label, const T* weight,
                                                     const TAcc* normalize_factor, T* output_data, fast_divmod C_fdm,
                                                     Tin C, CUDA_LONG N) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      if (*normalize_factor == TAcc(0.f)) {
        // normalize_factor is sum of labels' weights. Because zero
        // sum implies all weights are 0, the loss function should
        // be constant 0 and its corresponding gradient should be 0 as well.
        value[i] = T(0.f);
      } else {
        int row, d;
        C_fdm.divmod(id, row, d);
        CUDA_KERNEL_ASSERT(weight[row] == T(0.f) || (label[row] >= 0 && label[row] < C));
        value[i] = static_cast<T>(static_cast<TAcc>((IsReductionNone ? dY[row] : *dY) * weight[row]) *
                                  (_Exp(static_cast<TAcc>(log_prob[id])) - TAcc(1.f) * (TAcc)(d == label[row])) /
                                  (*normalize_factor));
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

template <typename T, typename TAcc, typename Tin>
void SoftmaxCrossEntropyLossGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const Tin* label,
                                     const T* weight, const TAcc* normalize_factor, size_t count, size_t label_depth,
                                     bool reduction_none, T* output_data) {
  CUDA_LONG N = static_cast<CUDA_LONG>(count * label_depth);
  int blocksPerGrid = CeilDiv(N, kElementsPerThread * kThreadsPerBlock);
  Tin C = static_cast<Tin>(label_depth);
  fast_divmod C_fdm(static_cast<int>(label_depth));
  if (reduction_none) {
    _WeightedSoftmaxCrossEntropyLossGrad<T, TAcc, Tin, true><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
        dY, log_prob, label, weight, normalize_factor, output_data, C_fdm, C, N);
  } else {
    _WeightedSoftmaxCrossEntropyLossGrad<T, TAcc, Tin, false><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
        dY, log_prob, label, weight, normalize_factor, output_data, C_fdm, C, N);
  }
}

#define INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(T, TAcc, Tin)                                                      \
  template void SoftmaxCrossEntropyLossGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const Tin* label, \
                                                const T* weight, const TAcc* normalize_factor, size_t count,           \
                                                size_t label_depth, bool reducation_none, T* output_data)

INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(float, float, int32_t);
INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(float, float, int64_t);
INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(half, float, int64_t);
INSTANTIATE_IMPL_SoftMaxEntropyLossGradImpl(BFloat16, float, int64_t);

#define INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(T, Tin)                                        \
  template void ComputeWeightsSoftmaxCrossEntropyImpl(cudaStream_t stream, const Tin* label, const T* weight, \
                                                      size_t count, size_t label_depth, int64_t ignore_index, \
                                                      T* weight_data_nd)

INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(float, int32_t);
INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(float, int64_t);
INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(half, int64_t);
INSTANTIATE_IMPL_ComputeWeightsSoftmaxCrossEntropyImpl(BFloat16, int64_t);

}  // namespace cuda
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
struct OpSoftmaxCrossEntropy {
  OpSoftmaxCrossEntropy(const T* log_prob_data, const T* label_data, T normalize_factor)
      : log_prob_data_(log_prob_data), label_data_(label_data), normalize_factor_(normalize_factor) {}

  __device__ __inline__ T operator()(CUDA_LONG idx) const {
    return -log_prob_data_[idx] * label_data_[idx] / normalize_factor_;
  }

  const T* log_prob_data_;
  const T* label_data_;
  T normalize_factor_;
};

template <typename T>
void SoftMaxCrossEntropyImpl(cudaStream_t stream, const T* log_prob, const T* label, size_t normalize_factor,
                             T* output_data, size_t count) {
  OpSoftmaxCrossEntropy<T> op(log_prob, label, static_cast<T>(normalize_factor));
  LaunchElementwiseKernel<T, decltype(op), CUDA_LONG>(stream, output_data, op, static_cast<CUDA_LONG>(count));
}

template void SoftMaxCrossEntropyImpl(cudaStream_t stream, const float* log_prob, const float* label,
                                      size_t normalize_factor, float* output_data, size_t count);

template <typename T>
struct OpSoftmaxCrossEntropyGrad {
  OpSoftmaxCrossEntropyGrad(const T* dY_data, const T* log_prob_data, const T* label_data, T normalize_factor)
      : dY_data_(dY_data),
        log_prob_data_(log_prob_data),
        label_data_(label_data),
        normalize_factor_(normalize_factor) {}

  __device__ __inline__ T operator()(CUDA_LONG idx) const {
    return (_Exp(log_prob_data_[idx]) - label_data_[idx]) * (*dY_data_) / normalize_factor_;
  }

  const T* dY_data_;
  const T* log_prob_data_;
  const T* label_data_;
  T normalize_factor_;
};

template <typename T>
void SoftMaxCrossEntropyGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const T* label,
                                 size_t normalize_factor, T* output_data, size_t count) {
  OpSoftmaxCrossEntropyGrad<T> op(dY, log_prob, label, static_cast<T>(normalize_factor));
  LaunchElementwiseKernel<T, decltype(op), CUDA_LONG>(stream, output_data, op, static_cast<CUDA_LONG>(count));
}

template void SoftMaxCrossEntropyGradImpl(cudaStream_t stream, const float* dY, const float* log_prob,
                                          const float* label, size_t normalize_factor, float* output_data,
                                          size_t count);

template <typename T, typename Tin, bool IsWeighted>
struct OpSparseSoftmaxCrossEntropy {
  OpSparseSoftmaxCrossEntropy(const T* log_prob_data, const Tin* label_data, const T* weight_data,
                              const T* normalize_factor_data, Tin D)
      : log_prob_data_(log_prob_data),
        label_data_(label_data),
        weight_data_(weight_data),
        normalize_factor_data_(normalize_factor_data),
        D_(D) {}

  __device__ __inline__ T operator()(CUDA_LONG idx) const {
    if (*normalize_factor_data_ != T(0.f)) {
      CUDA_KERNEL_ASSERT(label_data_[idx] >= 0 && label_data_[idx] < D_);
      return -log_prob_data_[idx * D_ + label_data_[idx]] * (IsWeighted ? weight_data_[idx] : T(1.f)) /
             (*normalize_factor_data_);
    }
    return T(0.f);
  }

  const T* log_prob_data_;
  const Tin* label_data_;
  const T* weight_data_;
  const T* normalize_factor_data_;
  Tin D_;
};

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyImpl(cudaStream_t stream, const T* log_prob, const Tin* label, const T* weight,
                                   const T* normalize_factor, T* output_data, size_t count, size_t label_depth) {
  if (weight) {
    OpSparseSoftmaxCrossEntropy<T, Tin, true> op(log_prob, label, weight, normalize_factor,
                                                 static_cast<Tin>(label_depth));
    LaunchElementwiseKernel<T, decltype(op), CUDA_LONG>(stream, output_data, op, static_cast<CUDA_LONG>(count));
  } else {
    OpSparseSoftmaxCrossEntropy<T, Tin, false> op(log_prob, label, nullptr, normalize_factor,
                                                  static_cast<Tin>(label_depth));
    LaunchElementwiseKernel<T, decltype(op), CUDA_LONG>(stream, output_data, op, static_cast<CUDA_LONG>(count));
  }
}

template <typename T, typename Tin, bool IsWeighted>
struct OpSparseSoftmaxCrossEntropyGrad {
  OpSparseSoftmaxCrossEntropyGrad(const T* dY_data, const T* log_prob_data, const Tin* label_data, const T* weight_data,
                                  const T* normalize_factor_data, fast_divmod D_fdm)
      : dY_data_(dY_data),
        log_prob_data_(log_prob_data),
        label_data_(label_data),
        weight_data_(weight_data),
        normalize_factor_data_(normalize_factor_data),
        D_fdm_(D_fdm) {}

  __device__ __inline__ T operator()(CUDA_LONG idx) const {
    if (*normalize_factor_data_ != T(0.f)) {
      int row, d;
      D_fdm_.divmod(idx, row, d);
      return (*dY_data_) * (IsWeighted ? weight_data_[row] : T(1.f)) *
             (_Exp(log_prob_data_[idx]) - (T)(d == label_data_[row])) / (*normalize_factor_data_);
    }
    return T(0.f);
  }

  const T* dY_data_;
  const T* log_prob_data_;
  const Tin* label_data_;
  const T* weight_data_;
  const T* normalize_factor_data_;
  fast_divmod D_fdm_;
};

template <typename T, typename Tin>
void SparseSoftmaxCrossEntropyGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const Tin* label,
                                       const T* weight, const T* normalize_factor, T* output_data, size_t count,
                                       size_t label_depth) {
  if (weight) {
    OpSparseSoftmaxCrossEntropyGrad<T, Tin, true> op(dY, log_prob, label, weight, normalize_factor,
                                                     fast_divmod(static_cast<int>(label_depth)));
    LaunchElementwiseKernel<T, decltype(op), CUDA_LONG>(stream, output_data, op,
                                                        static_cast<CUDA_LONG>(count * label_depth));
  } else {
    OpSparseSoftmaxCrossEntropyGrad<T, Tin, false> op(dY, log_prob, label, nullptr, normalize_factor,
                                                      fast_divmod(static_cast<int>(label_depth)));
    LaunchElementwiseKernel<T, decltype(op), CUDA_LONG>(stream, output_data, op,
                                                        static_cast<CUDA_LONG>(count * label_depth));
  }
}

#define SPECIALIZED_SPARSE_SOFTMAX_ENTROPY_IMPL(T, Tin)                                                         \
  template void SparseSoftmaxCrossEntropyImpl(cudaStream_t stream, const T* log_prob, const Tin* label,         \
                                              const T* weight, const T* normalize_factor, T* output_data,       \
                                              size_t count, size_t label_depth);                                \
  template void SparseSoftmaxCrossEntropyGradImpl(cudaStream_t stream, const T* dY, const T* log_prob,          \
                                                  const Tin* label, const T* weight, const T* normalize_factor, \
                                                  T* output_data, size_t count, size_t label_depth)

SPECIALIZED_SPARSE_SOFTMAX_ENTROPY_IMPL(float, int32_t);
SPECIALIZED_SPARSE_SOFTMAX_ENTROPY_IMPL(float, int64_t);

#undef SPECIALIZED_SPARSE_SOFTMAX_ENTROPY_IMPL

}  // namespace cuda
}  // namespace onnxruntime

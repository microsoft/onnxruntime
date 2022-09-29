// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin, bool IsWeighted>
struct OpSoftmaxCrossEntropyWeights {
  OpSoftmaxCrossEntropyWeights(const Tin* label_data, const T* weight_data, Tin C, Tin ignore_index)
      : label_data_(label_data), weight_data_(weight_data), C_(C), ignore_index_(ignore_index) {}

  __device__ __inline__ T operator()(CUDA_LONG idx) const {
    if (label_data_[idx] != ignore_index_) {
      if (IsWeighted) {
        CUDA_KERNEL_ASSERT(label_data_[idx] >= 0 && label_data_[idx] < C_);
        return weight_data_[label_data_[idx]];
      }
      return T(1.f);
    }
    return T(0.f);
  }

  const Tin* label_data_;
  const T* weight_data_;
  Tin C_;
  Tin ignore_index_;
};

template <typename T, typename Tin>
void ComputeSoftmaxCrossEntropyWeightsImpl(cudaStream_t stream, const Tin* label, const T* weight, size_t count,
                                           size_t label_depth, int64_t ignore_index, T* weight_data_nd) {
  if (weight) {
    OpSoftmaxCrossEntropyWeights<T, Tin, true> op(label, weight, static_cast<Tin>(label_depth),
                                                  static_cast<Tin>(ignore_index));
    LaunchElementwiseKernel<T, decltype(op)>(stream, weight_data_nd, op, count);
  } else {
    OpSoftmaxCrossEntropyWeights<T, Tin, false> op(label, nullptr, static_cast<Tin>(label_depth),
                                                   static_cast<Tin>(ignore_index));
    LaunchElementwiseKernel<T, decltype(op)>(stream, weight_data_nd, op, count);
  }
}

#define INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(T, Tin)                                                          \
  template void ComputeSoftmaxCrossEntropyWeightsImpl(cudaStream_t stream, const Tin* label, const T* weight, \
                                                      size_t count, size_t label_depth, int64_t ignore_index, \
                                                      T* weight_data_nd)

INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(float, int32_t);
INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(float, int64_t);
INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(half, int64_t);
INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(BFloat16, int64_t);

#undef INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL

template <typename T, typename TAcc, typename Tin>
struct OpWeightedSoftmaxCrossEntropyLoss {
  OpWeightedSoftmaxCrossEntropyLoss(const T* log_prob_data, const Tin* label_data, const T* weight_data,
                                    const TAcc* normalize_factor_data, Tin C, Tin ignore_index)
      : log_prob_data_(log_prob_data),
        label_data_(label_data),
        weight_data_(weight_data),
        normalize_factor_data_(normalize_factor_data),
        C_(C),
        ignore_index_(ignore_index) {}

  __device__ __inline__ T operator()(CUDA_LONG idx) const {
    if (label_data_[idx] != ignore_index_) {
      CUDA_KERNEL_ASSERT(label_data_[idx] >= 0 && label_data_[idx] < C_);
      return static_cast<T>(static_cast<TAcc>(-log_prob_data_[idx * C_ + label_data_[idx]] * weight_data_[idx]) /
                            (*normalize_factor_data_));
    }
    return T(0.f);
  }

  const T* log_prob_data_;
  const Tin* label_data_;
  const T* weight_data_;
  const TAcc* normalize_factor_data_;
  Tin C_;
  Tin ignore_index_;
};

template <typename T, typename TAcc, typename Tin>
void SoftmaxCrossEntropyLossImpl(cudaStream_t stream, const T* log_prob, const Tin* label, const T* weight,
                                 const TAcc* normalize_factor, size_t count, size_t label_depth, int64_t ignore_index,
                                 T* output_data) {
  OpWeightedSoftmaxCrossEntropyLoss<T, TAcc, Tin> op(log_prob, label, weight, normalize_factor,
                                                     static_cast<Tin>(label_depth), static_cast<Tin>(ignore_index));
  LaunchElementwiseKernel<T, decltype(op)>(stream, output_data, op, count);
}

template <typename T, typename TAcc, typename Tin, bool IsReductionNone, bool HasBias>
struct OpWeightedSoftmaxCrossEntropyLossGrad {
  OpWeightedSoftmaxCrossEntropyLossGrad(const T* dY_data, const T* log_prob_data, const Tin* label_data,
                                        const T* weight_data, const TAcc* normalize_factor_data, const T* bias_data,
                                        Tin C)
      : dY_data_(dY_data),
        log_prob_data_(log_prob_data),
        label_data_(label_data),
        weight_data_(weight_data),
        normalize_factor_data_(normalize_factor_data),
        bias_data_(bias_data),
        C_(C) {
    C_fdm_ = fast_divmod(static_cast<int>(C));
  }

  __device__ __inline__ T operator()(CUDA_LONG idx) const {
    // normalize_factor is sum of labels' weights. Because zero sum implies all weights are 0, the loss function should
    // be constant 0 and its corresponding gradient should be 0 as well.
    T result = T(0.f);
    if (*normalize_factor_data_ != TAcc(0.f)) {
      int row, d;
      C_fdm_.divmod(idx, row, d);
      CUDA_KERNEL_ASSERT(weight_data_[row] == T(0.f) || (label_data_[row] >= 0 && label_data_[row] < C_));
      result = static_cast<T>(static_cast<TAcc>((IsReductionNone ? dY_data_[row] : *dY_data_) * weight_data_[row]) *
                              (_Exp(static_cast<TAcc>(log_prob_data_[idx])) - (TAcc)(d == label_data_[row])) /
                              (*normalize_factor_data_));
    }
    return HasBias ? result + bias_data_[idx] : result;
  }

  const T* dY_data_;
  const T* log_prob_data_;
  const Tin* label_data_;
  const T* weight_data_;
  const TAcc* normalize_factor_data_;
  const T* bias_data_;
  Tin C_;
  fast_divmod C_fdm_;
};

template <typename T, typename TAcc, typename Tin>
void SoftmaxCrossEntropyLossGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const Tin* label,
                                     const T* weight, const TAcc* normalize_factor, const T* bias_data, size_t count,
                                     size_t label_depth, bool reduction_none, T* output_data) {
#define LAUNCH_WEIGHTED_SOFTMAX_CROSS_ENTROPY_LOSS_GRAD_KERNEL(is_reduction_none, has_bias)     \
  OpWeightedSoftmaxCrossEntropyLossGrad<T, TAcc, Tin, is_reduction_none, has_bias> op(          \
      dY, log_prob, label, weight, normalize_factor, bias_data, static_cast<Tin>(label_depth)); \
  LaunchElementwiseKernel<T, decltype(op)>(stream, output_data, op, count * label_depth)
  if (reduction_none) {
    if (bias_data) {
      LAUNCH_WEIGHTED_SOFTMAX_CROSS_ENTROPY_LOSS_GRAD_KERNEL(true, true);
    } else {
      LAUNCH_WEIGHTED_SOFTMAX_CROSS_ENTROPY_LOSS_GRAD_KERNEL(true, false);
    }
  } else {
    if (bias_data) {
      LAUNCH_WEIGHTED_SOFTMAX_CROSS_ENTROPY_LOSS_GRAD_KERNEL(false, true);
    } else {
      LAUNCH_WEIGHTED_SOFTMAX_CROSS_ENTROPY_LOSS_GRAD_KERNEL(false, false);
    }
  }
#undef LAUNCH_WEIGHTED_SOFTMAX_CROSS_ENTROPY_LOSS_GRAD_KERNEL
}

#define INSTANTIATE_SCE_LOSS_IMPL(T, TAcc, Tin)                                                                        \
  template void SoftmaxCrossEntropyLossImpl(cudaStream_t stream, const T* log_prob, const Tin* label, const T* weight, \
                                            const TAcc* normalize_factor, size_t count, size_t label_depth,            \
                                            int64_t ignore_index, T* output_data);                                     \
  template void SoftmaxCrossEntropyLossGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const Tin* label, \
                                                const T* weight, const TAcc* normalize_factor, const T* bias_data,     \
                                                size_t count, size_t label_depth, bool reducation_none,                \
                                                T* output_data)

INSTANTIATE_SCE_LOSS_IMPL(float, float, int32_t);
INSTANTIATE_SCE_LOSS_IMPL(float, float, int64_t);
INSTANTIATE_SCE_LOSS_IMPL(half, float, int64_t);
INSTANTIATE_SCE_LOSS_IMPL(BFloat16, float, int64_t);

#undef INSTANTIATE_SCE_LOSS_IMPL

}  // namespace cuda
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, typename TLabel, typename TOut, bool IsWeighted, typename TIndex>
struct OpSoftmaxCrossEntropyWeights {
  OpSoftmaxCrossEntropyWeights(const TLabel* label_data, const T* weight_data, TLabel C, TLabel ignore_index)
      : label_data_(label_data), weight_data_(weight_data), C_(C), ignore_index_(ignore_index) {}

  __device__ __inline__ TOut operator()(TIndex idx) const {
    if (label_data_[idx] != ignore_index_) {
      if (IsWeighted) {
        CUDA_KERNEL_ASSERT(label_data_[idx] >= 0 && label_data_[idx] < C_);
        return TOut(weight_data_[label_data_[idx]]);
      }
      return TOut(1.f);
    }
    return TOut(0.f);
  }

  const TLabel* label_data_;
  const T* weight_data_;
  TLabel C_;
  TLabel ignore_index_;
};

template <typename T, typename TLabel, typename TOut>
void ComputeSoftmaxCrossEntropyWeightsImpl(cudaStream_t stream, const TLabel* label, const T* weight, size_t count,
                                           size_t label_depth, int64_t ignore_index, TOut* weight_data_nd) {
  if (weight) {
    typedef OpSoftmaxCrossEntropyWeights<T, TLabel, TOut, true, size_t> OP_Type;
    OP_Type op(label, weight, static_cast<TLabel>(label_depth), static_cast<TLabel>(ignore_index));
    LaunchElementwiseKernel<TOut, OP_Type, size_t>(stream, weight_data_nd, op, count);
  } else {
    typedef OpSoftmaxCrossEntropyWeights<T, TLabel, TOut, false, size_t> OP_Type;
    OP_Type op(label, nullptr, static_cast<TLabel>(label_depth), static_cast<TLabel>(ignore_index));
    LaunchElementwiseKernel<TOut, OP_Type, size_t>(stream, weight_data_nd, op, count);
  }
}

#define INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(T, TLabel, TOut)                                                    \
  template void ComputeSoftmaxCrossEntropyWeightsImpl(cudaStream_t stream, const TLabel* label, const T* weight, \
                                                      size_t count, size_t label_depth, int64_t ignore_index,    \
                                                      TOut* weight_data_nd)

INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(float, int32_t, float);
INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(float, int64_t, float);
INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(half, int32_t, float);
INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(half, int64_t, float);
INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(half, int64_t, half);
INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL(BFloat16, int64_t, BFloat16);

#undef INSTANTIATE_COMPUTE_SCE_WEIGHTS_IMPL

template <typename T, typename TAcc, typename TLabel, typename TIndex>
struct OpWeightedSoftmaxCrossEntropyLoss {
  OpWeightedSoftmaxCrossEntropyLoss(const T* log_prob_data, const TLabel* label_data, const T* weight_data,
                                    const TAcc* normalize_factor_data, TLabel C, TLabel ignore_index)
      : log_prob_data_(log_prob_data),
        label_data_(label_data),
        weight_data_(weight_data),
        normalize_factor_data_(normalize_factor_data),
        C_(C),
        ignore_index_(ignore_index) {}

  __device__ __inline__ T operator()(TIndex idx) const {
    if (label_data_[idx] != ignore_index_) {
      CUDA_KERNEL_ASSERT(label_data_[idx] >= 0 && label_data_[idx] < C_);
      T ret = static_cast<T>(static_cast<TAcc>(-log_prob_data_[idx * C_ + label_data_[idx]] * weight_data_[idx]) /
                             (*normalize_factor_data_));
      return ret;
    }
    return T(0.f);
  }

  const T* log_prob_data_;
  const TLabel* label_data_;
  const T* weight_data_;
  const TAcc* normalize_factor_data_;
  TLabel C_;
  TLabel ignore_index_;
};

template <typename T, typename TAcc, typename TLabel>
void SoftmaxCrossEntropyLossImpl(cudaStream_t stream, const T* log_prob, const TLabel* label, const T* weight,
                                 const TAcc* normalize_factor, size_t count, size_t label_depth, int64_t ignore_index,
                                 T* output_data) {
  typedef OpWeightedSoftmaxCrossEntropyLoss<T, TAcc, TLabel, size_t> OP_Type;
  OP_Type op(log_prob, label, weight, normalize_factor,
             static_cast<TLabel>(label_depth),
             static_cast<TLabel>(ignore_index));
  LaunchElementwiseKernel<T, OP_Type, size_t>(stream, output_data, op, count);
}

template <typename T, typename TAcc, typename TLabel, typename TOut, bool IsReductionNone, bool HasBias, typename TIndex>
struct OpWeightedSoftmaxCrossEntropyLossGrad {
  OpWeightedSoftmaxCrossEntropyLossGrad(const T* dY_data, const T* log_prob_data, const TLabel* label_data,
                                        const T* weight_data, const TAcc* normalize_factor_data, const TOut* bias_data,
                                        TLabel C)
      : dY_data_(dY_data),
        log_prob_data_(log_prob_data),
        label_data_(label_data),
        weight_data_(weight_data),
        normalize_factor_data_(normalize_factor_data),
        bias_data_(bias_data),
        C_(C) {
    C_fdm_ = DivMod(static_cast<TIndex>(C));
  }

  __device__ __inline__ TOut operator()(TIndex idx) const {
    // normalize_factor is sum of labels' weights. Because zero sum implies all weights are 0, the loss function should
    // be constant 0 and its corresponding gradient should be 0 as well.
    TAcc result = TAcc(0.f);
    if (*normalize_factor_data_ != TAcc(0.f)) {
      TIndex row, d;
      C_fdm_.divmod(idx, row, d);
      CUDA_KERNEL_ASSERT(weight_data_[row] == T(0.f) || (label_data_[row] >= 0 && label_data_[row] < C_));
      result = static_cast<TAcc>((IsReductionNone ? dY_data_[row] : *dY_data_) * weight_data_[row]) *
               (_Exp(static_cast<TAcc>(log_prob_data_[idx])) - (TAcc)(d == label_data_[row])) /
               (*normalize_factor_data_);
    }
    return HasBias ? static_cast<TOut>(result + static_cast<TAcc>(bias_data_[idx])) : static_cast<TOut>(result);
  }

  const T* dY_data_;
  const T* log_prob_data_;
  const TLabel* label_data_;
  const T* weight_data_;
  const TAcc* normalize_factor_data_;
  const TOut* bias_data_;
  TLabel C_;
  DivMod<TIndex> C_fdm_;
};

template <typename T, typename TAcc, typename TLabel, typename TOut>
void SoftmaxCrossEntropyLossGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const TLabel* label,
                                     const T* weight, const TAcc* normalize_factor, const TOut* bias_data, size_t count,
                                     size_t label_depth, bool reduction_none, TOut* output_data) {
#define LAUNCH_WEIGHTED_SOFTMAX_CROSS_ENTROPY_LOSS_GRAD_KERNEL(is_reduction_none, has_bias)                              \
  uint64_t total_count = count * label_depth;                                                                            \
  if (total_count <= static_cast<uint64_t>(std::numeric_limits<int>::max())) {                                           \
    typedef OpWeightedSoftmaxCrossEntropyLossGrad<T, TAcc, TLabel, TOut, is_reduction_none, has_bias, int> OP_Type;      \
    OP_Type op(dY, log_prob, label, weight, normalize_factor, bias_data, static_cast<TLabel>(label_depth));              \
    LaunchElementwiseKernel<TOut, OP_Type, int>(stream, output_data, op, static_cast<int>(total_count));                 \
  } else {                                                                                                               \
    typedef OpWeightedSoftmaxCrossEntropyLossGrad<T, TAcc, TLabel, TOut, is_reduction_none, has_bias, uint64_t> OP_Type; \
    OP_Type op(dY, log_prob, label, weight, normalize_factor, bias_data, static_cast<TLabel>(label_depth));              \
    LaunchElementwiseKernel<TOut, decltype(op), uint64_t>(stream, output_data, op, total_count);                         \
  }
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

#define INSTANTIATE_SCE_LOSS_IMPL(T, TAcc, TLabel)                                                                        \
  template void SoftmaxCrossEntropyLossImpl(cudaStream_t stream, const T* log_prob, const TLabel* label, const T* weight, \
                                            const TAcc* normalize_factor, size_t count, size_t label_depth,               \
                                            int64_t ignore_index, T* output_data);

INSTANTIATE_SCE_LOSS_IMPL(float, float, int32_t);
INSTANTIATE_SCE_LOSS_IMPL(float, float, int64_t);
INSTANTIATE_SCE_LOSS_IMPL(half, float, int64_t);
INSTANTIATE_SCE_LOSS_IMPL(BFloat16, float, int64_t);

#undef INSTANTIATE_SCE_LOSS_IMPL

#define INSTANTIATE_SCE_LOSS_GRAD_IMPL(T, TAcc, TLabel, TOut)                                                             \
  template void SoftmaxCrossEntropyLossGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const TLabel* label, \
                                                const T* weight, const TAcc* normalize_factor, const TOut* bias_data,     \
                                                size_t count, size_t label_depth, bool reducation_none,                   \
                                                TOut* output_data)

INSTANTIATE_SCE_LOSS_GRAD_IMPL(float, float, int32_t, float);
INSTANTIATE_SCE_LOSS_GRAD_IMPL(float, float, int32_t, half);
INSTANTIATE_SCE_LOSS_GRAD_IMPL(float, float, int64_t, float);
INSTANTIATE_SCE_LOSS_GRAD_IMPL(float, float, int64_t, half);
INSTANTIATE_SCE_LOSS_GRAD_IMPL(half, float, int64_t, half);
INSTANTIATE_SCE_LOSS_GRAD_IMPL(BFloat16, float, int64_t, BFloat16);

#undef INSTANTIATE_SCE_LOSS_GRAD_IMPL

}  // namespace cuda
}  // namespace onnxruntime

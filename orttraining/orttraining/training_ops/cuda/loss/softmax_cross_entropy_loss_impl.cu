// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/elementwise_impl.cuh"
#include "orttraining/training_ops/cuda/math/softmax_grad_impl.cuh"

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
               (d == label_data_[row] ? expm1(static_cast<TAcc>(log_prob_data_[idx]))
                                      : _Exp(static_cast<TAcc>(log_prob_data_[idx]))) /
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

template <typename T, typename TAcc, typename TLabel, typename TOut, bool ReductionNone, bool HasBias, int ThreadsPerRow>
__global__ void SoftmaxCrossEntropyLossGradLowPrecision(
    const T* dY, const T* log_prob, const TLabel* label, const T* weight,
    const TAcc* normalize_factor, const TOut* bias, size_t count, size_t classes, TOut* output) {
  const size_t row = size_t(blockIdx.x) * (256 / ThreadsPerRow) + threadIdx.x / ThreadsPerRow;
  const int lane = threadIdx.x % ThreadsPerRow;
  const size_t offset = row * classes;
  const bool active = row < count && *normalize_factor != TAcc(0) && weight[row] != T(0);
  const TLabel target = active ? label[row] : TLabel(-1);
  CUDA_KERNEL_ASSERT(!active || (target >= 0 && static_cast<size_t>(target) < classes));
  const TAcc scale = active ? static_cast<TAcc>(ReductionNone ? dY[row] : *dY) *
                                  static_cast<TAcc>(weight[row]) / *normalize_factor
                            : TAcc(0);
  const bool dominant = active && static_cast<TAcc>(log_prob[offset + target]) > TAcc(-0.6931471805599453);
  TAcc tail = 0;
  for (size_t j = lane; row < count && j < classes; j += ThreadsPerRow) {
    if (!active) {
      output[offset + j] = HasBias ? bias[offset + j] : TOut(0);
    } else if (j != static_cast<size_t>(target)) {
      const TAcc probability = _Exp(static_cast<TAcc>(log_prob[offset + j]));
      tail += probability;
      const TAcc result = scale * probability + (HasBias ? static_cast<TAcc>(bias[offset + j]) : TAcc(0));
      output[offset + j] = static_cast<TOut>(result);
    }
  }
  if constexpr (ThreadsPerRow <= 32) {
    // Padded and ignored rows must participate: logical warps share the full hardware shuffle mask.
    TAcc values[1]{tail};
    warp_reduce<TAcc, 1, ThreadsPerRow, Add>(values);
    tail = values[0];
  } else {
    if (dominant) tail = SoftmaxGradientBlockReduce<TAcc, Add>(tail);
  }
  if (active && lane == 0) {
    // The target log-probability can round to zero in fp16 while its gradient, after scaling, is representable.
    const TAcc difference = dominant ? -tail : expm1(static_cast<TAcc>(log_prob[offset + target]));
    const TAcc result = scale * difference + (HasBias ? static_cast<TAcc>(bias[offset + target]) : TAcc(0));
    output[offset + target] = static_cast<TOut>(result);
  }
}

template <typename T, typename TAcc, typename TLabel, typename TOut>
void SoftmaxCrossEntropyLossGradImpl(cudaStream_t stream, const T* dY, const T* log_prob, const TLabel* label,
                                     const T* weight, const TAcc* normalize_factor, const TOut* bias_data, size_t count,
                                     size_t label_depth, bool reduction_none, TOut* output_data) {
  if (count == 0 || label_depth == 0) return;
  if constexpr (!std::is_same_v<T, float>) {
#define LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH(reduction, bias, width)                                          \
  SoftmaxCrossEntropyLossGradLowPrecision<T, TAcc, TLabel, TOut, reduction, bias, width>                     \
      <<<static_cast<unsigned int>(count / (256 / width) + (count % (256 / width) != 0)), 256, 0, stream>>>( \
          dY, log_prob, label, weight, normalize_factor, bias_data, count, label_depth, output_data)
#define LAUNCH_LOW_PRECISION_SCE_GRAD(reduction, bias)         \
  if (label_depth <= 1) {                                      \
    LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH(reduction, bias, 1);   \
  } else if (label_depth <= 2) {                               \
    LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH(reduction, bias, 2);   \
  } else if (label_depth <= 4) {                               \
    LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH(reduction, bias, 4);   \
  } else if (label_depth <= 8) {                               \
    LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH(reduction, bias, 8);   \
  } else if (label_depth <= 16) {                              \
    LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH(reduction, bias, 16);  \
  } else if (label_depth <= 256) {                             \
    LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH(reduction, bias, 32);  \
  } else {                                                     \
    LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH(reduction, bias, 256); \
  }
    if (reduction_none) {
      if (bias_data) {
        LAUNCH_LOW_PRECISION_SCE_GRAD(true, true);
      } else {
        LAUNCH_LOW_PRECISION_SCE_GRAD(true, false);
      }
    } else {
      if (bias_data) {
        LAUNCH_LOW_PRECISION_SCE_GRAD(false, true);
      } else {
        LAUNCH_LOW_PRECISION_SCE_GRAD(false, false);
      }
    }
#undef LAUNCH_LOW_PRECISION_SCE_GRAD
#undef LAUNCH_LOW_PRECISION_SCE_GRAD_WIDTH
    return;
  }
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

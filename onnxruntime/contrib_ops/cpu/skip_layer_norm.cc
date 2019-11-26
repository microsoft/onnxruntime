// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "skip_layer_norm.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SkipLayerNormalization,                                     \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SkipLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

template <typename T>
SkipLayerNorm<T>::SkipLayerNorm(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info) {
}

template <typename T>
static void LayerNormOneTask(int64_t row_idx,
                            int64_t hidden_size,
                            const T* __restrict p_input,
                            const T* __restrict p_skip,
                            const T* __restrict p_bias,
                            const T* __restrict p_gamma,
                            const T* __restrict p_beta,
                            T* __restrict p_output) {
  p_input = p_input + row_idx * hidden_size;
  p_skip = p_skip + row_idx * hidden_size;
  p_output = p_output + row_idx * hidden_size;

  T mean = 0;
  T mean_square = 0;

  for (int64_t h = 0; h < hidden_size; h++) {
    T value = p_input[h] + p_skip[h];
    if (nullptr != p_bias) {
      value += p_bias[h];
    }
    p_output[h] = value;
    mean += value;
    mean_square += value * value;
  }

  mean = mean / hidden_size;
  mean_square = sqrt(mean_square / hidden_size - mean * mean + float(1e-12));

  for (int64_t h = 0; h < hidden_size; h++) {
    p_output[h] = (p_output[h] - mean) / mean_square * p_gamma[h] + p_beta[h];
  }
}

template <typename T>
Status SkipLayerNorm<T>::Compute(OpKernelContext* p_ctx) const {
  const Tensor* input = p_ctx->Input<Tensor>(0);
  const Tensor* skip = p_ctx->Input<Tensor>(1);
  const Tensor* gamma = p_ctx->Input<Tensor>(2);
  const Tensor* beta = p_ctx->Input<Tensor>(3);
  const Tensor* bias = p_ctx->Input<Tensor>(4);
  Tensor* output = p_ctx->Output(0, input->Shape());

  const auto input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 3 dimensions, got ", input_dims.size());
  }

  if (input->Shape() != skip->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "skip is expected to have same shape as input");
  }

  const auto gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimension, got ", gamma_dims.size());
  }
  if (gamma_dims[0] != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Last dimension of gamma and input does not match");
  }

  const auto beta_dims = beta->Shape().GetDims();
  if (beta_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have 1 dimension, got ", beta_dims.size());
  }
  if (beta_dims[0] != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Last dimension of beta and input does not match");
  }

  if (nullptr != bias) {
    const auto bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "bias is expected to have 1 dimension, got ", bias_dims.size());
    }
    if (bias_dims[0] != input_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Last dimension of bias and input does not match");
    }
  }

  int64_t batch_size = input_dims[0];
  int64_t sequence_length = input_dims[1];
  int64_t hidden_size = input_dims[2];
  int64_t task_count = batch_size * sequence_length;

  const T* p_input = input->Data<T>();
  const T* p_skip = skip->Data<T>();
  const T* p_gamma = gamma->Data<T>();
  const T* p_beta = beta->Data<T>();
  const T* p_bias = bias == nullptr ? nullptr : bias->Data<T>();

  T* p_output = output->MutableData<T>();

  if (concurrency::ThreadPool* tp = p_ctx->GetOperatorThreadPool()) {
    int32_t block_count = tp->NumThreads() + 1;
    tp->ParallelFor(block_count, [task_count,
                                  hidden_size,
                                  block_count,
                                  p_input,
                                  p_skip,
                                  p_bias,
                                  p_gamma,
                                  p_beta,
                                  p_output](int32_t blk_idx) {
      int64_t task_start = blk_idx * task_count / block_count;
      int64_t task_end = (blk_idx + 1) * task_count / block_count;
      for (int64_t task_idx = task_start; task_idx < task_end; task_idx++) {
        LayerNormOneTask(task_idx,
                        hidden_size,
                        p_input,
                        p_skip,
                        p_bias,
                        p_gamma,
                        p_beta,
                        p_output);
      }
    });
  } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int task_idx = 0; task_idx < task_count; task_idx++) {
      LayerNormOneTask(task_idx,
                      hidden_size,
                      p_input,
                      p_skip,
                      p_bias,
                      p_gamma,
                      p_beta,
                      p_output);
    }
  }

  return Status::OK();
}  // namespace contrib

}  // namespace contrib
}  // namespace onnxruntime

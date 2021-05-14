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
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(epsilon_ >= 0);
}

template <typename T>
Status SkipLayerNorm<T>::Compute(OpKernelContext* p_ctx) const {
  const Tensor* input = p_ctx->Input<Tensor>(0);
  const Tensor* skip = p_ctx->Input<Tensor>(1);
  const Tensor* gamma = p_ctx->Input<Tensor>(2);
  const Tensor* beta = p_ctx->Input<Tensor>(3);
  const Tensor* bias = p_ctx->Input<Tensor>(4);
  Tensor* output = p_ctx->Output(0, input->Shape());

  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 3 dimensions, got ", input_dims.size());
  }

  if (input->Shape() != skip->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "skip is expected to have same shape as input");
  }

  const auto& gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimension, got ", gamma_dims.size());
  }
  if (gamma_dims[0] != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Last dimension of gamma and input does not match");
  }

  if (nullptr != beta) {
    const auto& beta_dims = beta->Shape().GetDims();
    if (beta_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "beta is expected to have 1 dimension, got ", beta_dims.size());
    }
    if (beta_dims[0] != input_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Last dimension of beta and input does not match");
    }
  }

  if (nullptr != bias) {
    const auto& bias_dims = bias->Shape().GetDims();
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

  const T* input_data = input->Data<T>();
  const T* skip_data = skip->Data<T>();
  const T* gamma_data = gamma->Data<T>();
  const T* beta_data = beta == nullptr ? nullptr : beta->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  T* output_data = output->MutableData<T>();

  concurrency::ThreadPool::TryBatchParallelFor(p_ctx->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
                                               [&](ptrdiff_t task_idx) {
                                                 const T* p_input = input_data + task_idx * hidden_size;
                                                 const T* p_skip = skip_data + task_idx * hidden_size;
                                                 T* p_output = output_data + task_idx * hidden_size;

                                                 T mean = 0;
                                                 T mean_square = 0;

                                                 for (int64_t h = 0; h < hidden_size; h++) {
                                                   T value = p_input[h] + p_skip[h];
                                                   if (nullptr != bias_data) {
                                                     value += bias_data[h];
                                                   }
                                                   p_output[h] = value;
                                                   mean += value;
                                                   mean_square += value * value;
                                                 }

                                                 mean = mean / hidden_size;
                                                 mean_square = sqrt(mean_square / hidden_size - mean * mean + epsilon_);

                                                 for (int64_t h = 0; h < hidden_size; h++) {
                                                   if (nullptr == beta_data) {
                                                     p_output[h] = (p_output[h] - mean) / mean_square * gamma_data[h];
                                                   } else {
                                                     p_output[h] = (p_output[h] - mean) / mean_square * gamma_data[h] + beta_data[h];
                                                   }
                                                 }
                                               }, 0);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

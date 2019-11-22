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

static void ComputeWithParallelFor(int32_t M,
                                   int32_t N,
                                   const T* __restrict p_input,
                                   const T* __restrict p_skip,
                                   const T* __restrict p_bias,
                                   const T* __restrict p_gamma,
                                   const T* __restrict p_beta,
                                   T* __restrict p_output,
                                   concurrency::ThreadPool* tp) {
  if (tp != nullptr) {
    int32_t task_count = tp->NumThreads() + 1;

    tp->ParallelFor(task_count, [M,
                                 N,
                                 task_count,
                                 p_input, p_skip, p_bias,
                                 p_gamma, p_beta,
                                 p_output](int t) {
      int32_t start_idx = t * N / task_count;
      int32_t end_idx = (t + 1) * N / task_count;
      for (int i = start_idx; i < end_idx; i++) {
        const T* start = p_input + i * M;
        const T* start_skip = p_skip + i * M;
        const T* src = start;
        const T* src_skip = start_skip;
        const T* src_bias = p_bias;
        T* dest = p_output + i * M;
        T mean = 0;
        T mean_square = 0;
        for (int32_t j = 0; j < M; j++) {
          T value = (*src++ + *src_skip++ + *src_bias++);
          *dest++ = value;
          mean += value;
          mean_square += value * value;
        }

        mean = mean / M;
        mean_square = sqrt(mean_square / M - mean * mean + float(1e-12));
        dest = p_output + i * M;
        src = start;
        src_skip = start_skip;
        src_bias = p_bias;
        for (int32_t j = 0; j < M; j++, dest++) {
          *dest = (*dest - mean) / mean_square * p_gamma[j] + p_beta[j];
        }
      }
    });
  } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++) {
      const T* start = p_input + i * M;
      const T* start_skip = p_skip + i * M;
      const T* src = start;
      const T* src_skip = start_skip;
      const T* src_bias = p_bias;
      T* dest = p_output + i * M;
      T mean = 0;
      T mean_square = 0;
      for (int32_t j = 0; j < M; j++) {
        T value = (*src++ + *src_skip++ + *src_bias++);
        *dest++ = value;
        mean += value;
        mean_square += value * value;
      }

      mean = mean / M;
      mean_square = sqrt(mean_square / M - mean * mean + float(1e-12));
      dest = p_output + i * M;
      src = start;
      src_skip = start_skip;
      src_bias = p_bias;
      for (int32_t j = 0; j < M; j++, dest++) {
        *dest = (*dest - mean) / mean_square * p_gamma[j] + p_beta[j];
      }
    }
  }
}

template <typename T>
Status SkipLayerNorm<T>::Compute(OpKernelContext* p_op_kernel_context) const {
  const Tensor* input = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* skip = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* gamma = p_op_kernel_context->Input<Tensor>(2);
  const Tensor* beta = p_op_kernel_context->Input<Tensor>(3);
  const Tensor* bias = p_op_kernel_context->Input<Tensor>(4);
  Tensor* output = p_op_kernel_context->Output(0, input->Shape());

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

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);
  int hidden_size = static_cast<int>(input_dims[2]);
  //int element_count = batch_size * sequence_length * hidden_size;
  //size_t element_size = sizeof(T);

  ComputeWithParallelFor<T>(static_cast<int32_t>(hidden_size),
                            static_cast<int32_t>(batch_size * sequence_length),
                            input->template Data<T>(),
                            skip->template Data<T>(),
                            bias != nullptr ? bias->template Data<T>() : nullptr,
                            gamma->template Data<T>(),
                            beta->template Data<T>(),
                            output->template MutableData<T>(),
                            p_op_kernel_context->GetOperatorThreadPool());
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

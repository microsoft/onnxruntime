// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "layer_norm.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LayerNormalization,                                         \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LayerNorm<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

template <typename T>
LayerNorm<T>::LayerNorm(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
}

template <typename T>
static void ComputeWithParallelFor(int32_t M, int32_t N, T epsilon, const T* p_input, const T* p_gamma, const T* p_beta, T* p_mean, T* p_stdev, T* p_output, concurrency::ThreadPool& tp) {
  int32_t task_count = tp.NumThreads() + 1;
  tp.ParallelFor(task_count, [M,
                              N,
                              task_count,
                              epsilon,
                              p_input,
                              p_gamma, p_beta,
                              p_mean,
                              p_stdev,
                              p_output](int t) {
    int32_t start_idx = t * N / task_count;
    int32_t end_idx = (t + 1) * N / task_count;
    for (int i = start_idx; i < end_idx; i++) {
      const T* start = p_input + i * M;
      const T* src = start;

      T sum = 0;
      T sum_square = 0;
      for (int32_t j = 0; j < M; j++) {
        T value = *src++;
        sum += value;
        sum_square += value * value;
      }

      p_mean[i] = sum / M;
      p_stdev[i] = sqrt(sum_square / M - p_mean[i] * p_mean[i] + epsilon);

      T* dest = p_output + i * M;
      src = start;
      for (int32_t j = 0; j < M; j++) {
        *dest++ = (*src++ - p_mean[i]) / p_stdev[i] * p_gamma[j] + p_beta[j];
      }
    }
  });
}

template <typename T>
Status LayerNorm<T>::Compute(OpKernelContext* p_op_kernel_context) const {
  // Inputs
  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* bias = p_op_kernel_context->Input<Tensor>(2);
  auto X_data = X->template Data<T>();
  auto scale_data = scale->template Data<T>();
  auto bias_data = bias->template Data<T>();

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  auto N = x_shape.SizeToDimension(axis);
  auto M = x_shape.SizeFromDimension(axis);

  // Outputs
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);
  auto Y_data = Y->template MutableData<T>();

  std::vector<int64_t> mean_inv_std_var_dim;
  mean_inv_std_var_dim.reserve(x_shape.NumDimensions());
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_var_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_var_dim.emplace_back(1);
    }
  }

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(p_op_kernel_context->GetTempSpaceAllocator(&alloc));

  T* mean_data = nullptr;
  BufferUniquePtr mean_data_buf_ptr;

  Tensor* mean = p_op_kernel_context->Output(1, TensorShape(mean_inv_std_var_dim));
  if (mean != nullptr) {
    mean_data = mean->template MutableData<T>();
  } else {
    auto mean_data_buf = alloc->Alloc(sizeof(T) * N);
    mean_data_buf_ptr = BufferUniquePtr(mean_data_buf, BufferDeleter(alloc));
    mean_data = static_cast<T*>(mean_data_buf_ptr.get());
  }

  T* inv_std_var_data = nullptr;
  BufferUniquePtr inv_std_var_data_buf_ptr;

  Tensor* inv_std_var = p_op_kernel_context->Output(2, TensorShape(mean_inv_std_var_dim));
  if (inv_std_var != nullptr) {
    inv_std_var_data = inv_std_var->template MutableData<T>();
  } else {
    auto inv_std_var_data_buf = alloc->Alloc(sizeof(T) * N);
    inv_std_var_data_buf_ptr = BufferUniquePtr(inv_std_var_data_buf, BufferDeleter(alloc));
    inv_std_var_data = static_cast<T*>(inv_std_var_data_buf_ptr.get());
  }

  concurrency::ThreadPool* tp = p_op_kernel_context->GetOperatorThreadPool();
  if (nullptr != tp) {
    ComputeWithParallelFor(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<T>(epsilon_), X_data, scale_data, bias_data, mean_data, inv_std_var_data, Y_data, *tp);
    return Status::OK();
  }

  ConstEigenArrayMap<T> X_arr(X_data, M, N);
  for (int i = 0; i < N; ++i) {
    mean_data[i] = X_arr.col(i).mean();
    inv_std_var_data[i] = X_arr.col(i).square().mean() - mean_data[i] * mean_data[i];
  }

  // Compute Y = ((x - mean) * (inv_var) * scale + bias
  EigenArrayMap<T> Y_arr(Y_data, M, N);

  ConstEigenVectorArrayMap<T> mean_arr(mean_data, N);
  EigenVectorArrayMap<T> inv_std_var_arr(inv_std_var_data, N);
  inv_std_var_arr = (inv_std_var_arr + epsilon_).sqrt().inverse();

  Y_arr = (X_arr.rowwise() - mean_arr.transpose()).rowwise() * inv_std_var_arr.transpose();

  ConstEigenVectorArrayMap<T> scale_arr(scale_data, M);
  ConstEigenVectorArrayMap<T> bias_arr(bias_data, M);
  Y_arr = (Y_arr.colwise() * scale_arr).colwise() + bias_arr;

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

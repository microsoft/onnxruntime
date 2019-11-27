// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm.h"

#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"

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
static void LayerNormOneTask(int64_t batch_idx,
                             int64_t norm_size,
                             T epsilon,
                             const T* __restrict p_input,
                             const T* __restrict p_gamma,
                             const T* __restrict p_beta,
                             T* __restrict p_output,
                             T* __restrict p_mean,
                             T* __restrict p_var) {
  p_input = p_input + batch_idx * norm_size;
  p_output = p_output + batch_idx * norm_size;

  T mean = 0;
  T mean_square = 0;

  for (int64_t h = 0; h < norm_size; h++) {
    mean += p_input[h];
    mean_square += p_input[h] * p_input[h];
  }

  mean = mean / norm_size;
  mean_square = sqrt(mean_square / norm_size - mean * mean + epsilon);

  for (int64_t h = 0; h < norm_size; h++) {
    p_output[h] = (p_input[h] - mean) / mean_square * p_gamma[h] + p_beta[h];
  }

  p_mean[batch_idx] = mean;
  p_var[batch_idx] = mean_square;
}

template <typename T>
Status LayerNorm<T>::Compute(OpKernelContext* p_ctx) const {
  // Inputs
  const Tensor* X = p_ctx->Input<Tensor>(0);
  const Tensor* scale = p_ctx->Input<Tensor>(1);
  const Tensor* bias = p_ctx->Input<Tensor>(2);
  auto X_data = X->template Data<T>();
  auto scale_data = scale->template Data<T>();
  auto bias_data = bias->template Data<T>();

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  auto batch_size = x_shape.SizeToDimension(axis);
  auto norm_size = x_shape.SizeFromDimension(axis);

  Tensor* Y = p_ctx->Output(0, x_shape);
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
  ORT_RETURN_IF_ERROR(p_ctx->GetTempSpaceAllocator(&alloc));

  T* mean_data = nullptr;
  BufferUniquePtr mean_data_buf_ptr;

  Tensor* mean = p_ctx->Output(1, TensorShape(mean_inv_std_var_dim));
  if (mean != nullptr) {
    mean_data = mean->template MutableData<T>();
  } else {
    auto mean_data_buf = alloc->Alloc(sizeof(T) * batch_size);
    mean_data_buf_ptr = BufferUniquePtr(mean_data_buf, BufferDeleter(alloc));
    mean_data = static_cast<T*>(mean_data_buf_ptr.get());
  }

  T* inv_std_var_data = nullptr;
  BufferUniquePtr inv_std_var_data_buf_ptr;

  Tensor* inv_std_var = p_ctx->Output(2, TensorShape(mean_inv_std_var_dim));
  if (inv_std_var != nullptr) {
    inv_std_var_data = inv_std_var->template MutableData<T>();
  } else {
    auto inv_std_var_data_buf = alloc->Alloc(sizeof(T) * batch_size);
    inv_std_var_data_buf_ptr = BufferUniquePtr(inv_std_var_data_buf, BufferDeleter(alloc));
    inv_std_var_data = static_cast<T*>(inv_std_var_data_buf_ptr.get());
  }

  if (concurrency::ThreadPool* tp = p_ctx->GetOperatorThreadPool()) {
    int block_count = tp->NumThreads() + 1;
    tp->ParallelFor(block_count, [block_count,
                                  batch_size,
                                  norm_size,
                                  epsilon = epsilon_, X_data,
                                  scale_data,
                                  bias_data,
                                  Y_data,
                                  mean_data,
                                  inv_std_var_data](int32_t blk_idx) {
      int64_t batch_start = blk_idx * batch_size / block_count;
      int64_t batch_end = (blk_idx + 1) * batch_size / block_count;
      for (int64_t batch_idx = batch_start; batch_idx < batch_end; batch_idx++) {
        LayerNormOneTask(batch_idx,
                       norm_size,
                       static_cast<T>(epsilon),
                       X_data,
                       scale_data,
                       bias_data,
                       Y_data,
                       mean_data,
                       inv_std_var_data);
      }
    });
  } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      LayerNormOneTask(batch_idx,
                     norm_size,
                     static_cast<T>(epsilon_),
                     X_data,
                     scale_data,
                     bias_data,
                     Y_data,
                     mean_data,
                     inv_std_var_data);
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

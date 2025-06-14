// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/group_norm.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

// Opset 18-20 registrations (without stash_type)
#define REGISTER_ONNX_KERNEL_TYPED_VERSIONED(T)                                                    \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(GroupNormalization, 18, 20, T,                        \
                                           KernelDefBuilder()                                      \
                                               .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                           GroupNorm);

// Opset 21+ registrations (with stash_type)
#define REGISTER_ONNX_KERNEL_TYPED_21(T)                                                          \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(GroupNormalization, 21, T,                                      \
                                 KernelDefBuilder()                                              \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),      \
                                 GroupNorm);

REGISTER_ONNX_KERNEL_TYPED_VERSIONED(float)
REGISTER_ONNX_KERNEL_TYPED_VERSIONED(double)
REGISTER_ONNX_KERNEL_TYPED_VERSIONED(MLFloat16)

REGISTER_ONNX_KERNEL_TYPED_21(float)
REGISTER_ONNX_KERNEL_TYPED_21(double)
REGISTER_ONNX_KERNEL_TYPED_21(MLFloat16)

GroupNorm::GroupNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr("num_groups", &num_groups_).IsOK());
  
  // stash_type is optional in opset 21, default to 1 (float32)
  if (!op_kernel_info.GetAttr("stash_type", &stash_type_).IsOK()) {
    stash_type_ = 1;
  }
}

Status GroupNorm::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* scale = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);

  ORT_RETURN_IF_ERROR(ComputeHelper(context, X, scale, bias));
  return Status::OK();
}

template<typename T>
Status GroupNorm::ComputeImpl(OpKernelContext* context, const Tensor* X, const Tensor* scale, const Tensor* bias) const {
  const auto& x_shape = X->Shape();
  const int64_t N = x_shape[0];  // batch size
  const int64_t C = x_shape[1];  // channels
  
  // Validate that channels are divisible by num_groups
  ORT_RETURN_IF_NOT(C % num_groups_ == 0, "Number of channels must be divisible by num_groups");
  
  const int64_t channels_per_group = C / num_groups_;
  
  // Calculate spatial dimensions (H*W*... for everything after batch and channel dims)
  int64_t spatial_size = 1;
  for (size_t i = 2; i < x_shape.NumDimensions(); ++i) {
    spatial_size *= x_shape[i];
  }
  
  Tensor* Y = context->Output(0, x_shape);
  
  const T* x_data = X->Data<T>();
  const T* scale_data = scale->Data<T>();
  const T* bias_data = bias->Data<T>();
  T* y_data = Y->MutableData<T>();
  
  // Process each batch and group
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  
  concurrency::ThreadPool::TryBatchParallelFor(
      tp, static_cast<int32_t>(N * num_groups_),
      [&](ptrdiff_t idx) {
        const int64_t batch_idx = idx / num_groups_;
        const int64_t group_idx = idx % num_groups_;
        
        const int64_t group_start_channel = group_idx * channels_per_group;
        const int64_t group_end_channel = group_start_channel + channels_per_group;
        
        // Calculate mean and variance for this group
        double sum = 0.0;
        double sum_sq = 0.0;
        const int64_t group_size = channels_per_group * spatial_size;
        
        for (int64_t c = group_start_channel; c < group_end_channel; ++c) {
          const T* channel_data = x_data + batch_idx * C * spatial_size + c * spatial_size;
          for (int64_t s = 0; s < spatial_size; ++s) {
            const double val = static_cast<double>(channel_data[s]);
            sum += val;
            sum_sq += val * val;
          }
        }
        
        const double mean = sum / group_size;
        const double variance = sum_sq / group_size - mean * mean;
        const double inv_std = 1.0 / std::sqrt(variance + static_cast<double>(epsilon_));
        
        // Apply normalization: y = scale * (x - mean) / std + bias
        for (int64_t c = group_start_channel; c < group_end_channel; ++c) {
          const T* channel_x_data = x_data + batch_idx * C * spatial_size + c * spatial_size;
          T* channel_y_data = y_data + batch_idx * C * spatial_size + c * spatial_size;
          
          const T scale_val = scale_data[c];
          const T bias_val = bias_data[c];
          
          for (int64_t s = 0; s < spatial_size; ++s) {
            const double normalized = (static_cast<double>(channel_x_data[s]) - mean) * inv_std;
            const double result = normalized * static_cast<double>(scale_val) + static_cast<double>(bias_val);
            channel_y_data[s] = static_cast<T>(static_cast<float>(result));
          }
        }
      },
      0);
  
  return Status::OK();
}

Status GroupNorm::ComputeHelper(OpKernelContext* context, const Tensor* X, const Tensor* scale, const Tensor* bias) const {
  const auto element_type = X->DataType();
  
  if (element_type == DataTypeImpl::GetType<float>()) {
    return ComputeImpl<float>(context, X, scale, bias);
  } else if (element_type == DataTypeImpl::GetType<double>()) {
    return ComputeImpl<double>(context, X, scale, bias);
  } else if (element_type == DataTypeImpl::GetType<MLFloat16>()) {
    return ComputeImpl<MLFloat16>(context, X, scale, bias);
  }
  
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GroupNorm only supports float, double, and float16 data types");
}

}  // namespace onnxruntime
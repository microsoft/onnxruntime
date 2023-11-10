// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm_impl.h"

#include "core/common/safeint.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

LayerNormImpl::LayerNormImpl(const OpKernelInfo& op_kernel_info, bool simplified, bool contrib_op)
    : OpKernel(op_kernel_info), simplified_{simplified}, contrib_op_{contrib_op} {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
}

namespace layernormimpl_internal {
template <typename T, typename U>
Status ComputeImpl(OpKernelContext* p_ctx, int64_t orig_axis, float epsilon, bool simplified) {
  // Inputs
  const Tensor* X = p_ctx->Input<Tensor>(0);
  const Tensor* scale = p_ctx->Input<Tensor>(1);
  const Tensor* bias = p_ctx->Input<Tensor>(2);
  auto X_data = X->Data<T>();
  auto scale_data = scale->Data<T>();
  auto bias_data = (simplified || nullptr == bias) ? nullptr : bias->Data<T>();

  const TensorShape& x_shape = X->Shape();
  const int64_t axis = HandleNegativeAxis(orig_axis, x_shape.NumDimensions());
  auto norm_count = x_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis));
  auto norm_size = x_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis));

  const auto scale_size = scale->Shape().Size();
  const auto bias_size = (bias_data) ? bias->Shape().Size() : 0;
  if (scale_size != norm_size || (bias_data && bias_size != norm_size)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Size of X.shape()[axis:] == ", norm_size,
                           ". Size of scale and bias (if provided) must match this. Got scale size of ",
                           scale_size, " and bias size of ", bias_size);
  }

  Tensor* Y = p_ctx->Output(0, x_shape);
  auto Y_data = Y->MutableData<T>();

  std::vector<int64_t> mean_inv_std_dev_dim;
  mean_inv_std_dev_dim.reserve(x_shape.NumDimensions());
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_dev_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_dev_dim.emplace_back(1);
    }
  }

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(p_ctx->GetTempSpaceAllocator(&alloc));

  int output_index = 1;

  U* mean_data = nullptr;
  if (!simplified) {
    Tensor* mean = p_ctx->Output(output_index++, TensorShape(mean_inv_std_dev_dim));
    if (mean != nullptr) {
      mean_data = mean->MutableData<U>();
    }
  }

  U* inv_std_dev_data = nullptr;
  Tensor* inv_std_dev = p_ctx->Output(output_index, TensorShape(mean_inv_std_dev_dim));
  if (inv_std_dev != nullptr) {
    inv_std_dev_data = inv_std_dev->MutableData<U>();
  }

  concurrency::ThreadPool::TryBatchParallelFor(
      p_ctx->GetOperatorThreadPool(), static_cast<int32_t>(norm_count),
      [&](ptrdiff_t task_idx) {
        const T* p_input = X_data + task_idx * norm_size;
        T* p_output = Y_data + task_idx * norm_size;

        T mean = 0;
        T mean_square = 0;

        for (int64_t h = 0; h < norm_size; h++) {
          mean += p_input[h];
          mean_square += p_input[h] * p_input[h];
        }

        mean = mean / norm_size;
        if (simplified) {
          mean_square = sqrt(mean_square / norm_size + epsilon);
        } else {
          mean_square = sqrt(mean_square / norm_size - mean * mean + epsilon);
        }

        for (int64_t h = 0; h < norm_size; h++) {
          if (simplified) {
            p_output[h] = p_input[h] / mean_square * scale_data[h];
          } else if (nullptr == bias) {
            p_output[h] = (p_input[h] - mean) / mean_square * scale_data[h];
          } else {
            p_output[h] = (p_input[h] - mean) / mean_square * scale_data[h] + bias_data[h];
          }
        }

        if (mean_data != nullptr) {
          // ONNX spec doesn't support 'double' for 'U' so when 'T' == double, 'U' == float and we need to narrow
          mean_data[task_idx] = gsl::narrow_cast<U>(mean);
        }

        if (inv_std_dev_data != nullptr) {
          inv_std_dev_data[task_idx] = gsl::narrow_cast<U>(1 / mean_square);
        }
      },
      0);

  return Status::OK();
}

template <typename T>
struct SrcDispatcher {
  Status operator()(OpKernelContext* p_ctx, int64_t orig_axis, float epsilon, bool simplified, bool contrib_op) const {
    // the contrib op kernel was always registered with the same type for all constraints.
    // our implementation of the onnx op only supports 'float' as the U constraint.
#if !defined(DISABLE_CONTRIB_OPS)
    if (contrib_op) {
      return ComputeImpl<T, T>(p_ctx, orig_axis, epsilon, simplified);
    } else
#else
    ORT_UNUSED_PARAMETER(contrib_op);
#endif
    {
      return ComputeImpl<T, float>(p_ctx, orig_axis, epsilon, simplified);
    }
  }
};
}  // namespace

Status LayerNormImpl::Compute(OpKernelContext* p_ctx) const {
  using namespace layernormimpl_internal;
  const auto elem_type = p_ctx->Input<Tensor>(0)->GetElementType();

  using SupportedTypeList = boost::mp11::mp_list<float, double>;

  utils::MLTypeCallDispatcherFromTypeList<SupportedTypeList> t_disp(elem_type);
  return t_disp.InvokeRet<Status, SrcDispatcher>(p_ctx, axis_, epsilon_, simplified_, contrib_op_);
}

}  // namespace onnxruntime

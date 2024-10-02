// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm_impl.h"

#include "core/common/safeint.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/util/force_inline.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace {

template <typename T,
          typename U,
          typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>, void>>
void ComputeJob(
    const T* X_data,
    const T* scale_data,
    const T* bias_data,
    const ptrdiff_t task_idx,
    const int64_t norm_size,
    float epsilon,
    bool simplified,
    T* Y_data,
    U* mean_data,
    U* inv_std_dev_data) {
  const T* p_input = X_data + task_idx * norm_size;
  T* p_output = Y_data + task_idx * norm_size;

  T mean(0.0f);
  T mean_square(0.0f);

  for (int64_t h = 0; h < norm_size; h++) {
    p_output[h] = p_input[h];
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
      p_output[h] = p_output[h] / mean_square * scale_data[h];
    } else if (nullptr == bias_data) {
      p_output[h] = (p_output[h] - mean) / mean_square * scale_data[h];
    } else {
      p_output[h] = (p_output[h] - mean) / mean_square * scale_data[h] + bias_data[h];
    }
  }

  if (mean_data != nullptr) {
    // ONNX spec doesn't support 'double' for 'U' so when 'T' == double, 'U' == float and we need to narrow
    mean_data[task_idx] = gsl::narrow_cast<float>(mean);
  }

  if (inv_std_dev_data != nullptr) {
    inv_std_dev_data[task_idx] = gsl::narrow_cast<float>(1 / mean_square);
  }
}

template <typename U>
void ComputeJob(
    const MLFloat16* X_data,
    const MLFloat16* scale_data,
    const MLFloat16* bias_data,
    const ptrdiff_t task_idx,
    const int64_t norm_size,
    float epsilon,
    bool simplified,
    MLFloat16* Y_data,
    U* mean_data,
    U* inv_std_dev_data) {
  const MLFloat16* p_input = X_data + task_idx * norm_size;
  MLFloat16* p_output = Y_data + task_idx * norm_size;

  float mean(0.0f);
  float mean_square(0.0f);

  const size_t num_elems = static_cast<size_t>(norm_size);
  float* float_input = new float[num_elems];
  MlasConvertHalfToFloatBuffer(p_input, float_input, num_elems);

  float* float_output = new float[num_elems];
  for (size_t h = 0; h < num_elems; h++) {
    float_output[h] = float_input[h];
    mean += float_input[h];
    mean_square += float_input[h] * float_input[h];
  }

  mean = mean / norm_size;
  if (simplified) {
    mean_square = sqrt(mean_square / norm_size + epsilon);
  } else {
    mean_square = sqrt(mean_square / norm_size - mean * mean + epsilon);
  }

  float* float_scale = float_input;  // overwrite float_input with scale values, since they have the same size
  MlasConvertHalfToFloatBuffer(scale_data, float_scale, num_elems);
  float* float_bias = new float[num_elems];
  MlasConvertHalfToFloatBuffer(bias_data, float_bias, num_elems);
  for (size_t h = 0; h < num_elems; h++) {
    if (simplified) {
      float_output[h] = float_output[h] / mean_square * float_scale[h];
    } else if (nullptr == bias_data) {
      float_output[h] = (float_output[h] - mean) / mean_square * float_scale[h];
    } else {
      float_output[h] = (float_output[h] - mean) / mean_square * float_scale[h] + float_bias[h];
    }
  }
  delete[] float_scale;  // also deletes float_input
  delete[] float_bias;

  MlasConvertFloatToHalfBuffer(float_output, p_output, num_elems);
  delete[] float_output;

  if (mean_data != nullptr) {
    // ONNX spec doesn't support 'double' for 'U' so when 'T' == double, 'U' == float and we need to narrow
    mean_data[task_idx] = MLFloat16(mean);
  }

  if (inv_std_dev_data != nullptr) {
    inv_std_dev_data[task_idx] = MLFloat16(1 / mean_square);
  }
}

}  // namespace

LayerNormImpl::LayerNormImpl(const OpKernelInfo& op_kernel_info, bool simplified, bool contrib_op)
    : OpKernel(op_kernel_info), simplified_{simplified}, contrib_op_{contrib_op} {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
}

template <typename T, typename U>
Status LayerNormImpl::ComputeImpl(OpKernelContext* p_ctx, int64_t orig_axis, float epsilon, bool simplified) const {
  // Inputs
  const Tensor* X = p_ctx->Input<Tensor>(0);
  const Tensor* scale = p_ctx->Input<Tensor>(1);
  const Tensor* bias = p_ctx->Input<Tensor>(2);
  const T* X_data = X->Data<T>();
  const T* scale_data = scale->Data<T>();
  const T* bias_data = (simplified || nullptr == bias) ? nullptr : bias->Data<T>();

  const TensorShape& x_shape = X->Shape();
  const TensorShape& scale_shape = scale->Shape();
  const TensorShape& bias_shape = bias->Shape();
  Tensor* Y = p_ctx->Output(0, x_shape);
  T* Y_data = Y->MutableData<T>();

  const int64_t axis = HandleNegativeAxis(orig_axis, x_shape.NumDimensions());

  std::vector<int64_t> mean_inv_std_dev_dim;
  mean_inv_std_dev_dim.reserve(x_shape.NumDimensions());
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_dev_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_dev_dim.emplace_back(1);
    }
  }

  int output_index = 1;
  U* mean_data = nullptr;
  if (!simplified) {
    Tensor* mean = p_ctx->Output(output_index++, TensorShape(mean_inv_std_dev_dim));
    mean_data = mean->MutableData<U>();
  }

  U* inv_std_dev_data = nullptr;
  Tensor* inv_std_dev = p_ctx->Output(output_index, TensorShape(mean_inv_std_dev_dim));
  if (inv_std_dev != nullptr) {
    inv_std_dev_data = inv_std_dev->MutableData<U>();
  }

  onnxruntime::concurrency::ThreadPool* thread_pool = p_ctx->GetOperatorThreadPool();

  return ComputeWithoutContext<T, U>(X_data, x_shape, scale_data, scale_shape, bias_data, bias_shape,
                                     Y_data, mean_data, inv_std_dev_data, thread_pool, axis, epsilon, simplified);
}

Status LayerNormImpl::Compute(OpKernelContext* p_ctx) const {
  const auto elem_type = p_ctx->Input<Tensor>(0)->GetElementType();

  using SupportedTypeList = boost::mp11::mp_list<float, double, MLFloat16>;

  utils::MLTypeCallDispatcherFromTypeList<SupportedTypeList> t_disp(elem_type);
  return t_disp.InvokeRet<Status, SrcDispatcher>(this, p_ctx, axis_, epsilon_, simplified_, contrib_op_);
}

template <typename T, typename U>
Status LayerNormImpl::ComputeWithoutContext(
    const T* X_data,
    const TensorShape& x_shape,
    const T* scale_data,
    const TensorShape& scale_shape,
    const T* bias_data,
    const TensorShape& bias_shape,
    T* Y_data,
    U* mean_data,
    U* inv_std_dev_data,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    int64_t axis,
    float epsilon,
    bool simplified) const {
  int64_t norm_count = x_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis));
  int64_t norm_size = x_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis));

  const auto scale_size = scale_shape.Size();
  const auto bias_size = (bias_data) ? bias_shape.Size() : 0;
  if (scale_size != norm_size || (bias_data && bias_size != norm_size)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Size of X.shape()[axis:] == ", norm_size,
                           ". Size of scale and bias (if provided) must match this. Got scale size of ",
                           scale_size, " and bias size of ", bias_size);
  }

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool, static_cast<int32_t>(norm_count),
      [&](ptrdiff_t task_idx) {
        ComputeJob(X_data, scale_data, bias_data, task_idx, norm_size, epsilon, simplified,
                   Y_data, mean_data, inv_std_dev_data);
      },
      0);

  return Status::OK();
}

}  // namespace onnxruntime

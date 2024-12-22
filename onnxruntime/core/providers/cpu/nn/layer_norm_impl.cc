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
    const float* scale_float_ptr,
    const float* bias_float_ptr,
    float epsilon,
    bool simplified,
    T* Y_data,
    U* mean_data,
    U* inv_std_dev_data,
    AllocatorPtr alloc) {
  ORT_UNUSED_PARAMETER(scale_float_ptr);  // only used in MLFloat16 overload
  ORT_UNUSED_PARAMETER(bias_float_ptr);   // only used in MLFloat16 overload
  ORT_UNUSED_PARAMETER(alloc);

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
    const float* scale_float_ptr,
    const float* bias_float_ptr,
    float epsilon,
    bool simplified,
    MLFloat16* Y_data,
    U* mean_data,
    U* inv_std_dev_data,
    AllocatorPtr alloc) {
  ORT_UNUSED_PARAMETER(scale_data);  // only used in float/double overload
  ORT_UNUSED_PARAMETER(bias_data);   // only used in float/double overload

  const MLFloat16* p_input = X_data + task_idx * norm_size;
  MLFloat16* p_output = Y_data + task_idx * norm_size;

  float mean(0.0f);
  float mean_square(0.0f);

  const size_t num_elems = static_cast<size_t>(norm_size);
  IAllocatorUniquePtr<float> input_float_uptr = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
  MlasConvertHalfToFloatBuffer(p_input, input_float_uptr.get(), num_elems);

  IAllocatorUniquePtr<float> output_float_uptr = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
  float* output_float_ptr = output_float_uptr.get();

  const float* input_float_ptr = input_float_uptr.get();
  for (size_t h = 0; h < num_elems; h++) {
    output_float_ptr[h] = input_float_ptr[h];
    mean += input_float_ptr[h];
    mean_square += input_float_ptr[h] * input_float_ptr[h];
  }

  mean = mean / norm_size;
  if (simplified) {
    mean_square = sqrt(mean_square / norm_size + epsilon);
  } else {
    mean_square = sqrt(mean_square / norm_size - mean * mean + epsilon);
  }

  for (size_t h = 0; h < num_elems; h++) {
    if (simplified) {
      output_float_ptr[h] = output_float_ptr[h] / mean_square * scale_float_ptr[h];
    } else if (nullptr == bias_float_ptr) {
      output_float_ptr[h] = (output_float_ptr[h] - mean) / mean_square * scale_float_ptr[h];
    } else {
      output_float_ptr[h] = (output_float_ptr[h] - mean) / mean_square * scale_float_ptr[h] + bias_float_ptr[h];
    }
  }

  MlasConvertFloatToHalfBuffer(output_float_ptr, p_output, num_elems);

  if (mean_data != nullptr) {
    // ONNX spec doesn't support 'double' for 'U' so when 'T' == double, 'U' == float and we need to narrow
    mean_data[task_idx] = MLFloat16(mean);
  }

  if (inv_std_dev_data != nullptr) {
    inv_std_dev_data[task_idx] = MLFloat16(1 / mean_square);
  }
}

void ConvertMLFloat16ToFloatIfNeeded(const Tensor& tensor, AllocatorPtr alloc, IAllocatorUniquePtr<float>& dest, bool& is_packed) {
  if (tensor.GetElementType() == utils::ToTensorProtoElementType<MLFloat16>()) {
    auto tensor_data_ptr = tensor.Data<MLFloat16>();
    auto tensor_size = static_cast<size_t>(tensor.Shape().Size());
    auto float_ptr = IAllocator::MakeUniquePtr<float>(alloc, tensor_size, true);

    MlasConvertHalfToFloatBuffer(tensor_data_ptr, float_ptr.get(), tensor_size);
    dest = std::move(float_ptr);
    is_packed = true;
  }
}

}  // namespace

LayerNormImpl::LayerNormImpl(const OpKernelInfo& op_kernel_info, bool simplified, bool contrib_op)
    : OpKernel(op_kernel_info),
      simplified_{simplified},
      contrib_op_{contrib_op},
      prepacked_scale_fp32_data_(nullptr),
      prepacked_scale_fp32_size_(0),
      prepacked_bias_fp32_data_(nullptr),
      prepacked_bias_fp32_size_(0) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
}

template <typename T, typename U>
Status LayerNormImpl::ComputeImpl(OpKernelContext* p_ctx, int64_t orig_axis, float epsilon, bool simplified) const {
  // Inputs
  const Tensor* X = p_ctx->Input<Tensor>(0);
  const Tensor* scale = prepacked_scale_fp32_data_ ? nullptr : p_ctx->Input<Tensor>(1);
  const Tensor* bias = prepacked_bias_fp32_data_ ? nullptr : p_ctx->Input<Tensor>(2);
  const T* X_data = X->Data<T>();
  const T* scale_data = scale ? scale->Data<T>() : nullptr;
  const T* bias_data = (simplified || nullptr == bias) ? nullptr : bias->Data<T>();

  const TensorShape& x_shape = X->Shape();
  size_t scale_size = scale ? static_cast<size_t>(scale->Shape().Size()) : prepacked_scale_fp32_size_;
  size_t bias_size = bias ? static_cast<size_t>(bias->Shape().Size()) : prepacked_bias_fp32_size_;
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
    if (mean != nullptr) {
      mean_data = mean->MutableData<U>();
    }
  }

  U* inv_std_dev_data = nullptr;
  Tensor* inv_std_dev = p_ctx->Output(output_index, TensorShape(mean_inv_std_dev_dim));
  if (inv_std_dev != nullptr) {
    inv_std_dev_data = inv_std_dev->MutableData<U>();
  }

  onnxruntime::concurrency::ThreadPool* thread_pool = p_ctx->GetOperatorThreadPool();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(p_ctx->GetTempSpaceAllocator(&alloc));
  return ComputeWithoutContext<T, U>(X_data, x_shape, scale_data, scale_size, bias_data, bias_size, Y_data, mean_data,
                                     inv_std_dev_data, thread_pool, axis, epsilon, simplified, alloc);
}

Status LayerNormImpl::Compute(OpKernelContext* p_ctx) const {
  const auto elem_type = p_ctx->Input<Tensor>(0)->GetElementType();

  using SupportedTypeList = boost::mp11::mp_list<float, double, MLFloat16>;

  utils::MLTypeCallDispatcherFromTypeList<SupportedTypeList> t_disp(elem_type);
  return t_disp.InvokeRet<Status, SrcDispatcher>(this, p_ctx, axis_, epsilon_, simplified_, contrib_op_);
}

Status LayerNormImpl::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                              bool& is_packed, PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);

  is_packed = false;
  if (input_idx == 1) {  // scale
    prepacked_scale_fp32_size_ = static_cast<size_t>(tensor.Shape().Size());
    ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_scale_fp32_data_, is_packed);
  } else if (input_idx == 2) {  // bias
    prepacked_bias_fp32_size_ = static_cast<size_t>(tensor.Shape().Size());
    ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_bias_fp32_data_, is_packed);
  }

  return Status::OK();
}

template <typename T, typename U>
Status LayerNormImpl::ComputeWithoutContext(
    const T* X_data,
    const TensorShape& x_shape,
    const T* scale_data,
    size_t scale_size,
    const T* bias_data,
    size_t bias_size,
    T* Y_data,
    U* mean_data,
    U* inv_std_dev_data,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    int64_t axis,
    float epsilon,
    bool simplified,
    AllocatorPtr alloc) const {
  int64_t norm_count = x_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis));
  int64_t norm_size = x_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis));

  if (static_cast<int64_t>(scale_size) != norm_size || (bias_data && static_cast<int64_t>(bias_size) != norm_size)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Size of X.shape()[axis:] == ", norm_size,
                           ". Size of scale and bias (if provided) must match this. Got scale size of ",
                           scale_size, " and bias size of ", bias_size);
  }

  IAllocatorUniquePtr<float> scale_fp32;
  IAllocatorUniquePtr<float> bias_fp32;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    if (prepacked_scale_fp32_data_ == nullptr) {
      const size_t num_elems = static_cast<size_t>(norm_size);
      scale_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      MlasConvertHalfToFloatBuffer(scale_data, scale_fp32.get(), num_elems);
    }
    if (prepacked_bias_fp32_data_ == nullptr && bias_data) {
      const size_t num_elems = static_cast<size_t>(norm_size);
      bias_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      MlasConvertHalfToFloatBuffer(bias_data, bias_fp32.get(), num_elems);
    }
  }

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool, static_cast<int32_t>(norm_count),
      [&](ptrdiff_t task_idx) {
        ComputeJob(X_data, scale_data, bias_data, task_idx, norm_size,
                   prepacked_scale_fp32_data_ ? prepacked_scale_fp32_data_.get() : scale_fp32.get(),
                   prepacked_bias_fp32_data_ ? prepacked_bias_fp32_data_.get() : bias_fp32.get(),
                   epsilon, simplified, Y_data, mean_data, inv_std_dev_data, alloc);
      },
      0);

  return Status::OK();
}

}  // namespace onnxruntime

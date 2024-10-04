// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.h"
#include "quantize_linear.cuh"

namespace onnxruntime {
namespace cuda {

void ValidateBlockQuantizationShapes(const TensorShape& input_shape,
                                     const TensorShape& scale_shape,
                                     const Tensor* zero_point,
                                     size_t axis_no_neg,
                                     int64_t block_size_) {
  ORT_ENFORCE(scale_shape.NumDimensions() == input_shape.NumDimensions(),
              "scale and input must have the same rank for blocked quantization");

  for (size_t i = 0, ndim = input_shape.NumDimensions(); i < ndim; ++i) {
    if (i == static_cast<size_t>(axis_no_neg)) {
      ORT_ENFORCE(scale_shape[i] == (input_shape[i] + block_size_ - 1) / block_size_,
                  "scale must be ceil(Di/block_size) on the quantize axis i for blocked quantization");
    } else {
      ORT_ENFORCE(scale_shape[i] == input_shape[i],
                  "scale and input must have the same shape despite the quantize axis for blocked quantization");
    }
  }

  if (zero_point) {
    ORT_ENFORCE(zero_point->Shape() == scale_shape,
                "zero_point and scale must have the same shape for blocked quantization");
  }
}

template <class T, class U>
Status CudaQuantizeLinear(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point,
                          size_t num_of_element, bool /*saturate*/) {
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(scale);
  ORT_UNUSED_PARAMETER(zero_point);
  ORT_UNUSED_PARAMETER(num_of_element);
  ORT_NOT_IMPLEMENTED("Unsupported quantization type.");
}

template <class T, class U>
Status CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point,
                              size_t num_of_element, size_t batch_size, size_t n_scales, bool /*saturate*/) {
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(scale);
  ORT_UNUSED_PARAMETER(zero_point);
  ORT_UNUSED_PARAMETER(num_of_element);
  ORT_UNUSED_PARAMETER(batch_size);
  ORT_UNUSED_PARAMETER(n_scales);
  ORT_NOT_IMPLEMENTED("Unsupported quantization type.");
}

template <typename T, typename U>
Status CudaQuantizeLinearBlock(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point,
                               size_t num_of_element, size_t K, size_t N, size_t block_size, bool /*saturate*/) {
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(scale);
  ORT_UNUSED_PARAMETER(zero_point);
  ORT_UNUSED_PARAMETER(num_of_element);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(N);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_NOT_IMPLEMENTED("Unsupported quantization type.");
}

template <class U>
Status CudaQuantizeLinear(cudaStream_t stream, const U* input, int8_t* output, const U* scale,
                          const int8_t* zero_point, size_t num_of_element, bool /*saturate*/) {
  return CudaQuantizeLinearStd(stream, input, output, scale, zero_point, num_of_element);
}

template <class U>
Status CudaQuantizeLinear(cudaStream_t stream, const U* input, uint8_t* output, const U* scale,
                          const uint8_t* zero_point, size_t num_of_element, bool /*saturate*/) {
  return CudaQuantizeLinearStd(stream, input, output, scale, zero_point, num_of_element);
}

template <class U>
Status CudaQuantizeLinear(cudaStream_t stream, const U* input, Int4x2* output, const U* scale,
                          const Int4x2* zero_point, size_t num_of_element, bool /*saturate*/) {
  return CudaQuantizeLinearStdInt4(stream, input, reinterpret_cast<int8_t*>(output), scale,
                                   zero_point ? reinterpret_cast<const int8_t*>(zero_point) : nullptr,
                                   num_of_element);
}

template <class U>
Status CudaQuantizeLinear(cudaStream_t stream, const U* input, UInt4x2* output, const U* scale,
                          const UInt4x2* zero_point, size_t num_of_element, bool /*saturate*/) {
  return CudaQuantizeLinearStdInt4(stream, input, reinterpret_cast<uint8_t*>(output), scale,
                                   zero_point ? reinterpret_cast<const uint8_t*>(zero_point) : nullptr,
                                   num_of_element);
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <class U>
Status CudaQuantizeLinear(cudaStream_t stream, const U* input, Float8E4M3FN* output, const U* scale,
                          const Float8E4M3FN* zero_point, size_t num_of_element, bool saturate) {
  return CudaQuantizeLinearSat(stream, input, output, scale, zero_point, num_of_element, saturate);
}

template <class U>
Status CudaQuantizeLinear(cudaStream_t stream, const U* input, Float8E5M2* output, const U* scale,
                          const Float8E5M2* zero_point, size_t num_of_element, bool saturate) {
  return CudaQuantizeLinearSat(stream, input, output, scale, zero_point, num_of_element, saturate);
}

template <class U>
Status CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, Float8E4M3FN* output, const U* scale,
                              const Float8E4M3FN* zero_point, size_t num_of_element,
                              size_t batch_size, size_t n_scales, bool saturate) {
  return CudaQuantizeLinearAxisSat(stream, input, output, scale, zero_point, num_of_element, batch_size,
                                   n_scales, saturate);
}

template <class U>
Status CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, Float8E5M2* output, const U* scale,
                              const Float8E5M2* zero_point, size_t num_of_element,
                              size_t batch_size, size_t n_scales, bool saturate) {
  return CudaQuantizeLinearAxisSat(stream, input, output, scale, zero_point, num_of_element, batch_size,
                                   n_scales, saturate);
}

#endif

template <class U>
Status CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, int8_t* output, const U* scale,
                              const int8_t* zero_point, size_t num_of_element,
                              size_t batch_size, size_t n_scales, bool /*saturate*/) {
  return CudaQuantizeLinearAxisStd(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}

template <class U>
Status CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, uint8_t* output, const U* scale,
                              const uint8_t* zero_point, size_t num_of_element,
                              size_t batch_size, size_t n_scales, bool /*saturate*/) {
  return CudaQuantizeLinearAxisStd(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}

template <class U>
Status CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, Int4x2* output, const U* scale,
                              const Int4x2* zero_point, size_t num_of_element,
                              size_t batch_size, size_t n_scales, bool /*saturate*/) {
  return CudaQuantizeLinearAxisStdInt4(stream, input, reinterpret_cast<int8_t*>(output), scale,
                                       zero_point ? reinterpret_cast<const int8_t*>(zero_point) : nullptr,
                                       num_of_element, batch_size, n_scales);
}

template <class U>
Status CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, UInt4x2* output, const U* scale,
                              const UInt4x2* zero_point, size_t num_of_element,
                              size_t batch_size, size_t n_scales, bool /*saturate*/) {
  return CudaQuantizeLinearAxisStdInt4(stream, input, reinterpret_cast<uint8_t*>(output), scale,
                                       zero_point ? reinterpret_cast<const uint8_t*>(zero_point) : nullptr,
                                       num_of_element, batch_size, n_scales);
}

template <typename U>
Status CudaQuantizeLinearBlock(cudaStream_t stream,
                               const U* input, Int4x2* output, const U* scale, const Int4x2* zero_point,
                               size_t num_of_element, size_t K, size_t N, size_t block_size, bool /*saturate*/) {
  return CudaQuantizeLinearBlockStdInt4(stream, input, reinterpret_cast<int8_t*>(output), scale,
                                        zero_point ? reinterpret_cast<const int8_t*>(zero_point) : nullptr,
                                        num_of_element, K, N, block_size);
}

template <typename U>
Status CudaQuantizeLinearBlock(cudaStream_t stream,
                               const U* input, UInt4x2* output, const U* scale, const UInt4x2* zero_point,
                               size_t num_of_element, size_t K, size_t N, size_t block_size, bool /*saturate*/) {
  return CudaQuantizeLinearBlockStdInt4(stream, input, reinterpret_cast<uint8_t*>(output), scale,
                                        zero_point ? reinterpret_cast<const uint8_t*>(zero_point) : nullptr,
                                        num_of_element, K, N, block_size);
}

template <class T, class U>
Status QuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<U>::MappedType CudaU;

  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);

  auto& y = *ctx->Output(0, x.Shape());

  const auto& x_shape = x.Shape();
  const auto num_of_elements = x_shape.Size();

  const CudaU* input = reinterpret_cast<const CudaU*>(x.Data<U>());
  T* output = y.MutableData<T>();

  if (IsScalarOr1ElementVector(&y_scale)) {  // per-tensor quantization
    ORT_ENFORCE(y_zero_point == nullptr || IsScalarOr1ElementVector(y_zero_point),
                "y_zero_point must be a scalar or 1D tensor of size 1.");
    ORT_ENFORCE(block_size_ == 0, "block_size must be 0 for per-tensor quantization.");

    const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());

    ORT_RETURN_IF_ERROR(CudaQuantizeLinear(Stream(ctx), input, output, scale, zero_point, num_of_elements, saturate_));
    return Status::OK();
  } else if (block_size_ == 0) {  // per-axis quantization
    ORT_ENFORCE(y_scale.Shape().NumDimensions() == 1);
    ORT_ENFORCE(y_zero_point == nullptr || (y_scale.Shape().Size() == y_zero_point->Shape().Size() &&
                                            y_zero_point->Shape().NumDimensions() == 1),
                "scale and zero_point must have the same shape.");
    ORT_ENFORCE(x_shape.NumDimensions() > 1);
    int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    ORT_ENFORCE(y_scale.Shape().Size() == x_shape[axis], "scale must have ", x_shape[axis], " elements (axis=", axis, ").");

    const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());

    ORT_RETURN_IF_ERROR(CudaQuantizeLinearAxis(Stream(ctx), input, output, scale, zero_point, num_of_elements,
                                               x_shape.SizeToDimension(axis), y_scale.Shape().Size(), saturate_));
    return Status::OK();
  } else {  // blocked quantization
    // validate shape
    size_t axis_no_neg = SafeInt<size_t>(HandleNegativeAxis(axis_, x_shape.NumDimensions()));
    const auto& y_scale_shape = y_scale.Shape();

    ValidateBlockQuantizationShapes(x_shape,
                                    y_scale_shape,
                                    y_zero_point,
                                    axis_no_neg,
                                    block_size_);

    // compute
    const T* zero_point = y_zero_point ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());

    ORT_RETURN_IF_ERROR(CudaQuantizeLinearBlock(Stream(ctx), input, output, scale, zero_point,
                                                num_of_elements, x_shape[axis_no_neg],
                                                x_shape.SizeFromDimension(axis_no_neg + 1),
                                                block_size_, saturate_));
    return Status::OK();
  }
}

template <class T, class U>
Status CudaDequantizeLinear(cudaStream_t stream, const T* input, U* output, const U* scale,
                            const T* zero_point, size_t num_of_element) {
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(scale);
  ORT_UNUSED_PARAMETER(zero_point);
  ORT_UNUSED_PARAMETER(num_of_element);
  ORT_NOT_IMPLEMENTED("Unsupported quantization type.");
}

template <class T, class U>
Status CudaDequantizeLinearAxis(cudaStream_t stream, const T* input, U* output, const U* scale,
                                const T* zero_point, size_t num_of_element,
                                size_t batch_size, size_t n_scales) {
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(scale);
  ORT_UNUSED_PARAMETER(zero_point);
  ORT_UNUSED_PARAMETER(num_of_element);
  ORT_UNUSED_PARAMETER(batch_size);
  ORT_UNUSED_PARAMETER(n_scales);
  ORT_NOT_IMPLEMENTED("Unsupported quantization type.");
}

template <class T, class U>
Status CudaDequantizeLinearBlockInt4(cudaStream_t stream, const T* input, U* output, const U* scale,
                                     const T* zero_point, size_t num_of_element, size_t K, size_t N,
                                     size_t block_size) {
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(scale);
  ORT_UNUSED_PARAMETER(zero_point);
  ORT_UNUSED_PARAMETER(num_of_element);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(N);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_NOT_IMPLEMENTED("Unsupported quantization type.");
}

template <class U>
Status CudaDequantizeLinear(cudaStream_t stream, const int8_t* input, U* output, const U* scale,
                            const int8_t* zero_point, size_t num_of_element) {
  return CudaDequantizeLinearStd(stream, input, output, scale, zero_point, num_of_element);
}

template <class U>
Status CudaDequantizeLinear(cudaStream_t stream, const uint8_t* input, U* output, const U* scale,
                            const uint8_t* zero_point, size_t num_of_element) {
  return CudaDequantizeLinearStd(stream, input, output, scale, zero_point, num_of_element);
}

template <class U>
Status CudaDequantizeLinear(cudaStream_t stream, const Int4x2* input, U* output, const U* scale,
                            const Int4x2* zero_point, size_t num_of_element) {
  return CudaDequantizeLinearStdInt4(stream, reinterpret_cast<const int8_t*>(input), output, scale,
                                     zero_point ? reinterpret_cast<const int8_t*>(zero_point) : nullptr,
                                     num_of_element);
}

template <class U>
Status CudaDequantizeLinear(cudaStream_t stream, const UInt4x2* input, U* output, const U* scale,
                            const UInt4x2* zero_point, size_t num_of_element) {
  return CudaDequantizeLinearStdInt4(stream, reinterpret_cast<const uint8_t*>(input), output, scale,
                                     zero_point ? reinterpret_cast<const uint8_t*>(zero_point) : nullptr,
                                     num_of_element);
}

#if !defined(DISABLE_FLOAT8_TYPES)
template <class U>
Status CudaDequantizeLinear(cudaStream_t stream, const Float8E4M3FN* input, U* output, const U* scale,
                            const Float8E4M3FN* zero_point, size_t num_of_element) {
  return CudaDequantizeLinearSat(stream, input, output, scale, zero_point, num_of_element);
}

template <class U>
Status CudaDequantizeLinear(cudaStream_t stream, const Float8E5M2* input, U* output, const U* scale,
                            const Float8E5M2* zero_point, size_t num_of_element) {
  return CudaDequantizeLinearSat(stream, input, output, scale, zero_point, num_of_element);
}
#endif

template <class U>
Status CudaDequantizeLinearAxis(cudaStream_t stream, const int8_t* input, U* output, const U* scale,
                                const int8_t* zero_point, size_t num_of_element,
                                size_t batch_size, size_t n_scales) {
  return CudaDequantizeLinearAxisStd(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}

template <class U>
Status CudaDequantizeLinearAxis(cudaStream_t stream, const uint8_t* input, U* output, const U* scale,
                                const uint8_t* zero_point, size_t num_of_element,
                                size_t batch_size, size_t n_scales) {
  return CudaDequantizeLinearAxisStd(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}

template <class U>
Status CudaDequantizeLinearAxis(cudaStream_t stream, const Int4x2* input, U* output, const U* scale,
                                const Int4x2* zero_point, size_t num_of_element,
                                size_t batch_size, size_t n_scales) {
  return CudaDequantizeLinearAxisStdInt4(stream, reinterpret_cast<const int8_t*>(input), output, scale,
                                         zero_point ? reinterpret_cast<const int8_t*>(zero_point) : nullptr,
                                         num_of_element, batch_size, n_scales);
}

template <class U>
Status CudaDequantizeLinearAxis(cudaStream_t stream, const UInt4x2* input, U* output, const U* scale,
                                const UInt4x2* zero_point, size_t num_of_element,
                                size_t batch_size, size_t n_scales) {
  return CudaDequantizeLinearAxisStdInt4(stream, reinterpret_cast<const uint8_t*>(input), output, scale,
                                         zero_point ? reinterpret_cast<const uint8_t*>(zero_point) : nullptr,
                                         num_of_element, batch_size, n_scales);
}

#if !defined(DISABLE_FLOAT8_TYPES)
template <class U>
Status CudaDequantizeLinearAxis(cudaStream_t stream, const Float8E4M3FN* input, U* output, const U* scale,
                                const Float8E4M3FN* zero_point, size_t num_of_element,
                                size_t batch_size, size_t n_scales) {
  return CudaDequantizeLinearAxisSat(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}

template <class U>
Status CudaDequantizeLinearAxis(cudaStream_t stream, const Float8E5M2* input, U* output, const U* scale,
                                const Float8E5M2* zero_point, size_t num_of_element,
                                size_t batch_size, size_t n_scales) {
  return CudaDequantizeLinearAxisSat(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}
#endif

template <class U>
Status CudaDequantizeLinearBlockInt4(cudaStream_t stream, const UInt4x2* input, U* output, const U* scale,
                                     const UInt4x2* zero_point, size_t num_of_element, size_t K, size_t N,
                                     size_t block_size) {
  return CudaDequantizeLinearBlockStdInt4(stream, reinterpret_cast<const uint8_t*>(input), output, scale,
                                          zero_point ? reinterpret_cast<const uint8_t*>(zero_point) : nullptr,
                                          num_of_element, K, N, block_size);
}

template <class U>
Status CudaDequantizeLinearBlockInt4(cudaStream_t stream, const Int4x2* input, U* output, const U* scale,
                                     const Int4x2* zero_point, size_t num_of_element, size_t K, size_t N,
                                     size_t block_size) {
  return CudaDequantizeLinearBlockStdInt4(stream, reinterpret_cast<const int8_t*>(input), output, scale,
                                          zero_point ? reinterpret_cast<const int8_t*>(zero_point) : nullptr,
                                          num_of_element, K, N, block_size);
}

template <class T, class U>
Status DequantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<U>::MappedType CudaU;

  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);

  const auto& x_shape = x.Shape();
  const auto num_of_elements = x_shape.Size();

  auto& y = *ctx->Output(0, x_shape);

  const T* input = x.Data<T>();
  CudaU* output = reinterpret_cast<CudaU*>(y.MutableData<U>());

  if (IsScalarOr1ElementVector(&y_scale)) {
    ORT_ENFORCE(y_zero_point == nullptr || IsScalarOr1ElementVector(y_zero_point), "y_zero_point must be a scalar or 1D tensor of size 1.");

    const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());

    ORT_RETURN_IF_ERROR(CudaDequantizeLinear(Stream(ctx), input, output, scale, zero_point, num_of_elements));

    return Status::OK();
  } else if (block_size_ == 0) {  // per axis quantization
    ORT_ENFORCE(y_scale.Shape().NumDimensions() == 1);
    ORT_ENFORCE(y_zero_point == nullptr || (y_scale.Shape().Size() == y_zero_point->Shape().Size() && y_zero_point->Shape().NumDimensions() == 1), "scale and zero_point must have the same shape.");
    ORT_ENFORCE(x_shape.NumDimensions() > 1);
    int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    ORT_ENFORCE(y_scale.Shape().Size() == x_shape[axis], "scale must have ", x_shape[axis], " elements (axis=", axis, ").");

    const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());

    ORT_RETURN_IF_ERROR(CudaDequantizeLinearAxis(Stream(ctx), input, output, scale, zero_point, num_of_elements,
                                                 x_shape.SizeToDimension(axis), y_scale.Shape().Size()));
    return Status::OK();
  } else {  // blocked quantization
    // validate shape
    auto axis_no_neg = SafeInt<size_t>(HandleNegativeAxis(axis_, x_shape.NumDimensions()));
    const auto& y_scale_shape = y_scale.Shape();

    ValidateBlockQuantizationShapes(x_shape,
                                    y_scale_shape,
                                    y_zero_point,
                                    axis_no_neg,
                                    block_size_);

    // compute
    const T* zero_point = y_zero_point ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());

    ORT_RETURN_IF_ERROR(CudaDequantizeLinearBlockInt4(Stream(ctx), input, output, scale, zero_point,
                                                      num_of_elements, x_shape[axis_no_neg],
                                                      x_shape.SizeFromDimension(axis_no_neg + 1),
                                                      block_size_));

    return Status::OK();
  }
}

// register QuantizeLinear kernels
#define REGISTER_Q_KERNEL_TYPED_10_12(T)                              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                            \
      QuantizeLinear,                                                 \
      kOnnxDomain,                                                    \
      10, 12,                                                         \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T, float>);

#define REGISTER_Q_KERNEL_TYPED_13_18(T)                              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                            \
      QuantizeLinear,                                                 \
      kOnnxDomain,                                                    \
      13, 18,                                                         \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T, float>);

REGISTER_Q_KERNEL_TYPED_10_12(int8_t)
REGISTER_Q_KERNEL_TYPED_10_12(uint8_t)
REGISTER_Q_KERNEL_TYPED_13_18(int8_t)
REGISTER_Q_KERNEL_TYPED_13_18(uint8_t)

#define REGISTER_Q_KERNEL_TWO_TYPED_19_20(T, U)                    \
  ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(                     \
      QuantizeLinear,                                              \
      kOnnxDomain,                                                 \
      19, 20,                                                      \
      T, U,                                                        \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<U>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      QuantizeLinear<T, U>);

REGISTER_Q_KERNEL_TWO_TYPED_19_20(int8_t, float)
REGISTER_Q_KERNEL_TWO_TYPED_19_20(uint8_t, float)
REGISTER_Q_KERNEL_TWO_TYPED_19_20(int8_t, MLFloat16)
REGISTER_Q_KERNEL_TWO_TYPED_19_20(uint8_t, MLFloat16)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_Q_KERNEL_TWO_TYPED_19_20(Float8E4M3FN, float)
REGISTER_Q_KERNEL_TWO_TYPED_19_20(Float8E5M2, float)
REGISTER_Q_KERNEL_TWO_TYPED_19_20(Float8E4M3FN, MLFloat16)
REGISTER_Q_KERNEL_TWO_TYPED_19_20(Float8E5M2, MLFloat16)
#endif

#define REGISTER_Q_KERNEL_TWO_TYPED_21(T, U)                       \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                               \
      QuantizeLinear,                                              \
      kOnnxDomain,                                                 \
      21,                                                          \
      T, U,                                                        \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<U>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      QuantizeLinear<T, U>);

REGISTER_Q_KERNEL_TWO_TYPED_21(uint8_t, float)
REGISTER_Q_KERNEL_TWO_TYPED_21(int8_t, float)
REGISTER_Q_KERNEL_TWO_TYPED_21(uint8_t, MLFloat16)
REGISTER_Q_KERNEL_TWO_TYPED_21(int8_t, MLFloat16)
REGISTER_Q_KERNEL_TWO_TYPED_21(UInt4x2, float)
REGISTER_Q_KERNEL_TWO_TYPED_21(Int4x2, float)
REGISTER_Q_KERNEL_TWO_TYPED_21(UInt4x2, MLFloat16)
REGISTER_Q_KERNEL_TWO_TYPED_21(Int4x2, MLFloat16)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_Q_KERNEL_TWO_TYPED_21(Float8E4M3FN, float)
REGISTER_Q_KERNEL_TWO_TYPED_21(Float8E5M2, float)
REGISTER_Q_KERNEL_TWO_TYPED_21(Float8E4M3FN, MLFloat16)
REGISTER_Q_KERNEL_TWO_TYPED_21(Float8E5M2, MLFloat16)
#endif

// register DequantizeLinear kernels
#define REGISTER_DQ_KERNEL_TYPED_10_12(T)                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      DequantizeLinear,                                           \
      kOnnxDomain,                                                \
      10, 12,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T, float>);

#define REGISTER_DQ_KERNEL_TYPED_13_18(T)                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      DequantizeLinear,                                           \
      kOnnxDomain,                                                \
      13, 18,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T, float>);

REGISTER_DQ_KERNEL_TYPED_10_12(int8_t)
REGISTER_DQ_KERNEL_TYPED_10_12(uint8_t)
REGISTER_DQ_KERNEL_TYPED_13_18(int8_t)
REGISTER_DQ_KERNEL_TYPED_13_18(uint8_t)

#define REGISTER_DQ_KERNEL_TWO_TYPED_19_20(T, U)                   \
  ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(                     \
      DequantizeLinear,                                            \
      kOnnxDomain,                                                 \
      19, 20,                                                      \
      T, U,                                                        \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<U>()), \
      DequantizeLinear<T, U>);

REGISTER_DQ_KERNEL_TWO_TYPED_19_20(int8_t, float)
REGISTER_DQ_KERNEL_TWO_TYPED_19_20(uint8_t, float)
REGISTER_DQ_KERNEL_TWO_TYPED_19_20(int8_t, MLFloat16)
REGISTER_DQ_KERNEL_TWO_TYPED_19_20(uint8_t, MLFloat16)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DQ_KERNEL_TWO_TYPED_19_20(Float8E4M3FN, float)
REGISTER_DQ_KERNEL_TWO_TYPED_19_20(Float8E5M2, float)
REGISTER_DQ_KERNEL_TWO_TYPED_19_20(Float8E4M3FN, MLFloat16)
REGISTER_DQ_KERNEL_TWO_TYPED_19_20(Float8E5M2, MLFloat16)
#endif

#define REGISTER_DQ_KERNEL_TWO_TYPED_21(T, U)                      \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                               \
      DequantizeLinear,                                            \
      kOnnxDomain,                                                 \
      21,                                                          \
      T, U,                                                        \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<U>()), \
      DequantizeLinear<T, U>);

REGISTER_DQ_KERNEL_TWO_TYPED_21(uint8_t, float)
REGISTER_DQ_KERNEL_TWO_TYPED_21(int8_t, float)
REGISTER_DQ_KERNEL_TWO_TYPED_21(uint8_t, MLFloat16)
REGISTER_DQ_KERNEL_TWO_TYPED_21(int8_t, MLFloat16)
REGISTER_DQ_KERNEL_TWO_TYPED_21(UInt4x2, float)
REGISTER_DQ_KERNEL_TWO_TYPED_21(Int4x2, float)
REGISTER_DQ_KERNEL_TWO_TYPED_21(UInt4x2, MLFloat16)
REGISTER_DQ_KERNEL_TWO_TYPED_21(Int4x2, MLFloat16)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DQ_KERNEL_TWO_TYPED_21(Float8E4M3FN, float)
REGISTER_DQ_KERNEL_TWO_TYPED_21(Float8E5M2, float)
REGISTER_DQ_KERNEL_TWO_TYPED_21(Float8E4M3FN, MLFloat16)
REGISTER_DQ_KERNEL_TWO_TYPED_21(Float8E5M2, MLFloat16)
#endif

// specialize QuantizeLinear::ComputeInternal and DequantizeLinear::ComputeInternal
#define SPECIALIZED_QDQ_COMPUTE(T, U)                                                \
  template Status QuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const; \
  template Status DequantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_QDQ_COMPUTE(int8_t, float)
SPECIALIZED_QDQ_COMPUTE(uint8_t, float)
SPECIALIZED_QDQ_COMPUTE(int8_t, MLFloat16)
SPECIALIZED_QDQ_COMPUTE(uint8_t, MLFloat16)
SPECIALIZED_QDQ_COMPUTE(Int4x2, float)
SPECIALIZED_QDQ_COMPUTE(UInt4x2, float)
SPECIALIZED_QDQ_COMPUTE(Int4x2, MLFloat16)
SPECIALIZED_QDQ_COMPUTE(UInt4x2, MLFloat16)

#if !defined(DISABLE_FLOAT8_TYPES)
SPECIALIZED_QDQ_COMPUTE(Float8E4M3FN, float)
SPECIALIZED_QDQ_COMPUTE(Float8E4M3FN, MLFloat16)
SPECIALIZED_QDQ_COMPUTE(Float8E5M2, float)
SPECIALIZED_QDQ_COMPUTE(Float8E5M2, MLFloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime

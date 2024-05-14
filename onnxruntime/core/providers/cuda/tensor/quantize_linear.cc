// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.h"
#include "quantize_linear.cuh"

namespace onnxruntime {
namespace cuda {

template <class T, class U>
typename std::enable_if<boost::mp11::mp_set_contains<TypeList<int8_t, uint8_t>, T>::value, Status>::type
CudaQuantizeLinear(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element, bool /*saturate*/) {
  return CudaQuantizeLinearStd(stream, input, output, scale, zero_point, num_of_element);
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <class T, class U>
typename std::enable_if<boost::mp11::mp_set_contains<TypeList<Float8E4M3FN, Float8E5M2>, T>::value, Status>::type
CudaQuantizeLinear(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element, bool saturate) {
  return CudaQuantizeLinearSat(stream, input, output, scale, zero_point, num_of_element, saturate);
}

template <class T, class U>
typename std::enable_if<boost::mp11::mp_set_contains<TypeList<Float8E4M3FN, Float8E5M2>, T>::value, Status>::type
CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element,
                       size_t batch_size, size_t n_scales, bool saturate) {
  return CudaQuantizeLinearAxisSat(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales, saturate);
}

#endif

template <class T, class U>
typename std::enable_if<boost::mp11::mp_set_contains<TypeList<int8_t, uint8_t>, T>::value, Status>::type
CudaQuantizeLinearAxis(cudaStream_t stream, const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element,
                       size_t batch_size, size_t n_scales, bool /*saturate*/) {
  return CudaQuantizeLinearAxisStd(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}

template <class T, class U>
Status QuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<U>::MappedType CudaU;

  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);

  auto& y = *ctx->Output(0, x.Shape());

  const auto& x_shape = x.Shape();

  const CudaU* input = reinterpret_cast<const CudaU*>(x.Data<U>());
  T* output = y.MutableData<T>();

  if (IsScalarOr1ElementVector(&y_scale)) {
    ORT_ENFORCE(y_zero_point == nullptr || IsScalarOr1ElementVector(y_zero_point),
                "y_zero_point must be a scalar or 1D tensor of size 1.");

    const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());
    const auto num_of_elements = x_shape.Size();

    ORT_RETURN_IF_ERROR(CudaQuantizeLinear(Stream(ctx), input, output, scale, zero_point, num_of_elements, saturate_));
    return Status::OK();
  } else {
    ORT_ENFORCE(y_scale.Shape().NumDimensions() == 1);
    ORT_ENFORCE(y_zero_point == nullptr || (y_scale.Shape().Size() == y_zero_point->Shape().Size() &&
                                            y_zero_point->Shape().NumDimensions() == 1),
                "scale and zero_point must have the same shape.");
    ORT_ENFORCE(x_shape.NumDimensions() > 1);
    int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    ORT_ENFORCE(y_scale.Shape().Size() == x_shape[axis], "scale must have ", x_shape[axis], " elements (axis=", axis, ").");

    const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());
    const auto num_of_elements = x_shape.Size();

    ORT_RETURN_IF_ERROR(CudaQuantizeLinearAxis(Stream(ctx), input, output, scale, zero_point, num_of_elements,
                                               x_shape.SizeToDimension(axis), y_scale.Shape().Size(), saturate_));
    return Status::OK();
  }
}

template <class T, class U>
typename std::enable_if<boost::mp11::mp_set_contains<TypeList<int8_t, uint8_t>, T>::value, Status>::type
CudaDequantizeLinear(cudaStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element) {
  return CudaDequantizeLinearStd(stream, input, output, scale, zero_point, num_of_element);
}

#if !defined(DISABLE_FLOAT8_TYPES)
template <class T, class U>
typename std::enable_if<boost::mp11::mp_set_contains<TypeList<Float8E4M3FN, Float8E5M2>, T>::value, Status>::type
CudaDequantizeLinear(cudaStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element) {
  return CudaDequantizeLinearSat(stream, input, output, scale, zero_point, num_of_element);
}
#endif

template <class T, class U>
typename std::enable_if<boost::mp11::mp_set_contains<TypeList<int8_t, uint8_t>, T>::value, Status>::type
CudaDequantizeLinearAxis(cudaStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element,
                         size_t batch_size, size_t n_scales) {
  return CudaDequantizeLinearAxisStd(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}

#if !defined(DISABLE_FLOAT8_TYPES)
template <class T, class U>
typename std::enable_if<boost::mp11::mp_set_contains<TypeList<Float8E4M3FN, Float8E5M2>, T>::value, Status>::type
CudaDequantizeLinearAxis(cudaStream_t stream, const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element,
                         size_t batch_size, size_t n_scales) {
  return CudaDequantizeLinearAxisSat(stream, input, output, scale, zero_point, num_of_element, batch_size, n_scales);
}
#endif

template <class T, class U>
Status DequantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<U>::MappedType CudaU;

  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);

  const auto& x_shape = x.Shape();

  auto& y = *ctx->Output(0, x_shape);

  const T* input = x.Data<T>();
  CudaU* output = reinterpret_cast<CudaU*>(y.MutableData<U>());

  if (IsScalarOr1ElementVector(&y_scale)) {
    ORT_ENFORCE(y_zero_point == nullptr || IsScalarOr1ElementVector(y_zero_point), "y_zero_point must be a scalar or 1D tensor of size 1.");

    const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());
    const auto num_of_elements = x_shape.Size();

    ORT_RETURN_IF_ERROR(CudaDequantizeLinear(Stream(ctx), input, output, scale, zero_point, num_of_elements));

    return Status::OK();
  } else {
    ORT_ENFORCE(y_scale.Shape().NumDimensions() == 1);
    ORT_ENFORCE(y_zero_point == nullptr || (y_scale.Shape().Size() == y_zero_point->Shape().Size() && y_zero_point->Shape().NumDimensions() == 1), "scale and zero_point must have the same shape.");
    ORT_ENFORCE(x_shape.NumDimensions() > 1);
    int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
    ORT_ENFORCE(y_scale.Shape().Size() == x_shape[axis], "scale must have ", x_shape[axis], " elements (axis=", axis, ").");

    const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
    const CudaU* scale = reinterpret_cast<const CudaU*>(y_scale.Data<U>());
    const auto num_of_elements = x_shape.Size();

    ORT_RETURN_IF_ERROR(CudaDequantizeLinearAxis(Stream(ctx), input, output, scale, zero_point, num_of_elements,
                                                 x_shape.SizeToDimension(axis), y_scale.Shape().Size()));
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

#define REGISTER_Q_KERNEL_TYPED_19(T)                                     \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                      \
      QuantizeLinear,                                                     \
      kOnnxDomain,                                                        \
      19,                                                                 \
      T, float,                                                           \
      kCudaExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),        \
      QuantizeLinear<T, float>);                                          \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                      \
      QuantizeLinear,                                                     \
      kOnnxDomain,                                                        \
      19,                                                                 \
      T, MLFloat16,                                                       \
      kCudaExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),        \
      QuantizeLinear<T, MLFloat16>);

REGISTER_Q_KERNEL_TYPED_19(int8_t)
REGISTER_Q_KERNEL_TYPED_19(uint8_t)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_Q_KERNEL_TYPED_19(Float8E4M3FN)
REGISTER_Q_KERNEL_TYPED_19(Float8E5M2)
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

#define REGISTER_DQ_KERNEL_TYPED_19(T)                                     \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                       \
      DequantizeLinear,                                                    \
      kOnnxDomain,                                                         \
      19,                                                                  \
      T, float,                                                            \
      kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),     \
      DequantizeLinear<T, float>);                                         \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                       \
      DequantizeLinear,                                                    \
      kOnnxDomain,                                                         \
      19,                                                                  \
      T, MLFloat16,                                                        \
      kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<MLFloat16>()), \
      DequantizeLinear<T, MLFloat16>);

REGISTER_DQ_KERNEL_TYPED_19(int8_t)
REGISTER_DQ_KERNEL_TYPED_19(uint8_t)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DQ_KERNEL_TYPED_19(Float8E4M3FN)
REGISTER_DQ_KERNEL_TYPED_19(Float8E5M2)
#endif

// specialize QuantizeLinear::ComputeInternal and DequantizeLinear::ComputeInternal
#define SPECIALIZED_QDQ_COMPUTE(T, U)                                                \
  template Status QuantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const; \
  template Status DequantizeLinear<T, U>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_QDQ_COMPUTE(int8_t, float)
SPECIALIZED_QDQ_COMPUTE(uint8_t, float)
SPECIALIZED_QDQ_COMPUTE(int8_t, MLFloat16)
SPECIALIZED_QDQ_COMPUTE(uint8_t, MLFloat16)

#if !defined(DISABLE_FLOAT8_TYPES)
SPECIALIZED_QDQ_COMPUTE(Float8E4M3FN, float)
SPECIALIZED_QDQ_COMPUTE(Float8E4M3FN, MLFloat16)
SPECIALIZED_QDQ_COMPUTE(Float8E5M2, float)
SPECIALIZED_QDQ_COMPUTE(Float8E5M2, MLFloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime

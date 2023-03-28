// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define COMPILED_IN_CAST

#include "cast_op.h"
#include "cast_op.cuh"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

const std::vector<MLDataType>& CastOpTypeConstraints() {
  // Must be done as a local static for a shared provider, to avoid the prefast warning:
  // Global initializer calls a non-constexpr function 'onnxruntime::DataTypeImpl::GetTensorType<onnxruntime::MLFloat16>'
  // In a shared provider, GetTensorType is a function call into Onnxruntime and isn't constexpr
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<BFloat16>(),
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<double>(),
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<int16_t>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<int64_t>(),
      DataTypeImpl::GetTensorType<uint8_t>(),
      DataTypeImpl::GetTensorType<uint16_t>(),
      DataTypeImpl::GetTensorType<uint32_t>(),
      DataTypeImpl::GetTensorType<uint64_t>(),
      DataTypeImpl::GetTensorType<bool>(),
      DataTypeImpl::GetTensorType<Float8E4M3FN>(),
      DataTypeImpl::GetTensorType<Float8E4M3FNUZ>(),
      DataTypeImpl::GetTensorType<Float8E5M2>(),
      DataTypeImpl::GetTensorType<Float8E5M2FNUZ>()};
  return types;
}

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      6, 8,                                                       \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      9, 12,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      13, 18,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Cast,                                                       \
      kOnnxDomain,                                                \
      19,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);

#define CASE(TP_TYPE, DstT)                                                                 \
  case TP_TYPE:                                                                             \
    std::cout << "Previous cast to_=" << to_ << "\n";                                       \
    if (count > 0) {                                                                        \
      Impl_Cast<CudaSrcT, typename ToCudaType<DstT>::MappedType>(                           \
          Stream(context),                                                                  \
          x_data,                                                                           \
          reinterpret_cast<typename ToCudaType<DstT>::MappedType*>(Y->MutableData<DstT>()), \
          count);                                                                           \
    }                                                                                       \
    break;

template <typename SrcT>
Status Cast<SrcT>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<SrcT>::MappedType CudaSrcT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<SrcT>());
  size_t count = shape.Size();

  std::cout << "Cast0 to=" << to_ << "\n";

  switch (to_) {
    CASE(TensorProto_DataType_FLOAT16, MLFloat16)
    CASE(TensorProto_DataType_BFLOAT16, BFloat16)
    CASE(TensorProto_DataType_FLOAT, float)
    CASE(TensorProto_DataType_DOUBLE, double)
    CASE(TensorProto_DataType_INT8, int8_t)
    CASE(TensorProto_DataType_INT16, int16_t)
    CASE(TensorProto_DataType_INT32, int32_t)
    CASE(TensorProto_DataType_INT64, int64_t)
    CASE(TensorProto_DataType_UINT8, uint8_t)
    CASE(TensorProto_DataType_UINT16, uint16_t)
    CASE(TensorProto_DataType_UINT32, uint32_t)
    CASE(TensorProto_DataType_UINT64, uint64_t)
    CASE(TensorProto_DataType_BOOL, bool)
    CASE(TensorProto_DataType_FLOAT8E4M3FN, Float8E4M3FN)
    CASE(TensorProto_DataType_FLOAT8E4M3FNUZ, Float8E4M3FNUZ)
    CASE(TensorProto_DataType_FLOAT8E5M2, Float8E5M2)
    CASE(TensorProto_DataType_FLOAT8E5M2FNUZ, Float8E5M2FNUZ)
    case TensorProto_DataType_STRING:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet.");
    case TensorProto_DataType_UNDEFINED:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_);
  }
  return Status::OK();
}

template <>
Status Cast<float>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<float>::MappedType CudaSrcT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<float>());
  size_t count = shape.Size();

  std::cout << "CastFloat to=" << to_ << "\n";

  switch (to_) {
    CASE(TensorProto_DataType_FLOAT16, MLFloat16)
    CASE(TensorProto_DataType_BFLOAT16, BFloat16)
    CASE(TensorProto_DataType_FLOAT, float)
    CASE(TensorProto_DataType_DOUBLE, double)
    CASE(TensorProto_DataType_INT8, int8_t)
    CASE(TensorProto_DataType_INT16, int16_t)
    CASE(TensorProto_DataType_INT32, int32_t)
    CASE(TensorProto_DataType_INT64, int64_t)
    CASE(TensorProto_DataType_UINT8, uint8_t)
    CASE(TensorProto_DataType_UINT16, uint16_t)
    CASE(TensorProto_DataType_UINT32, uint32_t)
    CASE(TensorProto_DataType_UINT64, uint64_t)
    CASE(TensorProto_DataType_BOOL, bool)
    // CASE(TensorProto_DataType_FLOAT8E4M3FN, Float8E4M3FN)
    CASE(TensorProto_DataType_FLOAT8E4M3FNUZ, Float8E4M3FNUZ)
    CASE(TensorProto_DataType_FLOAT8E5M2, Float8E5M2)
    CASE(TensorProto_DataType_FLOAT8E5M2FNUZ, Float8E5M2FNUZ)
    case TensorProto_DataType_STRING:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet.");
    case TensorProto_DataType_UNDEFINED:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");
    case TensorProto_DataType_FLOAT8E4M3FN:
      std::cout << "CudaCast to=" << to_ << "\n";
      if (count > 0) {
        return CudaCast<Float8E4M3FN, float>(
            Stream(context),
            x_data,
            reinterpret_cast<Float8E4M3FN*>(Y->MutableData<Float8E4M3FN>()),
            count);
      }
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_);
  }
  return Status::OK();
}

template <>
Status Cast<Float8E4M3FN>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<Float8E4M3FN>::MappedType CudaSrcT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<Float8E4M3FN>());
  size_t count = shape.Size();

  std::cout << "CastFloat8E4M3FN to=" << to_ << "\n";

  switch (to_) {
    CASE(TensorProto_DataType_FLOAT16, MLFloat16)
    case TensorProto_DataType_FLOAT:
      std::cout << "CudaCast2 to=" << to_ << "\n";
      if (count > 0) {
        return CudaCast<float, Float8E4M3FN>(
            Stream(context),
            x_data,
            reinterpret_cast<float*>(Y->MutableData<float>()),
            count);
      }
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_);
  }
  return Status::OK();
}

#define SPECIALIZE_IMPL(T) \
  REGISTER_KERNEL_TYPED(T) \
  template Status Cast<T>::ComputeInternal(OpKernelContext* context) const;

SPECIALIZE_IMPL(MLFloat16)
SPECIALIZE_IMPL(float)
SPECIALIZE_IMPL(double)
SPECIALIZE_IMPL(int8_t)
SPECIALIZE_IMPL(int16_t)
SPECIALIZE_IMPL(int32_t)
SPECIALIZE_IMPL(int64_t)
SPECIALIZE_IMPL(uint8_t)
SPECIALIZE_IMPL(uint16_t)
SPECIALIZE_IMPL(uint32_t)
SPECIALIZE_IMPL(uint64_t)
SPECIALIZE_IMPL(bool)
SPECIALIZE_IMPL(BFloat16)

#define REGISTER_KERNEL_TYPED_19(T)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Cast,                                                       \
      kOnnxDomain,                                                \
      19,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);

#define SPECIALIZE_IMPL_19(T) \
  REGISTER_KERNEL_TYPED_19(T) \
  template Status Cast<T>::ComputeInternal(OpKernelContext* context) const;

SPECIALIZE_IMPL_19(Float8E4M3FN)
SPECIALIZE_IMPL_19(Float8E4M3FNUZ)
SPECIALIZE_IMPL_19(Float8E5M2)
SPECIALIZE_IMPL_19(Float8E5M2FNUZ)

}  // namespace cuda
}  // namespace onnxruntime

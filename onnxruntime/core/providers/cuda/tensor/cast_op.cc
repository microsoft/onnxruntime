// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cast_op.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

const std::vector<MLDataType>& CastOpTypeConstraints() {
  // Must be done as a local static for a shared provider, to avoid the prefast warning:
  // Global initializer calls a non-constexpr function 'onnxruntime::DataTypeImpl::GetTensorType<onnxruntime::MLFloat16>'
  // In a shared provider, GetTensorType is a function call into Onnxruntime and isn't constexpr
  static std::vector<MLDataType> types {
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
        DataTypeImpl::GetTensorType<bool>()
#if !defined(DISABLE_FLOAT8_TYPES)
            ,
        DataTypeImpl::GetTensorType<Float8E4M3FN>(), DataTypeImpl::GetTensorType<Float8E5M2>()
#endif
  };
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
    if (count > 0) {                                                                        \
      Impl_Cast<CudaSrcT, typename ToCudaType<DstT>::MappedType>(                           \
          Stream(context),                                                                  \
          x_data,                                                                           \
          reinterpret_cast<typename ToCudaType<DstT>::MappedType*>(Y->MutableData<DstT>()), \
          count);                                                                           \
    }                                                                                       \
    break;

#if !defined(DISABLE_FLOAT8_TYPES)

#define CASE_CHECKNOSAT(TP_TYPE, DstT)                                                      \
  case TP_TYPE:                                                                             \
    if (count > 0) {                                                                        \
      ORT_ENFORCE(!saturate_, "saturate_=False is only supported for float and float16.");  \
      Impl_Cast<CudaSrcT, typename ToCudaType<DstT>::MappedType>(                           \
          Stream(context),                                                                  \
          x_data,                                                                           \
          reinterpret_cast<typename ToCudaType<DstT>::MappedType*>(Y->MutableData<DstT>()), \
          count);                                                                           \
    }                                                                                       \
    break;

#define CASE_SAT(TP_TYPE, DstT)                                                             \
  case TP_TYPE:                                                                             \
    if (count > 0) {                                                                        \
      Impl_CastSat<CudaSrcT, typename ToCudaType<DstT>::MappedType>(                        \
          Stream(context),                                                                  \
          x_data,                                                                           \
          reinterpret_cast<typename ToCudaType<DstT>::MappedType*>(Y->MutableData<DstT>()), \
          count,                                                                            \
          saturate_);                                                                       \
    }                                                                                       \
    break;

#endif

template <typename SrcT>
Status Cast<SrcT>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<SrcT>::MappedType CudaSrcT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<SrcT>());
  size_t count = shape.Size();

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
    // By default saturate is true. Case saturate False is only supported for float, float16 for the CUDA provider.
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_CHECKNOSAT(TensorProto_DataType_FLOAT8E4M3FN, Float8E4M3FN)
    CASE_CHECKNOSAT(TensorProto_DataType_FLOAT8E5M2, Float8E5M2)
#endif
    case TensorProto_DataType_STRING:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet.");
    case TensorProto_DataType_UNDEFINED:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_);
  }
  return Status::OK();
}

#if !defined(DISABLE_FLOAT8_TYPES)

#define COMPUTE_INTERNAL_FL16_32(FLOAT_TYPE)                                                            \
  template <>                                                                                           \
  Status Cast<FLOAT_TYPE>::ComputeInternal(OpKernelContext* context) const {                            \
    typedef typename ToCudaType<FLOAT_TYPE>::MappedType CudaSrcT;                                       \
    const Tensor* X = context->Input<Tensor>(0);                                                        \
    const TensorShape& shape = X->Shape();                                                              \
    Tensor* Y = context->Output(0, shape);                                                              \
    const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<FLOAT_TYPE>());                      \
    size_t count = shape.Size();                                                                        \
    switch (to_) {                                                                                      \
      CASE(TensorProto_DataType_FLOAT16, MLFloat16)                                                     \
      CASE(TensorProto_DataType_BFLOAT16, BFloat16)                                                     \
      CASE(TensorProto_DataType_FLOAT, float)                                                           \
      CASE(TensorProto_DataType_DOUBLE, double)                                                         \
      CASE(TensorProto_DataType_INT8, int8_t)                                                           \
      CASE(TensorProto_DataType_INT16, int16_t)                                                         \
      CASE(TensorProto_DataType_INT32, int32_t)                                                         \
      CASE(TensorProto_DataType_INT64, int64_t)                                                         \
      CASE(TensorProto_DataType_UINT8, uint8_t)                                                         \
      CASE(TensorProto_DataType_UINT16, uint16_t)                                                       \
      CASE(TensorProto_DataType_UINT32, uint32_t)                                                       \
      CASE(TensorProto_DataType_UINT64, uint64_t)                                                       \
      CASE(TensorProto_DataType_BOOL, bool)                                                             \
      CASE_SAT(TensorProto_DataType_FLOAT8E4M3FN, Float8E4M3FN)                                         \
      CASE_SAT(TensorProto_DataType_FLOAT8E5M2, Float8E5M2)                                             \
      case TensorProto_DataType_STRING:                                                                 \
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet."); \
      case TensorProto_DataType_UNDEFINED:                                                              \
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");  \
      default:                                                                                          \
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_);             \
    }                                                                                                   \
    return Status::OK();                                                                                \
  }

#else

#define COMPUTE_INTERNAL_FL16_32(FLOAT_TYPE)                                                            \
  template <>                                                                                           \
  Status Cast<FLOAT_TYPE>::ComputeInternal(OpKernelContext* context) const {                            \
    typedef typename ToCudaType<FLOAT_TYPE>::MappedType CudaSrcT;                                       \
    const Tensor* X = context->Input<Tensor>(0);                                                        \
    const TensorShape& shape = X->Shape();                                                              \
    Tensor* Y = context->Output(0, shape);                                                              \
    const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<FLOAT_TYPE>());                      \
    size_t count = shape.Size();                                                                        \
    switch (to_) {                                                                                      \
      CASE(TensorProto_DataType_FLOAT16, MLFloat16)                                                     \
      CASE(TensorProto_DataType_BFLOAT16, BFloat16)                                                     \
      CASE(TensorProto_DataType_FLOAT, float)                                                           \
      CASE(TensorProto_DataType_DOUBLE, double)                                                         \
      CASE(TensorProto_DataType_INT8, int8_t)                                                           \
      CASE(TensorProto_DataType_INT16, int16_t)                                                         \
      CASE(TensorProto_DataType_INT32, int32_t)                                                         \
      CASE(TensorProto_DataType_INT64, int64_t)                                                         \
      CASE(TensorProto_DataType_UINT8, uint8_t)                                                         \
      CASE(TensorProto_DataType_UINT16, uint16_t)                                                       \
      CASE(TensorProto_DataType_UINT32, uint32_t)                                                       \
      CASE(TensorProto_DataType_UINT64, uint64_t)                                                       \
      CASE(TensorProto_DataType_BOOL, bool)                                                             \
      case TensorProto_DataType_STRING:                                                                 \
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet."); \
      case TensorProto_DataType_UNDEFINED:                                                              \
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");  \
      default:                                                                                          \
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_);             \
    }                                                                                                   \
    return Status::OK();                                                                                \
  }

#endif

COMPUTE_INTERNAL_FL16_32(float)
COMPUTE_INTERNAL_FL16_32(MLFloat16)

// TODO: enable BFLOAT16 in another PR.
/*
#if defined(USE_CUDA)
COMPUTE_INTERNAL_FL16_32(BFloat16)
#endif
*/

#if !defined(DISABLE_FLOAT8_TYPES)

#define COMPUTE_INTERNAL_FL8(FLOAT_TYPE)                                                    \
  template <>                                                                               \
  Status Cast<FLOAT_TYPE>::ComputeInternal(OpKernelContext* context) const {                \
    typedef typename ToCudaType<FLOAT_TYPE>::MappedType CudaSrcT;                           \
    const Tensor* X = context->Input<Tensor>(0);                                            \
    const TensorShape& shape = X->Shape();                                                  \
    Tensor* Y = context->Output(0, shape);                                                  \
    const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<FLOAT_TYPE>());          \
    size_t count = shape.Size();                                                            \
    switch (to_) {                                                                          \
      case TensorProto_DataType_FLOAT16:                                                    \
        if (count > 0) {                                                                    \
          Impl_Cast<FLOAT_TYPE, half>(                                                      \
              Stream(context),                                                              \
              x_data,                                                                       \
              reinterpret_cast<half*>(Y->MutableData<MLFloat16>()),                         \
              count);                                                                       \
        }                                                                                   \
        break;                                                                              \
      case TensorProto_DataType_BFLOAT16:                                                   \
        if (count > 0) {                                                                    \
          Impl_Cast<FLOAT_TYPE, half>(                                                      \
              Stream(context),                                                              \
              x_data,                                                                       \
              reinterpret_cast<half*>(Y->MutableData<BFloat16>()),                          \
              count);                                                                       \
        }                                                                                   \
        break;                                                                              \
      case TensorProto_DataType_FLOAT:                                                      \
        if (count > 0) {                                                                    \
          Impl_Cast<FLOAT_TYPE, float>(                                                     \
              Stream(context),                                                              \
              x_data,                                                                       \
              reinterpret_cast<float*>(Y->MutableData<float>()),                            \
              count);                                                                       \
        }                                                                                   \
        break;                                                                              \
      default:                                                                              \
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_); \
    }                                                                                       \
    return Status::OK();                                                                    \
  }

COMPUTE_INTERNAL_FL8(Float8E4M3FN)
COMPUTE_INTERNAL_FL8(Float8E5M2)

#endif

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

#if !defined(DISABLE_FLOAT8_TYPES)

#define SPECIALIZE_IMPL_19(T) \
  REGISTER_KERNEL_TYPED_19(T) \
  template Status Cast<T>::ComputeInternal(OpKernelContext* context) const;

SPECIALIZE_IMPL_19(Float8E4M3FN)
SPECIALIZE_IMPL_19(Float8E5M2)

#endif

///////////////////////////////////////////////////////////////////
// The section below implements CastLike.
///////////////////////////////////////////////////////////////////

#define REGISTER_CASTLIKE_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      CastLike,                                                   \
      kOnnxDomain,                                                \
      15,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      CastLike<T>);

REGISTER_CASTLIKE_KERNEL_TYPED(MLFloat16)
REGISTER_CASTLIKE_KERNEL_TYPED(float)
REGISTER_CASTLIKE_KERNEL_TYPED(double)
REGISTER_CASTLIKE_KERNEL_TYPED(int8_t)
REGISTER_CASTLIKE_KERNEL_TYPED(int16_t)
REGISTER_CASTLIKE_KERNEL_TYPED(int32_t)
REGISTER_CASTLIKE_KERNEL_TYPED(int64_t)
REGISTER_CASTLIKE_KERNEL_TYPED(uint8_t)
REGISTER_CASTLIKE_KERNEL_TYPED(uint16_t)
REGISTER_CASTLIKE_KERNEL_TYPED(uint32_t)
REGISTER_CASTLIKE_KERNEL_TYPED(uint64_t)
REGISTER_CASTLIKE_KERNEL_TYPED(bool)
REGISTER_CASTLIKE_KERNEL_TYPED(BFloat16)

template <typename SrcT>
Status CastLike<SrcT>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<SrcT>::MappedType CudaSrcT;
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* liked = context->Input<Tensor>(1);
  const auto liked_element_type = liked->GetElementType();
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<SrcT>());
  size_t count = shape.Size();

  switch (liked_element_type) {
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
      // By default saturate is true. Case saturate False is only supported for float, float16 for the CUDA provider.

    case TensorProto_DataType_STRING:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet.");
    case TensorProto_DataType_UNDEFINED:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", liked_element_type, ". Search for e.g., TensorProto_DataType_FLOAT for supported element type.");
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

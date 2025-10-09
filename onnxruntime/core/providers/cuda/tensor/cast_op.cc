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
      DataTypeImpl::GetTensorType<bool>()
#if !defined(DISABLE_FLOAT8_TYPES)
          ,
      DataTypeImpl::GetTensorType<Float8E4M3FN>(), DataTypeImpl::GetTensorType<Float8E5M2>()
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
                                                       ,
      DataTypeImpl::GetTensorType<Float4E2M1x2>()
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
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      19, 20,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      21, 22,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Cast,                                                       \
      kOnnxDomain,                                                \
      23,                                                         \
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

#if !defined(DISABLE_FLOAT4_TYPES)

#define CASE_BYTE_PACKED(TP_TYPE, SrcT, DstT)                \
  case TP_TYPE:                                              \
    if (count > 0) {                                         \
      return cast_helper_impl::CudaCastPairwise<DstT, SrcT>( \
          Stream(context),                                   \
          X->Data<SrcT>(),                                   \
          Y->MutableData<DstT>(),                            \
          count);                                            \
    }                                                        \
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
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unimplemented 'to' argument value: ", to_);
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
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_SAT(TensorProto_DataType_FLOAT8E4M3FN, Float8E4M3FN)
    CASE_SAT(TensorProto_DataType_FLOAT8E5M2, Float8E5M2)
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
    CASE_BYTE_PACKED(TensorProto_DataType_FLOAT4E2M1, float, Float4E2M1x2)
#endif
    case TensorProto_DataType_STRING:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet.");
    case TensorProto_DataType_UNDEFINED:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unimplemented 'to' argument value: ", to_);
  }
  return Status::OK();
}

template <>
Status Cast<MLFloat16>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<MLFloat16>::MappedType CudaSrcT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<MLFloat16>());
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
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_SAT(TensorProto_DataType_FLOAT8E4M3FN, Float8E4M3FN)
    CASE_SAT(TensorProto_DataType_FLOAT8E5M2, Float8E5M2)
#endif
    case TensorProto_DataType_STRING:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet.");
    case TensorProto_DataType_UNDEFINED:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unimplemented 'to' argument value: ", to_);
  }
  return Status::OK();
}

// TODO: enable BFLOAT16 in another PR.
/*
#if defined(USE_CUDA)
COMPUTE_INTERNAL_FL16_32(BFloat16)
#endif
*/

#if !defined(DISABLE_FLOAT8_TYPES)

#define COMPUTE_INTERNAL_FL8(FLOAT8_TYPE)                                           \
  template <>                                                                       \
  Status Cast<FLOAT8_TYPE>::ComputeInternal(OpKernelContext* context) const {       \
    typedef typename ToCudaType<FLOAT8_TYPE>::MappedType CudaSrcT;                  \
    const Tensor* X = context->Input<Tensor>(0);                                    \
    const TensorShape& shape = X->Shape();                                          \
    Tensor* Y = context->Output(0, shape);                                          \
    const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->Data<FLOAT8_TYPE>()); \
    size_t count = shape.Size();                                                    \
    switch (to_) {                                                                  \
      case TensorProto_DataType_FLOAT16:                                            \
        if (count > 0) {                                                            \
          Impl_Cast<FLOAT8_TYPE, half>(                                             \
              Stream(context),                                                      \
              x_data,                                                               \
              reinterpret_cast<half*>(Y->MutableData<MLFloat16>()),                 \
              count);                                                               \
        }                                                                           \
        break;                                                                      \
      case TensorProto_DataType_BFLOAT16:                                           \
        if (count > 0) {                                                            \
          Impl_Cast<FLOAT8_TYPE, half>(                                             \
              Stream(context),                                                      \
              x_data,                                                               \
              reinterpret_cast<half*>(Y->MutableData<BFloat16>()),                  \
              count);                                                               \
        }                                                                           \
        break;                                                                      \
      case TensorProto_DataType_FLOAT:                                              \
        if (count > 0) {                                                            \
          Impl_Cast<FLOAT8_TYPE, float>(                                            \
              Stream(context),                                                      \
              x_data,                                                               \
              reinterpret_cast<float*>(Y->MutableData<float>()),                    \
              count);                                                               \
        }                                                                           \
        break;                                                                      \
      default:                                                                      \
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,                        \
                               "Unimplemented 'to' argument value: ", to_);         \
    }                                                                               \
    return Status::OK();                                                            \
  }

COMPUTE_INTERNAL_FL8(Float8E4M3FN)
COMPUTE_INTERNAL_FL8(Float8E5M2)

#endif

#if !defined(DISABLE_FLOAT4_TYPES)

template <>
Status Cast<Float4E2M1x2>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  size_t count = shape.Size();

  switch (to_) {
    CASE_BYTE_PACKED(TensorProto_DataType_FLOAT, Float4E2M1x2, float);
    case TensorProto_DataType_STRING:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet.");
    case TensorProto_DataType_UNDEFINED:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unimplemented 'to' argument value: ", to_);
  }
  return Status::OK();
}

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

#define REGISTER_KERNEL_TYPED_19_TO_22(T)                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      19, 20,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      21, 22,                                                     \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", CastOpTypeConstraints()),         \
      Cast<T>);

#define REGISTER_KERNEL_TYPED_23(T, OutputTypeConstraints)        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Cast,                                                       \
      kOnnxDomain,                                                \
      23,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", OutputTypeConstraints),           \
      Cast<T>);

#if !defined(DISABLE_FLOAT8_TYPES)

#define SPECIALIZE_IMPL_19_TO_23(T)                    \
  REGISTER_KERNEL_TYPED_19_TO_22(T)                    \
  REGISTER_KERNEL_TYPED_23(T, CastOpTypeConstraints()) \
  template Status Cast<T>::ComputeInternal(OpKernelContext* context) const;

SPECIALIZE_IMPL_19_TO_23(Float8E4M3FN)
SPECIALIZE_IMPL_19_TO_23(Float8E5M2)

#endif

#if !defined(DISABLE_FLOAT4_TYPES)
REGISTER_KERNEL_TYPED_23(Float4E2M1x2, {DataTypeImpl::GetTensorType<float>()})
template Status Cast<Float4E2M1x2>::ComputeInternal(OpKernelContext* context) const;
#endif

}  // namespace cuda
}  // namespace onnxruntime

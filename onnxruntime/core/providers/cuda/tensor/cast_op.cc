// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cast_op.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

const DeleteOnUnloadPtr<std::vector<MLDataType>> castOpTypeConstraints = new std::vector<MLDataType> {
  DataTypeImpl::GetTensorType<MLFloat16>(),
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
      DataTypeImpl::GetTensorType<BFloat16>(),
#endif
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
};

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      6, 8,                                                       \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", *castOpTypeConstraints),          \
      Cast<T>);                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      9, 12,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", *castOpTypeConstraints),          \
      Cast<T>);                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Cast,                                                       \
      kOnnxDomain,                                                \
      13,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", *castOpTypeConstraints),          \
      Cast<T>);

template <typename SrcT>
Status Cast<SrcT>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<SrcT>::MappedType CudaSrcT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  const auto* x_data = reinterpret_cast<const CudaSrcT*>(X->template Data<SrcT>());
  size_t count = shape.Size();

#define CASE(TP_TYPE, DstT)                                                                          \
  case TP_TYPE:                                                                                      \
    if (count > 0) {                                                                                 \
      Impl_Cast<CudaSrcT, typename ToCudaType<DstT>::MappedType>(                                    \
          Stream(),                                                                                  \
          x_data,                                                                                    \
          reinterpret_cast<typename ToCudaType<DstT>::MappedType*>(Y->template MutableData<DstT>()), \
          count);                                                                                    \
    }                                                                                                \
    break;

  switch (to_) {
    CASE(TensorProto_DataType_FLOAT16, MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    CASE(TensorProto_DataType_BFLOAT16, BFloat16)
#endif
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
    case TensorProto_DataType_STRING:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Casting to and from strings is not supported yet.");
    case TensorProto_DataType_UNDEFINED:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cast op must have 'to' argument of type DataType");
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
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
SPECIALIZE_IMPL(BFloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime

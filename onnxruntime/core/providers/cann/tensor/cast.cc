// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include "core/providers/cann/tensor/cast.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

aclDataType getACLTypeByMap(ONNX_NAMESPACE::TensorProto_DataType type) {
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return ACL_FLOAT16;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return ACL_FLOAT;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return ACL_DOUBLE;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return ACL_INT8;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return ACL_INT16;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return ACL_INT32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return ACL_INT64;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return ACL_UINT8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return ACL_UINT16;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return ACL_UINT32;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return ACL_UINT64;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return ACL_BOOL;

    default:
      return ACL_DT_UNDEFINED;
  }
}

template <typename T>
Status Cast<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);

  Tensor* Y = context->Output(0, X->Shape());

  aclFormat format = ACL_FORMAT_ND;
  const aclDataType aclTypeX = getACLType<T>();
  const aclDataType aclTypeY = getACLTypeByMap(to_);
  ORT_ENFORCE(aclTypeY != ACL_DT_UNDEFINED, "unsupported type");

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "dst_type", aclTypeY));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclTypeX, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclTypeY, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(X->DataRaw()), X->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableDataRaw(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("Cast",
                                              prepare.inputDesc_.size(),
                                              prepare.inputDesc_.data(),
                                              prepare.inputBuffers_.data(),
                                              prepare.outputDesc_.size(),
                                              prepare.outputDesc_.data(),
                                              prepare.outputBuffers_.data(),
                                              prepare.opAttr_,
                                              ACL_ENGINE_SYS,
                                              ACL_COMPILE_SYS,
                                              NULL,
                                              Stream()));

  return Status::OK();
}

const std::vector<MLDataType> castOpTypeConstraints = {
    DataTypeImpl::GetTensorType<MLFloat16>(),
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
    DataTypeImpl::GetTensorType<bool>()};

#define REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Cast,                                                       \
      kOnnxDomain,                                                \
      startver,                                                   \
      endver,                                                     \
      T,                                                          \
      kCannExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", castOpTypeConstraints),           \
      Cast<T>);

#define REGISTER_CAST_TYPED_KERNEL(ver, T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Cast,                                                       \
      kOnnxDomain,                                                \
      ver,                                                        \
      T,                                                          \
      kCannExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", castOpTypeConstraints),           \
      Cast<T>);

// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// X: bfloat16
// F: float
// D: double
// O: bool

#define REGISTER_CAST_VERSIONED_BWUZCSILHFDO(startver, endver)      \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, uint8_t)   \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, uint16_t)  \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, uint32_t)  \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, uint64_t)  \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, int8_t)    \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, int16_t)   \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, int32_t)   \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, int64_t)   \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, MLFloat16) \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, float)     \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, double)    \
  REGISTER_CAST_VERSIONED_TYPED_KERNEL(startver, endver, bool)

#define REGISTER_CAST_BWUZCSILHFDO(ver)      \
  REGISTER_CAST_TYPED_KERNEL(ver, uint8_t)   \
  REGISTER_CAST_TYPED_KERNEL(ver, uint16_t)  \
  REGISTER_CAST_TYPED_KERNEL(ver, uint32_t)  \
  REGISTER_CAST_TYPED_KERNEL(ver, uint64_t)  \
  REGISTER_CAST_TYPED_KERNEL(ver, int8_t)    \
  REGISTER_CAST_TYPED_KERNEL(ver, int16_t)   \
  REGISTER_CAST_TYPED_KERNEL(ver, int32_t)   \
  REGISTER_CAST_TYPED_KERNEL(ver, int64_t)   \
  REGISTER_CAST_TYPED_KERNEL(ver, MLFloat16) \
  REGISTER_CAST_TYPED_KERNEL(ver, float)     \
  REGISTER_CAST_TYPED_KERNEL(ver, double)    \
  REGISTER_CAST_TYPED_KERNEL(ver, bool)

REGISTER_CAST_VERSIONED_BWUZCSILHFDO(6, 8)
REGISTER_CAST_VERSIONED_BWUZCSILHFDO(9, 12)
REGISTER_CAST_BWUZCSILHFDO(13)

}  // namespace cann
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/cast_op.h"
#include <sstream>
#include "core/common/common.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

const std::vector<MLDataType> castOpTypeConstraints{
    DataTypeImpl::GetTensorType<bool>(),
    DataTypeImpl::GetTensorType<float>(),
    DataTypeImpl::GetTensorType<double>(),
    DataTypeImpl::GetTensorType<uint8_t>(),
    DataTypeImpl::GetTensorType<uint16_t>(),
    DataTypeImpl::GetTensorType<uint32_t>(),
    DataTypeImpl::GetTensorType<uint64_t>(),
    DataTypeImpl::GetTensorType<int8_t>(),
    DataTypeImpl::GetTensorType<int16_t>(),
    DataTypeImpl::GetTensorType<int32_t>(),
    DataTypeImpl::GetTensorType<int64_t>(),
    DataTypeImpl::GetTensorType<MLFloat16>()};

#define ADD_FROM_CAST_OP(in_type)                                                                                                  \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                                                                  \
      Cast,                                                                                                                        \
      6,                                                                                                                           \
      in_type,                                                                                                                     \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", castOpTypeConstraints), \
      Cast<in_type>);                                                                                                              \
                                                                                                                                   \
  template <>                                                                                                                      \
  Status Cast<in_type>::Compute(OpKernelContext* context) const {                                                                  \
    const Tensor* X = context->Input<Tensor>(0);                                                                                   \
    if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");                                    \
    const TensorShape& shape = X->Shape();                                                                                         \
    Tensor* Y = context->Output(0, TensorShape(shape));                                                                            \
                                                                                                                                   \
    switch (to_) {                                                                                                                 \
      case TensorProto_DataType_BOOL:                                                                                              \
        CastData<in_type, bool>(X, Y, shape);                                                                                      \
        break;                                                                                                                     \
      case TensorProto_DataType_INT16:                                                                                             \
        CastData<in_type, int16_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_INT32:                                                                                             \
        CastData<in_type, int32_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_INT64:                                                                                             \
        CastData<in_type, int64_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT8:                                                                                             \
        CastData<in_type, uint8_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT16:                                                                                            \
        CastData<in_type, uint16_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT32:                                                                                            \
        CastData<in_type, uint32_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT64:                                                                                            \
        CastData<in_type, uint64_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_FLOAT:                                                                                             \
        CastData<in_type, float>(X, Y, shape);                                                                                     \
        break;                                                                                                                     \
      case TensorProto_DataType_DOUBLE:                                                                                            \
        CastData<in_type, double>(X, Y, shape);                                                                                    \
        break;                                                                                                                     \
      case TensorProto_DataType_INT8:                                                                                              \
        CastData<in_type, int8_t>(X, Y, shape);                                                                                    \
        break;                                                                                                                     \
      case TensorProto_DataType_FLOAT16:                                                                                           \
        if (std::is_same<in_type, float>::value) {                                                                                 \
          CastData<float, MLFloat16>(X, Y, shape);                                                                                 \
        } else {                                                                                                                   \
          auto st = CastFloat16Data<in_type, MLFloat16>(X, Y, shape, context);                                                     \
          if (!st.IsOK()) return st;                                                                                               \
        }                                                                                                                          \
        break;                                                                                                                     \
      case TensorProto_DataType_STRING:                                                                                            \
        ONNXRUNTIME_THROW("Casting to and from strings is not supported yet."); /*break;*/                                         \
      case TensorProto_DataType_UNDEFINED:                                                                                         \
        ONNXRUNTIME_THROW("Cast op must have 'to' argument of type DataType"); /*break;*/                                          \
      default:                                                                                                                     \
        ONNXRUNTIME_THROW("Unexpected 'to' argument value: ", to_);                                                                \
    }                                                                                                                              \
    return Status::OK();                                                                                                           \
  }

ADD_FROM_CAST_OP(uint8_t);
ADD_FROM_CAST_OP(uint16_t);
ADD_FROM_CAST_OP(uint32_t);
ADD_FROM_CAST_OP(uint64_t);
ADD_FROM_CAST_OP(int8_t);
ADD_FROM_CAST_OP(int16_t);
ADD_FROM_CAST_OP(int32_t);
ADD_FROM_CAST_OP(int64_t);
ADD_FROM_CAST_OP(bool);
ADD_FROM_CAST_OP(float);
ADD_FROM_CAST_OP(double);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Cast,
    6,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>()).TypeConstraint("T2", castOpTypeConstraints),
    Cast<MLFloat16>);

template <>
Status Cast<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, TensorShape(shape));
  Status st;
  switch (to_) {
    case TensorProto_DataType_BOOL:
      st = CastFloat16Data<MLFloat16, bool>(X, Y, shape, context);
      break;
    case TensorProto_DataType_INT16:
      st = CastFloat16Data<MLFloat16, int16_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_INT32:
      st = CastFloat16Data<MLFloat16, int32_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_INT64:
      st = CastFloat16Data<MLFloat16, int64_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_UINT8:
      st = CastFloat16Data<MLFloat16, uint8_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_UINT16:
      st = CastFloat16Data<MLFloat16, uint16_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_UINT32:
      st = CastFloat16Data<MLFloat16, uint32_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_UINT64:
      st = CastFloat16Data<MLFloat16, uint64_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_FLOAT:
      CastData<MLFloat16, float>(X, Y, shape);
      break;
    case TensorProto_DataType_FLOAT16: {
        auto X_type = X->DataType();
        const void* source = X->DataRaw(X_type);
        void* target = Y->MutableDataRaw(X_type);
        // if source and target pointers are not equal, we need to copy the data.
        if (target != source) {
          memcpy(target, source, shape.Size() * X_type->Size());
        }
        st = Status::OK();
      break;
    }
    case TensorProto_DataType_DOUBLE:
      st = CastFloat16Data<MLFloat16, double>(X, Y, shape, context);
      break;
    case TensorProto_DataType_INT8:
      st = CastFloat16Data<MLFloat16, int8_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_STRING:
      ONNXRUNTIME_THROW("Casting to and from strings is not supported yet."); /*break;*/
    case TensorProto_DataType_UNDEFINED:
      ONNXRUNTIME_THROW("Cast op must have 'to' argument of type DataType"); /*break;*/
    default:
      ONNXRUNTIME_THROW("Unexpected 'to' argument value: ", to_);
  }
  return st;
}

}  //namespace onnxruntime

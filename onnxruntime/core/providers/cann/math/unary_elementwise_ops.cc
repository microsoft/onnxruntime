// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/math/unary_elementwise_ops.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status UnaryElementwise::Prepare(OpKernelContext* ctx, CannPreparation& prepare) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  if (op_name_ == "Log" || op_name_ == "Exp") {
    CANN_RETURN_IF_ERROR(aclopSetAttrFloat(prepare.opAttr_, "base", -1.0f));
    CANN_RETURN_IF_ERROR(aclopSetAttrFloat(prepare.opAttr_, "scale", 1.0f));
    CANN_RETURN_IF_ERROR(aclopSetAttrFloat(prepare.opAttr_, "shift", 0.0f));
  }

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(X->DataRaw()), X->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableDataRaw(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

#define REGISTER_ELEMENTWISE_TYPED_COMPUTE(x, T)                               \
  template <>                                                                  \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {               \
    CannPreparation prepare;                                                   \
    ORT_RETURN_IF_ERROR(Prepare<T>(context, prepare));                         \
    CANN_RETURN_IF_ERROR(aclopCompileAndExecute(#x,                            \
                                                prepare.inputDesc_.size(),     \
                                                prepare.inputDesc_.data(),     \
                                                prepare.inputBuffers_.data(),  \
                                                prepare.outputDesc_.size(),    \
                                                prepare.outputDesc_.data(),    \
                                                prepare.outputBuffers_.data(), \
                                                prepare.opAttr_,               \
                                                ACL_ENGINE_SYS,                \
                                                ACL_COMPILE_SYS,               \
                                                NULL,                          \
                                                Stream()));                    \
    return Status::OK();                                                       \
  }

#define REGISTER_ELEMENTWISE_TYPED_KERNEL(x, ver, T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define REGISTER_ELEMENTWISE_VERSIONED_TYPED_KERNEL(x, startver, endver, T)                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      endver,                                                                              \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, T) \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED_KERNEL(name, startver, endver, T)

#define REGISTER_ELEMENTWISE_TYPED(name, ver, T)  \
  REGISTER_ELEMENTWISE_TYPED_KERNEL(name, ver, T) \
  REGISTER_ELEMENTWISE_TYPED_COMPUTE(name, T)

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

#define REGISTER_ELEMENTWISE_VERSIONED_H(name, startver, endver) \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, MLFloat16)

#define REGISTER_ELEMENTWISE_VERSIONED_HF(name, startver, endver) \
  REGISTER_ELEMENTWISE_VERSIONED_H(name, startver, endver)        \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, float)

#define REGISTER_ELEMENTWISE_VERSIONED_IHF(name, startver, endver) \
  REGISTER_ELEMENTWISE_VERSIONED_HF(name, startver, endver)        \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int32_t)

#define REGISTER_ELEMENTWISE_VERSIONED_CIHF(name, startver, endver) \
  REGISTER_ELEMENTWISE_VERSIONED_IHF(name, startver, endver)        \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int8_t)

#define REGISTER_ELEMENTWISE_H(name, ver) \
  REGISTER_ELEMENTWISE_TYPED(name, ver, MLFloat16)

#define REGISTER_ELEMENTWISE_HF(name, ver) \
  REGISTER_ELEMENTWISE_H(name, ver)        \
  REGISTER_ELEMENTWISE_TYPED(name, ver, float)

#define REGISTER_ELEMENTWISE_IHF(name, ver) \
  REGISTER_ELEMENTWISE_HF(name, ver)        \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int32_t)

#define REGISTER_ELEMENTWISE_CIHF(name, ver) \
  REGISTER_ELEMENTWISE_IHF(name, ver)        \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int8_t)

REGISTER_ELEMENTWISE_VERSIONED_HF(Abs, 1, 5)
REGISTER_ELEMENTWISE_VERSIONED_IHF(Abs, 6, 12)
REGISTER_ELEMENTWISE_IHF(Abs, 13)

REGISTER_ELEMENTWISE_VERSIONED_HF(Neg, 1, 5)
REGISTER_ELEMENTWISE_VERSIONED_CIHF(Neg, 6, 12)
REGISTER_ELEMENTWISE_CIHF(Neg, 13)

REGISTER_ELEMENTWISE_VERSIONED_HF(Floor, 1, 5)
REGISTER_ELEMENTWISE_VERSIONED_HF(Floor, 6, 12)
REGISTER_ELEMENTWISE_HF(Floor, 13)

REGISTER_ELEMENTWISE_VERSIONED_HF(Ceil, 1, 5)
REGISTER_ELEMENTWISE_VERSIONED_HF(Ceil, 6, 12)
REGISTER_ELEMENTWISE_HF(Ceil, 13)

REGISTER_ELEMENTWISE_VERSIONED_HF(Reciprocal, 1, 5)
REGISTER_ELEMENTWISE_VERSIONED_HF(Reciprocal, 6, 12)
REGISTER_ELEMENTWISE_HF(Reciprocal, 13)

REGISTER_ELEMENTWISE_VERSIONED_HF(Sqrt, 1, 5)
REGISTER_ELEMENTWISE_VERSIONED_HF(Sqrt, 6, 12)
REGISTER_ELEMENTWISE_HF(Sqrt, 13)

REGISTER_ELEMENTWISE_VERSIONED_HF(Log, 1, 5)
REGISTER_ELEMENTWISE_VERSIONED_HF(Log, 6, 12)
REGISTER_ELEMENTWISE_HF(Log, 13)

REGISTER_ELEMENTWISE_VERSIONED_H(Exp, 1, 5)
REGISTER_ELEMENTWISE_VERSIONED_H(Exp, 6, 12)
REGISTER_ELEMENTWISE_H(Exp, 13)

REGISTER_ELEMENTWISE_VERSIONED_HF(Erf, 9, 12)
REGISTER_ELEMENTWISE_HF(Erf, 13)

REGISTER_ELEMENTWISE_HF(Round, 11)

REGISTER_ELEMENTWISE_HF(Sin, 7)

REGISTER_ELEMENTWISE_HF(Cos, 7)

}  // namespace cann
}  // namespace onnxruntime

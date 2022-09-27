// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/math/binary_elementwise_ops.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status BinaryElementwise::Prepare(OpKernelContext* ctx, CannPreparation& prepare) const {
  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  Tensor* C = ctx->Output(0, A->Shape());

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, A->Shape().NumDimensions(), A->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, aclType, A->Shape().NumDimensions(), A->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, A->Shape().NumDimensions(), A->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(A->template Data<T>()), A->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(B->template Data<T>()), B->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, C->template MutableData<T>(), C->SizeInBytes());
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

#define REGISTER_ELEMENTWISE_TYPED_KERNEL(x, class_name, ver, T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      class_name<T>);

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

#define REGISTER_ELEMENTWISE_TYPED(name, ver, T)        \
  REGISTER_ELEMENTWISE_TYPED_KERNEL(name, name, ver, T) \
  REGISTER_ELEMENTWISE_TYPED_COMPUTE(name, T)

#define REGISTER_ELEMENTWISE_VERSIONED_ILHFD(name, startver, endver)      \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int32_t)   \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int64_t)   \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, float)     \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, double)

#define REGISTER_ELEMENTWISE_VERSIONED_IHF(name, startver, endver)        \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int32_t)   \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, float)

#define REGISTER_ELEMENTWISE_BCSILHFD(name, ver)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, uint8_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int8_t)    \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int16_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int32_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int64_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, MLFloat16) \
  REGISTER_ELEMENTWISE_TYPED(name, ver, float)     \
  REGISTER_ELEMENTWISE_TYPED(name, ver, double)

#define REGISTER_ELEMENTWISE_IHF(name, ver)        \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int32_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, MLFloat16) \
  REGISTER_ELEMENTWISE_TYPED(name, ver, float)

#define REGISTER_ELEMENTWISE_BCSIHF(name, ver)     \
  REGISTER_ELEMENTWISE_TYPED(name, ver, uint8_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int8_t)    \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int16_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int32_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, MLFloat16) \
  REGISTER_ELEMENTWISE_TYPED(name, ver, float)

#define REGISTER_ELEMENTWISE_ILHFD(name, ver)      \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int32_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int64_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, MLFloat16) \
  REGISTER_ELEMENTWISE_TYPED(name, ver, float)     \
  REGISTER_ELEMENTWISE_TYPED(name, ver, double)

REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Add, 7, 12)
REGISTER_ELEMENTWISE_VERSIONED_IHF(Sub, 7, 12)
REGISTER_ELEMENTWISE_VERSIONED_IHF(Mul, 7, 12)
REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Div, 7, 12)

REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Add, 13, 13)
REGISTER_ELEMENTWISE_VERSIONED_IHF(Sub, 13, 13)
REGISTER_ELEMENTWISE_VERSIONED_IHF(Mul, 13, 13)
REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Div, 13, 13)

REGISTER_ELEMENTWISE_BCSILHFD(Add, 14)
REGISTER_ELEMENTWISE_IHF(Sub, 14)
REGISTER_ELEMENTWISE_BCSIHF(Mul, 14)
REGISTER_ELEMENTWISE_ILHFD(Div, 14)

}  // namespace cann
}  // namespace onnxruntime

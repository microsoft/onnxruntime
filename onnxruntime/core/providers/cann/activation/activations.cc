// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/activation/activations.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status Activations::Prepare(OpKernelContext* ctx, CannPreparation& prepare) const {
  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  const Tensor* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(X->DataRaw()), X->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableDataRaw(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

#define REGISTER_ACTIVATION_TYPED_COMPUTE(x, T)                                \
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

#define REGISTER_ACTIVATION_TYPED_KERNEL(x, class_name, ver, T)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      class_name<T>);

#define REGISTER_ACTIVATION_VERSIONED_TYPED_KERNEL(x, startver, endver, T)                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      endver,                                                                              \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define REGISTER_ACTIVATION_VERSIONED_TYPED(name, startver, endver, T) \
  REGISTER_ACTIVATION_VERSIONED_TYPED_KERNEL(name, startver, endver, T)

#define REGISTER_ACTIVATION_TYPED(name, ver, T)        \
  REGISTER_ACTIVATION_TYPED_KERNEL(name, name, ver, T) \
  REGISTER_ACTIVATION_TYPED_COMPUTE(name, T)

#define REGISTER_ACTIVATION_VERSIONED_HFD(name, startver, endver)        \
  REGISTER_ACTIVATION_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  REGISTER_ACTIVATION_VERSIONED_TYPED(name, startver, endver, float)     \
  REGISTER_ACTIVATION_VERSIONED_TYPED(name, startver, endver, double)

#define REGISTER_ACTIVATION_CSIHFD(name, ver)     \
  REGISTER_ACTIVATION_TYPED(name, ver, int8_t)    \
  REGISTER_ACTIVATION_TYPED(name, ver, int16_t)   \
  REGISTER_ACTIVATION_TYPED(name, ver, int32_t)   \
  REGISTER_ACTIVATION_TYPED(name, ver, int64_t)   \
  REGISTER_ACTIVATION_TYPED(name, ver, MLFloat16) \
  REGISTER_ACTIVATION_TYPED(name, ver, float)     \
  REGISTER_ACTIVATION_TYPED(name, ver, double)

REGISTER_ACTIVATION_VERSIONED_HFD(Relu, 6, 12)

REGISTER_ACTIVATION_VERSIONED_HFD(Relu, 13, 13)

REGISTER_ACTIVATION_CSIHFD(Relu, 14)

}  // namespace cann
}  // namespace onnxruntime

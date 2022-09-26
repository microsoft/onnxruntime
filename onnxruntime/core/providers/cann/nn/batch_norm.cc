// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/nn/batch_norm.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cann {

template <typename T>
Status BatchNorm<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);
  const Tensor* B = ctx->Input<Tensor>(2);
  const Tensor* mean = ctx->Input<Tensor>(3);
  const Tensor* var = ctx->Input<Tensor>(4);

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, spatial_ == 1));

  const TensorShape& x_shape = X->Shape();
  const TensorShape& channel_shape = mean->Shape();

  Tensor* Y = ctx->Output(0, x_shape);
  Tensor* running_mean = ctx->Output(1, channel_shape);
  Tensor* running_var = ctx->Output(2, channel_shape);
  Tensor* saved_mean = ctx->Output(3, channel_shape);
  Tensor* saved_var = ctx->Output(4, channel_shape);

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "epsilon", epsilon_));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, scale->Shape().NumDimensions(), scale->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, B->Shape().NumDimensions(), B->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, mean->Shape().NumDimensions(), mean->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, var->Shape().NumDimensions(), var->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);
    if (running_mean && running_var && saved_mean && saved_var) {
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, running_mean->Shape().NumDimensions(),
                              running_mean->Shape().GetDims().data(), format);
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, running_var->Shape().NumDimensions(),
                              running_var->Shape().GetDims().data(), format);
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, saved_mean->Shape().NumDimensions(),
                              saved_mean->Shape().GetDims().data(), format);
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, saved_var->Shape().NumDimensions(),
                              saved_var->Shape().GetDims().data(), format);
    } else {
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
      CANN_PREPARE_OUTPUTDESC(prepare, ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    }

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(X->template Data<T>()), X->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(scale->template Data<T>()), scale->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(B->template Data<T>()), B->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(mean->template Data<T>()), mean->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(var->template Data<T>()), var->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->template MutableData<T>(), Y->SizeInBytes());
    if (running_mean && running_var && saved_mean && saved_var) {
      CANN_PREPARE_OUTPUTBUFFER(prepare, running_mean->template MutableData<T>(), running_mean->SizeInBytes());
      CANN_PREPARE_OUTPUTBUFFER(prepare, running_var->template MutableData<T>(), running_var->SizeInBytes());
      CANN_PREPARE_OUTPUTBUFFER(prepare, saved_mean->template MutableData<T>(), saved_mean->SizeInBytes());
      CANN_PREPARE_OUTPUTBUFFER(prepare, saved_var->template MutableData<T>(), saved_var->SizeInBytes());
    } else {
      CANN_PREPARE_OUTPUTBUFFER(prepare, nullptr, 0);
      CANN_PREPARE_OUTPUTBUFFER(prepare, nullptr, 0);
      CANN_PREPARE_OUTPUTBUFFER(prepare, nullptr, 0);
      CANN_PREPARE_OUTPUTBUFFER(prepare, nullptr, 0);
    }
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("BatchNormalization",
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

#define REGISTER_KERNEL_TYPED(T)                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      BatchNormalization,                                          \
      kOnnxDomain,                                                 \
      7, 8,                                                        \
      T,                                                           \
      kCannExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);                                               \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      BatchNormalization,                                          \
      kOnnxDomain,                                                 \
      9, 13,                                                       \
      T,                                                           \
      kCannExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);                                               \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      BatchNormalization,                                          \
      kOnnxDomain,                                                 \
      14, 14,                                                      \
      T,                                                           \
      kCannExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())   \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<T>()),  \
      BatchNorm<T>);                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      BatchNormalization,                                          \
      kOnnxDomain,                                                 \
      15,                                                          \
      T,                                                           \
      kCannExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      BatchNorm<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)

}  // namespace cann
}  // namespace onnxruntime

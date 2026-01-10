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

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var));

  // There is only one output in inference mode
  Tensor* Y = ctx->Output(0, X->Shape());

  IAllocatorUniquePtr<void> pbatch_mean = GetScratchBuffer<void>(mean->SizeInBytes(), ctx->GetComputeStream());
  IAllocatorUniquePtr<void> pbatch_variance = GetScratchBuffer<void>(var->SizeInBytes(), ctx->GetComputeStream());
  IAllocatorUniquePtr<void> preserver_space_1 = GetScratchBuffer<void>(mean->SizeInBytes(), ctx->GetComputeStream());
  IAllocatorUniquePtr<void> preserver_space_2 = GetScratchBuffer<void>(var->SizeInBytes(), ctx->GetComputeStream());

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_NCHW;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrFloat(prepare.opAttr_, "epsilon", epsilon_));
  CANN_RETURN_IF_ERROR(aclopSetAttrString(prepare.opAttr_, "data_format", "NCHW"));
  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "is_training", is_training_mode_));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, 1, scale->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, 1, B->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, 1, mean->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, 1, var->Shape().GetDims().data(), format);

    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, 1, mean->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, 1, var->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, 1, mean->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, ACL_FLOAT, 1, var->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(X->DataRaw()), X->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(scale->DataRaw()), scale->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(B->DataRaw()), B->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(mean->DataRaw()), mean->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(var->DataRaw()), var->SizeInBytes());

    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableData<T>(), Y->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, pbatch_mean.get(), mean->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, pbatch_variance.get(), var->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, preserver_space_1.get(), mean->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, preserver_space_2.get(), var->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("BatchNorm",
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
                                              Stream(ctx)));

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

REGISTER_KERNEL_TYPED(float)

}  // namespace cann
}  // namespace onnxruntime

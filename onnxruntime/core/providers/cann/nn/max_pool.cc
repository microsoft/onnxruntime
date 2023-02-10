// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/nn/max_pool.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status MaxPool<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  const auto X_dims = X_shape.GetDims();

  if (X_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  auto kernel_shape = pool_attrs_.kernel_shape;
  auto strides = pool_attrs_.strides;
  auto pads = pool_attrs_.pads;

  if (pool_attrs_.global_pooling) {
    kernel_shape.assign(X_dims.begin() + 2, X_dims.end());
    strides.assign(kernel_shape.size(), 1);
    pads.assign(kernel_shape.size() * 2, 0);
  }

  kernel_shape.insert(kernel_shape.begin(), {1, 1});
  strides.insert(strides.begin(), {1, 1});

  auto Y_dims = pool_attrs_.SetOutputSize(X_shape, X_shape[1], &pads);
  TensorShape Y_shape(Y_dims);
  Tensor* Y = context->Output(0, Y_shape);
  if (Y_shape.Size() == 0)
    return Status::OK();

  std::unordered_map<AutoPadType, const char*> padding_mode = {
      {AutoPadType::NOTSET, "CALCULATED"},
      {AutoPadType::SAME_UPPER, "SAME"},
      {AutoPadType::SAME_LOWER, "SAME"},
      {AutoPadType::VALID, "VALID"}};

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "ksize", kernel_shape.size(), kernel_shape.data()));
  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "strides", strides.size(), strides.data()));
  CANN_RETURN_IF_ERROR(aclopSetAttrString(prepare.opAttr_, "data_format", "NCHW"));
  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "pads", pads.size(), pads.data()));
  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "ceil_mode", pool_attrs_.ceil_mode));
  if (!pool_attrs_.global_pooling) {
    CANN_RETURN_IF_ERROR(aclopSetAttrString(prepare.opAttr_, "padding_mode", padding_mode[pool_attrs_.auto_pad]));
    CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "global_pooling", false));
  } else {
    CANN_RETURN_IF_ERROR(aclopSetAttrString(prepare.opAttr_, "padding_mode", "VALID"));
    CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "global_pooling", true));
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

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("MaxPoolV3",
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

#define REGISTER_POOL_TYPED_KERNEL(x, T, ver)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MaxPool<T>);

#define REGISTER_POOL_VERSIONED_TYPED_KERNEL(x, T, startver, endver) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                           \
      x,                                                             \
      kOnnxDomain,                                                   \
      startver,                                                      \
      endver,                                                        \
      T,                                                             \
      kCannExecutionProvider,                                        \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),    \
      MaxPool<T>);

REGISTER_POOL_VERSIONED_TYPED_KERNEL(MaxPool, MLFloat16, 1, 7)
REGISTER_POOL_VERSIONED_TYPED_KERNEL(MaxPool, float, 1, 7)
REGISTER_POOL_VERSIONED_TYPED_KERNEL(MaxPool, double, 1, 7)

REGISTER_POOL_TYPED_KERNEL(GlobalMaxPool, MLFloat16, 1)
REGISTER_POOL_TYPED_KERNEL(GlobalMaxPool, float, 1)
REGISTER_POOL_TYPED_KERNEL(GlobalMaxPool, double, 1)

}  // namespace cann
}  // namespace onnxruntime

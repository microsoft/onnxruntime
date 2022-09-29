// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/nn/pool.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status Pool<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  auto kernel_shape = pool_attrs_.kernel_shape;
  auto pads = pool_attrs_.pads;
  auto strides = pool_attrs_.strides;

  if (pool_attrs_.global_pooling) {
    kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
    pads.assign(kernel_shape.size(), 0);
    strides.assign(kernel_shape.size(), 1);
  }

  auto y_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  TensorShape y_shape(y_dims);
  Tensor* Y = context->Output(0, y_shape);
  if (y_shape.Size() == 0)
    return Status::OK();

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  if (!pool_attrs_.global_pooling) {
    CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "auto_pad", static_cast<int64_t>(pool_attrs_.auto_pad)));
    CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "kernel_shape", pool_attrs_.kernel_shape.size(),
                                             pool_attrs_.kernel_shape.data()));
    CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "strides", pool_attrs_.strides.size(),
                                             pool_attrs_.strides.data()));
    CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "pads", pool_attrs_.pads.size(),
                                             pool_attrs_.pads.data()));
    CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "ceil_mode", pool_attrs_.ceil_mode));
  }

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(X->template Data<T>()), X->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->template MutableData<T>(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute(op_name_.c_str(),
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

#define REGISTER_POOL_TYPED_KERNEL(x, T, startver)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pool<T>);

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
      Pool<T>);

REGISTER_POOL_VERSIONED_TYPED_KERNEL(AveragePool, float, 7, 9)
REGISTER_POOL_VERSIONED_TYPED_KERNEL(AveragePool, MLFloat16, 7, 9)
REGISTER_POOL_VERSIONED_TYPED_KERNEL(AveragePool, float, 10, 10)
REGISTER_POOL_VERSIONED_TYPED_KERNEL(AveragePool, MLFloat16, 10, 10)
REGISTER_POOL_TYPED_KERNEL(AveragePool, float, 11)
REGISTER_POOL_TYPED_KERNEL(AveragePool, MLFloat16, 11)

REGISTER_POOL_TYPED_KERNEL(GlobalAveragePool, float, 1)
REGISTER_POOL_TYPED_KERNEL(GlobalAveragePool, MLFloat16, 1)

REGISTER_POOL_VERSIONED_TYPED_KERNEL(MaxPool, float, 1, 7)
REGISTER_POOL_VERSIONED_TYPED_KERNEL(MaxPool, MLFloat16, 1, 7)

REGISTER_POOL_TYPED_KERNEL(GlobalMaxPool, float, 1)
REGISTER_POOL_TYPED_KERNEL(GlobalMaxPool, double, 1)
REGISTER_POOL_TYPED_KERNEL(GlobalMaxPool, MLFloat16, 1)

}  // namespace cann
}  // namespace onnxruntime

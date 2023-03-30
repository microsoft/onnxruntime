// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/tensor/flatten.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status Flatten<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();

  auto axis = axis_;
  if (axis < 0) {
    axis = HandleNegativeAxis(axis, X_shape.NumDimensions());  // handle negative and enforce axis is valid
  }
  ORT_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.NumDimensions()) >= axis, "The rank of input tensor must be >= axis");

  Tensor* Y = ctx->Output(0, {X_shape.SizeToDimension(axis), X_shape.SizeFromDimension(axis)});

  const void* source = X->DataRaw();
  void* target = Y->MutableDataRaw();
  if (target != source) {
    const aclDataType aclType = getACLType<T>();
    aclFormat format = ACL_FORMAT_ND;

    CannPreparation prepare;

    CANN_RETURN_IF_ERROR(aclopSetAttrInt(prepare.opAttr_, "axis", axis_));

    ORT_TRY {
      CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
      CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

      CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(X->DataRaw()), X->SizeInBytes());
      CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableDataRaw(), Y->SizeInBytes());
    }
    ORT_CATCH(const std::exception& e) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
    }

    CANN_RETURN_IF_ERROR(aclopCompileAndExecute("Flatten",
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
  }

  return Status::OK();
}

#define REGISTER_FLATTEN_TYPED_KERNEL(ver, T)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Flatten,                                                                             \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Flatten<T>);

#define REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, T)                       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Flatten,                                                                             \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      endver,                                                                              \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Flatten<T>);

#define REGISTER_FLATTEN_VERSIONED_HF(startver, endver)                \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, MLFloat16) \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, float)

#define REGISTER_FLATTEN_VERSIONED_BWUZCSILHF(startver, endver)        \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, uint8_t)   \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, uint16_t)  \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, uint32_t)  \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, uint64_t)  \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, int8_t)    \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, int16_t)   \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, int32_t)   \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, int64_t)   \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, MLFloat16) \
  REGISTER_FLATTEN_VERSIONED_TYPED_KERNEL(startver, endver, float)

#define REGISTER_FLATTEN_BWUZCSILHF(ver)        \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, uint8_t)   \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, uint16_t)  \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, uint32_t)  \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, uint64_t)  \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, int8_t)    \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, int16_t)   \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, int32_t)   \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, int64_t)   \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, MLFloat16) \
  REGISTER_FLATTEN_TYPED_KERNEL(ver, float)

REGISTER_FLATTEN_VERSIONED_HF(1, 8)
REGISTER_FLATTEN_VERSIONED_BWUZCSILHF(9, 10)
REGISTER_FLATTEN_VERSIONED_BWUZCSILHF(11, 12)
REGISTER_FLATTEN_BWUZCSILHF(13)

}  // namespace cann
}  // namespace onnxruntime

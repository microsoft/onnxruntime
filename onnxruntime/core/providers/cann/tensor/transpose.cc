// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/tensor/transpose.h"

namespace onnxruntime {
namespace cann {

template <typename T>
Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  int32_t rank = gsl::narrow_cast<int32_t>(X->Shape().NumDimensions());

  TensorShapeVector Y_dims(rank);
  InlinedVector<size_t> default_perm(rank);
  const InlinedVector<size_t>* perm = nullptr;
  const auto& status = ComputeOutputShape(*X, Y_dims, default_perm, perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{Y_dims};
  Tensor* Y = ctx->Output(0, output_shape);

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_,
                                           "perm",
                                           perm->size(),
                                           reinterpret_cast<const int64_t*>(perm->data())));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, X->Shape().NumDimensions(), X->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(X->DataRaw()), X->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableDataRaw(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("TransposeD",
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

#define REGISTER_TRANSPOSE_TYPED_KERNEL(T, ver)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Transpose,                                                                           \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Transpose<T>);

REGISTER_TRANSPOSE_TYPED_KERNEL(uint8_t, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(uint16_t, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(uint32_t, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(uint64_t, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(int8_t, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(int16_t, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(int32_t, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(int64_t, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(MLFloat16, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(float, 1)
REGISTER_TRANSPOSE_TYPED_KERNEL(bool, 1)

}  // namespace cann
}  // namespace onnxruntime

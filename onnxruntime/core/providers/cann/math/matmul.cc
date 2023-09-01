// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/math/matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace cann {

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape()));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "adj_x1", 0));
  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "adj_x2", 0));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, A->Shape().NumDimensions(), A->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, aclType, B->Shape().NumDimensions(), B->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(A->DataRaw()), A->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(B->DataRaw()), B->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->MutableDataRaw(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("BatchMatMul",
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

#define REGISTER_MATMUL_TYPED_KERNEL(T, ver)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      MatMul,                                                                              \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

#define REGISTER_MATMUL_VERSIONED_TYPED_KERNEL(T, startver, endver) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                          \
      MatMul,                                                       \
      kOnnxDomain,                                                  \
      startver,                                                     \
      endver,                                                       \
      T,                                                            \
      kCannExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
      MatMul<T>);

REGISTER_MATMUL_VERSIONED_TYPED_KERNEL(MLFloat16, 1, 8)
REGISTER_MATMUL_VERSIONED_TYPED_KERNEL(float, 1, 8)

REGISTER_MATMUL_VERSIONED_TYPED_KERNEL(MLFloat16, 9, 12)
REGISTER_MATMUL_VERSIONED_TYPED_KERNEL(float, 9, 12)
REGISTER_MATMUL_VERSIONED_TYPED_KERNEL(int32_t, 9, 12)

REGISTER_MATMUL_TYPED_KERNEL(MLFloat16, 13)
REGISTER_MATMUL_TYPED_KERNEL(BFloat16, 13)
REGISTER_MATMUL_TYPED_KERNEL(float, 13)
REGISTER_MATMUL_TYPED_KERNEL(int32_t, 13)

}  // namespace cann
}  // namespace onnxruntime

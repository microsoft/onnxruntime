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

  bool transa = trans_A_;
  bool transb = trans_B_;
  if (A->Shape().NumDimensions() == 1) {
    transa = false;
  }
  if (B->Shape().NumDimensions() == 1) {
    transb = false;
  }

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape(), transa, transb, trans_batch_a_, trans_batch_b_, false));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "transpose_x1", transa));
  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "transpose_x2", transb));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, A->Shape().NumDimensions(), A->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, aclType, B->Shape().NumDimensions(), B->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(A->template Data<T>()), A->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(B->template Data<T>()), B->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, nullptr, 0);
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->template MutableData<T>(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("MatMul",
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

#define REGISTER_MATMUL_TYPED_KERNEL(T, startver)                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      MatMul,                                                                              \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
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
REGISTER_MATMUL_TYPED_KERNEL(float, 13)
REGISTER_MATMUL_TYPED_KERNEL(int32_t, 13)

}  // namespace cann
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/math/gemm.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status Gemm<T>::Prepare(OpKernelContext* ctx, CannPreparation& prepare) const {
  const auto* A = ctx->Input<Tensor>(0);
  const auto* B = ctx->Input<Tensor>(1);
  const auto* C = ctx->Input<Tensor>(2);

  GemmHelper helper(A->Shape(), trans_A_, B->Shape(), trans_B_, C != nullptr ? C->Shape() : TensorShape({}));
  if (!helper.State().IsOK())
    return helper.State();

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  auto* Y = ctx->Output(0, {M, N});

  TensorShape shape{1};

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "transpose_a", trans_A_));
  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "transpose_b", trans_B_));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, A->Shape().NumDimensions(), A->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, aclType, B->Shape().NumDimensions(), B->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, aclType, C->Shape().NumDimensions(), C->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, shape.NumDimensions(), shape.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, ACL_FLOAT, shape.NumDimensions(), shape.GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, Y->Shape().NumDimensions(), Y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(A->template Data<T>()), A->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(B->template Data<T>()), B->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<T*>(C->template Data<T>()), C->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<float*>(&alpha_), sizeof(float));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<float*>(&beta_), sizeof(float));
    CANN_PREPARE_OUTPUTBUFFER(prepare, Y->template MutableData<T>(), Y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

#define REGISTER_GEMM_TYPED_COMPUTE(T)                                         \
  template <>                                                                  \
  Status Gemm<T>::ComputeInternal(OpKernelContext* context) const {            \
    CannPreparation prepare;                                                   \
    ORT_RETURN_IF_ERROR(Prepare(context, prepare));                            \
    CANN_RETURN_IF_ERROR(aclopCompileAndExecute("GEMM",                        \
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

#define REGISTER_GEMM_TYPED_KERNEL(ver, T)                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Gemm,                                                                                \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

#define REGISTER_GEMM_VERSIONED_TYPED_KERNEL(startver, endver, T)                          \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Gemm,                                                                                \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      endver,                                                                              \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

#define REGISTER_GEMM_VERSIONED_TYPED(startver, endver, T) \
  REGISTER_GEMM_VERSIONED_TYPED_KERNEL(startver, endver, T)

#define REGISTER_GEMM_TYPED(ver, T)  \
  REGISTER_GEMM_TYPED_KERNEL(ver, T) \
  REGISTER_GEMM_TYPED_COMPUTE(T)

REGISTER_GEMM_VERSIONED_TYPED(7, 8, MLFloat16)
REGISTER_GEMM_VERSIONED_TYPED(9, 10, MLFloat16)
REGISTER_GEMM_VERSIONED_TYPED(11, 12, MLFloat16)
REGISTER_GEMM_TYPED(13, MLFloat16)

}  // namespace cann
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/math/binary_elementwise_ops.h"
#include <vector>
#include <algorithm>
#include <string>

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status BinaryElementwise::Prepare(OpKernelContext* ctx, CannPreparation& prepare) const {
  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), A->Shape(), B->Shape(), output_shape));
  Tensor* C = ctx->Output(0, output_shape);

  void* A_data = const_cast<void*>(A->DataRaw());
  void* B_data = const_cast<void*>(B->DataRaw());

  if (A->Shape() != C->Shape()) {
    IAllocatorUniquePtr<void> pA = GetScratchBuffer<void>(C->SizeInBytes(), ctx->GetComputeStream());
    ORT_RETURN_IF_ERROR(Broadcast<T>(A, C, pA.get(), Stream(ctx)));
    A_data = pA.get();
  }

  if (B->Shape() != C->Shape()) {
    IAllocatorUniquePtr<void> pB = GetScratchBuffer<void>(C->SizeInBytes(), ctx->GetComputeStream());
    ORT_RETURN_IF_ERROR(Broadcast<T>(B, C, pB.get(), Stream(ctx)));
    B_data = pB.get();
  }

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, C->Shape().NumDimensions(), C->Shape().GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, aclType, C->Shape().NumDimensions(), C->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, C->Shape().NumDimensions(), C->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, A_data, C->SizeInBytes());
    CANN_PREPARE_INPUTBUFFER(prepare, B_data, C->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, C->MutableDataRaw(), C->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  return Status::OK();
}

#define REGISTER_ELEMENTWISE_TYPED_COMPUTE(x, T)                               \
  template <>                                                                  \
  Status x<T>::ComputeInternal(OpKernelContext* ctx) const {                   \
    CannPreparation prepare;                                                   \
    ORT_RETURN_IF_ERROR(Prepare<T>(ctx, prepare));                             \
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
                                                Stream(ctx)));                 \
    return Status::OK();                                                       \
  }

#define REGISTER_ELEMENTWISE_TYPED_KERNEL(x, ver, T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define REGISTER_ELEMENTWISE_VERSIONED_TYPED_KERNEL(x, startver, endver, T)                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      endver,                                                                              \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, T) \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED_KERNEL(name, startver, endver, T)

#define REGISTER_ELEMENTWISE_TYPED(name, ver, T)  \
  REGISTER_ELEMENTWISE_TYPED_KERNEL(name, ver, T) \
  REGISTER_ELEMENTWISE_TYPED_COMPUTE(name, T)

// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// X: bfloat16
// F: float
// D: double
// O: bool

#define REGISTER_ELEMENTWISE_VERSIONED_ILHFD(name, startver, endver)      \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int32_t)   \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, int64_t)   \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, float)     \
  REGISTER_ELEMENTWISE_VERSIONED_TYPED(name, startver, endver, double)

#define REGISTER_ELEMENTWISE_BCSILHFD(name, ver)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, uint8_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int8_t)    \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int16_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int32_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int64_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, MLFloat16) \
  REGISTER_ELEMENTWISE_TYPED(name, ver, float)     \
  REGISTER_ELEMENTWISE_TYPED(name, ver, double)

#define REGISTER_ELEMENTWISE_BWCSILHFD(name, ver)  \
  REGISTER_ELEMENTWISE_TYPED(name, ver, uint8_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, uint16_t)  \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int8_t)    \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int16_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int32_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, int64_t)   \
  REGISTER_ELEMENTWISE_TYPED(name, ver, MLFloat16) \
  REGISTER_ELEMENTWISE_TYPED(name, ver, float)     \
  REGISTER_ELEMENTWISE_TYPED(name, ver, double)

REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Add, 7, 12)
REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Sub, 7, 12)
REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Mul, 7, 12)
REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Div, 7, 12)

REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Add, 13, 13)
REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Sub, 13, 13)
REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Mul, 13, 13)
REGISTER_ELEMENTWISE_VERSIONED_ILHFD(Div, 13, 13)

REGISTER_ELEMENTWISE_BCSILHFD(Add, 14)
REGISTER_ELEMENTWISE_BWCSILHFD(Sub, 14)
REGISTER_ELEMENTWISE_BWCSILHFD(Mul, 14)
REGISTER_ELEMENTWISE_BWCSILHFD(Div, 14)

}  // namespace cann
}  // namespace onnxruntime

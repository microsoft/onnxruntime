// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "unary_elementwise_ops.h"
#include "unary_elementwise_ops_impl.h"

namespace onnxruntime {
namespace cuda {

Status UnaryElementwise::Prepare(OpKernelContext* context, UnaryElementwisePreparation* p) const {
  p->input_tensor = context->Input<Tensor>(0);
  p->output_tensor = context->Output(0, p->input_tensor->Shape());
  return Status::OK();
}

#define UNARY_ELEMENTWISE_REGISTER_VERSIONED_KERNEL(x, startver, endver, T)     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      x,                                                                        \
      kOnnxDomain,                                                              \
      startver,                                                                 \
      endver,                                                                   \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define UNARY_ELEMENTWISE_REGISTER_KERNEL(x, ver, T)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      x,                                                                        \
      kOnnxDomain,                                                              \
      ver,                                                                      \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define UNARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(x, ver, T)                                                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                                          \
      x,                                                                                                                                  \
      kOnnxDomain,                                                                                                                        \
      ver,                                                                                                                                \
      T,                                                                                                                                  \
      kCudaExecutionProvider,                                                                                                             \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()).TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()), \
      x<T>);

#define UNARY_ELEMENTWISE_COMPUTE(x, T)                                                                    \
  template <>                                                                                              \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                           \
    UnaryElementwisePreparation p;                                                                         \
    ORT_RETURN_IF_ERROR(UnaryElementwise::Prepare(context, &p));                                           \
    Impl_##x(                                                                                              \
        Stream(),                                                                                          \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(p.input_tensor->template Data<T>()),   \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(p.output_tensor->template MutableData<T>()), \
        p.output_tensor->Shape().Size());                                                                  \
                                                                                                           \
    return Status::OK();                                                                                   \
  }

#define UNARY_OP_VERSIONED_TYPED(name, startver, endver, T)               \
  UNARY_ELEMENTWISE_REGISTER_VERSIONED_KERNEL(name, startver, endver, T)

#define UNARY_OP_TYPED(name, ver, T)              \
  UNARY_ELEMENTWISE_REGISTER_KERNEL(name, ver, T) \
  UNARY_ELEMENTWISE_COMPUTE(name, T)

#define UNARY_LOGICALOP_TYPED(name, ver, T)                       \
  UNARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, T) \
  UNARY_ELEMENTWISE_COMPUTE(name, T)

// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

#define UNARY_OP_VERSIONED_HFD(name, startver, endver)        \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, float)     \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, double)

#define UNARY_OP_VERSIONED_CSILHFD(name, startver, endver)  \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, int8_t)  \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, int16_t) \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, int32_t) \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, int64_t) \
  UNARY_OP_VERSIONED_HFD(name, startver, endver)

#define UNARY_OP_VERSIONED_BWUZCSILHFD(name, startver, endver) \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, uint8_t)    \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, uint16_t)   \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, uint32_t)   \
  UNARY_OP_VERSIONED_TYPED(name, startver, endver, uint64_t)   \
  UNARY_OP_VERSIONED_CSILHFD(name, startver, endver)

#define UNARY_OP_HFD(name, ver)        \
  UNARY_OP_TYPED(name, ver, MLFloat16) \
  UNARY_OP_TYPED(name, ver, float)     \
  UNARY_OP_TYPED(name, ver, double)

#define UNARY_OP_CSILHFD(name, ver)  \
  UNARY_OP_TYPED(name, ver, int8_t)  \
  UNARY_OP_TYPED(name, ver, int16_t) \
  UNARY_OP_TYPED(name, ver, int32_t) \
  UNARY_OP_TYPED(name, ver, int64_t) \
  UNARY_OP_HFD(name, ver)

#define UNARY_OP_BWUZCSILHFD(name, ver) \
  UNARY_OP_TYPED(name, ver, uint8_t)    \
  UNARY_OP_TYPED(name, ver, uint16_t)   \
  UNARY_OP_TYPED(name, ver, uint32_t)   \
  UNARY_OP_TYPED(name, ver, uint64_t)   \
  UNARY_OP_CSILHFD(name, ver)

UNARY_OP_VERSIONED_BWUZCSILHFD(Abs, 6, 12)
UNARY_OP_VERSIONED_CSILHFD(Neg, 6, 12)
UNARY_OP_VERSIONED_HFD(Floor, 6, 12)
UNARY_OP_VERSIONED_HFD(Ceil, 6, 12)
UNARY_OP_VERSIONED_HFD(Reciprocal, 6, 12)
UNARY_OP_VERSIONED_HFD(Sqrt, 6, 12)
UNARY_OP_VERSIONED_HFD(Log, 6, 12)
UNARY_OP_VERSIONED_HFD(Exp, 6, 12)
UNARY_OP_VERSIONED_HFD(Erf, 9, 12)

UNARY_OP_BWUZCSILHFD(Abs, 13)
UNARY_OP_CSILHFD(Neg, 13)
UNARY_OP_HFD(Floor, 13)
UNARY_OP_HFD(Ceil, 13)
UNARY_OP_HFD(Reciprocal, 13)
UNARY_OP_HFD(Sqrt, 13)
UNARY_OP_HFD(Log, 13)
UNARY_OP_HFD(Exp, 13)
UNARY_OP_HFD(Erf, 13)

UNARY_LOGICALOP_TYPED(Not, 1, bool)
UNARY_OP_HFD(Round, 11)

}  // namespace cuda
}  // namespace onnxruntime

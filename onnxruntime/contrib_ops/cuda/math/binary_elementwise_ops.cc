// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "binary_elementwise_ops.h"
#include "binary_elementwise_ops_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace contrib {
namespace cuda {

#define CONTRIB_BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(x, ver, T)             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      x,                                                                        \
      kMSDomain,                                                                \
      ver,                                                                      \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define CONTRIB_BINARY_ELEMENTWISE_COMPUTE(x, T)                                                                 \
  template <>                                                                                                    \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                                 \
    BinaryElementwisePreparation prepare(this);                                                                  \
    ORT_RETURN_IF_ERROR(Prepare(context, &prepare));                                                             \
    ORT_RETURN_IF_ERROR(prepare.CopyToGpu());                                                                    \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                                \
        prepare.output_rank_or_simple_broadcast,                                                                 \
        prepare.lhs_padded_strides.GpuPtr(),                                                                     \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),     \
        prepare.rhs_padded_strides.GpuPtr(),                                                                     \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),     \
        prepare.fdm_output_strides.GpuPtr(),                                                                     \
        prepare.fdm_H,                                                                                           \
        prepare.fdm_C,                                                                                           \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()), \
        prepare.output_tensor->Shape().Size());                                                                  \
    return Status::OK();                                                                                         \
  }

#define CONTRIB_BINARY_OP_TYPED(name, ver, T)                    \
  CONTRIB_BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, T) \
  CONTRIB_BINARY_ELEMENTWISE_COMPUTE(name, T)

// since different ops has different types, we cannot use BINARY_OPS() directly
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

#define CONTRIB_BINARY_OP_HFD(name, ver)        \
  CONTRIB_BINARY_OP_TYPED(name, ver, MLFloat16) \
  CONTRIB_BINARY_OP_TYPED(name, ver, float)     \
  CONTRIB_BINARY_OP_TYPED(name, ver, double)

CONTRIB_BINARY_OP_HFD(AddGeluFusion, 1)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "complex_mul.h"
#include "complex_mul_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ComplexMul,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ComplexMul<T>);                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ComplexMulConj,                                             \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ComplexMulConj<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status ComplexMul<T>::ComputeInternal(OpKernelContext* context) const {
  for (int index = 0; index < context->InputCount(); ++index) {
    const Tensor* input = context->Input<Tensor>(index);
    TensorShape shape = input->Shape();
    int64_t last_dimension = shape[shape.NumDimensions() - 1];
    ORT_ENFORCE(last_dimension == 2, "The input_", index, " last demension is supposed to be 2, but get ", last_dimension);
  }

  BinaryElementwisePreparation prepare;
  Prepare(context, &prepare);
  ComplexMul_Impl<typename ToCudaType<T>::MappedType>(
      prepare.output_rank_or_simple_broadcast,
      &prepare.lhs_padded_strides,
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),
      &prepare.rhs_padded_strides,
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),
      &prepare.fdm_output_strides,
      prepare.fdm_H,
      prepare.fdm_C,
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),
      prepare.output_tensor->Shape().Size(),
      false);
  return Status::OK();
}

template <typename T>
Status ComplexMulConj<T>::ComputeInternal(OpKernelContext* context) const {
  for (int index = 0; index < context->InputCount(); ++index) {
    const Tensor* input = context->Input<Tensor>(index);
    TensorShape shape = input->Shape();
    int64_t last_dimension = shape[shape.NumDimensions() - 1];
    ORT_ENFORCE(last_dimension == 2, "The input_", index, " last demension is supposed to be 2, but get ", last_dimension);
  }

  BinaryElementwisePreparation prepare;
  Prepare(context, &prepare);
  ComplexMul_Impl<typename ToCudaType<T>::MappedType>(
      prepare.output_rank_or_simple_broadcast,
      &prepare.lhs_padded_strides,
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),
      &prepare.rhs_padded_strides,
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),
      &prepare.fdm_output_strides,
      prepare.fdm_H,
      prepare.fdm_C,
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),
      prepare.output_tensor->Shape().Size(),
      true);
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

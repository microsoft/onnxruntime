// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "where.h"
#include "where_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/math/ternary_elementwise_ops.h"

namespace onnxruntime {
namespace cuda {

// kernel builder functions
#define WHERE_TYPED_KERNEL_WITH_TYPE_NAME(T, TName)                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                          \
      Where,                                                        \
      kOnnxDomain,                                                  \
      9,                                                            \
      15,                                                           \
      TName,                                                        \
      kCudaExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                 \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
      Where<T>);                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      Where,                                                        \
      kOnnxDomain,                                                  \
      16,                                                           \
      TName,                                                        \
      kCudaExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                 \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
      Where<T>);

template <typename T>
Status Where<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto* const condition = context->Input<Tensor>(0);
  const auto* const X = context->Input<Tensor>(1);
  const auto* const Y = context->Input<Tensor>(2);
  ORT_ENFORCE(condition && X && Y, "condition, X, and Y inputs are required!");

  auto const& condition_shape = condition->Shape();
  auto const& X_shape = X->Shape();
  auto const& Y_shape = Y->Shape();

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), condition_shape, X_shape, Y_shape, output_shape));
  auto output_tensor = context->Output(0, output_shape);

  if (output_shape.Size() == 0)
    return Status::OK();

  TernaryElementwisePreparation prepare;
  prepare.a_tensor = condition;
  prepare.b_tensor = X;
  prepare.c_tensor = Y;
  prepare.output_tensor = output_tensor;

  ORT_RETURN_IF_ERROR(prepare.TernaryElementwiseBroadcastPrepareHelper(condition_shape, X_shape, Y_shape, output_shape));

  WhereImpl<CudaT>(
      Stream(),
      prepare.output_rank_or_simple_broadcast,
      prepare.a_index_type,
      prepare.a_padded_strides,
      reinterpret_cast<const bool*>(prepare.a_tensor->template Data<bool>()),
      prepare.b_index_type,
      prepare.b_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.b_tensor->template Data<T>()),
      prepare.c_index_type,
      prepare.c_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.c_tensor->template Data<T>()),
      prepare.fdm_output_strides,
      reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>()),
      output_tensor->Shape().Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE_WITH_NAME(T, TName) \
  WHERE_TYPED_KERNEL_WITH_TYPE_NAME(T, TName)   \
  template Status Where<T>::ComputeInternal(OpKernelContext* context) const;

#define SPECIALIZED_COMPUTE(T) \
  SPECIALIZED_COMPUTE_WITH_NAME(T, T)

SPECIALIZED_COMPUTE(uint8_t)
SPECIALIZED_COMPUTE(int32_t)
SPECIALIZED_COMPUTE(int64_t)
SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double_t)
SPECIALIZED_COMPUTE(MLFloat16)
}  // namespace cuda
}  // namespace onnxruntime

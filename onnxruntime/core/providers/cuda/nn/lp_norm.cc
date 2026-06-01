// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/nn/lp_norm.h"
#include "core/providers/cuda/nn/lp_norm_impl.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_LPNORM_VERSIONED_KERNEL(type, sinceVersion, endVersion)                      \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                    \
      LpNormalization,                                                                        \
      kOnnxDomain,                                                                            \
      sinceVersion,                                                                           \
      endVersion,                                                                             \
      type,                                                                                   \
      kCudaExecutionProvider,                                                                 \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      LpNorm<type>);

#define REGISTER_LPNORM_KERNEL(type, sinceVersion)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                              \
      LpNormalization,                                                                        \
      kOnnxDomain,                                                                            \
      sinceVersion,                                                                           \
      type,                                                                                   \
      kCudaExecutionProvider,                                                                 \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      LpNorm<type>);

REGISTER_LPNORM_VERSIONED_KERNEL(float, 1, 21)
REGISTER_LPNORM_VERSIONED_KERNEL(MLFloat16, 1, 21)

REGISTER_LPNORM_KERNEL(float, 22)
REGISTER_LPNORM_KERNEL(MLFloat16, 22)

template <typename T>
Status LpNorm<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const auto* input = context->Input<Tensor>(0);
  const TensorShape& input_shape = input->Shape();
  Tensor* output = context->Output(0, input_shape);

  const auto canonical_axis = HandleNegativeAxis(axis_, static_cast<int64_t>(input_shape.NumDimensions()));
  const int64_t norm_size = input_shape.GetDims()[onnxruntime::narrow<size_t>(canonical_axis)];
  const int64_t num_elements = input_shape.Size();
  if (num_elements == 0) {
    return Status::OK();
  }
  const int64_t num_norms = num_elements / norm_size;
  const int64_t stride = input_shape.SizeFromDimension(SafeInt<size_t>(canonical_axis) + 1);

  LpNormImpl<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      norm_size,
      num_norms,
      stride,
      static_cast<int>(p_));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

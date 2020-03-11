// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/tensor/gather_grad.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_KERNEL(
    GatherGrad,
    9,
    KernelDefBuilder()
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherGrad);

#define TYPED_GRAD_FUNCTION_CALL(T)                                      \
  if (T_type == DataTypeImpl::GetType<T>()) {                            \
    if (Tind_type == DataTypeImpl::GetType<int32_t>()) {                 \
      return ComputeImpl<T, int32_t>(data_shape, indices, grad, output); \
    }                                                                    \
    if (Tind_type == DataTypeImpl::GetType<int64_t>()) {                 \
      return ComputeImpl<T, int64_t>(data_shape, indices, grad, output); \
    }                                                                    \
  }

Status GatherGrad::Compute(OpKernelContext* context) const {
  const Tensor& shape = *context->Input<Tensor>(0);
  const Tensor& indices = *context->Input<Tensor>(1);
  const Tensor& grad = *context->Input<Tensor>(2);

  const TensorShape data_shape(shape.template Data<int64_t>(), shape.Shape().Size());
  Tensor& output = *context->Output(0, data_shape);
  memset(output.MutableDataRaw(), 0, output.SizeInBytes());

  MLDataType T_type = grad.DataType();
  MLDataType Tind_type = indices.DataType();
  TYPED_GRAD_FUNCTION_CALL(float);
  TYPED_GRAD_FUNCTION_CALL(double);

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for T or Tind not supported yet in GatherGrad.");
}

template <typename T, typename Tind>
Status GatherGrad::ComputeImpl(const TensorShape& data_shape, const Tensor& indices, const Tensor& grad, Tensor& output) const {
  const Tind* indices_data = indices.template Data<Tind>();
  const T* grad_data = grad.template Data<T>();
  T* output_data = output.template MutableData<T>();

  const int64_t axis = HandleNegativeAxis(axis_, data_shape.NumDimensions());
  const int64_t block_size = data_shape.SizeFromDimension(axis + 1);
  const int64_t N = indices.Shape().Size();
  const int64_t input_block_size = data_shape.SizeFromDimension(axis);
  const int64_t output_block_size = N * block_size;
  const int64_t indices_max = data_shape[axis];
  const int64_t grad_size = grad.Shape().Size();

  // Check the indices first in case there's a out of bound index.
  // We can't merge this code in the omp loop below as omp does not allow return in the loop
  for (int64_t i = 0; i < N; i++) {
    Tind idx = indices_data[i];
    if (idx < 0 || idx >= indices_max) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices element out of data bounds, idx=", idx,
                             " data_dim=", indices_max);
    }
  }

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t g = 0; g < grad_size; g++) {
    const int64_t input_block_index = g / output_block_size;
    const int64_t block_offset = g % output_block_size;
    const int64_t indices_index = block_offset / block_size;
    const int64_t offset = block_offset % block_size;
    const Tind idx = indices_data[indices_index];
    const int64_t input_index = input_block_index * input_block_size + idx * block_size + offset;
#ifdef USE_OPENMP
#pragma omp atomic
#endif
    output_data[input_index] += grad_data[g];
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

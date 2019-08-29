// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather
#include "core/providers/cpu/tensor/gather.h"
#include "core/common/common.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Gather,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

Status GatherBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  p.input_tensor = context->Input<Tensor>(0);
  const TensorShape& input_data_shape = p.input_tensor->Shape();
  p.indices_tensor = context->Input<Tensor>(1);
  const TensorShape& indices_shape = p.indices_tensor->Shape();

  p.axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  std::vector<int64_t> shape(indices_shape.GetDims().begin(), indices_shape.GetDims().end());
  shape.insert(shape.begin(), input_data_shape.GetDims().begin(), input_data_shape.GetDims().begin() + p.axis);
  shape.insert(shape.end(), input_data_shape.GetDims().begin() + p.axis + 1, input_data_shape.GetDims().end());

  p.output_tensor = context->Output(0, TensorShape(shape));

  return Status::OK();
}

template <typename Tin>
Status GatherCopyData(const Tensor* indices_tensor, const uint8_t* src_base, uint8_t* dst_base, bool is_string_type,
                      const size_t element_bytes, const int64_t block_size, const int64_t M,
                      const int64_t N, const int64_t data_batch_bytes, const int64_t gathered_batch_bytes,
                      const TensorShape& input_data_shape, const int64_t axis) {
  const Tin* indices_data = indices_tensor->template Data<Tin>();

  // Check the indices first in case there's a out of bound index.
  // We can't merge this code in the omp loop below as omp does not allow return in the loop
  for (int64_t i = 0; i < N; ++i) {
    Tin idx = indices_data[i];
    if (idx < 0 || idx >= input_data_shape[axis]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices element out of data bounds, idx=", idx,
                             " data_dim=", input_data_shape[axis]);
    }
  }

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t index = 0; index < M * N; ++index) {
    int64_t batch = index / N;
    int64_t i = index % N;

    const int64_t src_offset_batch = batch * data_batch_bytes;
    const int64_t dst_offset_batch = batch * gathered_batch_bytes;
    Tin idx = indices_data[i];
    const int64_t src_offset = src_offset_batch + idx * block_size;
    const int64_t dst_offset = dst_offset_batch + i * block_size;

    if (is_string_type) {
      reinterpret_cast<std::string*>(dst_base)[dst_offset / element_bytes] =
          reinterpret_cast<const std::string*>(src_base)[src_offset / element_bytes];
    } else {
      memcpy(dst_base + dst_offset, src_base + src_offset, block_size);
    }
  }

  return Status::OK();
}

Status Gather::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));

  const TensorShape& input_data_shape = p.input_tensor->Shape();

  bool is_string_type = p.input_tensor->DataType() == DataTypeImpl::GetType<std::string>();

  const size_t element_bytes = p.input_tensor->DataType()->Size();
  const int64_t block = input_data_shape.SizeFromDimension(p.axis + 1);
  const int64_t block_size = block * element_bytes;
  const int64_t M = input_data_shape.SizeToDimension(p.axis);
  const int64_t N = p.indices_tensor->Shape().Size();
  const int64_t data_batch_bytes = input_data_shape.SizeFromDimension(p.axis) * element_bytes;
  const int64_t gathered_batch_bytes = N * block * element_bytes;

  const auto* src_base = static_cast<const uint8_t*>(p.input_tensor->DataRaw());
  auto* dst_base = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());

  MLDataType Tind_type = p.indices_tensor->DataType();
  if (Tind_type == DataTypeImpl::GetType<int32_t>()) {
    return GatherCopyData<int32_t>(p.indices_tensor, src_base, dst_base, is_string_type, element_bytes,
                                   block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis);
  }
  if (Tind_type == DataTypeImpl::GetType<int64_t>()) {
    return GatherCopyData<int64_t>(p.indices_tensor, src_base, dst_base, is_string_type, element_bytes,
                                   block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
}

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

#define TYPED_GRAD_FUNCTION_CALL(T)                                           \
  if (T_type == DataTypeImpl::GetType<T>()) {                                 \
    if (Tind_type == DataTypeImpl::GetType<int32_t>()) {                      \
      return ComputeImpl<T, int32_t>(data_shape, indices, grad, output);      \
    }                                                                         \
    if (Tind_type == DataTypeImpl::GetType<int64_t>()) {                      \
      return ComputeImpl<T, int64_t>(data_shape, indices, grad, output);      \
    }                                                                         \
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
    output_data[input_index] += grad_data[g];
  }

  return Status::OK();
}

}  // namespace contrib

}  // namespace onnxruntime

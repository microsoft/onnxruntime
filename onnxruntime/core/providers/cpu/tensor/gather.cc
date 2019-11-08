// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather
#include "core/providers/cpu/tensor/gather.h"
#include "core/common/common.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Gather,
    1,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

ONNX_CPU_OPERATOR_KERNEL(
    Gather,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

Status GatherBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  p.input_tensor = context->Input<Tensor>(0);
  const TensorShape& input_data_shape = p.input_tensor->Shape();
  p.indices_tensor = context->Input<Tensor>(1);
  const TensorShape& indices_shape = p.indices_tensor->Shape();

  const auto input_rank = input_data_shape.NumDimensions();
  p.axis = HandleNegativeAxis(axis_, input_rank);

  std::vector<int64_t> shape;
  shape.reserve(input_rank - 1 + indices_shape.NumDimensions());

  // replace the dimension for p.axis with the shape from the indices
  for (int64_t i = 0; i < p.axis; ++i)
    shape.push_back(input_data_shape[i]);

  for (const auto dim : indices_shape.GetDims())
    shape.push_back(dim);

  for (int64_t i = p.axis + 1; i < static_cast<int64_t>(input_rank); ++i)
    shape.push_back(input_data_shape[i]);

  p.output_tensor = context->Output(0, TensorShape(std::move(shape)));

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
  auto axis_dim_limit = input_data_shape[axis];

  for (int64_t i = 0; i < N; ++i) {
    Tin idx = indices_data[i];
    if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "indices element out of data bounds, idx=", idx,
                             " must be within the inclusive range [", -axis_dim_limit,",", axis_dim_limit - 1, "]");
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
    idx = idx < 0 ? idx + static_cast<Tin>(axis_dim_limit) : idx;
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

  bool is_string_type = p.input_tensor->IsDataTypeString();

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
  if (utils::IsPrimitiveDataType<int32_t>(Tind_type)) {
    return GatherCopyData<int32_t>(p.indices_tensor, src_base, dst_base, is_string_type, element_bytes,
                                   block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis);
  }
  if (utils::IsPrimitiveDataType<int64_t>(Tind_type)) {
    return GatherCopyData<int64_t>(p.indices_tensor, src_base, dst_base, is_string_type, element_bytes,
                                   block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
}

}  // namespace onnxruntime

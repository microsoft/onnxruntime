// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "compress.h"
#include "compress_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Compress,
    kOnnxDomain,
    9, 10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Compress);

// explicit negative axis support
ONNX_OPERATOR_KERNEL_EX(
    Compress,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Compress);

Status Compress::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor);
  size_t rank = input_tensor->Shape().NumDimensions();
  auto& input_dimensions = input_tensor->Shape().GetDims();
  int64_t axis = 0;
  if (has_axis_) {
    axis = HandleNegativeAxis(axis_, rank);
  }

  const Tensor* condition = ctx->Input<Tensor>(1);
  ORT_ENFORCE(condition);
  auto condition_length = condition->Shape().Size();
  auto condition_data = condition->template Data<bool>();

  // if has axis, we need to compress on dimension[axis], otherwise compress on the flattened input data
  int64_t input_size = input_tensor->Shape().Size();
  int64_t compress_input_length = has_axis_ ? input_dimensions[axis] : input_size;
  int64_t valid_condition_length = compress_input_length < condition_length ? compress_input_length : condition_length;

  auto condition_cumulative_sum_buffer = GetScratchBuffer<int32_t>(valid_condition_length);
  auto condition_cumulative_sum = condition_cumulative_sum_buffer.get();
  size_t temp_storage_bytes = 0;
  CUDA_RETURN_IF_ERROR(CompressCalcPrefixSumTempStorageBytes(Stream(),
                                                             reinterpret_cast<const int8_t*>(condition_data),
                                                             condition_cumulative_sum,
                                                             static_cast<int>(valid_condition_length),
                                                             temp_storage_bytes));
  auto temp_buffer = GetScratchBuffer<uint8_t>(temp_storage_bytes);
  auto d_temp_storage = temp_buffer.get();
  CUDA_RETURN_IF_ERROR(CompressInclusivePrefixSum(Stream(),
                                                  d_temp_storage,
                                                  temp_storage_bytes,
                                                  reinterpret_cast<const int8_t*>(condition_data),
                                                  condition_cumulative_sum,
                                                  static_cast<int>(valid_condition_length)));

  // cudaMemcpyAsync from device memory to pageable host memory will return only once the copy has completed.
  int32_t positive_condition_count = 0;
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(&positive_condition_count, condition_cumulative_sum + valid_condition_length - 1, sizeof(int32_t), cudaMemcpyDeviceToHost, Stream()));

  std::vector<int64_t> output_dims(input_dimensions);
  if (has_axis_) {
    output_dims[axis] = positive_condition_count;
  } else {
    output_dims.resize(1);
    output_dims[0] = positive_condition_count;
  }

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  if (positive_condition_count <= 0) {
    return Status::OK();
  }

  auto element_bytes = input_tensor->DataType()->Size();

  int64_t axis_right_stride = 1;
  if (has_axis_) {
    for (auto i = static_cast<size_t>(axis + 1); i < rank; ++i) {
      axis_right_stride *= input_dimensions[i];
    }
  }

  ORT_RETURN_IF_ERROR(CompressImpl(Stream(),
                                   element_bytes,
                                   gsl::narrow_cast<int32_t>(valid_condition_length),
                                   gsl::narrow_cast<int32_t>(axis_right_stride),
                                   has_axis_ ? gsl::narrow_cast<int32_t>(input_dimensions[axis])
                                             : gsl::narrow_cast<int32_t>(input_size),
                                   gsl::narrow_cast<int32_t>(positive_condition_count),
                                   condition_cumulative_sum,
                                   condition_data,
                                   input_tensor->DataRaw(),
                                   output_tensor->MutableDataRaw(),
                                   input_size));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

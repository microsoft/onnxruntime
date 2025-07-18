// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_block_quantized.h"
#include "gather_block_quantized_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    GatherBlockQuantized,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<BFloat16>()})
        .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    GatherBlockQuantized);

GatherBlockQuantized::GatherBlockQuantized(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("gather_axis", &gather_axis_).IsOK());
  ORT_ENFORCE(info.GetAttr<int64_t>("quantize_axis", &quantize_axis_).IsOK());
  ORT_ENFORCE(info.GetAttr<int64_t>("block_size", &block_size_).IsOK());
  ORT_ENFORCE(info.GetAttr<int64_t>("bits", &bits_).IsOK());
  ORT_ENFORCE(bits_ == 4 || bits_ == 8, "bits attribute must be 4 or 8");
  ORT_ENFORCE(block_size_ >= 16 && (block_size_ & (block_size_ - 1)) == 0, "block_size must be a power of 2 and >= 16");
}

Status GatherBlockQuantized::ComputeInternal(OpKernelContext* context) const {
  const Tensor* data = context->Input<Tensor>(0);
  const Tensor* indices = context->Input<Tensor>(1);
  const Tensor* scales = context->Input<Tensor>(2);
  const Tensor* zero_points = context->Input<Tensor>(3);

  const auto& data_shape = data->Shape();
  const auto& indices_shape = indices->Shape();
  const auto data_rank = data_shape.NumDimensions();

  const int64_t gather_axis = HandleNegativeAxis(gather_axis_, data_rank);
  const int64_t quantize_axis = HandleNegativeAxis(quantize_axis_, data_rank);

  std::vector<int64_t> output_dims;
  output_dims.reserve(data_rank + indices_shape.NumDimensions() - 1);

  for (int64_t i = 0; i < gather_axis; ++i) {
    output_dims.push_back(data_shape[i]);
  }
  for (auto dim : indices_shape.GetDims()) {
    output_dims.push_back(dim);
  }
  for (int64_t i = gather_axis + 1; i < data_rank; ++i) {
    output_dims.push_back(data_shape[i]);
  }

  const int64_t components = (bits_ == 4) ? 2 : 1;
  if (components > 1) {
    int64_t quantize_output_dim_idx = (quantize_axis < gather_axis) ? quantize_axis : quantize_axis + indices_shape.NumDimensions() - 1;
    output_dims[quantize_output_dim_idx] *= components;
  }

  Tensor* output = context->Output(0, output_dims);

  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  const auto& scales_type = scales->GetElementType();
  if (scales_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    if (indices->IsDataType<int32_t>()) {
      return GatherBlockQuantizedImpl<MLFloat16, int32_t>(Stream(context), *this, data, indices, scales, zero_points, *output,
                                                          gather_axis, quantize_axis, block_size_, bits_);
    }
    if (indices->IsDataType<int64_t>()) {
      return GatherBlockQuantizedImpl<MLFloat16, int64_t>(Stream(context), *this, data, indices, scales, zero_points, *output,
                                                          gather_axis, quantize_axis, block_size_, bits_);
    }
  } else if (scales_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
    if (indices->IsDataType<int32_t>()) {
      return GatherBlockQuantizedImpl<BFloat16, int32_t>(Stream(context), *this, data, indices, scales, zero_points, *output,
                                                         gather_axis, quantize_axis, block_size_, bits_);
    }
    if (indices->IsDataType<int64_t>()) {
      return GatherBlockQuantizedImpl<BFloat16, int64_t>(Stream(context), *this, data, indices, scales, zero_points, *output,
                                                         gather_axis, quantize_axis, block_size_, bits_);
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unsupported combination of data types for GatherBlockQuantized.");
}

}  // namespace cuda
}  // namespace onnxruntime

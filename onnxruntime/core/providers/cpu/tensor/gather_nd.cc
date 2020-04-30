// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_nd.h"

namespace onnxruntime {

// Register a kernel for kMsDomain (contrib op) GatherND
#ifndef DISABLE_CONTRIB_OPS

namespace contrib {
// TODO: Remove this contrib kernel registration and the schema from the appropriate places
// once Keras Mask RCNN is shipped with all ONNX domain ops

// Currently this kernel is required to support Keras Mask-RCNN
ONNX_OPERATOR_KERNEL_EX(GatherND, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
                            // contrib spec supports `int32_t` and `int64_t` for indices
                            .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(),
                                                     DataTypeImpl::GetTensorType<int64_t>()}),
                        GatherND);

}  // namespace contrib

#endif

ONNX_CPU_OPERATOR_KERNEL(
    GatherND,
    11,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        // official ONNX spec only supports `int64_t` for indices
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    GatherND);

ONNX_CPU_OPERATOR_KERNEL(
    GatherND,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    GatherND);

template <typename Tind>
Status GatherNDBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indices_tensor = context->Input<Tensor>(1);
  ORT_ENFORCE(input_tensor != nullptr && indices_tensor != nullptr, "GatherNDBase PrepareForCompute: Input count mismatch");

  const auto& input_shape = input_tensor->Shape();
  const auto& indices_shape = indices_tensor->Shape();
  if (indices_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices tensor must has rank larger than 0");
  }

  const auto num_slice_dims = indices_shape[indices_shape.NumDimensions() - 1];
  const auto num_slices = indices_shape.SizeToDimension(indices_shape.NumDimensions() - 1);
  const auto slice_size = input_shape.SizeFromDimension(batch_dims_ + num_slice_dims);
  const auto num_batches = input_shape.SizeToDimension(batch_dims_);
  const auto input_batch_stride = input_shape.SizeFromDimension(batch_dims_);
  const auto num_slices_per_batch = num_slices / num_batches;

  int64_t last_indices_dimension = batch_dims_ + num_slice_dims;
  if (last_indices_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  std::vector<int64_t> shape(indices_shape.GetDims().begin(), indices_shape.GetDims().end() - 1);
  shape.insert(shape.end(), input_shape.GetDims().begin() + last_indices_dimension, input_shape.GetDims().end());
  auto* output_tensor = context->Output(0, TensorShape(std::move(shape)));

  std::vector<int64_t> sizes_from_slice_dims(num_slice_dims);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < num_slice_dims; ++i) {
    sizes_from_slice_dims[i] = input_shape.SizeFromDimension(batch_dims_ + i + 1);
  }

  int64_t err_index = 0;
  p.element_bytes = input_tensor->DataType()->Size();
  p.element_count_per_slice = slice_size;
  p.bytes_per_slice = p.element_bytes * p.element_count_per_slice;
  const auto* indices_data = indices_tensor->Data<Tind>();
  p.slice_offsets.assign(num_slices, 0LL);

  if (input_tensor->IsDataTypeString()) {
    p.input_str_base = static_cast<const std::string*>(input_tensor->DataRaw());
    p.output_str_base = static_cast<std::string*>(output_tensor->MutableDataRaw());
  } else {
    p.input_base = static_cast<const uint8_t*>(input_tensor->DataRaw());
    p.output_base = static_cast<uint8_t*>(output_tensor->MutableDataRaw());
  }

  // Compute the element_offset
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int64_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    const size_t batch_idx = slice_idx / num_slices_per_batch;
    const size_t input_base_offset = batch_idx * input_batch_stride;

    const auto* const slice_indices = indices_data + slice_idx * num_slice_dims;
    size_t relative_slice_offset = 0;
    for (int64_t dim_idx = 0; dim_idx < num_slice_dims; ++dim_idx) {
      int64_t index = static_cast<int64_t>(slice_indices[dim_idx]);
      const auto upper_limit = input_shape[batch_dims_ + dim_idx];
      const auto lower_limit = -upper_limit;
      if (index < lower_limit || index >= upper_limit) {
        err_index = index;
        break;
      }
      if (index < 0) index += upper_limit;

      relative_slice_offset += index * sizes_from_slice_dims[dim_idx];
    }

    p.slice_offsets[slice_idx] = input_base_offset + relative_slice_offset;
  }

  return err_index == 0 ? Status::OK()
                        : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid index found, index = ", err_index);
}

template Status GatherNDBase::PrepareForCompute<int32_t>(OpKernelContext*, Prepare&) const;
template Status GatherNDBase::PrepareForCompute<int64_t>(OpKernelContext*, Prepare&) const;

Status GatherND::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(context->Input<Tensor>(1)->IsDataType<int32_t>()
                          ? PrepareForCompute<int32_t>(context, p)
                          : PrepareForCompute<int64_t>(context, p));

  return nullptr == p.input_str_base ? GatherNumber(p) : GatherString(p);
}

Status GatherND::GatherNumber(const Prepare& p) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int64_t slice_idx = 0; slice_idx < static_cast<int64_t>(p.slice_offsets.size()); ++slice_idx) {
    memcpy(p.output_base + slice_idx * p.bytes_per_slice, p.input_base + p.slice_offsets[slice_idx] * p.element_bytes,
           p.bytes_per_slice);
  }

  return Status::OK();
}

Status GatherND::GatherString(const Prepare& p) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int64_t slice_idx = 0; slice_idx < static_cast<int64_t>(p.slice_offsets.size()); ++slice_idx) {
    const int64_t slice_base_offset = slice_idx * p.element_count_per_slice;
    for (int64_t j = 0; j < static_cast<int64_t>(p.element_count_per_slice); ++j) {
      p.output_str_base[slice_base_offset + j] = p.input_str_base[p.slice_offsets[slice_idx] + j];
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime

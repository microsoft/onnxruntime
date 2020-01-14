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

ONNX_CPU_OPERATOR_KERNEL(GatherND, 11,
                         KernelDefBuilder()
                             .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
                             // official ONNX spec only supports `int64_t` for indices
                             .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
                         GatherND);

template <typename Tind>
Status GatherNDBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indices_tensor = context->Input<Tensor>(1);
  ORT_ENFORCE(input_tensor != nullptr && indices_tensor != nullptr, "GatherND op: Input count mismatch");

  const auto& input_shape = input_tensor->Shape();
  const auto& indices_shape = indices_tensor->Shape();
  if (indices_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices tensor must has rank larger than 0");
  }

  int64_t last_indices_dimension = indices_shape[indices_shape.NumDimensions() - 1];
  if (last_indices_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  std::vector<int64_t> shape(indices_shape.GetDims().begin(), indices_shape.GetDims().end() - 1);
  shape.insert(shape.end(), input_shape.GetDims().begin() + last_indices_dimension, input_shape.GetDims().end());
  auto* output_tensor = context->Output(0, TensorShape(std::move(shape)));
  std::vector<int64_t> element_counts(last_indices_dimension,
                                      0LL);  // Number of elements for each input dimension

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < last_indices_dimension; ++i) {
    element_counts[i] = input_shape.SizeFromDimension(i + 1);
  }

  int64_t err_index = 0;
  p.element_bytes = input_tensor->DataType()->Size();
  p.element_to_copy = input_shape.SizeFromDimension(last_indices_dimension);
  p.bytes_to_copy = p.element_bytes * p.element_to_copy;
  const auto* indices_data = indices_tensor->Data<Tind>();
  const int64_t offset_count = indices_shape.Size() / last_indices_dimension;  // Times to copy
  p.element_offsets.assign(offset_count, 0LL);

  if (input_tensor->IsDataTypeString()) {
    p.input_str_base = static_cast<const std::string*>(input_tensor->DataRaw());
    p.output_str_base = static_cast<std::string*>(output_tensor->MutableDataRaw());
  } else {
    p.input_base = static_cast<const uint8_t*>(input_tensor->DataRaw());
    p.output_base = static_cast<uint8_t*>(output_tensor->MutableDataRaw());
  }

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < offset_count; ++i) {
    for (int64_t j = 0; j < last_indices_dimension; ++j) {
      auto index = *(indices_data + i * last_indices_dimension + j);
      auto upper_limit = input_shape[j];
      auto lower_limit = -upper_limit;
      if (index < lower_limit || index >= upper_limit) {
        err_index = index;
      }
      if (index < 0) {
        index += static_cast<Tind>(upper_limit);
      }
      p.element_offsets[i] += index * element_counts[j];
    }
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
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(p.element_offsets.size()); ++i) {
    memcpy(p.output_base + i * p.bytes_to_copy, p.input_base + p.element_offsets[i] * p.element_bytes,
           p.bytes_to_copy);
  }

  return Status::OK();
}

Status GatherND::GatherString(const Prepare& p) const {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(p.element_offsets.size()); ++i) {
    for (int64_t j = 0; j < static_cast<int64_t>(p.element_to_copy); ++j) {
      p.output_str_base[i * p.element_to_copy + j] = p.input_str_base[p.element_offsets[i] + j];
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime

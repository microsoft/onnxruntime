// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scatter_nd.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    ScatterND,
    11,
    KernelDefBuilder()
        .TypeConstraint("T",    DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
    ScatterND);

template<typename Tind>
Status ScatterNDBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {

  auto input_tensor  = context->Input<Tensor>(0);
  auto indice_tensor = context->Input<Tensor>(1);
  auto update_tensor = context->Input<Tensor>(2);
  ORT_ENFORCE(input_tensor  != nullptr);
  ORT_ENFORCE(indice_tensor != nullptr);
  ORT_ENFORCE(update_tensor != nullptr);

  auto input_shape   = input_tensor->Shape();
  auto indice_shape  = indice_tensor->Shape();
  auto update_shape  = update_tensor->Shape();
  if (indice_shape.NumDimensions() == 0 || input_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "input tensor and indices tensor must has rank larger than 0. ",
      "input shape: ", input_shape, ", indices shape: ", indice_shape);
  }

  auto indice_rank = indice_shape.NumDimensions();
  auto last_indice_dimension = indice_shape[indice_rank - 1];
  if (last_indice_dimension > static_cast<int64_t>(input_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "last dimension of indices must not be larger than rank of input tensor");
  }

  bool is_update_shape_invalid = [&](){
    auto update_rank = update_shape.NumDimensions();
    auto input_rank = input_shape.NumDimensions();
    if (update_rank < indice_rank - 1) {
      return true;
    }
    if ((update_rank >= indice_rank - 1) &&
        (indice_rank - 1 >= 0) &&
        (indice_shape.Slice(0, indice_rank - 1) != update_shape.Slice(0, indice_rank - 1))) {
      return true;
    }
    if ((static_cast<int64_t>(input_rank) > last_indice_dimension) &&
        (update_rank >= indice_rank - 1) &&
        (input_shape.Slice(last_indice_dimension) != update_shape.Slice(indice_rank - 1))) {
      return true;
    }
    return false;
  }();
  if (is_update_shape_invalid) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "updates tensor should have shape equal to indices.shape[:-1] + data.shape[indices.shape[-1]:]. ",
      "updates shape: ", update_shape, ", indices shape: ", indice_shape, ", data shape: ", input_shape);
  }

  auto output_tensor = context->Output(0, TensorShape(input_shape));

  const auto* src_base = input_tensor->DataRaw();
  auto* dst_base = output_tensor->MutableDataRaw();
  bool is_string_type = utils::IsDataTypeString(input_tensor->DataType());

  // Re-use input for output. If input/output Tensor* are the same, do not copy.
  if (src_base != dst_base) {
    if (is_string_type) {
      const auto* str_begin = input_tensor->template Data<std::string>();
      const std::string* str_end = str_begin + input_shape.Size();
      auto* dst = output_tensor->template MutableData<std::string>();
      std::copy(str_begin, str_end, dst);
    } else {
      memcpy(dst_base, src_base, input_tensor->SizeInBytes());
    }
  }

  std::vector<int64_t> element_counts(last_indice_dimension, 0LL); // Number of elements for each input dimension

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < last_indice_dimension; ++i) {
    element_counts[i] = input_shape.SizeFromDimension(i + 1);
  }

  int64_t err_indice = 0;
  p.element_bytes    = input_tensor->DataType()->Size();
  p.element_to_copy  = input_shape.SizeFromDimension(last_indice_dimension);
  p.bytes_to_copy    = p.element_bytes * p.element_to_copy;
  auto indice_offset = static_cast<const Tind*>(indice_tensor->DataRaw());
  auto offset_count  = indice_shape.Size() / last_indice_dimension; // Times to copy
  p.element_offsets.assign(offset_count, 0LL);

  if (utils::IsDataTypeString(input_tensor->DataType())) {
    p.input_str_base  = static_cast<const std::string*>(update_tensor->DataRaw());
    p.output_str_base = static_cast<std::string*>(output_tensor->MutableDataRaw());
  } else {
    p.input_base      = static_cast<const uint8_t*>(update_tensor->DataRaw());
    p.output_base     = static_cast<uint8_t*>(output_tensor->MutableDataRaw());
  }

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < offset_count; ++i) {
    for (int64_t j = 0; j < last_indice_dimension; ++j) {
      auto indice = *(indice_offset + i * last_indice_dimension + j);
      if (indice < 0 || indice >= input_shape[j]) {
        err_indice = indice;
      }
      p.element_offsets[i] += indice * element_counts[j];
    }
  }
  return err_indice == 0 ? Status::OK() :
    ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid indice found, indice = ", err_indice);
}

template Status ScatterNDBase::PrepareForCompute<int64_t>(OpKernelContext*, Prepare&) const;

Status ScatterND::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute<int64_t>(context, p));
  return nullptr == p.input_str_base ? ScatterNumber(p) : ScatterString(p);
}

Status ScatterND::ScatterNumber(const Prepare& p) const {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(p.element_offsets.size()); ++i) {
    memcpy(p.output_base + p.element_offsets[i] * p.element_bytes,
           p.input_base + i * p.bytes_to_copy,
           p.bytes_to_copy);
  }
  return Status::OK();
}

Status ScatterND::ScatterString(const Prepare& p) const {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(p.element_offsets.size()); ++i) {
    for (int64_t j = 0; j < static_cast<int64_t>(p.element_to_copy); ++j) {
      p.output_str_base[p.element_offsets[i] + j] = p.input_str_base[i * p.element_to_copy + j];
    }
  }
  return Status::OK();
}

}
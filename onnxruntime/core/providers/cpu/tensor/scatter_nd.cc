// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/scatter_nd.h"

#include "core/framework/element_type_lists.h"
#include "core/platform/threadpool.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterND, Input, 0,
    element_type_lists::All);
}

using ScatterNDDataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterND, Input, 0);
using EnabledScatterNDDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterND, Input, 0);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterND,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<ScatterNDDataTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

ONNX_CPU_OPERATOR_KERNEL(
    ScatterND,
    13,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<ScatterNDDataTypes>(),
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

Status ScatterNDBase::ValidateShapes(const TensorShape& input_shape,
                                     const TensorShape& indice_shape,
                                     const TensorShape& update_shape) {
  auto input_rank = input_shape.NumDimensions();
  auto indice_rank = indice_shape.NumDimensions();
  auto update_rank = update_shape.NumDimensions();

  if (input_rank == 0 || indice_rank == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input tensor and indices tensor must has rank larger than 0. ",
                           "input shape: ", input_shape, ", indices shape: ", indice_shape);
  }

  auto last_indice_dimension = indice_shape[indice_rank - 1];
  if (last_indice_dimension > static_cast<int64_t>(input_rank)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  bool is_update_shape_invalid = [&]() {
    // Validate rank of update tensor
    // Per spec, the rank of the update tensor should be:
    // (Rank of input tensor) + (Rank of indices tensor) -1 - last_indice_dimension
    if (update_rank != (input_rank + indice_rank - 1 - static_cast<int64_t>(last_indice_dimension))) {
      return true;
    }

    // Validate shape of the update tensor
    // Part 1: The shape of the update tensor upto the indices rank - 1 (exclusive)
    // should match the shape of the indices tensor upto indices rank - 1 (exclusive)
    if (indice_shape.Slice(0, indice_rank - 1) != update_shape.Slice(0, indice_rank - 1)) {
      return true;
    }

    // Part 2: The shape of the update tensor after indices rank - 1 (inclusive)
    // should match the shape of the input tensor after `last_indice_dimension`
    if (input_shape.Slice(last_indice_dimension) != update_shape.Slice(indice_rank - 1)) {
      return true;
    }

    return false;
  }();

  if (is_update_shape_invalid) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "updates tensor should have shape equal to indices.shape[:-1] + data.shape[indices.shape[-1]:]. ",
                           "updates shape: ", update_shape, ", indices shape: ", indice_shape, ", data shape: ", input_shape);
  }

  return Status::OK();
}

Status ScatterNDBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indice_tensor = context->Input<Tensor>(1);
  const auto* update_tensor = context->Input<Tensor>(2);

  const auto& input_shape = input_tensor->Shape();
  const auto& indice_shape = indice_tensor->Shape();
  const auto& update_shape = update_tensor->Shape();

  ORT_RETURN_IF_ERROR(ValidateShapes(input_shape, indice_shape, update_shape));

  auto output_tensor = context->Output(0, input_shape);

  const auto* src_base = input_tensor->DataRaw();
  auto* dst_base = output_tensor->MutableDataRaw();
  const bool is_string_type = input_tensor->IsDataTypeString();

  auto last_indice_dimension = indice_shape[indice_shape.NumDimensions() - 1];

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

  std::vector<int64_t> element_counts(last_indice_dimension, 0LL);  // Number of elements for each input dimension

  TensorPitches input_strides(input_shape);
  for (int64_t i = 0; i < last_indice_dimension; ++i) {
    element_counts[i] = input_strides[i];
  }

  int64_t err_indice = 0;
  p.element_bytes = input_tensor->DataType()->Size();
  p.element_to_copy = input_shape.SizeFromDimension(last_indice_dimension);
  p.bytes_to_copy = p.element_bytes * p.element_to_copy;
  const int64_t* indice_offset = indice_tensor->template Data<int64_t>();
  auto offset_count = indice_shape.Size() / last_indice_dimension;  // Times to copy
  p.element_offsets.assign(offset_count, 0LL);

  if (input_tensor->IsDataTypeString()) {
    p.input_str_base = static_cast<const std::string*>(update_tensor->DataRaw());
    p.output_str_base = static_cast<std::string*>(output_tensor->MutableDataRaw());
  } else {
    p.input_base = static_cast<const uint8_t*>(update_tensor->DataRaw());
    p.output_base = static_cast<uint8_t*>(output_tensor->MutableDataRaw());
  }

  for (int64_t i = 0; i < offset_count; ++i) {
    for (int64_t j = 0; j < last_indice_dimension; ++j) {
      auto indice = *(indice_offset + i * last_indice_dimension + j);
      if (indice < 0 || indice >= input_shape[j]) {
        err_indice = indice;
      }
      p.element_offsets[i] += indice * element_counts[j];
    }
  }
  return err_indice == 0 ? Status::OK() : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid indice found, indice = ", err_indice);
}

Status ScatterND::Compute(OpKernelContext* context) const {
  Prepare p;
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));
  return nullptr == p.input_str_base ? ScatterNumber(p, tp) : ScatterString(p, tp);
}

Status ScatterND::ScatterNumber(const Prepare& p, concurrency::ThreadPool* tp) const {
  constexpr bool has_non_string_enabled_type =
      !boost::mp11::mp_empty<
          boost::mp11::mp_remove<
              EnabledScatterNDDataTypes,
              std::string>>::value;
  if (!has_non_string_enabled_type) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Non-string data types are not supported in this build.");
  }

  auto lambda = [&](int64_t i) {
    memcpy(p.output_base + p.element_offsets[i] * p.element_bytes,
           p.input_base + i * p.bytes_to_copy,
           p.bytes_to_copy);
  };
  concurrency::ThreadPool::TryParallelFor(tp, p.element_offsets.size(), static_cast<double>(p.bytes_to_copy),
                                          [&lambda](ptrdiff_t first, ptrdiff_t last) {
                                            for (int i = static_cast<int>(first), end = static_cast<int>(last); i < end; ++i) {
                                              lambda(i);
                                            }
                                          });
  return Status::OK();
}

Status ScatterND::ScatterString(const Prepare& p, concurrency::ThreadPool* tp) const {
  if (!utils::HasType<EnabledScatterNDDataTypes, std::string>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "String data type is not supported in this build.");
  }

  auto lambda = [&](int64_t i) {
    for (int64_t j = 0; j < static_cast<int64_t>(p.element_to_copy); ++j) {
      p.output_str_base[p.element_offsets[i] + j] = p.input_str_base[i * p.element_to_copy + j];
    }
  };
  concurrency::ThreadPool::TryParallelFor(tp, p.element_offsets.size(), static_cast<double>(p.element_to_copy),
                                          [&lambda](ptrdiff_t first, ptrdiff_t last) {
                                            for (int i = static_cast<int>(first), end = static_cast<int>(last); i < end; ++i) {
                                              lambda(i);
                                            }
                                          });
  return Status::OK();
}

}  // namespace onnxruntime

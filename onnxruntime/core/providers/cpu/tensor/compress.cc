// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/compress.h"
#include "core/providers/common.h"
using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Compress,
    9,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes())
                      .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Compress);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_KERNEL(
    Compress,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes())
                      .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Compress);

Status Compress::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  size_t rank = input_tensor->Shape().NumDimensions();
  auto& input_dimensions = input_tensor->Shape().GetDims();
  int64_t axis = axis_;
  if (has_axis_) {
    axis = HandleNegativeAxis(axis, rank);  // handle negative and enforce axis is valid
    ORT_ENFORCE(axis < static_cast<int64_t>(rank), "axis greater than input data dimension!");
  }

  const auto* condition = ctx->Input<Tensor>(1);
  auto condition_length = condition->Shape().Size();
  auto condition_data = condition->template Data<bool>();

  int64_t positive_condition_count = 0;
  // if has axis, we need to compress on dimension[axis], otherwise compress on the flattened input data
  int64_t compress_input_length = has_axis_ ? input_dimensions[axis] : input_tensor->Shape().Size();
  int64_t valid_condition_length = compress_input_length < condition_length ? compress_input_length : condition_length;

  // Figure out output shape
  for (int i = 0; i < valid_condition_length; ++i) {
    if (condition_data[i]) {
      ++positive_condition_count;
    }
  }

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

  const auto* input_data = static_cast<const uint8_t*>(input_tensor->DataRaw());
  auto* output_data = static_cast<uint8_t*>(output_tensor->MutableDataRaw());
  auto element_bytes = input_tensor->DataType()->Size();
  bool is_string_type = input_tensor->IsDataTypeString();
  int64_t output_index = 0;

  if (has_axis_) {
    int64_t axes_left_stride = 1;
    int64_t axes_right_stride = 1;
    for (int i = 0; i < axis; ++i) {
      axes_left_stride *= input_dimensions[i];
    }

    for (auto i = static_cast<size_t>(axis + 1); i < rank; ++i) {
      axes_right_stride *= input_dimensions[i];
    }
    int64_t axes_included_right_stride = axes_right_stride * input_dimensions[axis];
    int64_t axes_included_right_stride_bytes = axes_included_right_stride * element_bytes;
    ORT_ENFORCE(axes_right_stride >= 0 &&
                static_cast<uint64_t>(axes_right_stride) < std::numeric_limits<size_t>::max());
    size_t axes_right_stride_bytes = 0;
    if (!IAllocator::CalcMemSizeForArray(static_cast<size_t>(axes_right_stride), element_bytes,
                                         &axes_right_stride_bytes))
      return Status(ONNXRUNTIME, FAIL, "size overflow");
    for (int i = 0; i < axes_left_stride; ++i) {
      for (int j = 0; j < valid_condition_length; ++j) {
        if (!condition_data[j]) {
          continue;
        }
        if (is_string_type) {
          for (int idxItem = 0; idxItem < axes_right_stride; ++idxItem) {
            reinterpret_cast<std::string*>(output_data)[output_index + idxItem] =
              reinterpret_cast<const std::string*>(input_data)[i * axes_included_right_stride + j * axes_right_stride + idxItem];
          }
          output_index += axes_right_stride;
        } else {
          memcpy(output_data + output_index, input_data + i * axes_included_right_stride_bytes + j * axes_right_stride_bytes, axes_right_stride_bytes);
          output_index += axes_right_stride_bytes;
        }
      }
    }
  } else {
    for (int i = 0; i < valid_condition_length; ++i) {
      if (!condition_data[i]) {
        continue;
      }
      if (is_string_type) {
        reinterpret_cast<std::string*>(output_data)[output_index] = reinterpret_cast<const std::string*>(input_data)[i];
      } else {
        memcpy(output_data + output_index * element_bytes, input_data + i * element_bytes, element_bytes);
      }
      ++output_index;
    }
  }

return Status::OK();
}

}  // namespace onnxruntime

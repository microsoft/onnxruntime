// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdlib.h>
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
using namespace ::onnxruntime::common;
using namespace std;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Slice,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Slice);

namespace {
// std::clamp doesn't exist until C++17 so create a local version
template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
}  // namespace

Status SliceBase::PrepareForCompute(const size_t dimension_count, const std::vector<int64_t>& input_dimensions,
                                    std::vector<int64_t>& starts, std::vector<int64_t>& output_dims,
                                    int64_t& min_axis, int64_t& max_axis) const {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(axes_);
  if (!has_axes_) {
    //axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(starts.size());
    for (size_t i = 0; i < starts.size(); i++)
      axes[i] = i;
  }

  if (axes.size() > starts_.size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has more entries than the 'starts' attribute holds");
  if (axes.size() > ends_.size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has more entries than the 'ends' attribute holds");

  min_axis = max_axis = axes[0];
  // Iterate through the provided axes and override the start/end ranges
  for (size_t axesIndex = 0; axesIndex < axes.size(); axesIndex++) {
    auto axis = static_cast<size_t>(axes[axesIndex]);
    if (axis >= dimension_count)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    auto start = starts_[axesIndex];
    if (start < 0)
      start += input_dimensions[axis];
    starts[axis] = clamp(start, int64_t{0}, input_dimensions[axis]);

    auto end = ends_[axesIndex];
    if (end < 0)
      end += input_dimensions[axis];
    output_dims[axis] = clamp(end, int64_t{0}, input_dimensions[axis]) - starts[axis];
    if (output_dims[axis] < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'starts' and 'ends' values resulted in a negative dimension");

    if (axis < static_cast<size_t>(min_axis)) {
      min_axis = axis;
    }

    if (axis > static_cast<size_t>(max_axis)) {
      max_axis = axis;
    }
  }

  return Status::OK();
}

Status Slice::Compute(OpKernelContext* ctx) const {
  const Tensor* input_tensor_ptr = ctx->Input<Tensor>(0);
  ONNXRUNTIME_ENFORCE(input_tensor_ptr != nullptr);
  auto& input_dimensions = input_tensor_ptr->Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  const size_t dimension_count = input_dimensions.size();
  std::vector<int64_t> starts(dimension_count, 0);
  std::vector<int64_t> output_dims(input_dimensions);

  int64_t min_axis, max_axis;
  ONNXRUNTIME_RETURN_IF_ERROR(PrepareForCompute(dimension_count, input_dimensions, starts, output_dims, min_axis, max_axis));

  TensorShape output_shape(output_dims);
  auto output_tensor_ptr = ctx->Output(0, output_shape);
  int64_t output_size = output_shape.Size();
  if (0 == output_size) {
    return Status::OK();
  }

  auto output = output_tensor_ptr->MutableDataRaw();
  auto input = input_tensor_ptr->DataRaw();
  bool is_string_type = input_tensor_ptr->DataType() == DataTypeImpl::GetType<std::string>();

  int64_t input_size = input_tensor_ptr->Shape().Size();
  if (output_size == input_size) {  // means all input data are sliced
    if (is_string_type) {
      for (int i = 0; i < output_size; ++i) {
        reinterpret_cast<std::string*>(output)[i] = reinterpret_cast<const std::string*>(input)[i];
      }
    } else {
      memcpy(output, input, output_size * input_tensor_ptr->DataType()->Size());
    }
    return Status::OK();
  }

  TensorPitches input_pitches(input_dimensions);
  TensorPitches output_pitches(output_dims);

  // given input data with dimension [M*N* S1*S2*...*Sn *O*P], slice axes on S1*S2...Sn
  int64_t left_steps = 1;
  for (int i = 0; i < min_axis; ++i) {
    left_steps *= input_dimensions[i];
  }
  int64_t right_setps = input_pitches[max_axis];
  size_t element_bytes = input_tensor_ptr->DataType()->Size();
  int64_t copy_block_size = output_dims[max_axis] * right_setps;
  int64_t copy_block_bytes = copy_block_size * element_bytes;

  int64_t left_step_block_size = input_pitches[min_axis] * input_dimensions[min_axis];
  int64_t slice_steps = 1;
  for (int64_t i = min_axis; i < max_axis; ++i) {
    slice_steps *= output_dims[i];
  }

#pragma omp parallel for
  for (int left_index = 0; left_index < left_steps; ++left_index) {
    int64_t left_offset = left_index * left_step_block_size;
    for (int slice_step = 0; slice_step < slice_steps; ++slice_step) {
      int64_t initial = slice_step * copy_block_size;
      int64_t offset = 0;
      lldiv_t div_result;
      for (int64_t axis = 0; axis <= max_axis; ++axis) {
        div_result = div(initial, output_pitches[axis]);
        initial = div_result.rem;
        offset += (starts[axis] + div_result.quot) * input_pitches[axis];
      }
      int64_t input_pos = left_offset + offset;
      int64_t output_pos = (left_index * slice_steps + slice_step) * copy_block_size;

      if (is_string_type) {
        for (int i = 0; i < copy_block_size; ++i) {
          reinterpret_cast<std::string*>(output)[output_pos + i] = reinterpret_cast<const std::string*>(input)[input_pos + i];
        }
      } else {
        memcpy(reinterpret_cast<uint8_t*>(output) + output_pos * element_bytes, reinterpret_cast<const uint8_t*>(input) + input_pos * element_bytes, copy_block_bytes);
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime

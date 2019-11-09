// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"
#include <unordered_map>
#include <limits>

using namespace ::onnxruntime::common;
using namespace std;

namespace onnxruntime {
#define ADD_TYPED_SLICE_V9_OP(data_type)                                                \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                             \
      Slice,                                                                            \
      1, 9,                                                                             \
      data_type,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Slice<data_type, false>);

ADD_TYPED_SLICE_V9_OP(uint8_t);
ADD_TYPED_SLICE_V9_OP(uint16_t);
ADD_TYPED_SLICE_V9_OP(uint32_t);
ADD_TYPED_SLICE_V9_OP(uint64_t);
ADD_TYPED_SLICE_V9_OP(int8_t);
ADD_TYPED_SLICE_V9_OP(int16_t);
ADD_TYPED_SLICE_V9_OP(int32_t);
ADD_TYPED_SLICE_V9_OP(int64_t);
ADD_TYPED_SLICE_V9_OP(float);
ADD_TYPED_SLICE_V9_OP(double);
ADD_TYPED_SLICE_V9_OP(MLFloat16);
ADD_TYPED_SLICE_V9_OP(bool);
ADD_TYPED_SLICE_V9_OP(string);

#define ADD_TYPED_SLICE_V10_OP(data_type)                                                                                                                                                        \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                                                                                                                                      \
      Slice,                                                                                                                                                                                     \
      10,                                                                                                                                                                                        \
      10,                                                                                                                                                                                        \
      data_type,                                                                                                                                                                                 \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()).TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}), \
      Slice<data_type, true>);

ADD_TYPED_SLICE_V10_OP(uint8_t);
ADD_TYPED_SLICE_V10_OP(uint16_t);
ADD_TYPED_SLICE_V10_OP(uint32_t);
ADD_TYPED_SLICE_V10_OP(uint64_t);
ADD_TYPED_SLICE_V10_OP(int8_t);
ADD_TYPED_SLICE_V10_OP(int16_t);
ADD_TYPED_SLICE_V10_OP(int32_t);
ADD_TYPED_SLICE_V10_OP(int64_t);
ADD_TYPED_SLICE_V10_OP(float);
ADD_TYPED_SLICE_V10_OP(double);
ADD_TYPED_SLICE_V10_OP(MLFloat16);
ADD_TYPED_SLICE_V10_OP(bool);
ADD_TYPED_SLICE_V10_OP(string);

#define ADD_TYPED_SLICE_V11_OP(data_type)                                                                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                                                    \
      Slice,                                                                                                         \
      11,                                                                                                            \
      data_type,                                                                                                     \
      KernelDefBuilder()                                                                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())                                             \
          .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}), \
      Slice<data_type, true>);

ADD_TYPED_SLICE_V11_OP(uint8_t);
ADD_TYPED_SLICE_V11_OP(uint16_t);
ADD_TYPED_SLICE_V11_OP(uint32_t);
ADD_TYPED_SLICE_V11_OP(uint64_t);
ADD_TYPED_SLICE_V11_OP(int8_t);
ADD_TYPED_SLICE_V11_OP(int16_t);
ADD_TYPED_SLICE_V11_OP(int32_t);
ADD_TYPED_SLICE_V11_OP(int64_t);
ADD_TYPED_SLICE_V11_OP(float);
ADD_TYPED_SLICE_V11_OP(double);
ADD_TYPED_SLICE_V11_OP(MLFloat16);
ADD_TYPED_SLICE_V11_OP(bool);
ADD_TYPED_SLICE_V11_OP(string);

namespace {
// std::clamp doesn't exist until C++17 so create a local version
template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
}  // namespace

// Check if it's possible to combine innermost dimensions so we copy larger blocks.
// Sets flattened_output_dims to nullptr if it is not.
// Updates starts and steps to match flattened_output_dims if it is.
// e.g. if input shape is { 2, 2, 2 }, output shape is { 1, 2, 2 }, and the 'steps' value for the last two dims is 1,
// we are keeping all the data of the inner most two dimensions so can combine those into dims of { 1, 4 }
static void FlattenOutputDims(const std::vector<int64_t>& input_dimensions,
                              const std::vector<int64_t>& output_dims,
                              std::vector<int64_t>& starts,
                              std::vector<int64_t>& steps,
                              std::vector<int64_t>*& flattened_output_dims) {
  int num_to_combine = 0;
  for (int64_t i = static_cast<int64_t>(starts.size()) - 1; i >= 0; --i) {
    // if we're keeping all the data for the dimension and not reversing the direction we can potentially combine it
    if (steps[i] == 1 && input_dimensions[i] == output_dims[i])
      ++num_to_combine;
    else
      break;
  }

  if (num_to_combine > 1) {
    auto num_dims = output_dims.size() - num_to_combine + 1;
    *flattened_output_dims = output_dims;
    flattened_output_dims->resize(num_dims);

    int64_t dim_value = 1;
    for (size_t k = num_dims - 1, end = output_dims.size(); k < end; ++k) {
      dim_value *= output_dims[k];
    }

    flattened_output_dims->back() = dim_value;

    // the value of starts and steps for all the dims being combined are 0 and 1 respectively,
    // so we can just shrink via resize so the number of entries matches flattened_output_dims
    starts.resize(num_dims);
    steps.resize(num_dims);
  } else {
    flattened_output_dims = nullptr;
  }
}

// Slice V1-9 & DynamicSlice
Status SliceBase::PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                    const std::vector<int64_t>& raw_ends,
                                    const std::vector<int64_t>& raw_axes,
                                    const std::vector<int64_t>& input_dimensions,
                                    std::vector<int64_t>& starts,
                                    std::vector<int64_t>& steps,
                                    std::vector<int64_t>& output_dims,
                                    std::vector<int64_t>*& flattened_output_dims) const {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(raw_axes);
  if (axes.empty()) {
    //axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Iterate through the provided axes and override the start/end ranges
  std::unordered_set<int64_t> unique_axes;
  const auto& dimension_count = input_dimensions.size();
  for (size_t axis_index = 0, axes_count = axes.size(); axis_index < axes_count; ++axis_index) {
    auto axis = HandleNegativeAxis(axes[axis_index], dimension_count);  // handle negative and enforce axis is valid
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    if (unique_axes.find(axis) != unique_axes.end())
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has duplicates");
    unique_axes.insert(axis);

    // process start
    auto start = raw_starts[axis_index];
    if (start < 0)
      start += input_dimensions[axis];
    starts[axis] = clamp(start, int64_t{0}, input_dimensions[axis]);

    // process end
    auto end = raw_ends[axis_index];
    if (end < 0)
      end += input_dimensions[axis];

    // find output dim value for this axis
    auto temp = clamp(end, int64_t{0}, input_dimensions[axis]) - starts[axis];
    if (temp < 0)
      output_dims[axis] = 0;
    else
      output_dims[axis] = temp;
  }

  FlattenOutputDims(input_dimensions, output_dims, starts, steps, flattened_output_dims);

  return Status::OK();
}

// DynamicSlice & Slice V10
Status SliceBase::PrepareForCompute(const std::vector<int64_t>& raw_starts,
                                    const std::vector<int64_t>& raw_ends,
                                    const std::vector<int64_t>& raw_axes,
                                    const std::vector<int64_t>& raw_steps,
                                    const std::vector<int64_t>& input_dimensions,
                                    std::vector<int64_t>& starts,
                                    std::vector<int64_t>& steps,
                                    std::vector<int64_t>& output_dims,
                                    std::vector<int64_t>*& flattened_output_dims) const {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(raw_axes);

  if (axes.empty()) {
    // axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Iterate through the provided axes and override the start/end/steps ranges
  std::unordered_set<int64_t> unique_axes;
  const auto& dimension_count = input_dimensions.size();
  for (size_t axis_index = 0, axes_count = axes.size(); axis_index < axes_count; ++axis_index) {
    auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(dimension_count) : axes[axis_index];
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    if (unique_axes.find(axis) != unique_axes.end())
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has duplicates");
    unique_axes.insert(axis);

    // process step
    auto step = axis_index < raw_steps.size() ? raw_steps[axis_index] : 1;
    if (step == 0)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'step' value cannot be 0");
    steps[axis] = step;

    // process start
    auto start = raw_starts[axis_index];
    if (start < 0)
      start += input_dimensions[axis];
    if (step < 0)
      starts[axis] = clamp(start, int64_t{0}, input_dimensions[axis] - 1);
    else
      starts[axis] = clamp(start, int64_t{0}, input_dimensions[axis]);

    // process end
    auto end = raw_ends[axis_index];
    // INT_MAX has a special meaning for end according to spec
    // equivalent to 'None' in numpy
    // it represent slicing to the end of the dimension
    if (end == std::numeric_limits<int32_t>::max() ||
        end == std::numeric_limits<int64_t>::max()) {
      end = step < 0 ? -1 : input_dimensions[axis];
    }

    else {
      if (end < 0)
        end += input_dimensions[axis];
      if (step < 0)
        end = clamp(end, int64_t{-1}, input_dimensions[axis]);
      else
        end = clamp(end, int64_t{0}, input_dimensions[axis]);
    }

    // find output dim value for this axis
    auto temp = static_cast<int64_t>(ceil(1.0 * (end - starts[axis]) / step));
    if (temp < 0)
      output_dims[axis] = 0;
    else
      output_dims[axis] = temp;
  }

  FlattenOutputDims(input_dimensions, output_dims, starts, steps, flattened_output_dims);

  return Status::OK();
}

// Slice V10 & DynamicSlice
void SliceBase::FillVectorsFromInput(const OpKernelContext* context,
                                     std::vector<int64_t>& input_starts,
                                     std::vector<int64_t>& input_ends,
                                     std::vector<int64_t>& input_axes,
                                     std::vector<int64_t>& input_steps) const {
  const auto* start_tensor = context->Input<Tensor>(1);
  const auto* ends_tensor = context->Input<Tensor>(2);
  const auto* axes_tensor = context->Input<Tensor>(3);
  const Tensor* steps_tensor = nullptr;
  // check if this is Slice V10 - only Slice V10 has this optional input
  if (context->InputCount() == 5)
    steps_tensor = context->Input<Tensor>(4);

  ORT_ENFORCE(nullptr != start_tensor && start_tensor->Shape().NumDimensions() == 1, "Starts must be a 1-D array");
  ORT_ENFORCE(nullptr != ends_tensor && ends_tensor->Shape().NumDimensions() == 1, "Ends must be a 1-D array");
  ORT_ENFORCE(start_tensor->Shape() == ends_tensor->Shape(), "Starts and ends shape mismatch");
  ORT_ENFORCE(nullptr == axes_tensor || start_tensor->Shape() == axes_tensor->Shape(), "Starts and axes shape mismatch");
  ORT_ENFORCE(nullptr == steps_tensor || start_tensor->Shape() == steps_tensor->Shape(), "Starts and steps shape mismatch");

  const auto& dtype = start_tensor->DataType();
  const auto& size = start_tensor->Shape().Size();
  input_starts.resize(size);
  input_ends.resize(size);
  if (nullptr != axes_tensor)
    input_axes.resize(size);
  // Slice V10
  if (nullptr != steps_tensor)
    input_steps.resize(size);

  if (utils::IsPrimitiveDataType<int32_t>(dtype)) {
    std::copy(start_tensor->Data<int32_t>(), start_tensor->Data<int32_t>() + size, input_starts.begin());
    std::copy(ends_tensor->Data<int32_t>(), ends_tensor->Data<int32_t>() + size, input_ends.begin());
    if (nullptr != axes_tensor)
      std::copy(axes_tensor->Data<int32_t>(), axes_tensor->Data<int32_t>() + size, input_axes.begin());
    // Slice V10
    if (nullptr != steps_tensor)
      std::copy(steps_tensor->Data<int32_t>(), steps_tensor->Data<int32_t>() + size, input_steps.begin());
  }

  else if (utils::IsPrimitiveDataType<int64_t>(dtype)) {
    std::copy(start_tensor->Data<int64_t>(), start_tensor->Data<int64_t>() + size, input_starts.begin());
    std::copy(ends_tensor->Data<int64_t>(), ends_tensor->Data<int64_t>() + size, input_ends.begin());
    if (nullptr != axes_tensor)
      std::copy(axes_tensor->Data<int64_t>(), axes_tensor->Data<int64_t>() + size, input_axes.begin());
    // Slice V10
    if (nullptr != steps_tensor)
      std::copy(steps_tensor->Data<int64_t>(), steps_tensor->Data<int64_t>() + size, input_steps.begin());
  }

  // should not reach this as no kernel is registered for this condition to be triggered - just an additional safety check
  else {
    ORT_THROW("Data type for starts and ends inputs' need to be int32_t or int64_t, but instead got ", dtype);
  }
}

template <typename T>
Status SliceImpl(OpKernelContext* ctx,
                 const Tensor& input_tensor,
                 std::vector<int64_t>& output_dims,
                 std::vector<int64_t>* flattened_output_dims,
                 const std::vector<int64_t>& starts,
                 const std::vector<int64_t>& steps) {
  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  auto* output = output_tensor.template MutableData<T>();
  const auto* output_end = output + output_tensor.Shape().Size();

  SliceIterator<T> input_iterator =
      flattened_output_dims != nullptr
          ? SliceIterator<T>(input_tensor, *flattened_output_dims, starts, *flattened_output_dims, steps)
          : SliceIterator<T>(input_tensor, starts, output_dims, steps);

  if (input_iterator.SolitaryInnerStep()) {
    while (output < output_end) {
      output = input_iterator.CopyInnermostAxisSolitaryInnerStep(output);
    }
  } else {
    while (output < output_end) {
      output = input_iterator.CopyInnermostAxisNonSolitaryInnerStep(output);
    }
  }

  ORT_ENFORCE(output == output_end);

  return Status::OK();
}

template <typename T, bool dynamic>
Status Slice<T, dynamic>::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr, "Missing input tensor to be processed");
  const auto& input_tensor = *input_tensor_ptr;
  const auto& input_dimensions = input_tensor.Shape().GetDims();
  if (input_dimensions.empty()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");

  // Initialize the starts & ends to the actual tensor shape
  std::vector<int64_t> starts(input_dimensions.size(), 0);
  std::vector<int64_t> steps(input_dimensions.size(), 1);
  std::vector<int64_t> output_dims(input_dimensions);
  std::vector<int64_t> flattened_output_dims;
  std::vector<int64_t>* p_flattened_output_dims = &flattened_output_dims;

  // Slice V10 & DynamicSlice
  if (dynamic) {
    std::vector<int64_t> input_starts;
    std::vector<int64_t> input_ends;
    std::vector<int64_t> input_axes;
    std::vector<int64_t> input_steps;
    FillVectorsFromInput(ctx, input_starts, input_ends, input_axes, input_steps);
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps,
                                          input_dimensions, starts, steps, output_dims,
                                          p_flattened_output_dims));
  }
  // Slice V1-9
  else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_,
                                          input_dimensions, starts, steps, output_dims,
                                          p_flattened_output_dims));
  }

  return SliceImpl<T>(ctx, input_tensor, output_dims, p_flattened_output_dims, starts, steps);
}
}  // namespace onnxruntime

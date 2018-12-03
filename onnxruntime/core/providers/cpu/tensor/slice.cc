// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
using namespace ::onnxruntime::common;
using namespace std;

namespace onnxruntime {

#define ADD_TYPED_SLICE_OP(data_type)                                                   \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                       \
      Slice,                                                                            \
      1,                                                                                \
      data_type,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Slice<data_type>);

ADD_TYPED_SLICE_OP(uint8_t);
ADD_TYPED_SLICE_OP(uint16_t);
ADD_TYPED_SLICE_OP(uint32_t);
ADD_TYPED_SLICE_OP(uint64_t);
ADD_TYPED_SLICE_OP(int8_t);
ADD_TYPED_SLICE_OP(int16_t);
ADD_TYPED_SLICE_OP(int32_t);
ADD_TYPED_SLICE_OP(int64_t);
ADD_TYPED_SLICE_OP(float);
ADD_TYPED_SLICE_OP(double);
ADD_TYPED_SLICE_OP(MLFloat16);
ADD_TYPED_SLICE_OP(bool);
ADD_TYPED_SLICE_OP(string);

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
                                    std::vector<int64_t>& starts, std::vector<int64_t>& output_dims) const {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(axes_);
  if (!has_axes_) {
    //axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(starts.size());
    for (size_t i = 0; i < starts.size(); i++)
      axes[i] = i;

    if (axes.size() > starts_.size())
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has more entries than the 'starts' attribute holds");
    if (axes.size() > ends_.size())
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has more entries than the 'ends' attribute holds");
  }

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
  }

  return Status::OK();
}

template <typename T>
Status Slice<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* input_tensor_ptr = ctx->Input<Tensor>(0);
  ONNXRUNTIME_ENFORCE(input_tensor_ptr != nullptr);
  auto& input_tensor = *input_tensor_ptr;
  auto& input_dimensions = input_tensor.Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  const size_t dimension_count = input_dimensions.size();
  std::vector<int64_t> starts(dimension_count, 0);
  std::vector<int64_t> output_dims(input_dimensions);

  ONNXRUNTIME_RETURN_IF_ERROR(PrepareForCompute(dimension_count, input_dimensions, starts, output_dims));

  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto* output = output_tensor.template MutableData<T>();
  const auto* output_end = output + output_shape.Size();

  SliceIterator<T> input_iterator(input_tensor, starts, output_dims);
  while (output != output_end)
    *output++ = *input_iterator++;

  return Status::OK();
}

}  // namespace onnxruntime

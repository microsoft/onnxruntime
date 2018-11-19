// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "gsl/gsl_algorithm"
#include "core/providers/cpu/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Tile,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Tile<float>);

template <>
Status Tile<float>::Compute(OpKernelContext* ctx) const {
  const Tensor* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "Input count of Tile OP mismatch, the first one is empty");
  const Tensor& input_tensor = *tensor_pointer;
  tensor_pointer = ctx->Input<Tensor>(1);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "Input count of Tile OP mismatch, the second one is empty");
  const Tensor& repeats_tensor = *tensor_pointer;

  size_t dimension_count = input_tensor.Shape().NumDimensions();

  if (repeats_tensor.Shape().NumDimensions() != 1)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must be 1 dimensional");
  if (size_t(repeats_tensor.Shape().Size()) != input_tensor.Shape().NumDimensions())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must have the same length as the 'input' tensor");

  // Calculate the shape of the output tensor
  auto* repeats = repeats_tensor.template Data<int64_t>();
  std::vector<int64_t> output_dims = input_tensor.Shape().GetDims();
  for (auto axis = 0; axis < input_tensor.Shape().NumDimensions(); axis++)
    output_dims[axis] *= repeats[axis];
  TensorShape outputShape(output_dims);
  auto& output_tensor = *ctx->Output(0, outputShape);

  auto* output = output_tensor.template MutableData<float>();
  auto* input = input_tensor.template Data<float>();

  TensorPitches output_pitches(output_tensor);
  TensorAxisCounters input_counters(input_tensor);

  while (input_counters) {
    // Copy the input data over
    size_t input_pitch = input_tensor.Shape().GetDims().back();
    for (size_t i = 0; i < input_pitch; i++)
      *output++ = *input++;

    // Tile it for the innermost axis
    const auto* copy = output - input_tensor.Shape()[dimension_count - 1];
    for (int64_t repeat = (repeats[dimension_count - 1] - 1) * input_pitch; repeat-- > 0;)
      *output++ = *copy++;

    // Tile it in the other axes
    while (input_counters.Increment()) {
      ptrdiff_t pitch = output_pitches[input_counters.Axis()] * input_tensor.Shape()[input_counters.Axis()];
      copy = output - pitch;
      for (int64_t repeat = (repeats[input_counters.Axis()] - 1) * pitch; repeat-- > 0;) {
        *output++ = *copy++;
      }
    }
  }
  return Status::OK();
}
}  // namespace onnxruntime

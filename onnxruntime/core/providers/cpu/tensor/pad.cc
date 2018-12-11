// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif
#include "core/providers/cpu/tensor/pad.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Pad,
    2,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pad<float>);

// This is the general padding method to n-dimensionally do edge or reflection padding (based on the inputDelta values)
template <typename T>
static void PadAxis(T* output, T* input, ptrdiff_t input_delta, ptrdiff_t input_pitch, size_t block_size, size_t block_count) {
  for (size_t block_index = 0; block_index < block_count; block_index++) {
    for (size_t i = 0; i < block_size; i++) {
      *output++ = *input;
      input += input_delta;
    }
    input += input_pitch;
  }
}

// These are optimizations of PadAxis. The inner loop is removed since the innermost axis has a blockSize of 1,
// and inputPitch and inputDelta are just a single value added each iteration.
template <typename T>
static void PadInnermostAxis(T* output, T* input, ptrdiff_t input_delta, size_t block_count) {
  for (size_t block_index = 0; block_index < block_count; block_index++) {
    *output++ = *input;
    input += input_delta;
  }
}

// For constant padding, there is no input, just a size to write the constant to
template <typename T>
static void PadAxisConstant(T* output, T constant, size_t size) {
  for (size_t i = 0; i < size; i++)
    *output++ = constant;
}

template <>
Status Pad<float>::Compute(OpKernelContext* ctx) const {
  auto& input_tensor = *ctx->Input<Tensor>(0);
  std::vector<int64_t> output_dims(input_tensor.Shape().GetDims());
  size_t dimension_count = output_dims.size();

  ONNXRUNTIME_ENFORCE(dimension_count * 2 == pads_.size(), "'pads' attribute has wrong number of values");

  std::vector<int64_t> input_starts;
  std::vector<int64_t> input_extents;

  // Calculate output dimensions, and handle any negative padding
  for (size_t i = 0; i < dimension_count; i++) {
    input_starts.push_back(slices_[i]);
    input_extents.push_back(output_dims[i] + slices_[i] + slices_[i + dimension_count]);
    output_dims[i] += pads_[i] + pads_[i + dimension_count] + slices_[i] + slices_[i + dimension_count];
  }
  TensorShape output_shape(output_dims);

  SliceIterator<float> input(input_tensor, input_starts, input_extents);
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto* output = output_tensor.template MutableData<float>();

  TensorPitches output_pitches(output_tensor);
  size_t alignSkip = 0;  // Amount to skip to align to where the next input tensor data needs to be written

  // Initial skip, sum up the begin padding on each axis
  for (size_t i = 0; i < dimension_count; i++)
    alignSkip += pads_[i] * output_pitches[i];

  size_t inner_axis = dimension_count - 1;
  ExtentAxisCounters input_counters(input_extents);

  switch (mode_) {
    case Mode::Constant:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (input_counters) {
        output += alignSkip;
        {
          float* axisStart = output;
          output = input.CopyInnermostAxis(output);

          int64_t prePad = pads_[inner_axis];
          int64_t postPad = pads_[inner_axis + dimension_count];
          PadAxisConstant(axisStart - prePad, value_, prePad);
          PadAxisConstant(output, value_, postPad);
          output += postPad;
          alignSkip = prePad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = output_pitches[input_counters.Axis()];
          float* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
          int64_t prePad = pads_[input_counters.Axis()];
          int64_t postPad = pads_[input_counters.Axis() + dimension_count];
          PadAxisConstant(axisStart - prePad * inner_pitch, value_, prePad * inner_pitch);
          PadAxisConstant(output, value_, postPad * inner_pitch);
          output += inner_pitch * postPad;
          alignSkip += inner_pitch * prePad;
        }
      }
      break;

    case Mode::Edge:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (input_counters) {
        output += alignSkip;
        {
          float* axisStart = output;
          output = input.CopyInnermostAxis(output);

          int64_t prePad = pads_[inner_axis];
          int64_t postPad = pads_[inner_axis + dimension_count];
          PadAxisConstant(axisStart - prePad, *axisStart, prePad);
          PadAxisConstant(output, *(output - 1), postPad);
          output += postPad;
          alignSkip = prePad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = output_pitches[input_counters.Axis()];
          float* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
          int64_t prePad = pads_[input_counters.Axis()];
          int64_t postPad = pads_[input_counters.Axis() + dimension_count];
          PadAxis(axisStart - prePad * inner_pitch, axisStart, 1, -inner_pitch, inner_pitch, prePad);
          PadAxis(output, output - inner_pitch, 1, -inner_pitch, inner_pitch, postPad);
          output += inner_pitch * postPad;
          alignSkip += inner_pitch * prePad;
        }
      }
      break;

    case Mode::Reflect:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (input_counters) {
        output += alignSkip;
        {
          float* axisStart = output;
          output = input.CopyInnermostAxis(output);

          int64_t prePad = pads_[inner_axis];
          int64_t postPad = pads_[inner_axis + dimension_count];
          PadInnermostAxis(axisStart - prePad, axisStart + prePad, -1 /* inputDelta */, prePad);
          PadInnermostAxis(output, output - 2, -1 /* inputDelta */, postPad);
          output += postPad;
          alignSkip = prePad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = output_pitches[input_counters.Axis()];
          float* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
          int64_t prePad = pads_[input_counters.Axis()];
          int64_t postPad = pads_[input_counters.Axis() + dimension_count];
          PadAxis(axisStart - prePad * inner_pitch, axisStart + prePad * inner_pitch, 1, -inner_pitch * 2, inner_pitch, prePad);
          PadAxis(output, output - 2 * inner_pitch, 1, -inner_pitch * 2, inner_pitch, postPad);
          output += inner_pitch * postPad;
          alignSkip += inner_pitch * prePad;
        }
      }
      break;
  }

  return Status::OK();
}
};  // namespace onnxruntime

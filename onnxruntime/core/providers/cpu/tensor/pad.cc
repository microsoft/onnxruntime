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

// Flatten no padding inner most Axis, so one memcpy cover multiple Axis.
// For example, for a shape of [1,224,224,3] with padding [0,3,3,0,0,3,3,0], can be flatten as
// [1,224,224*3] with padding [0,3,3*3,0,3,3*3].
static void FlattenInnerShape(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& slices, std::vector<int64_t>& reshaped_dims) {
  size_t dims_count = input_dims.size();
  size_t inner_axis = dims_count - 1;
  size_t inner_size = 1;

  // Find all inner most dimensions that can be flattened.
  do {
    inner_size *= input_dims[inner_axis];

    if (inner_axis == 0)
      break;

    // Break on first Axis that has padding
    if (!(pads[inner_axis] == 0 && pads[inner_axis + dims_count] == 0 && slices[inner_axis] == 0 && slices[inner_axis + dims_count] == 0))
      break;

  } while (inner_axis-- > 0);

  reshaped_dims.resize(inner_axis + 1);
  std::copy(input_dims.begin(), input_dims.begin() + inner_axis + 1, reshaped_dims.begin());

  // Flatten inner axis.
  reshaped_dims[inner_axis] = inner_size;
}

static void ReshapePads(const std::vector<int64_t>& src_pad, size_t src_dim_count, size_t new_dim_count,
                        size_t inner_no_pad_size, std::vector<int64_t>& reshaped_pad) {
  size_t inner_axis = new_dim_count - 1;
  std::copy(src_pad.begin(), src_pad.begin() + inner_axis, reshaped_pad.begin());
  std::copy(src_pad.begin() + src_dim_count, src_pad.begin() + src_dim_count + inner_axis, reshaped_pad.begin() + new_dim_count);

  // Flatten inner axis.
  reshaped_pad[inner_axis] = src_pad[inner_axis] * inner_no_pad_size;
  reshaped_pad[inner_axis + new_dim_count] = src_pad[inner_axis + src_dim_count] * inner_no_pad_size;
}

template <>
Status PadCpuImpl<float>(OpKernelContext* ctx,
                         const std::vector<int64_t>& pads,
                         const std::vector<int64_t>& slices,
                         const Mode& mode,
                         float value) {
  auto& input_tensor = *ctx->Input<Tensor>(0);
  std::vector<int64_t> output_dims(input_tensor.Shape().GetDims());
  size_t dimension_count = output_dims.size();

  // make copy of raw_pads as it may be mutated below
  ORT_ENFORCE(dimension_count > 0, "Input tensor has no dimensions");
  ORT_ENFORCE(dimension_count * 2 == pads.size(), "'pads' has wrong number of values");

  // Reshape input dims
  std::vector<int64_t> reshaped_input_dims;
  FlattenInnerShape(output_dims, pads, slices, reshaped_input_dims);

  // Reshape padding
  size_t new_dims_count = reshaped_input_dims.size();
  size_t inner_axis = new_dims_count - 1;
  size_t inner_no_pad_size = reshaped_input_dims[inner_axis] / output_dims[inner_axis];
  std::vector<int64_t> reshaped_pad(2 * new_dims_count);
  std::vector<int64_t> reshaped_slice(2 * new_dims_count);
  ReshapePads(pads, dimension_count, new_dims_count, inner_no_pad_size, reshaped_pad);
  ReshapePads(slices, dimension_count, new_dims_count, inner_no_pad_size, reshaped_slice);

  std::vector<int64_t> reshaped_output_dims = reshaped_input_dims;
  std::vector<int64_t> input_starts;
  std::vector<int64_t> input_extents;

  // Calculate output dimensions, and handle any negative padding
  input_starts.reserve(2 * new_dims_count);
  input_extents.reserve(2 * new_dims_count);
  for (size_t i = 0; i < new_dims_count; i++) {
    input_starts.push_back(reshaped_slice[i]);
    input_extents.push_back(reshaped_input_dims[i] + reshaped_slice[i] + reshaped_slice[i + new_dims_count]);
    reshaped_output_dims[i] += reshaped_pad[i] + reshaped_pad[i + new_dims_count] + reshaped_slice[i] + reshaped_slice[i + new_dims_count];
  }

  for (size_t i = 0; i < dimension_count; i++) {
    output_dims[i] += pads[i] + pads[i + dimension_count] + slices[i] + slices[i + dimension_count];
  }
  TensorShape output_shape(output_dims);

  TensorShape input_shape(reshaped_input_dims);
  SliceIterator<float> input(input_tensor, input_shape, input_starts, input_extents, {});

  // output_shape need to keep original.
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto* output = output_tensor.template MutableData<float>();

  TensorPitches output_pitches(reshaped_output_dims);
  size_t alignSkip = 0;  // Amount to skip to align to where the next input tensor data needs to be written

  // Initial skip, sum up the begin padding on each axis
  for (size_t i = 0; i < new_dims_count; i++)
    alignSkip += reshaped_pad[i] * output_pitches[i];

  ExtentAxisCounters input_counters(input_extents);

  switch (mode) {
    case Mode::Constant:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (input_counters) {
        output += alignSkip;
        {
          float* axisStart = output;
          output = input.CopyInnermostAxisSolitaryInnerStep(output);

          int64_t prePad = reshaped_pad[inner_axis];
          int64_t postPad = reshaped_pad[inner_axis + new_dims_count];
          PadAxisConstant(axisStart - prePad, value, prePad);
          PadAxisConstant(output, value, postPad);
          output += postPad;
          alignSkip = prePad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = output_pitches[input_counters.Axis()];
          float* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
          int64_t prePad = reshaped_pad[input_counters.Axis()];
          int64_t postPad = reshaped_pad[input_counters.Axis() + new_dims_count];
          PadAxisConstant(axisStart - prePad * inner_pitch, value, prePad * inner_pitch);
          PadAxisConstant(output, value, postPad * inner_pitch);
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
          output = input.CopyInnermostAxisSolitaryInnerStep(output);

          int64_t prePad = reshaped_pad[inner_axis];
          int64_t postPad = reshaped_pad[inner_axis + new_dims_count];
          PadAxisConstant(axisStart - prePad, *axisStart, prePad);
          PadAxisConstant(output, *(output - 1), postPad);
          output += postPad;
          alignSkip = prePad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = output_pitches[input_counters.Axis()];
          float* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
          int64_t prePad = reshaped_pad[input_counters.Axis()];
          int64_t postPad = reshaped_pad[input_counters.Axis() + new_dims_count];
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
          output = input.CopyInnermostAxisSolitaryInnerStep(output);

          int64_t prePad = reshaped_pad[inner_axis];
          int64_t postPad = reshaped_pad[inner_axis + new_dims_count];
          PadInnermostAxis(axisStart - prePad, axisStart + prePad, -1 /* inputDelta */, prePad);
          PadInnermostAxis(output, output - 2, -1 /* inputDelta */, postPad);
          output += postPad;
          alignSkip = prePad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = output_pitches[input_counters.Axis()];
          float* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
          int64_t prePad = reshaped_pad[input_counters.Axis()];
          int64_t postPad = reshaped_pad[input_counters.Axis() + new_dims_count];
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

template <>
Status Pad<float>::Compute(OpKernelContext* ctx) const {
  return PadCpuImpl<float>(ctx, pads_, slices_, mode_, value_);
}
};  // namespace onnxruntime

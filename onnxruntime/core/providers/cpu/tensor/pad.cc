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

// Register a kernel for kMsDomain (contrib op) Pad
#ifndef DISABLE_CONTRIB_OPS

namespace contrib {
// TODO: Remove this contrib kernel registration and the schema from the appropriate places
// once Keras Mask RCNN is shipped with all ONNX domain ops

// Currently this kernel is required to support Keras Mask-RCNN
// only float type is supported
ONNX_OPERATOR_KERNEL_EX(Pad,
                        kMSDomain,
                        1,
                        kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        onnxruntime::Pad<float>);

}  // namespace contrib

#endif

// only float type is supported for opset-10
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    2, 10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pad<float>);

// The interface for the 'Pad' op was changed in opset-11
// 'pads' and 'value' (attributes previously) became inputs in this version
// The core logic remains the same

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Pad,
    11,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), Pad<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Pad,
    11,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), Pad<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Pad,
    11,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()), Pad<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Pad,
    11,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()), Pad<int64_t>);

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

Status PadBase::HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, TensorShape& output_shape) {
  switch (mode) {
    case Mode::Constant: {
      // default behavior is fine
      break;
    }
    case Mode::Edge: {
      // we need to override the default logic and set the output dim to 0 where the input dim is zero.
      // this is to match numpy behavior.
      for (size_t i = 0, end = input_shape.NumDimensions(); i < end; ++i) {
        if (input_shape[i] == 0)
          output_shape[i] = 0;
      }
      break;
    }
    case Mode::Reflect: {
      // match numpy behavior of failing if mode is 'reflect' and there's an attempt to pad a dimension with value of 0
      for (size_t i = 0, end = input_shape.NumDimensions(); i < end; ++i) {
        if (input_shape[i] == 0 && output_shape[i] > 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Cannot use 'reflect' mode to pad dimension with a value of 0. Input shape:",
                                 input_shape);
        }
      }
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected mode of ", static_cast<int>(mode));
  }

  return Status::OK();
}

// special handling for edge case where the input has one or more dims with value of 0
template <typename T>
static Status PadInputWithDimValueOfZero(OpKernelContext* ctx,
                                         const Mode& mode,
                                         const TensorShape& input_shape,
                                         std::vector<int64_t>& output_dims,
                                         T value) {
  TensorShape output_shape(output_dims);
  ORT_RETURN_IF_ERROR(PadBase::HandleDimValueZero(mode, input_shape, output_shape));

  auto& output_tensor = *ctx->Output(0, output_shape);

  // we need to add pads if mode is constant, otherwise the output has one or more dim values of 0 so is empty
  if (mode == Mode::Constant) {
    // we add pads with the default value to all dims including those with a value of 0
    auto* output = output_tensor.template MutableData<T>();
    std::fill_n(output, output_shape.Size(), value);
  }

  return Status::OK();
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

template <typename T>
Status PadCpuImpl(OpKernelContext* ctx,
                  const std::vector<int64_t>& pads,
                  const std::vector<int64_t>& slices,
                  const Mode& mode,
                  T value) {
  const auto& input_tensor = *ctx->Input<Tensor>(0);
  const auto& orig_input_shape = input_tensor.Shape();
  std::vector<int64_t> output_dims(orig_input_shape.GetDims());
  size_t data_rank = output_dims.size();

  // make copy of raw_pads as it may be mutated below
  ORT_ENFORCE(data_rank > 0, "Input tensor has no dimensions");
  ORT_ENFORCE(data_rank * 2 == pads.size(), "'pads' has wrong number of values");

  // Reshape input dims
  std::vector<int64_t> reshaped_input_dims;
  FlattenInnerShape(output_dims, pads, slices, reshaped_input_dims);

  // Reshape padding
  size_t new_dims_count = reshaped_input_dims.size();
  size_t inner_axis = new_dims_count - 1;
  size_t inner_no_pad_size = output_dims[inner_axis] > 0 ? reshaped_input_dims[inner_axis] / output_dims[inner_axis] : 0;
  std::vector<int64_t> reshaped_pad(2 * new_dims_count), reshaped_slice(2 * new_dims_count);
  ReshapePads(pads, data_rank, new_dims_count, inner_no_pad_size, reshaped_pad);
  ReshapePads(slices, data_rank, new_dims_count, inner_no_pad_size, reshaped_slice);

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

  for (size_t i = 0; i < data_rank; i++) {
    output_dims[i] += pads[i] + pads[i + data_rank] + slices[i] + slices[i + data_rank];
  }

  // special case an input with one or more dim values of 0. edge case that is easier to handle
  // separately than to complicate all the code for normal usage.
  if (orig_input_shape.Size() == 0) {
    return PadInputWithDimValueOfZero(ctx, mode, orig_input_shape, output_dims, value);
  }

  TensorShape input_shape(reshaped_input_dims);
  SliceIterator<T> input(input_tensor, input_shape, input_starts, input_extents, {});

  // output_shape need to keep original.
  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto* output = output_tensor.template MutableData<T>();

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
          T* axisStart = output;
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
          T* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
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
          T* axisStart = output;
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
          T* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
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
          T* axisStart = output;
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
          T* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
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

template <typename T>
Status Pad<T>::Compute(OpKernelContext* ctx) const {
  // kOnnxDomain Pad opset >= 11 (Or) kMsDomain opset == 1
  if (is_dynamic_) {
    const Tensor& input_tensor = *ctx->Input<Tensor>(0);
    size_t data_rank = input_tensor.Shape().NumDimensions();

    const Tensor& pads_tensor = *ctx->Input<Tensor>(1);
    const std::vector<int64_t>& pads_tensor_dims = pads_tensor.Shape().GetDims();
    ORT_ENFORCE(utils::IsPrimitiveDataType<int64_t>(pads_tensor.DataType()),
                "Pads tensor should be an INT64 tensor");
    ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
                "Pads tensor should be a 1D tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 * input_rank]");

    const int64_t* pads_tensor_raw_data = pads_tensor.template Data<int64_t>();
    size_t pads_size = static_cast<size_t>(pads_tensor.Shape().Size());
    ORT_ENFORCE(pads_size == 2 * data_rank,
                "Pads tensor size should be equal to twice the input dimension count ");

    std::vector<int64_t> pads;
    pads.reserve(2 * data_rank);
    for (size_t i = 0; i < pads_size; ++i) {
      pads.push_back(pads_tensor_raw_data[i]);
    }

    // Separate out any negative pads into the slices array
    std::vector<int64_t> slices(pads.size(), 0);
    for (size_t index = 0; index < pads.size(); index++) {
      if (pads[index] < 0) {
        slices[index] = pads[index];
        pads[index] = 0;
      }
    }

    T value = 0;
    const Tensor* value_tensor = ctx->Input<Tensor>(2);
    if (nullptr != value_tensor) {
      ORT_ENFORCE(utils::IsPrimitiveDataType<T>(value_tensor->DataType()) &&
                      value_tensor->Shape().Size() == 1,
                  "Value tensor should be a 1D tensor of size 1 with the same type as that of the input tensor");
      value = value_tensor->template Data<T>()[0];
    }

    return PadCpuImpl<T>(ctx, pads, slices, mode_, value);
  } else {
    // kOnnxDomain Pad opset < 11
    // In the earlier opset versions of Pad, the type for 'value' attribute was always float,
    // irrespective of the data type of the actual input to be padded
    return PadCpuImpl<T>(ctx, pads_, slices_, mode_, static_cast<T>(value_));
  }
}
};  // namespace onnxruntime

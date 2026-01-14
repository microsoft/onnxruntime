// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/pad.h"

#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math.h"

#include <functional>

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

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
                        onnxruntime::Pad);

}  // namespace contrib

#endif

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 2, Input, 0,
    float,
    double);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 11, Input, 0,
    float,
    double,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    int8_t,
    uint8_t);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 13, Input, 0,
    float,
    double,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    int8_t,
    uint8_t,
    bool);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 18, Input, 0,
    float,
    double,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    int8_t,
    uint8_t,
    bool);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 19, Input, 0,
    float,
    double,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    int8_t,
    uint8_t,
    bool);

// Opset 21 added int4 and uint4.
// TODO(adrianlizarraga): Implement int4 and uint4 support.
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 21, Input, 0,
    float,
    double,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    int8_t,
    uint8_t,
    bool);

// Opset 23 added support for float4e2m1.
// TODO: Add support for float4e2m1.
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 23, Input, 0,
    float,
    double,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    int8_t,
    uint8_t,
    bool);

// Opset 24
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 24, Input, 0,
    float,
    double,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    int8_t,
    uint8_t,
    bool);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 25, Input, 0,
    float,
    double,
    int32_t,
    int64_t,
    uint32_t,
    uint64_t,
    int8_t,
    uint8_t,
    bool);

ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 11, Input, 0, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 13, Input, 0, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 18, Input, 0, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 19, Input, 0, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 21, Input, 0, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 23, Input, 0, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 24, Input, 0, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 25, Input, 0, int32_t, int64_t);

}  // namespace op_kernel_type_control

using EnabledPad2Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 2, Input, 0);
using EnabledPad11Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 11, Input, 0);
using EnabledPad13Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 13, Input, 0);
using EnabledPad18Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 18, Input, 0);
using EnabledPad19Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 19, Input, 0);
using EnabledPad21Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 21, Input, 0);
using EnabledPad23Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 23, Input, 0);

using EnabledPad24Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 24, Input, 0);

using EnabledPad25Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 25, Input, 0);

using AllEnabledPadTypes =
    utils::TypeSetUnion<
        EnabledPad2Types,
        EnabledPad11Types,
        EnabledPad13Types>;

// only float type is supported for opset-10
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    2, 10,
    KernelDefBuilder().TypeConstraint(
        "T",
        BuildKernelDefConstraintsFromTypeList<EnabledPad2Types>()),
    Pad);

// The interface for the 'Pad' op was changed in opset-11
// 'pads' and 'value' (attributes previously) became inputs in this version
// The core logic remains the same

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    11, 12,
    KernelDefBuilder().TypeConstraint(
        "T",
        BuildKernelDefConstraintsFromTypeList<EnabledPad11Types>()),
    Pad);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    13, 17,
    KernelDefBuilder().TypeConstraint(
        "T",
        BuildKernelDefConstraintsFromTypeList<EnabledPad13Types>()),
    Pad);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    18, 18,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledPad18Types>()),
    Pad);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    19, 20,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledPad19Types>()),
    Pad);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    21, 22,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledPad21Types>()),
    Pad);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    23, 23,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledPad23Types>()),
    Pad);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Pad,
    24, 24,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledPad24Types>()),
    Pad);

ONNX_CPU_OPERATOR_KERNEL(
    Pad,
    25,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<EnabledPad25Types>()),
    Pad);

using PadsVector = PadBase::PadsVector;

Status PadBase::HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, const TensorShape& output_shape) {
  switch (mode) {
    case Mode::Constant: {
      // default behavior is fine
      break;
    }
    case Mode::Edge: {
      // match numpy behavior of failing if mode is 'edge' and there's an attempt to pad a dimension with value of 0
      for (size_t i = 0, end = input_shape.NumDimensions(); i < end; ++i) {
        if (input_shape[i] == 0 && output_shape[i] > 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Cannot use 'edge' mode to pad dimension with a value of 0. Input shape:",
                                 input_shape);
        }
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

static void ComputePadWithAxes(
    gsl::span<const int64_t> pads_tensor_raw_data,
    std::function<int64_t(size_t)> get_axis,
    size_t axes_size,
    size_t data_rank,
    PadsVector& pads) {
  for (size_t i = 0; i < axes_size; ++i) {
    const size_t axis = onnxruntime::narrow<size_t>(HandleNegativeAxis(get_axis(i), data_rank));
    pads[axis] = pads_tensor_raw_data[i];                          // xi_begin
    pads[data_rank + axis] = pads_tensor_raw_data[axes_size + i];  // xi_end
  }
}

void PadBase::ComputePads(OpKernelContext& ctx, size_t data_rank, gsl::span<const int64_t> pads_data,
                          PadsVector& pads) {
  pads.reserve(2 * data_rank);
  const Tensor* axes_tensor = ctx.Input<Tensor>(3);
  if (axes_tensor) {
    const size_t num_axes_dims = axes_tensor->Shape().NumDimensions();
    ORT_ENFORCE(num_axes_dims == 1, "Axes tensor should be a 1D tensor ");

    const int64_t num_axes = axes_tensor->Shape().Size();
    ORT_ENFORCE(pads_data.size() == narrow<size_t>(2 * num_axes),
                "Pads tensor size should be equal to twice the number of explicitly provided axes.");

    pads.resize(2 * data_rank, 0);
    if (axes_tensor->IsDataType<int32_t>()) {
      auto axes_data = axes_tensor->DataAsSpan<int32_t>();
      ComputePadWithAxes(
          pads_data,
          [axes_data](size_t idx) -> int64_t {
            return axes_data[idx];
          },
          axes_data.size(),
          data_rank,
          pads);
    } else if (axes_tensor->IsDataType<int64_t>()) {
      auto axes_data = axes_tensor->DataAsSpan<int64_t>();
      ComputePadWithAxes(
          pads_data,
          [axes_data](size_t idx) {
            return axes_data[idx];
          },
          axes_data.size(),
          data_rank,
          pads);
    }
  } else {
    ORT_ENFORCE(pads_data.size() == 2 * data_rank,
                "Pads tensor size should be equal to twice the input dimension count ");
    pads.assign(pads_data.begin(), pads_data.end());
  }
}

// Flatten no padding inner most Axis, so one memcpy cover multiple Axis.
// For example, for a shape of [1,224,224,3] with padding [0,3,3,0,0,3,3,0], can be flatten as
// [1,224,224*3] with padding [0,3,3*3,0,3,3*3].
void PadBase::FlattenInnerShape(gsl::span<const int64_t> input_dims, gsl::span<const int64_t> pads,
                                gsl::span<const int64_t> slices, TensorShapeVector& reshaped_dims) {
  const size_t dims_count = input_dims.size();
  size_t inner_axis = dims_count - 1;
  SafeInt<int64_t> inner_size = 1;

  // Find all inner most dimensions that can be flattened.
  do {
    inner_size *= input_dims[inner_axis];

    if (inner_axis == 0)
      break;

    // Break on first Axis that has padding
    if (!(pads[inner_axis] == 0 && pads[inner_axis + dims_count] == 0 &&
          slices[inner_axis] == 0 && slices[inner_axis + dims_count] == 0))
      break;

  } while (inner_axis-- > 0);

  reshaped_dims.reserve(inner_axis + 1);
  std::copy(input_dims.begin(), input_dims.begin() + inner_axis + 1, std::back_inserter(reshaped_dims));

  // Flatten inner axis.
  reshaped_dims[inner_axis] = inner_size;
}

void PadBase::ReshapePads(gsl::span<const int64_t> src_pad, size_t src_dim_count, size_t new_dim_count,
                          size_t inner_no_pad_size, PadsVector& reshaped_pad) {
  size_t inner_axis = new_dim_count - 1;
  std::copy(src_pad.begin(), src_pad.begin() + inner_axis, reshaped_pad.begin());
  std::copy(src_pad.begin() + src_dim_count, src_pad.begin() + src_dim_count + inner_axis,
            reshaped_pad.begin() + new_dim_count);

  // Flatten inner axis.
  reshaped_pad[inner_axis] = SafeInt<int64_t>(src_pad[inner_axis]) * inner_no_pad_size;
  reshaped_pad[inner_axis + new_dim_count] = SafeInt<int64_t>(src_pad[inner_axis + src_dim_count]) * inner_no_pad_size;
}

template <typename T>
struct OutputSink {
  void operator()(T* output, T value) const {
#ifdef _DEBUG
    if (output < beg || output >= end) {
      ORT_THROW("Pad OutputSink: Output pointer is out of range");
    }
#endif
    *output = value;
  }

#ifdef _DEBUG
  OutputSink(T* output, T* output_end)
      : beg(output), end(output_end) {}

  T* beg;
  T* end;
#else
  OutputSink(T* /* output */, T* /* output_end */) {}
#endif
};

// special handling for edge case where the input has one or more dims with value of 0
template <typename T>
static Status PadInputWithDimValueOfZero(OpKernelContext* ctx,
                                         const Mode& mode,
                                         const TensorShape& input_shape,
                                         TensorShapeVector& output_dims,
                                         T value) {
  TensorShape output_shape(output_dims);
  ORT_RETURN_IF_ERROR(PadBase::HandleDimValueZero(mode, input_shape, output_shape));

  auto& output_tensor = *ctx->Output(0, output_shape);

  // we need to add pads if mode is constant, otherwise the output has one or more dim values of 0 so is empty
  if (mode == Mode::Constant) {
    // we add pads with the default value to all dims including those with a value of 0
    auto* output = reinterpret_cast<T*>(output_tensor.MutableDataRaw());
    std::fill_n(output, output_shape.Size(), value);
  }

  return Status::OK();
}

// This is the general padding method to n-dimensionally do edge or reflection padding (based on the inputDelta values)
template <typename T>
static void PadAxis(OutputSink<T>& sink, T* output, T* input, ptrdiff_t input_delta, ptrdiff_t input_pitch,
                    size_t block_size, size_t block_count) {
  for (size_t block_index = 0; block_index < block_count; block_index++) {
    for (size_t i = 0; i < block_size; i++) {
      sink(output++, *input);
      input += input_delta;
    }
    input += input_pitch;
  }
}

// These are optimizations of PadAxis. The inner loop is removed since the innermost axis has a blockSize of 1,
// and inputPitch and inputDelta are just a single value added each iteration.
template <typename T>
static void PadInnermostAxis(OutputSink<T>& sink, T* output, T* input, ptrdiff_t input_delta, size_t block_count) {
  for (size_t block_index = 0; block_index < block_count; block_index++) {
    sink(output++, *input);
    input += input_delta;
  }
}

// For constant padding, there is no input, just a size to write the constant to
template <typename T>
static void PadAxisConstant(OutputSink<T>& sink, T* output, T constant, size_t size) {
  if (size == 1) {
    sink(output, constant);
  } else if (size == 2) {
    sink(output, constant);
    sink(output + 1, constant);
  } else {
    // This would be faster with SSE instructions.
    // That would mean to have an implementation for each type (uint8, uint32, uint64).
    T* end = output + size;
    for (; output != end;)
      sink(output++, constant);
  }
}

template <typename T>
static Status PadImpl(OpKernelContext* ctx,
                      const PadsVector& pads,
                      const PadsVector& slices,
                      const Mode& mode,
                      T value) {
  if (!utils::HasTypeWithSameSize<AllEnabledPadTypes, T>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input data type not supported in this build.");
  }

  const auto& input_tensor = *ctx->Input<Tensor>(0);
  const auto& orig_input_shape = input_tensor.Shape();
  auto output_dims(orig_input_shape.AsShapeVector());
  const size_t data_rank = output_dims.size();

  // make copy of raw_pads as it may be mutated below
  ORT_ENFORCE(data_rank > 0, "Input tensor has no dimensions");
  ORT_ENFORCE(data_rank * 2 == pads.size(), "'pads' has wrong number of values");

  // Reshape input dims
  TensorShapeVector reshaped_input_dims;
  if (PadBase::ShouldFlattenInnerShape(output_dims, pads, slices)) {
    PadBase::FlattenInnerShape(output_dims, pads, slices, reshaped_input_dims);
  } else {
    reshaped_input_dims = output_dims;
  }

  // Reshape padding
  const size_t new_dims_count = reshaped_input_dims.size();
  const size_t inner_axis = new_dims_count - 1;
  const size_t inner_no_pad_size = narrow<size_t>(output_dims[inner_axis] > 0
                                                      ? reshaped_input_dims[inner_axis] / output_dims[inner_axis]
                                                      : 0);
  PadsVector reshaped_pad(2 * new_dims_count), reshaped_slice(2 * new_dims_count);
  PadBase::ReshapePads(pads, data_rank, new_dims_count, inner_no_pad_size, reshaped_pad);
  PadBase::ReshapePads(slices, data_rank, new_dims_count, inner_no_pad_size, reshaped_slice);

  TensorShapeVector reshaped_output_dims = reshaped_input_dims;
  TensorShapeVector input_starts;
  TensorShapeVector effective_input_extents;

  // Calculate reshaped output dimensions, and handle any negative padding
  input_starts.reserve(new_dims_count);
  effective_input_extents.reserve(new_dims_count);
  for (size_t i = 0; i < new_dims_count; i++) {
    // Starts for every dimension. If slice is negative, we need to start further in, handled by the SliceIterator
    input_starts.push_back(-1 * reshaped_slice[i]);
    // Do not allow negative extents
    int64_t extent = std::max<int64_t>(SafeInt<int64_t>(reshaped_input_dims[i]) +
                                           reshaped_slice[i] + reshaped_slice[i + new_dims_count],
                                       0LL);
    effective_input_extents.push_back(extent);
    reshaped_output_dims[i] += SafeInt<int64_t>(reshaped_pad[i]) + reshaped_pad[i + new_dims_count] +
                               reshaped_slice[i] + reshaped_slice[i + new_dims_count];
  }

  // Compute true output dimensions
  for (size_t i = 0; i < data_rank; i++) {
    output_dims[i] += SafeInt<int64_t>(pads[i]) + pads[i + data_rank] + slices[i] + slices[i + data_rank];
  }

  // If the input is empty, but output shape may not be, need padding only
  // this is expected for constant mode only, otherwise the output is empty
  // no error
  if (orig_input_shape.Size() == 0) {
    return PadInputWithDimValueOfZero(ctx, mode, orig_input_shape, output_dims, value);
  }

  // output_shape needs to keep original.
  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);

  const SafeInt<size_t> total_output_elems(output_shape.Size());
  auto* output = reinterpret_cast<T*>(output_tensor.MutableDataRaw());
  auto* output_end = output + static_cast<size_t>(total_output_elems);
  OutputSink<T> sink(output, output_end);

  // Early constant-fill: if any effective input extent is zero (input is not empty), no data to copy
  // only padding if any for constant mode, for other modes it is an error
  const bool no_effective_data_to_copy = std::any_of(effective_input_extents.begin(), effective_input_extents.end(),
                                                     [](int64_t v) { return v == 0; });

  if (no_effective_data_to_copy) {
    if (mode == Mode::Constant) {
      PadAxisConstant<T>(sink, output, value, total_output_elems);
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Pad: invalid mode: ", static_cast<int>(mode), " with zero effective input extent");
  }

  // Special case for Reflect mode: ensure all extents >= 2 after slicing
  // otherwise reflection is not possible. Matches numpy behavior as ONNX only
  // implies that this would be wrong as the start and end positions should be distinct
  // values and with 0 there is not one, and with 1 reflection degenerates into ambiguity.
  if (mode == Mode::Reflect) {
    for (size_t i = 0; i < new_dims_count; ++i) {
      const int64_t extent = effective_input_extents[i];  // length after slicing
      const bool reflect_on_axis =
          (reshaped_pad[i] > 0) || (reshaped_pad[i + new_dims_count] > 0);
      if (reflect_on_axis && extent < 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Pad reflect requires axis length >= 2 after slicing. Input shape:",
                               orig_input_shape);
      }
    }
  }

  TensorPitches output_pitches(reshaped_output_dims);
  // Initial skip, sum up the start padding on each axis
  SafeInt<size_t> align_skip = 0;
  for (size_t i = 0; i < new_dims_count; i++) {
    const auto inc = SafeInt<int64_t>(reshaped_pad[i]) * output_pitches[i];
    align_skip += inc;
  }

  // Validate coverage: pre + copy + post == total
  SafeInt<size_t> copy_elems = 1;
  for (size_t i = 0, lim = effective_input_extents.size(); i < lim; ++i) {
    // All extents are positive here due to the no_data_to_copy check above
    copy_elems *= effective_input_extents[i];
  }

  const size_t prepad_elems = align_skip;
  const size_t postpad_elems = SafeInt<size_t>(total_output_elems) - prepad_elems - copy_elems;
  ORT_RETURN_IF_ERROR(PadBase::ValidateTotalElementsCoverage(
      total_output_elems, prepad_elems, copy_elems, postpad_elems));

  TensorShape input_shape(reshaped_input_dims);
  SliceIterator<T> input(input_tensor, input_shape, input_starts, effective_input_extents, {});

  ExtentAxisCounters input_counters(effective_input_extents);

  switch (mode) {
    case Mode::Constant:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (input_counters) {
        output += align_skip;
        {
          T* axis_start = output;
          // Compute the actual number of data elements to copy on the innermost axis (after cropping).
          const size_t inner_extent = onnxruntime::narrow<size_t>(effective_input_extents[inner_axis]);

          // Copy innermost block. IMPORTANT: do not rely on the returned 'output' to be end-of-the extent.
          ORT_IGNORE_RETURN_VALUE(input.CopyInnermostAxisSolitaryInnerStep(output));

          const SafeInt<size_t> pre_pad = reshaped_pad[inner_axis];
          const SafeInt<size_t> post_pad = reshaped_pad[inner_axis + new_dims_count];
          if (pre_pad > 0) {
            /// Pre-pad(innermost) retro-fill remains valid(write before row_start).
            PadAxisConstant(sink, axis_start - static_cast<size_t>(pre_pad), value, pre_pad);
          }
          if (post_pad > 0) {
            PadAxisConstant(sink, axis_start + inner_extent, value, post_pad);
          }
          output = axis_start + inner_extent + static_cast<size_t>(post_pad);
          align_skip = pre_pad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = onnxruntime::narrow<std::ptrdiff_t>(output_pitches[input_counters.Axis()]);
          T* axis_start = output - inner_pitch * effective_input_extents[input_counters.Axis()];
          const SafeInt<size_t> pre_pad = reshaped_pad[input_counters.Axis()];
          const SafeInt<size_t> post_pad = reshaped_pad[input_counters.Axis() + new_dims_count];
          if (pre_pad > 0) {
            PadAxisConstant(sink, axis_start - static_cast<ptrdiff_t>(pre_pad * inner_pitch), value, pre_pad * inner_pitch);
          }
          if (post_pad > 0) {
            PadAxisConstant(sink, output, value, post_pad * inner_pitch);
          }
          output += inner_pitch * post_pad;
          align_skip += inner_pitch * pre_pad;
        }
      }
      break;

    case Mode::Edge:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (input_counters) {
        output += align_skip;
        {
          const SafeInt<size_t> inner_extent = effective_input_extents[inner_axis];
          T* axis_start = output;
          T* axis_end = axis_start + onnxruntime::narrow<ptrdiff_t>(inner_extent);
          output = input.CopyInnermostAxisSolitaryInnerStep(output);

          const SafeInt<size_t> pre_pad = reshaped_pad[inner_axis];
          const SafeInt<size_t> post_pad = reshaped_pad[inner_axis + new_dims_count];
          if (inner_no_pad_size == 1) {
            if (pre_pad > 0) {
              PadAxisConstant(sink, axis_start - static_cast<size_t>(pre_pad), *axis_start, pre_pad);
            }
            if (post_pad > 0) {
              PadAxisConstant(sink, output, *(output - 1), post_pad);
            }
          } else {
            // When inner_most axis(es) do not need pad, above PadAxisConstant() do not fit for Edge mode.
            // Also general loop below after handling first pad axis with non-pad axis works fine.
            if (pads[inner_axis] > 0) {
              PadAxis(sink, axis_start - static_cast<size_t>(pre_pad), axis_start, 1, -ptrdiff_t(inner_no_pad_size), inner_no_pad_size,
                      onnxruntime::narrow<size_t>(pads[inner_axis]));
            }
            if (pads[inner_axis + data_rank] > 0) {
              PadAxis(sink, output, output - inner_no_pad_size, 1, -ptrdiff_t(inner_no_pad_size), inner_no_pad_size,
                      onnxruntime::narrow<size_t>(pads[inner_axis + data_rank]));
            }
          }
          output = axis_end + static_cast<size_t>(post_pad);
          align_skip = pre_pad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = onnxruntime::narrow<std::ptrdiff_t>(output_pitches[input_counters.Axis()]);
          T* axis_start = output - inner_pitch * effective_input_extents[input_counters.Axis()];
          const SafeInt<size_t> pre_pad = reshaped_pad[input_counters.Axis()];
          const SafeInt<size_t> post_pad = reshaped_pad[input_counters.Axis() + new_dims_count];
          if (pre_pad > 0) {
            PadAxis(sink, axis_start - static_cast<size_t>(pre_pad) * inner_pitch, axis_start, 1, -inner_pitch, inner_pitch,
                    pre_pad);
          }
          if (post_pad > 0) {
            PadAxis(sink, output, output - inner_pitch, 1, -inner_pitch, inner_pitch, post_pad);
          }
          output += inner_pitch * post_pad;
          align_skip += inner_pitch * pre_pad;
        }
      }
      break;

    case Mode::Reflect:
    case Mode::Wrap:
      // Loop over the output tensor, writing out padding between the blocks of copied data
      // On loop entry, 'pad' is already set to the first continuous block of padding, and
      // after every pass through the inner loop it gets set to the next continuous pad size.
      while (input_counters) {
        output += align_skip;
        {
          T* axis_start = output;
          output = input.CopyInnermostAxisSolitaryInnerStep(output);

          const SafeInt<size_t> pre_pad = reshaped_pad[inner_axis];
          const SafeInt<size_t> post_pad = reshaped_pad[inner_axis + new_dims_count];
          if (inner_no_pad_size == 1) {
            if (mode == Mode::Reflect) {
              if (pre_pad > 0) {
                PadInnermostAxis(sink, axis_start - static_cast<size_t>(pre_pad),
                                 axis_start + static_cast<size_t>(pre_pad), -1 /* inputDelta */, pre_pad);
              }
              if (post_pad > 0) {
                PadInnermostAxis(sink, output, output - 2, -1 /* inputDelta */, post_pad);
              }
            } else {
              if (pre_pad > 0) {
                PadInnermostAxis(sink, axis_start - static_cast<size_t>(pre_pad),
                                 output - static_cast<size_t>(pre_pad), 1 /* inputDelta */, pre_pad);
              }
              if (post_pad > 0) {
                PadInnermostAxis(sink, output, axis_start, 1 /* inputDelta */, post_pad);
              }
            }
          } else {
            // When inner_most axis(es) do not need pad, Above PadInnermostAxis() do not fit for Reflect mode.
            if (mode == Mode::Reflect) {
              PadAxis(sink,
                      axis_start - static_cast<size_t>(pre_pad),
                      axis_start + static_cast<size_t>(pre_pad),
                      1,
                      -ptrdiff_t(inner_no_pad_size * 2),
                      inner_no_pad_size,
                      onnxruntime::narrow<size_t>(pads[inner_axis]));
              PadAxis(sink,
                      output,
                      output - 2 * inner_no_pad_size,
                      1,
                      -ptrdiff_t(inner_no_pad_size * 2),
                      inner_no_pad_size,
                      onnxruntime::narrow<size_t>(pads[inner_axis + data_rank]));
            } else {
              PadAxis(sink,
                      axis_start - static_cast<size_t>(pre_pad),
                      output - pads[inner_axis] * inner_no_pad_size,
                      1,
                      0,
                      inner_no_pad_size,
                      onnxruntime::narrow<size_t>(pads[inner_axis]));
              PadAxis(sink,
                      output,
                      axis_start,
                      1,
                      0,
                      inner_no_pad_size,
                      onnxruntime::narrow<size_t>(pads[inner_axis + data_rank]));
            }
          }
          output += post_pad;
          align_skip = pre_pad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = onnxruntime::narrow<std::ptrdiff_t>(output_pitches[input_counters.Axis()]);
          T* axis_start = output - inner_pitch * effective_input_extents[input_counters.Axis()];
          const SafeInt<size_t> pre_pad = reshaped_pad[input_counters.Axis()];
          const SafeInt<size_t> post_pad = reshaped_pad[input_counters.Axis() + new_dims_count];
          if (mode == Mode::Reflect) {
            PadAxis(sink,
                    axis_start - static_cast<size_t>(pre_pad) * inner_pitch,
                    axis_start + static_cast<size_t>(pre_pad) * inner_pitch,
                    1,
                    -inner_pitch * 2,
                    inner_pitch,
                    pre_pad);
            PadAxis(sink,
                    output,
                    output - 2 * inner_pitch,
                    1,
                    -inner_pitch * 2,
                    inner_pitch,
                    post_pad);
          } else {
            PadAxis(sink,
                    axis_start - static_cast<size_t>(pre_pad) * inner_pitch,
                    output - static_cast<size_t>(pre_pad) * inner_pitch,
                    1,
                    0,
                    inner_pitch,
                    pre_pad);
            PadAxis(sink,
                    output,
                    axis_start,
                    1,
                    0,
                    inner_pitch,
                    post_pad);
          }
          output += inner_pitch * post_pad;
          align_skip += inner_pitch * pre_pad;
        }
      }
      break;
  }

  return Status::OK();
}

union PadValue {
  uint64_t u64;
  uint32_t u32;
  uint8_t u8;
  double f64;
  float f32;
};

static PadValue PadValueFromFloat(float value, MLDataType data_type) {
  PadValue result;
  if (data_type == DataTypeImpl::GetType<float>()) {
    result.f32 = value;
  } else if (data_type == DataTypeImpl::GetType<double>()) {
    result.f64 = value;
  } else {
    ORT_THROW("Unsupported input data type of ", data_type);
  }
  return result;
}

Status Pad::Compute(OpKernelContext* ctx) const {
  const Tensor& input_tensor = *ctx->Input<Tensor>(0);
  MLDataType data_type = input_tensor.DataType();
  const auto element_size = data_type->Size();
  PadsVector pads;
  PadsVector slices;
  const PadsVector* pads_to_use;
  const PadsVector* slices_to_use;
  PadValue value;

  // kOnnxDomain Pad opset >= 11 (Or) kMsDomain opset == 1
  if (is_dynamic_) {
    size_t data_rank = input_tensor.Shape().NumDimensions();

    const Tensor& pads_tensor = *ctx->Input<Tensor>(1);
    auto pads_tensor_dims = pads_tensor.Shape().GetDims();
    ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
                "Pads tensor should be a 1D tensor of shape [2 * num_axes] "
                "or a 2D tensor of shape [1, 2 * num_axes]");

    const auto pads_data = pads_tensor.DataAsSpan<int64_t>();

    // Compute Pads by applying axes if specified otherwise copy the supplied pads.
    PadBase::ComputePads(*ctx, data_rank, pads_data, pads);

    // Separate out any negative pads into the slices array
    PadBase::SeparateNegativeToSlices(pads, slices);

    value.u64 = 0U;
    const Tensor* value_tensor = ctx->Input<Tensor>(2);
    if (nullptr != value_tensor) {
      ORT_ENFORCE(value_tensor->DataType() == data_type &&
                      value_tensor->Shape().Size() == 1,
                  "Value tensor should be a 1D tensor of size 1 with the same type as that of the input tensor");
      const void* value_data = value_tensor->DataRaw();
      switch (element_size) {
        case sizeof(uint32_t):
          value.u32 = reinterpret_cast<const uint32_t*>(value_data)[0];
          break;
        case sizeof(uint64_t):
          value.u64 = reinterpret_cast<const uint64_t*>(value_data)[0];
          break;
        case sizeof(uint8_t):
          value.u8 = reinterpret_cast<const uint8_t*>(value_data)[0];
          break;
        default:
          ORT_THROW("Unsupported input data type of ", data_type);
      }
    }

    pads_to_use = &pads;
    slices_to_use = &slices;
  } else {
    // kOnnxDomain Pad opset < 11
    // In the earlier opset versions of Pad, the type for 'value' attribute was always float,
    // irrespective of the data type of the actual input to be padded
    value = PadValueFromFloat(value_, data_type);
    pads_to_use = &pads_;
    slices_to_use = &slices_;
  }

  Status pad_status{};
  switch (element_size) {
    case sizeof(uint32_t):
      pad_status = PadImpl<uint32_t>(ctx, *pads_to_use, *slices_to_use, mode_, value.u32);
      break;
    case sizeof(uint64_t):
      pad_status = PadImpl<uint64_t>(ctx, *pads_to_use, *slices_to_use, mode_, value.u64);
      break;
    case sizeof(uint8_t):
      pad_status = PadImpl<uint8_t>(ctx, *pads_to_use, *slices_to_use, mode_, value.u8);
      break;
    default:
      pad_status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input data type of ", data_type);
      break;
  }
  return pad_status;
}
};  // namespace onnxruntime

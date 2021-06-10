// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/pad.h"

#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"
#include "core/util/math.h"

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

ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 11, Input, 0, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Pad, 13, Input, 0, int32_t, int64_t);
}  // namespace op_kernel_type_control

using Pad2Types = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 2, Input, 0);
using EnabledPad2Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 2, Input, 0);
using Pad11Types = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 11, Input, 0);
using EnabledPad11Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 11, Input, 0);
using Pad13Types = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 13, Input, 0);
using EnabledPad13Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Pad, 13, Input, 0);

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
        BuildKernelDefConstraintsFromTypeList<Pad2Types>(),
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
        BuildKernelDefConstraintsFromTypeList<Pad11Types>(),
        BuildKernelDefConstraintsFromTypeList<EnabledPad11Types>()),
    Pad);

ONNX_CPU_OPERATOR_KERNEL(
    Pad,
    13,
    KernelDefBuilder()
        .TypeConstraint(
            "T",
            BuildKernelDefConstraintsFromTypeList<Pad13Types>(),
            BuildKernelDefConstraintsFromTypeList<EnabledPad13Types>()),
    Pad);

// This is the general padding method to n-dimensionally do edge or reflection padding (based on the inputDelta values)
template <typename T>
static void PadAxis(T* output, T* input, ptrdiff_t input_delta, ptrdiff_t input_pitch,
                    size_t block_size, size_t block_count) {
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
  if (size == 1) {
    *output = constant;
  } else if (size == 2) {
    *output = constant;
    *(output + 1) = constant;
  } else {
    // This would be faster with SSE instructions.
    // That would mean to have an implementation for each type (uint8, uint32, uint64).
    T* end = output + size;
    for (; output != end;)
      *output++ = constant;
  }
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
    auto* output = reinterpret_cast<T*>(output_tensor.MutableDataRaw());
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
    inner_size *= static_cast<size_t>(input_dims[inner_axis]);

    if (inner_axis == 0)
      break;

    // Break on first Axis that has padding
    if (!(pads[inner_axis] == 0 && pads[inner_axis + dims_count] == 0 &&
          slices[inner_axis] == 0 && slices[inner_axis + dims_count] == 0))
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
  std::copy(src_pad.begin() + src_dim_count, src_pad.begin() + src_dim_count + inner_axis,
            reshaped_pad.begin() + new_dim_count);

  // Flatten inner axis.
  reshaped_pad[inner_axis] = src_pad[inner_axis] * inner_no_pad_size;
  reshaped_pad[inner_axis + new_dim_count] = src_pad[inner_axis + src_dim_count] * inner_no_pad_size;
}

template <typename T>
static Status PadImpl(OpKernelContext* ctx,
                      const std::vector<int64_t>& pads,
                      const std::vector<int64_t>& slices,
                      const Mode& mode,
                      T value) {
  if (!utils::HasTypeWithSameSize<AllEnabledPadTypes, T>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input data type not supported in this build.");
  }

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
  size_t inner_no_pad_size = output_dims[inner_axis] > 0
                                 ? reshaped_input_dims[inner_axis] / output_dims[inner_axis]
                                 : 0;
  std::vector<int64_t> reshaped_pad(2 * new_dims_count), reshaped_slice(2 * new_dims_count);
  ReshapePads(pads, data_rank, new_dims_count, inner_no_pad_size, reshaped_pad);
  ReshapePads(slices, data_rank, new_dims_count, inner_no_pad_size, reshaped_slice);

  std::vector<int64_t> reshaped_output_dims = reshaped_input_dims;
  std::vector<int64_t> input_starts;
  std::vector<int64_t> input_extents;

  // Calculate output dimensions, and handle any negative padding
  input_starts.reserve(new_dims_count);
  input_extents.reserve(new_dims_count);
  for (size_t i = 0; i < new_dims_count; i++) {
    input_starts.push_back(-1 * reshaped_slice[i]);
    input_extents.push_back(reshaped_input_dims[i] + reshaped_slice[i] + reshaped_slice[i + new_dims_count]);
    reshaped_output_dims[i] += reshaped_pad[i] + reshaped_pad[i + new_dims_count] +
                               reshaped_slice[i] + reshaped_slice[i + new_dims_count];
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
  auto* output = reinterpret_cast<T*>(output_tensor.MutableDataRaw());

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
          if (inner_no_pad_size == 1) {
            PadAxisConstant(axisStart - prePad, *axisStart, prePad);
            PadAxisConstant(output, *(output - 1), postPad);
          } else {
            // When inner_most axis(es) do not need pad, above PadAxisConstant() do not fit for Edge mode.
            // Also general loop below after handling first pad axis with non-pad axis works fine.
            PadAxis(axisStart - prePad, axisStart, 1, -ptrdiff_t(inner_no_pad_size), inner_no_pad_size, pads[inner_axis]);
            PadAxis(output, output - inner_no_pad_size, 1, -ptrdiff_t(inner_no_pad_size), inner_no_pad_size, pads[inner_axis + data_rank]);
          }
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
          if (inner_no_pad_size == 1) {
            PadInnermostAxis(axisStart - prePad, axisStart + prePad, -1 /* inputDelta */, prePad);
            PadInnermostAxis(output, output - 2, -1 /* inputDelta */, postPad);
          } else {
            // When inner_most axis(es) do not need pad, Above PadInnermostAxis() do not fit for Reflect mode.
            PadAxis(axisStart - prePad, axisStart + prePad, 1, -ptrdiff_t(inner_no_pad_size * 2), inner_no_pad_size, pads[inner_axis]);
            PadAxis(output, output - 2 * inner_no_pad_size, 1, -ptrdiff_t(inner_no_pad_size * 2), inner_no_pad_size, pads[inner_axis + data_rank]);
          }
          output += postPad;
          alignSkip = prePad;
        }
        // Calculate the size of the next block of padding (skipping over the innermost axis since that's already done)
        while (input_counters.Increment()) {
          ptrdiff_t inner_pitch = output_pitches[input_counters.Axis()];
          T* axisStart = output - inner_pitch * input_extents[input_counters.Axis()];
          int64_t prePad = reshaped_pad[input_counters.Axis()];
          int64_t postPad = reshaped_pad[input_counters.Axis() + new_dims_count];
          PadAxis(axisStart - prePad * inner_pitch, axisStart + prePad * inner_pitch, 1, -inner_pitch * 2,
                  inner_pitch, prePad);
          PadAxis(output, output - 2 * inner_pitch, 1, -inner_pitch * 2, inner_pitch, postPad);
          output += inner_pitch * postPad;
          alignSkip += inner_pitch * prePad;
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
  std::vector<int64_t> pads;
  std::vector<int64_t> slices;
  const std::vector<int64_t>* pads_to_use;
  const std::vector<int64_t>* slices_to_use;
  PadValue value;

  // kOnnxDomain Pad opset >= 11 (Or) kMsDomain opset == 1
  if (is_dynamic_) {
    size_t data_rank = input_tensor.Shape().NumDimensions();

    const Tensor& pads_tensor = *ctx->Input<Tensor>(1);
    const std::vector<int64_t>& pads_tensor_dims = pads_tensor.Shape().GetDims();
    ORT_ENFORCE(pads_tensor.IsDataType<int64_t>(),
                "Pads tensor should be an INT64 tensor");
    ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
                "Pads tensor should be a 1D tensor of shape [2 * input_rank] "
                "or a 2D tensor of shape [1, 2 * input_rank]");

    const int64_t* pads_tensor_raw_data = pads_tensor.template Data<int64_t>();
    size_t pads_size = static_cast<size_t>(pads_tensor.Shape().Size());
    ORT_ENFORCE(pads_size == 2 * data_rank,
                "Pads tensor size should be equal to twice the input dimension count ");

    pads.reserve(2 * data_rank);
    for (size_t i = 0; i < pads_size; ++i) {
      pads.push_back(pads_tensor_raw_data[i]);
    }

    // Separate out any negative pads into the slices array
    slices = std::vector<int64_t>(pads.size(), 0);
    for (size_t index = 0; index < pads.size(); index++) {
      if (pads[index] < 0) {
        slices[index] = pads[index];
        pads[index] = 0;
      }
    }

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

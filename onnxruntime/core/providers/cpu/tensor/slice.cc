// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"

#include <limits>
#include <unordered_map>

#include "core/common/narrow.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

namespace op_kernel_type_control {
// we're using one set of types for all opsets
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 0,
    element_type_lists::All);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 0, int32_t, int64_t);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 1, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 1, int32_t, int64_t);
}  // namespace op_kernel_type_control

namespace {
using EnabledDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                        Slice, Input, 0);
using EnabledIndicesTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                           Slice, Input, 1);
}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    1, 9,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>()),
    Slice1);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    10, 10,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

ONNX_CPU_OPERATOR_KERNEL(
    Slice,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

// Coalesce contiguous non-slice dimensions into a single dimension.
// Set p_flattened_input_dims_ and p_flattened_output_dims_ to nullptr if nothing coalesced.
// Updates starts and steps to match the new dimensions.
// e.g. if input shape is { 2, 2, 2, 2, 2 }, output shape is { 2, 2, 1, 2, 2 },
// and the 'steps' value for all dims is 1 except dim-2, then the input shape is coalesced to { 4, 2, 4 }
// and the output shape is coalesced to { 4, 1, 4 }.
Status SliceBase::FlattenOutputDims(gsl::span<const int64_t> input_dimensions, gsl::span<const int64_t> output_dims,
                                    TensorShapeVector& starts, TensorShapeVector& ends, TensorShapeVector& steps,
                                    TensorShapeVector*& p_flattened_input_dims, TensorShapeVector*& p_flattened_output_dims) {
  size_t cur = 0;
  size_t nxt = 0;
  while (true) {
    // Skip all leading slicing dims.
    while (nxt < starts.size() && (steps[nxt] != 1 || input_dimensions[nxt] != output_dims[nxt])) {
      p_flattened_input_dims->emplace_back(input_dimensions[nxt]);
      p_flattened_output_dims->emplace_back(output_dims[nxt]);
      starts[cur] = starts[nxt];
      ends[cur] = ends[nxt];
      steps[cur] = steps[nxt];
      ++cur;
      ++nxt;
    }
    if (nxt == starts.size()) {
      break;
    }
    // Coalesce contiguous non-slicing dims.
    int64_t running_size = 1;
    while (nxt < starts.size() && steps[nxt] == 1 && input_dimensions[nxt] == output_dims[nxt]) {
      running_size *= input_dimensions[nxt];
      ++nxt;
    }
    if (running_size > 1) {
      p_flattened_input_dims->emplace_back(running_size);
      p_flattened_output_dims->emplace_back(running_size);
      starts[cur] = 0LL;
      ends[cur] = running_size;
      steps[cur] = 1LL;
      ++cur;
    }
  }

  // No actual slice dim, and all dims are size 1.
  if (cur == 0) {
    p_flattened_input_dims->emplace_back(1LL);
    p_flattened_output_dims->emplace_back(1LL);
    starts[cur] = 0LL;
    ends[cur] = 1LL;
    steps[cur] = 1LL;
    ++cur;
  }

  if (p_flattened_output_dims->size() == output_dims.size()) {
    p_flattened_input_dims->clear();
    p_flattened_output_dims->clear();
    p_flattened_input_dims = nullptr;
    p_flattened_output_dims = nullptr;
  } else {
    starts.resize(cur);
    ends.resize(cur);
    steps.resize(cur);
  }

  return Status::OK();
}

// Slice V1-9 & DynamicSlice
Status SliceBase::PrepareForCompute(gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
                                    gsl::span<const int64_t> raw_axes,
                                    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, compute_metadata));
  ORT_RETURN_IF_ERROR(FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                                        compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_input_dims_,
                                        compute_metadata.p_flattened_output_dims_));
  return Status::OK();
}

// DynamicSlice & Slice V10
Status SliceBase::PrepareForCompute(gsl::span<const int64_t> raw_starts, gsl::span<const int64_t> raw_ends,
                                    gsl::span<const int64_t> raw_axes, gsl::span<const int64_t> raw_steps,
                                    SliceOp::PrepareForComputeMetadata& compute_metadata) {
  ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));
  ORT_RETURN_IF_ERROR(FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                                        compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_input_dims_,
                                        compute_metadata.p_flattened_output_dims_));

  return Status::OK();
}
namespace {
template <typename T>
void CopyData(const Tensor& start_tensor,
              const Tensor& ends_tensor,
              const Tensor* axes_tensor,
              const Tensor* steps_tensor,
              TensorShapeVector& input_starts,
              TensorShapeVector& input_ends,
              TensorShapeVector& input_axes,
              TensorShapeVector& input_steps) {
  auto start_data = start_tensor.DataAsSpan<T>();
  std::copy(start_data.begin(), start_data.end(), std::back_inserter(input_starts));
  auto ends_data = ends_tensor.DataAsSpan<T>();
  std::copy(ends_data.begin(), ends_data.end(), std::back_inserter(input_ends));
  if (nullptr != axes_tensor) {
    auto axes_data = axes_tensor->DataAsSpan<T>();
    std::copy(axes_data.begin(), axes_data.end(), std::back_inserter(input_axes));
  }
  // Slice V10
  if (nullptr != steps_tensor) {
    auto steps_data = steps_tensor->DataAsSpan<T>();
    std::copy(steps_data.begin(), steps_data.end(), std::back_inserter(input_steps));
  }
}
}  // namespace

// Slice V10 & DynamicSlice
Status SliceBase::FillVectorsFromInput(const Tensor& start_tensor,
                                       const Tensor& ends_tensor,
                                       const Tensor* axes_tensor,
                                       const Tensor* steps_tensor,
                                       TensorShapeVector& input_starts,
                                       TensorShapeVector& input_ends,
                                       TensorShapeVector& input_axes,
                                       TensorShapeVector& input_steps) {
  ORT_RETURN_IF_NOT(start_tensor.Shape().NumDimensions() == 1, "Starts must be a 1-D array");
  ORT_RETURN_IF_NOT(ends_tensor.Shape().NumDimensions() == 1, "Ends must be a 1-D array");
  ORT_RETURN_IF_NOT(start_tensor.Shape() == ends_tensor.Shape(), "Starts and ends shape mismatch");
  ORT_RETURN_IF_NOT(nullptr == axes_tensor || start_tensor.Shape() == axes_tensor->Shape(),
                    "Starts and axes shape mismatch");
  ORT_RETURN_IF_NOT(nullptr == steps_tensor || start_tensor.Shape() == steps_tensor->Shape(),
                    "Starts and steps shape mismatch");

  const auto size = narrow<size_t>(start_tensor.Shape().Size());
  input_starts.reserve(size);
  input_ends.reserve(size);
  if (nullptr != axes_tensor)
    input_axes.reserve(size);
  // Slice V10
  if (nullptr != steps_tensor)
    input_steps.reserve(size);

  // check for type reduction of supported indices types
  constexpr bool int32_enabled = utils::HasType<EnabledIndicesTypes, int32_t>();
  constexpr bool int64_enabled = utils::HasType<EnabledIndicesTypes, int64_t>();

  if (int32_enabled && start_tensor.IsDataType<int32_t>()) {
    CopyData<int32_t>(start_tensor, ends_tensor, axes_tensor, steps_tensor, input_starts, input_ends, input_axes, input_steps);
  } else if (int64_enabled && start_tensor.IsDataType<int64_t>()) {
    CopyData<int64_t>(start_tensor, ends_tensor, axes_tensor, steps_tensor, input_starts, input_ends, input_axes, input_steps);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Data type for starts and ends inputs' is not supported in this build. Got ",
                           start_tensor.DataType());
  }

  return Status::OK();
}

template <typename T>
static Status SliceImpl(OpKernelContext* ctx,
                        const Tensor& input_tensor,
                        SliceOp::PrepareForComputeMetadata& compute_metadata) {
  TensorShape output_shape(compute_metadata.output_dims_);
  auto& output_tensor = *ctx->Output(0, output_shape);

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  // use MutableDataRaw as actual data type in tensor may not match as we templatize on data size
  T* output = reinterpret_cast<T*>(output_tensor.MutableDataRaw());
  const auto* output_end = output + output_tensor.Shape().Size();

  auto create_output = [&output, &output_end](SliceIterator<T>& slice_input_iterator) {
    while (output < output_end) {
      output = slice_input_iterator.CopyContiguousInnermostAxes(output);
    }

    ORT_ENFORCE(output == output_end);
  };

  if (compute_metadata.p_flattened_input_dims_) {
    // If we were able to coalesce the input and output shapes, use the new shapes.
    auto input_iterator2 =
        SliceIterator<T>(input_tensor, TensorShape(compute_metadata.flattened_input_dims_), compute_metadata.starts_,
                         compute_metadata.flattened_output_dims_, compute_metadata.steps_);
    create_output(input_iterator2);
  } else {
    auto input_iterator2 = SliceIterator<T>(input_tensor, compute_metadata.starts_, compute_metadata.output_dims_,
                                            compute_metadata.steps_);
    create_output(input_iterator2);
  }

  return Status::OK();
}

template <typename EnabledTypes, typename T>
static inline bool CallSliceImplIfEnabled(OpKernelContext* ctx,
                                          const Tensor& input_tensor,
                                          SliceOp::PrepareForComputeMetadata& compute_metadata,
                                          Status& status) {
  constexpr bool enabled = utils::HasTypeWithSameSize<EnabledTypes, T>();
  if constexpr (enabled) {
    status = SliceImpl<T>(ctx, input_tensor, compute_metadata);
  }

  return enabled;
}

Status SliceBase::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  const auto& input_tensor = *input_tensor_ptr;
  const auto input_dimensions = input_tensor.Shape().GetDims();

  if (input_dimensions.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");
  }

  SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);

  // Slice V10 & DynamicSlice
  if (dynamic_) {
    TensorShapeVector input_starts;
    TensorShapeVector input_ends;
    TensorShapeVector input_axes;
    TensorShapeVector input_steps;
    ORT_RETURN_IF_ERROR(FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2),
                                             ctx->Input<Tensor>(3), ctx->Input<Tensor>(4),
                                             input_starts, input_ends,
                                             input_axes, input_steps));

    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));
  }
  // Slice V1-9
  else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_, compute_metadata));
  }

  Status status = Status::OK();

  bool supported = false;
  if (input_tensor.IsDataTypeString()) {
    if (utils::HasType<EnabledDataTypes, std::string>()) {
      supported = true;
      status = SliceImpl<std::string>(ctx, input_tensor, compute_metadata);
    }
  } else {
    const auto element_size = input_tensor.DataType()->Size();
    // call SliceImpl
    switch (element_size) {
      case sizeof(uint32_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint32_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint64_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint64_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint16_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint16_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint8_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint8_t>(ctx, input_tensor, compute_metadata, status);
        break;
      default:
        // leave 'supported' as false
        break;
    }
  }

  if (!supported) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input data type of ", input_tensor.DataType());
  }

  return status;
}

}  // namespace onnxruntime

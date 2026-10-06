// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/cuda/tensor/slice_impl.h"

#include <cstdint>
#include <limits>

namespace onnxruntime {
namespace cuda {
// this really doesn't need to be a typed registration as the indices come from attributes and can only be int64.
// leaving as in maintain original incorrect registration setup (pre 02/2022).
#define REGISTER_VERSIONED_TYPED_SLICE(TIND)                             \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                               \
      Slice,                                                             \
      kOnnxDomain,                                                       \
      1, 9,                                                              \
      TIND,                                                              \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), \
      Slice<false>);

REGISTER_VERSIONED_TYPED_SLICE(int64_t)

#define REGISTER_V10_TYPED_SLICE(TIND)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      10, 10,                                                           \
      TIND,                                                             \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<true>);

REGISTER_V10_TYPED_SLICE(int32_t)
REGISTER_V10_TYPED_SLICE(int64_t)

#define REGISTER_V12_TYPED_SLICE(TIND)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      11, 12,                                                           \
      TIND,                                                             \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<true>);

REGISTER_V12_TYPED_SLICE(int32_t)
REGISTER_V12_TYPED_SLICE(int64_t)

#define REGISTER_V13_TYPED_SLICE(TIND)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      13,                                                               \
      TIND,                                                             \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 4)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<true>);

REGISTER_V13_TYPED_SLICE(int32_t)
REGISTER_V13_TYPED_SLICE(int64_t)

static Status SliceImpCore(cudaStream_t stream,
                           const void* input_data, void* output_data,
                           size_t element_size, size_t dimension_count,
                           const TArray<int64_t>& starts_buffer, const TArray<int64_t>& steps_buffer,
                           const TArray<int64_t>& input_strides, const TArray<fast_divmod>& output_strides,
                           const TensorShape& output_shape) {
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  return SliceImpl(stream,
                   element_size,
                   gsl::narrow_cast<int32_t>(dimension_count),
                   starts_buffer,
                   steps_buffer,
                   input_strides,
                   output_strides,
                   input_data,
                   output_data,
                   output_shape.Size());
}

static bool TryConvertInt64ToSizeT(int64_t value, size_t& converted_value) {
  if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) {
    return false;
  }

  converted_value = static_cast<size_t>(value);
  return true;
}

static bool TryMultiply(size_t lhs, size_t rhs, size_t& product) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }

  product = lhs * rhs;
  return true;
}

// Detect whether a step-1 slice selects a single contiguous block of the input tensor.
// This is the case when the slice only trims leading dimensions: scanning from the right,
// every axis is fully included until the first trimmed ("pivot") axis, and every axis to the
// left of the pivot selects exactly one element. When true, the output is the contiguous
// sub-region input_ptr + offset_in_elements and can be produced with a single device-to-device
// memcpy instead of the per-element slice kernel.
static bool TryComputeContiguousSliceOffset(gsl::span<const int64_t> input_dims,
                                            gsl::span<const int64_t> output_dims,
                                            const TArray<int64_t>& starts_buffer,
                                            const TArray<int64_t>& steps_buffer,
                                            size_t& offset_in_elements) {
  const int32_t rank = static_cast<int32_t>(input_dims.size());
  if (rank == 0 || static_cast<int32_t>(output_dims.size()) != rank ||
      starts_buffer.Size() != rank || steps_buffer.Size() != rank) {
    return false;
  }

  // Only step-1 slices can be contiguous.
  for (int32_t i = 0; i < rank; ++i) {
    if (steps_buffer[i] != 1) {
      return false;
    }
  }

  // Find the first trimmed axis scanning from the right (the pivot).
  int32_t pivot = -1;
  for (int32_t i = rank - 1; i >= 0; --i) {
    if (output_dims[i] != input_dims[i]) {
      pivot = i;
      break;
    }
  }

  // Every axis to the left of the pivot must select exactly one element, otherwise the
  // selected region is split into multiple non-adjacent blocks.
  for (int32_t i = 0; i < pivot; ++i) {
    if (output_dims[i] != 1) {
      return false;
    }
  }

  // Compute the offset of the first selected element (in elements).
  size_t offset = 0;
  size_t stride = 1;
  for (int32_t i = rank - 1; i >= 0; --i) {
    size_t start = 0;
    size_t input_dim = 0;
    if (!TryConvertInt64ToSizeT(starts_buffer[i], start) ||
        !TryConvertInt64ToSizeT(input_dims[i], input_dim) ||
        start > input_dim) {
      return false;
    }

    size_t offset_increment = 0;
    if (!TryMultiply(start, stride, offset_increment) ||
        offset > std::numeric_limits<size_t>::max() - offset_increment ||
        !TryMultiply(stride, input_dim, stride)) {
      return false;
    }

    offset += offset_increment;
  }
  offset_in_elements = offset;
  return true;
}

namespace SliceCuda {

static Status ComputeSliceStrides(const TensorShape& input_shape, TArray<int64_t>& input_strides,
                                  TArray<fast_divmod>& output_strides,
                                  SliceOp::PrepareForComputeMetadata& compute_metadata) {
  // If we were able to coalesce the input and output shapes, use the new shapes to compute the strides.
  const auto input_dimensions = input_shape.GetDims();
  size_t rank = compute_metadata.p_flattened_input_dims_ ? compute_metadata.p_flattened_input_dims_->size()
                                                         : input_dimensions.size();
  input_strides.SetSize(gsl::narrow_cast<int32_t>(rank));
  const gsl::span<int64_t> input_strides_span = gsl::make_span(input_strides.Data(), input_strides.Size());
  if (compute_metadata.p_flattened_input_dims_) {
    ORT_ENFORCE(TensorPitches::Calculate(input_strides_span, compute_metadata.flattened_input_dims_));
  } else {
    ORT_ENFORCE(TensorPitches::Calculate(input_strides_span, input_dimensions));
  }

  const auto output_dims =
      gsl::make_span(compute_metadata.p_flattened_output_dims_ != nullptr ? compute_metadata.flattened_output_dims_
                                                                          : compute_metadata.output_dims_);
  TensorPitches original_output_strides(output_dims);
  output_strides.SetSize(gsl::narrow_cast<int32_t>(original_output_strides.size()));
  for (int32_t i = 0, limit = static_cast<int32_t>(original_output_strides.size()); i < limit; ++i) {
    output_strides[i] = fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
  }

  return Status::OK();
}

Status Impl(cudaStream_t stream,
            const void* input_data,
            const TensorShape& input_shape,
            void* output_data,
            SliceOp::PrepareForComputeMetadata& compute_metadata,
            size_t element_size) {
  const auto input_dimensions = input_shape.GetDims();
  size_t dimension_count = input_dimensions.size();

  TArray<int64_t> starts_buffer(compute_metadata.starts_);
  TArray<int64_t> steps_buffer(compute_metadata.steps_);
  TArray<int64_t> input_strides;
  TArray<fast_divmod> output_strides;

  ORT_RETURN_IF_ERROR(ComputeSliceStrides(input_shape, input_strides, output_strides, compute_metadata));

  TensorShape output_shape(compute_metadata.output_dims_);

  ORT_RETURN_IF_ERROR(SliceImpCore(stream,
                                   input_data,
                                   output_data,
                                   element_size,
                                   gsl::narrow_cast<int32_t>(dimension_count),
                                   starts_buffer,
                                   steps_buffer,
                                   input_strides,
                                   output_strides,
                                   output_shape));

  return Status::OK();
}
}  // namespace SliceCuda

template <bool dynamic>
Status Slice<dynamic>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = GetSlicedOrUnslicedTensor(ctx);
  ORT_ENFORCE(nullptr != input_tensor);
  const auto& input_shape = input_tensor->Shape();
  const auto input_dimensions = input_shape.GetDims();
  if (input_dimensions.empty()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");

  SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);

  if (dynamic) {
    TensorShapeVector input_starts, input_ends, input_axes, input_steps;
    ORT_RETURN_IF_ERROR(FillInputVectors(ctx, input_starts, input_ends, input_axes, input_steps));
    ORT_RETURN_IF_ERROR(SliceBase::PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));

  } else {
    ORT_RETURN_IF_ERROR(SliceBase::PrepareForCompute(StartsAttribute(), EndsAttribute(), AxesAttribute(), compute_metadata));
  }

  TensorShape output_shape(compute_metadata.output_dims_);

  TArray<int64_t> starts_buffer(compute_metadata.starts_);
  TArray<int64_t> steps_buffer(compute_metadata.steps_);
  TArray<int64_t> input_strides;
  TArray<fast_divmod> output_strides;

  ORT_RETURN_IF_ERROR(SliceCuda::ComputeSliceStrides(input_shape, input_strides, output_strides, compute_metadata));

  gsl::span<const int64_t> sliced_input_dims = input_dimensions;
  gsl::span<const int64_t> sliced_output_dims = compute_metadata.output_dims_;
  if (compute_metadata.p_flattened_input_dims_) {
    sliced_input_dims = compute_metadata.flattened_input_dims_;
    sliced_output_dims = compute_metadata.flattened_output_dims_;
  }

  // It may seem that we may use `SliceImpCore()` directly, but we need to go through `CallSliceImp()` because
  // `ComputeInternal()` is shared between the inferencing and training kernels and the training kernel overrides
  // `CallSliceImp()`
  ORT_RETURN_IF_ERROR(CallSliceImp(input_tensor->DataType()->Size(), input_dimensions.size(),
                                   sliced_input_dims, sliced_output_dims, starts_buffer,
                                   steps_buffer, input_strides,
                                   output_strides, ctx,
                                   output_shape));

  return Status::OK();
}

template <bool dynamic>
const Tensor* Slice<dynamic>::GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const {
  return ctx->Input<Tensor>(0);
}

template <bool dynamic>
Status Slice<dynamic>::FillInputVectors(OpKernelContext* ctx, TensorShapeVector& input_starts,
                                        TensorShapeVector& input_ends, TensorShapeVector& input_axes,
                                        TensorShapeVector& input_steps) const {
  return SliceBase::FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2), ctx->Input<Tensor>(3),
                                         ctx->Input<Tensor>(4), input_starts, input_ends, input_axes, input_steps);
}

template <bool dynamic>
Status Slice<dynamic>::CallSliceImp(size_t element_size, size_t dimension_count,
                                    gsl::span<const int64_t> sliced_input_dims,
                                    gsl::span<const int64_t> sliced_output_dims,
                                    const TArray<int64_t>& starts_buffer,
                                    const TArray<int64_t>& steps_buffer, const TArray<int64_t>& input_strides,
                                    const TArray<fast_divmod>& output_strides, OpKernelContext* ctx,
                                    const TensorShape& output_shape) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  auto* output_tensor = ctx->Output(0, output_shape);

  const int64_t output_size = output_shape.Size();
  if (output_size == 0) {
    return Status::OK();
  }

  // Fast path: when the slice selects a single contiguous block of the input (only leading
  // dimensions are trimmed and all steps are 1), we can copy the block directly with a single
  // device-to-device memcpy and skip the per-element slice kernel entirely.
  size_t offset_in_elements = 0;
  size_t output_elements = 0;
  size_t input_elements = 0;
  if (TryComputeContiguousSliceOffset(sliced_input_dims, sliced_output_dims,
                                      starts_buffer, steps_buffer, offset_in_elements)) {
    if (TryConvertInt64ToSizeT(output_size, output_elements) &&
        TryConvertInt64ToSizeT(input_tensor->Shape().Size(), input_elements) &&
        offset_in_elements <= input_elements &&
        output_elements <= input_elements - offset_in_elements) {
      size_t byte_offset = 0;
      size_t copy_size = 0;
      if (TryMultiply(offset_in_elements, element_size, byte_offset) &&
          TryMultiply(output_elements, element_size, copy_size)) {
        const char* input_data = static_cast<const char*>(input_tensor->DataRaw()) + byte_offset;
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_tensor->MutableDataRaw(), input_data,
                                             copy_size,
                                             cudaMemcpyDeviceToDevice, Stream(ctx)));
        return Status::OK();
      }
    }
  }

  return SliceImpCore(Stream(ctx),
                      input_tensor->DataRaw(),
                      output_tensor->MutableDataRaw(),
                      element_size,
                      gsl::narrow_cast<int32_t>(dimension_count),
                      starts_buffer,
                      steps_buffer,
                      input_strides,
                      output_strides,
                      output_shape);
}

Status FuncSlice(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const CudaKernel* cuda_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const Tensor* input,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& steps,
    Tensor* output) {
  gsl::span<const int64_t> starts_span = gsl::make_span(starts.data(), starts.size());
  gsl::span<const int64_t> ends_span = gsl::make_span(ends.data(), ends.size());
  gsl::span<const int64_t> axes_span = gsl::make_span(axes.data(), axes.size());
  gsl::span<const int64_t> steps_span = gsl::make_span(steps.data(), steps.size());
  const auto& input_shape = input->Shape();
  const auto input_dimensions = input_shape.GetDims();

  SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);

  ORT_RETURN_IF_ERROR(SliceBase::PrepareForCompute(starts_span, ends_span, axes_span, steps_span, compute_metadata));

  TensorShape output_shape(compute_metadata.output_dims_);

  TArray<int64_t> starts_buffer(compute_metadata.starts_);
  TArray<int64_t> steps_buffer(compute_metadata.steps_);
  TArray<int64_t> input_strides;
  TArray<fast_divmod> output_strides;

  ORT_RETURN_IF_ERROR(SliceCuda::ComputeSliceStrides(input_shape, input_strides, output_strides, compute_metadata));

  ORT_RETURN_IF_ERROR(SliceImpl(
      cuda_kernel->Stream(ctx),
      input->DataType()->Size(),
      gsl::narrow_cast<int32_t>(input_dimensions.size()),
      starts_buffer,
      steps_buffer,
      input_strides,
      output_strides,
      input->DataRaw(),
      output->MutableDataRaw(),
      output_shape.Size()));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/cuda/tensor/slice_impl.h"

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
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));

  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(StartsAttribute(), EndsAttribute(), AxesAttribute(), compute_metadata));
  }

  TensorShape output_shape(compute_metadata.output_dims_);

  TArray<int64_t> starts_buffer(compute_metadata.starts_);
  TArray<int64_t> steps_buffer(compute_metadata.steps_);
  TArray<int64_t> input_strides;
  TArray<fast_divmod> output_strides;

  ORT_RETURN_IF_ERROR(SliceCuda::ComputeSliceStrides(input_shape, input_strides, output_strides, compute_metadata));

  // It may seem that we may use `SliceImpCore()` directly, but we need to go through `CallSliceImp()` because
  // `ComputeInternal()` is shared between the inferencing and training kernels and the training kernel overrides
  // `CallSliceImp()`
  ORT_RETURN_IF_ERROR(CallSliceImp(input_tensor->DataType()->Size(), input_dimensions.size(), starts_buffer,
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
  return FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2), ctx->Input<Tensor>(3),
                              ctx->Input<Tensor>(4), input_starts, input_ends, input_axes, input_steps);
}

template <bool dynamic>
Status Slice<dynamic>::CallSliceImp(size_t element_size, size_t dimension_count, const TArray<int64_t>& starts_buffer,
                                    const TArray<int64_t>& steps_buffer, const TArray<int64_t>& input_strides,
                                    const TArray<fast_divmod>& output_strides, OpKernelContext* ctx,
                                    const TensorShape& output_shape) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  auto* output_tensor = ctx->Output(0, output_shape);

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

  ORT_RETURN_IF_ERROR(
      SliceOp::PrepareForComputeHelper(starts_span, ends_span, axes_span, steps_span, compute_metadata));

  ORT_RETURN_IF_ERROR(SliceBase::FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                                                   compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_input_dims_,
                                                   compute_metadata.p_flattened_output_dims_));

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

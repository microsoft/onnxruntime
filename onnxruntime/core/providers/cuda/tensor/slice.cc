// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/tensor/slice_impl.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_VERSIONED_TYPED_SLICE(TIND)                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      1, 9,                                                             \
      TIND,                                                             \
      kCudaExecutionProvider,                                           \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<false>);

REGISTER_VERSIONED_TYPED_SLICE(int32_t)
REGISTER_VERSIONED_TYPED_SLICE(int64_t)
REGISTER_VERSIONED_TYPED_SLICE(float)

#define REGISTER_V10_TYPED_SLICE(TIND)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      10, 10,                                                           \
      TIND,                                                             \
      kCudaExecutionProvider,                                           \
      KernelDefBuilder()                                                \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(2)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(3)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(4)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<true>);

REGISTER_V10_TYPED_SLICE(int32_t)
REGISTER_V10_TYPED_SLICE(int64_t)
REGISTER_V10_TYPED_SLICE(float)

#define REGISTER_V11_TYPED_SLICE(TIND)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      11,                                                               \
      TIND,                                                             \
      kCudaExecutionProvider,                                           \
      KernelDefBuilder()                                                \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(2)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(3)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(4)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<true>);

REGISTER_V11_TYPED_SLICE(int32_t)
REGISTER_V11_TYPED_SLICE(int64_t)
REGISTER_V11_TYPED_SLICE(float)

static Status SliceImpCore(const void* input_data, void* output_data,
                           size_t element_size, size_t dimension_count,
                           const TArray<int64_t>& starts_buffer, const TArray<int64_t>& steps_buffer,
                           const TArray<int64_t>& input_strides, const TArray<fast_divmod>& output_strides,
                           TensorShape output_shape) {
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  return SliceImpl(element_size,
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

Status Impl(const void* input_data,
            const TensorShape& input_shape,
            void* output_data,
            SliceOp::PrepareForComputeMetadata& prepare_metadata,
            size_t element_size) {
  const auto& input_dimensions = input_shape.GetDims();
  size_t dimension_count = input_dimensions.size();
  // if we are able to flatten the output dims we updated 'starts' and 'steps' to match the smaller number of dims.
  // update dimension_count to match.
  if (prepare_metadata.p_flattened_output_dims) {
    dimension_count = prepare_metadata.p_flattened_output_dims->size();
  }

  TArray<int64_t> starts_buffer(prepare_metadata.starts);
  TArray<int64_t> steps_buffer(prepare_metadata.steps);
  TArray<int64_t> input_strides(gsl::narrow_cast<int32_t>(dimension_count));
  const gsl::span<int64_t> input_strides_span = gsl::make_span(input_strides.Data(), input_strides.Size());
  if (prepare_metadata.p_flattened_output_dims != nullptr) {
    // we were able to flatten the innermost dimensions as they're being copied in full to the output.
    // do the same flattening to the innermost input dimensions in order to calculate pitches that match
    // the flattened output dimensions.
    int64_t aggregated_last_dim = 1;
    for (size_t i = dimension_count - 1, end = input_dimensions.size(); i < end; ++i) {
      aggregated_last_dim *= input_dimensions[i];
    }

    auto flattened_input_dims(input_dimensions);
    flattened_input_dims.resize(dimension_count);
    flattened_input_dims.back() = aggregated_last_dim;
    ORT_ENFORCE(TensorPitches::Calculate(input_strides_span, flattened_input_dims));
  } else {
    ORT_ENFORCE(TensorPitches::Calculate(input_strides_span, input_dimensions));
  }

  TensorPitches original_output_strides(
      prepare_metadata.p_flattened_output_dims != nullptr ? prepare_metadata.flattened_output_dims : prepare_metadata.output_dims);
  TArray<fast_divmod> output_strides(gsl::narrow_cast<int32_t>(original_output_strides.size()));
  for (int32_t i = 0; i < static_cast<int32_t>(original_output_strides.size()); ++i) {
    output_strides[i] = fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
  }

  ORT_RETURN_IF_ERROR(SliceImpCore(input_data,
                                   output_data,
                                   element_size,
                                   gsl::narrow_cast<int32_t>(dimension_count),
                                   starts_buffer,
                                   steps_buffer,
                                   input_strides,
                                   output_strides,
                                   TensorShape(prepare_metadata.output_dims)));

  return Status::OK();
}
}  // namespace SliceCuda

template <bool dynamic>
Status Slice<dynamic>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = GetSlicedOrUnslicedTensor(ctx);
  ORT_ENFORCE(nullptr != input_tensor);
  auto& input_dimensions = input_tensor->Shape().GetDims();
  if (input_dimensions.empty()) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");

  SliceOp::PrepareForComputeMetadata prepare_metadata(input_dimensions);

  if (dynamic) {
    std::vector<int64_t> input_starts, input_ends, input_axes, input_steps;
    FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2), ctx->Input<Tensor>(3),
                         ctx->Input<Tensor>(4), input_starts, input_ends, input_axes, input_steps);
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes,
                                          input_steps, input_dimensions, prepare_metadata));

  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(StartsAttribute(), EndsAttribute(), AxesAttribute(),
                                          input_dimensions, prepare_metadata));
  }

  auto* output_tensor = ctx->Output(0, prepare_metadata.output_dims);

  return SliceCuda::Impl(input_tensor->DataRaw(),
                         input_tensor->Shape(),
                         output_tensor->MutableDataRaw(),
                         prepare_metadata,
                         input_tensor->DataType()->Size());
}

template <bool dynamic>
const Tensor* Slice<dynamic>::GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const {
  return ctx->Input<Tensor>(0);
}

template <bool dynamic>
void Slice<dynamic>::FillInputVectors(OpKernelContext* ctx, std::vector<int64_t>& input_starts,
                                      std::vector<int64_t>& input_ends, std::vector<int64_t>& input_axes,
                                      std::vector<int64_t>& input_steps) const {
  FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2), ctx->Input<Tensor>(3),
                       ctx->Input<Tensor>(4), input_starts, input_ends, input_axes, input_steps);
}

template <bool dynamic>
Status Slice<dynamic>::CallSliceImp(size_t element_size, size_t dimension_count, const TArray<int64_t>& starts_buffer,
                                    const TArray<int64_t>& steps_buffer, const TArray<int64_t>& input_strides,
                                    const TArray<fast_divmod>& output_strides, OpKernelContext* ctx,
                                    TensorShape output_shape) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  auto* output_tensor = ctx->Output(0, output_shape);

  return SliceImpCore(input_tensor->DataRaw(),
                      output_tensor->MutableDataRaw(),
                      element_size,
                      gsl::narrow_cast<int32_t>(dimension_count),
                      starts_buffer,
                      steps_buffer,
                      input_strides,
                      output_strides,
                      output_shape);
}

}  // namespace cuda
}  // namespace onnxruntime

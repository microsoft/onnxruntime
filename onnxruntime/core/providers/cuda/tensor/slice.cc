// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "slice_impl.h"

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
      Slice<TIND, false>);

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
      Slice<TIND, true>);

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
      Slice<TIND, true>);

REGISTER_V11_TYPED_SLICE(int32_t)
REGISTER_V11_TYPED_SLICE(int64_t)
REGISTER_V11_TYPED_SLICE(float)

template <typename Tind, bool dynamic>
Status Slice<Tind, dynamic>::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != input_tensor);
<<<<<<< HEAD
  auto& input_dims = input_tensor->Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  const int32_t rank = gsl::narrow_cast<int32_t>(input_dims.size());
  std::vector<int64_t> starts(rank, 0);
  std::vector<int64_t> steps(rank, 1);
  std::vector<int64_t> output_dims(input_dims);
=======
  const auto& input_dimensions = input_tensor->Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  size_t dimension_count = input_dimensions.size();
  std::vector<int64_t> starts(dimension_count, 0);
  std::vector<int64_t> steps(dimension_count, 1);
  std::vector<int64_t> output_dims(input_dimensions);
  std::vector<int64_t> flattened_output_dims;
  std::vector<int64_t>* p_flattened_output_dims = &flattened_output_dims;
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677

  if (dynamic) {
    std::vector<int64_t> input_starts, input_ends, input_axes, input_steps;
    FillVectorsFromInput(ctx, input_starts, input_ends, input_axes, input_steps);
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes,
<<<<<<< HEAD
                        input_steps, input_dims, starts, steps, output_dims));

  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_, 
	                    input_dims, starts, output_dims));
=======
                                          input_steps, input_dimensions, starts, steps, output_dims,
                                          p_flattened_output_dims));

  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_,
                                          input_dimensions, starts, steps, output_dims,
                                          p_flattened_output_dims));
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677
  }

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  int64_t output_size = output_shape.Size();
  if (output_size == 0) {
    return Status::OK();
  }
<<<<<<< HEAD

  ORT_ENFORCE(rank <= MAX_ARRAY_SIZE);
  TArray<int64_t> starts_buffer(gsl::narrow_cast<int32_t>(starts.size()));
  for (auto i = 0; i < starts.size(); ++i) {
    starts_buffer.data_[i] = starts[i];
  }
  TArray<int64_t> steps_buffer(gsl::narrow_cast<int32_t>(steps.size()));
  for (auto i = 0; i < steps.size(); ++i) {
    steps_buffer.data_[i] = steps[i];
  }

  TensorPitches original_input_strides(input_dims);
  TArray<int64_t> input_strides(gsl::narrow_cast<int32_t>(original_input_strides.size()));
  for (auto i = 0; i < original_input_strides.size(); ++i) {
    input_strides.data_[i] = original_input_strides[i];
  }

  TensorPitches original_output_strides(output_dims);
  TArray<fast_divmod> output_strides(gsl::narrow_cast<int32_t>(original_output_strides.size()));
  for (auto i = 0; i < original_output_strides.size(); ++i) {
    output_strides.data_[i] = fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
=======

  // if we are able to flatten the output dims we updated 'starts' and 'steps' to match the smaller number of dims.
  // update dimension_count to match.
  if (p_flattened_output_dims != nullptr) {
    dimension_count = flattened_output_dims.size();
  }

  CudaAsyncBuffer<int64_t> starts_buffer(this, dimension_count);
  gsl::span<int64_t> starts_buffer_span = starts_buffer.CpuSpan();
  for (size_t i = 0; i < dimension_count; ++i) {
    starts_buffer_span[i] = starts[i];
  }
  starts_buffer.CopyToGpu();

  CudaAsyncBuffer<int64_t> steps_buffer(this, dimension_count);
  gsl::span<int64_t> steps_buffer_span = steps_buffer.CpuSpan();
  for (size_t i = 0; i < dimension_count; ++i) {
    steps_buffer_span[i] = steps[i];
  }
  steps_buffer.CopyToGpu();

  CudaAsyncBuffer<int64_t> input_strides(this, dimension_count);
  if (p_flattened_output_dims != nullptr) {
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
    ORT_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), flattened_input_dims));
  } else {
    ORT_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), input_dimensions));
  }

  input_strides.CopyToGpu();

  TensorPitches output_pitches(p_flattened_output_dims != nullptr ? flattened_output_dims : output_dims);

  CudaAsyncBuffer<fast_divmod> div_strides(this, dimension_count);
  gsl::span<fast_divmod> div_strides_span = div_strides.CpuSpan();
  for (size_t i = 0; i < dimension_count; ++i) {
    div_strides_span[i] = fast_divmod(gsl::narrow_cast<int>(output_pitches[i]));
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677
  }

  size_t element_size = input_tensor->DataType()->Size();

  ORT_RETURN_IF_ERROR(SliceImpl(element_size,
<<<<<<< HEAD
                              rank,
                              &starts_buffer,
                              &steps_buffer,
                              &input_strides,
                              &output_strides,
                              input_tensor->DataRaw(),
                              output_tensor->MutableDataRaw(),
                              output_size));
=======
                                gsl::narrow_cast<int32_t>(dimension_count),
                                starts_buffer.GpuPtr(),
                                steps_buffer.GpuPtr(),
                                input_strides.GpuPtr(),
                                div_strides.GpuPtr(),
                                input_tensor->DataRaw(),
                                output_tensor->MutableDataRaw(),
                                output_size));
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "slice_impl.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_VERSIONED_TYPED_SLICE(TIND)                                              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                \
      Slice,                                                                              \
      kOnnxDomain,                                                                        \
      1, 9,                                                                               \
      TIND,                                                                               \
      kCudaExecutionProvider,                                                             \
      KernelDefBuilder().TypeConstraint("T",    DataTypeImpl::AllFixedSizeTensorTypes()). \
                         TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()),     \
      Slice<TIND,false>);

REGISTER_VERSIONED_TYPED_SLICE(int32_t) 
REGISTER_VERSIONED_TYPED_SLICE(int64_t)

#define REGISTER_V10_TYPED_SLICE(TIND)                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                          \
      Slice,                                                                              \
      kOnnxDomain,                                                                        \
      10,                                                                                 \
      TIND,                                                                               \
      kCudaExecutionProvider,                                                             \
      KernelDefBuilder().InputMemoryType<OrtMemTypeCPUInput>(1).                          \
                         InputMemoryType<OrtMemTypeCPUInput>(2).                          \
                         InputMemoryType<OrtMemTypeCPUInput>(3).                          \
                         InputMemoryType<OrtMemTypeCPUInput>(4).                          \
                         TypeConstraint("T",    DataTypeImpl::AllFixedSizeTensorTypes()). \
                         TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()),     \
      Slice<TIND,true>);

REGISTER_V10_TYPED_SLICE(int32_t) 
REGISTER_V10_TYPED_SLICE(int64_t)

template<typename Tind, bool dynamic>
Status Slice<Tind, dynamic>::ComputeInternal(OpKernelContext* ctx) const {
  auto input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != input_tensor);
  auto& input_dims = input_tensor->Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  const int32_t rank = gsl::narrow_cast<int32_t>(input_dims.size());
  std::vector<int64_t> starts(rank, 0);
  std::vector<int64_t> steps(rank, 1);
  std::vector<int64_t> output_dims(input_dims);

  if (dynamic) {
    std::vector<int64_t> input_starts, input_ends, input_axes, input_steps;
    FillVectorsFromInput(ctx, input_starts, input_ends, input_axes, input_steps);
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes,
                        input_steps, input_dims, starts, steps, output_dims));

  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_, 
	                    input_dims, starts, output_dims));
  }

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  int64_t output_size = output_shape.Size();
  if (output_size == 0) {
    return Status::OK();
  }

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
  }

  size_t element_size = input_tensor->DataType()->Size();

  ORT_RETURN_IF_ERROR(SliceImpl(element_size,
                              rank,
                              &starts_buffer,
                              &steps_buffer,
                              &input_strides,
                              &output_strides,
                              input_tensor->DataRaw(),
                              output_tensor->MutableDataRaw(),
                              output_size));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

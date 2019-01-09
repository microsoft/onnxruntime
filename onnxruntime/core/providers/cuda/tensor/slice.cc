// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "slice_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_TYPED_SLICE(NAME, TIND, DYNAMIC)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                          \
      NAME,                                                                               \
      kOnnxDomain,                                                                        \
      1,                                                                                  \
      TIND,                                                                               \
      kCudaExecutionProvider,                                                             \
      KernelDefBuilder().TypeConstraint("T",    DataTypeImpl::AllFixedSizeTensorTypes()). \
                         TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()),     \
      Slice<TIND,DYNAMIC>);

REGISTER_TYPED_SLICE(Slice,        int32_t, false) 
REGISTER_TYPED_SLICE(Slice,        int64_t, false) 
REGISTER_TYPED_SLICE(DynamicSlice, int32_t, true ) 
REGISTER_TYPED_SLICE(DynamicSlice, int64_t, true )

template <typename Tind, bool dynamic>
void Slice<Tind, dynamic>::FillVectorsFromInput(const OpKernelContext* context,
                                                std::vector<int64_t>&  input_starts,
                                                std::vector<int64_t>&  input_ends,
                                                std::vector<int64_t>&  input_axes) const {

  ORT_ENFORCE(context->Input<Tensor>(1) != nullptr, "Required starts input is missing");
  ORT_ENFORCE(context->Input<Tensor>(2) != nullptr, "Required ends input is missing");
  auto starts_tensor_ptr = context->Input<Tensor>(1);
  auto ends_tensor_ptr   = context->Input<Tensor>(2);

  ORT_ENFORCE(starts_tensor_ptr->Shape().NumDimensions() == 1, "Starts input must be a 1-D array");
  ORT_ENFORCE(ends_tensor_ptr->Shape().NumDimensions()   == 1, "ends input must be a 1-D array");
  ORT_ENFORCE(starts_tensor_ptr->Shape().Size() == ends_tensor_ptr->Shape().Size(), "Found mismatch between starts and ends input");
  auto size = starts_tensor_ptr->Shape().Size();
  Tind* buffer = new Tind[size];

  cudaMemcpy(buffer, starts_tensor_ptr->DataRaw(), size*sizeof(Tind), cudaMemcpyDeviceToHost);
  for (auto i = 0; i < size; ++i) input_starts.push_back(buffer[i]);
  cudaMemcpy(buffer, ends_tensor_ptr->DataRaw(),   size*sizeof(Tind), cudaMemcpyDeviceToHost);
  for (auto i = 0; i < ends_tensor_ptr->Shape().Size(); ++i) input_ends.push_back(buffer[i]);

  if (context->Input<Tensor>(3) != nullptr) {
    auto axes_tensor_ptr = context->Input<Tensor>(3);
    ORT_ENFORCE(size == axes_tensor_ptr->Shape().Size(), "Found mismatch between starts and axes input");
    cudaMemcpy(buffer, axes_tensor_ptr->DataRaw(), size*sizeof(Tind), cudaMemcpyDeviceToHost);
    for (auto i = 0; i < size; ++i) input_axes.push_back(buffer[i]);
  }

  delete[] buffer;
}

template<typename Tind, bool dynamic>
Status Slice<Tind, dynamic>::ComputeInternal(OpKernelContext* ctx) const {
  auto input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != input_tensor);
  auto& input_dimensions = input_tensor->Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  const size_t dimension_count = input_dimensions.size();
  std::vector<int64_t> starts(dimension_count, 0);
  std::vector<int64_t> output_dims(input_dimensions);

  if (dynamic) {
    std::vector<int64_t> input_starts, input_ends, input_axes;
    FillVectorsFromInput(ctx, input_starts, input_ends, input_axes);
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes,
                        dimension_count, input_dimensions, starts, output_dims));

  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_, 
	                    dimension_count, input_dimensions, starts, output_dims));
  }

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  int64_t output_size = output_shape.Size();
  if (output_size == 0) {
    return Status::OK();
  }
  int device_id = 0;
  CudaAsyncBuffer<int64_t> starts_buffer(this, device_id, dimension_count);
  gsl::span<int64_t> starts_buffer_span = starts_buffer.CpuSpan();
  for (int i = 0; i < dimension_count; ++i) {
    starts_buffer_span[i] = starts[i];
  }
  starts_buffer.CopyToGpu();

  CudaAsyncBuffer<int64_t> input_strides(this, device_id, dimension_count);
  ORT_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), input_dimensions));
  input_strides.CopyToGpu();

  TensorPitches output_pitches(output_dims);

  CudaAsyncBuffer<fast_divmod> div_strides(this, device_id, dimension_count);
  gsl::span<fast_divmod> div_strides_span = div_strides.CpuSpan();
  for (int i = 0; i < dimension_count; ++i) {
    div_strides_span[i] = fast_divmod(gsl::narrow_cast<int>(output_pitches[i]));
  }
  div_strides.CopyToGpu();

  size_t element_size = input_tensor->DataType()->Size();

  ORT_RETURN_IF_ERROR(SliceImpl(element_size,
                              gsl::narrow_cast<int32_t>(dimension_count),
                              starts_buffer.GpuPtr(),
                              input_strides.GpuPtr(),
                              div_strides.GpuPtr(),
                              input_tensor->DataRaw(),
                              output_tensor->MutableDataRaw(),
                              output_size));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

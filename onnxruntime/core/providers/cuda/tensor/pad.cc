// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pad.h"
#include "pad_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/tensor/pad.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      2, 10,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);

template <typename T>
Status Pad<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto& input_tensor = *ctx->Input<Tensor>(0);
  auto const& input_shape = input_tensor.Shape();
  auto dimension_count = input_shape.NumDimensions();
  CudaAsyncBuffer<int64_t> input_dims(this, input_shape.GetDims());
  CudaAsyncBuffer<int64_t> input_strides(this, dimension_count);
  CudaAsyncBuffer<int64_t> lower_pads(this, dimension_count);
  CudaAsyncBuffer<int64_t> upper_pads(this, dimension_count);
  CudaAsyncBuffer<fast_divmod> fdm_output_strides(this, dimension_count);

  TensorPitches::Calculate(input_strides.CpuSpan(), input_shape.GetDims());
  std::vector<int64_t> output_dims(input_shape.GetDims());

  ORT_ENFORCE(dimension_count * 2 == pads_.size(), "'pads' attribute has wrong number of values");

  // Calculate output dimensions, and handle any negative padding
  auto lower_pads_span = lower_pads.CpuSpan();
  auto upper_pads_span = upper_pads.CpuSpan();
  for (size_t i = 0; i < dimension_count; i++) {
    lower_pads_span[i] = pads_[i] + slices_[i];
    upper_pads_span[i] = pads_[i + dimension_count] + slices_[i + dimension_count];
    output_dims[i] += lower_pads_span[i] + upper_pads_span[i];
  }
  TensorShape output_shape(output_dims);

  // special case when there is a dim value of 0 in the shape. behavior depends on mode
  if (input_shape.Size() == 0) {
    ORT_RETURN_IF_ERROR(PadBase::HandleDimValueZero(mode_, input_shape, output_shape));
  }

  auto& output_tensor = *ctx->Output(0, output_shape);
  ORT_ENFORCE(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_dims));
  ORT_RETURN_IF_ERROR(input_dims.CopyToGpu());
  ORT_RETURN_IF_ERROR(input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(lower_pads.CopyToGpu());
  ORT_RETURN_IF_ERROR(upper_pads.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());

  PadImpl(
      dimension_count,
      input_dims.GpuPtr(),
      input_strides.GpuPtr(),
      lower_pads.GpuPtr(),
      upper_pads.GpuPtr(),
      value_,
      static_cast<int>(mode_),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(input_tensor.template Data<T>()),
      fdm_output_strides.GpuPtr(),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_tensor.template MutableData<T>()),
      output_tensor.Shape().Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Pad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
};  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "upsample.h"
#include "upsample_impl.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Upsample,                                                   \
      kOnnxDomain,                                                \
      7,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Upsample<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(int32_t)

template <typename T>
Status Upsample<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != X);
  const std::vector<int64_t>& X_dims = X->Shape().GetDims();
  auto rank = X_dims.size();
  if (rank == 0)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Upsample: input tensor cannot be scalar.");

  if (rank != scales_.size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Upsample: input tensor's dimension does not match the scales.");

  std::vector<int64_t> Y_dims;
  for (std::size_t i = 0; i < rank; i++) {
    Y_dims.push_back(static_cast<int64_t>(scales_[i] * X_dims[i]));
  }
  Tensor* Y = context->Output(0, Y_dims);
  typedef typename ToCudaType<T>::MappedType CudaT;

  // kernel
  int device_id = 0;
  TensorPitches input_pitches(X_dims);
  CudaAsyncBuffer<int64_t> input_strides(this, device_id, rank);
  gsl::span<int64_t> input_stride_span = input_strides.CpuSpan();

  TensorPitches output_pitches(Y_dims);
  CudaAsyncBuffer<fast_divmod> output_div_pitches(this, device_id, rank);
  gsl::span<fast_divmod> div_strides_span = output_div_pitches.CpuSpan();

  
  CudaAsyncBuffer<fast_divmod> scales_div(this, device_id, rank);
  gsl::span<fast_divmod> scales_div_span = scales_div.CpuSpan();

  for (int i = 0; i < rank; ++i) {
    input_stride_span[i] = input_pitches[i];
    div_strides_span[i] = fast_divmod(gsl::narrow_cast<int>(output_pitches[i]));
    scales_div_span[i] = fast_divmod(gsl::narrow_cast<int>(ceil(scales_[i])));
  }
  input_strides.CopyToGpu();
  output_div_pitches.CopyToGpu();
  scales_div.CopyToGpu();

  size_t output_count = Y->Shape().Size();

  if (UpsampleMode::LINEAR == mode_) {
    if (rank != 4)
      return Status(ONNXRUNTIME, FAIL, "Upsample: linear mode upsample only support 4-D tensor with NCHW layout");
  }

  UpampleImpl(mode_,
              rank,
              (UpsampleMode::LINEAR == mode_) ? X_dims[2] : 0,
              input_strides.GpuPtr(),
              output_div_pitches.GpuPtr(),
              scales_div.GpuPtr(),
              reinterpret_cast<const CudaT*>(X->template Data<T>()),
              reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
              output_count);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

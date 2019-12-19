// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "upsample.h"
#include "upsample_impl.h"
#include "core/providers/cuda/tensor/resize_impl.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Upsample,                                                   \
      kOnnxDomain,                                                \
      7,                                                          \
      9,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Upsample<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(int32_t)
REGISTER_KERNEL_TYPED(uint8_t)

template <typename T>
Status Upsample<T>::BaseCompute(OpKernelContext* context,
                                const std::vector<float>& roi,
                                const std::vector<float>& scales,
                                const std::vector<int64_t>& output_dims) const {
  const Tensor* X = context->Input<Tensor>(0);
  const std::vector<int64_t>& X_dims = X->Shape().GetDims();
  auto rank = X_dims.size();

  ORT_ENFORCE(output_dims.size() == rank, "Rank of input and output tensor should be same.");
  if (rank == 0)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  is_resize_ ? "Resize: input tensor cannot be scalar." : "Upsample: input tensor cannot be scalar.");
  if (rank != scales.size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  is_resize_ ? "Resize: input tensor's dimension does not match the scales." : "Upsample: input tensor's dimension does not match the scales.");
  if (roi.size() != 2 * X->Shape().GetDims().size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Resize: size of roi array should be 2 * N where N is the rank of input tensor X.");

  Tensor* Y = context->Output(0, output_dims);
  typedef typename ToCudaType<T>::MappedType CudaT;

  // kernel
  TensorPitches input_pitches(X_dims);
  CudaAsyncBuffer<int64_t> input_strides(this, rank);
  gsl::span<int64_t> input_stride_span = input_strides.CpuSpan();

  TensorPitches output_pitches(output_dims);
  CudaAsyncBuffer<fast_divmod> output_div_pitches(this, rank);
  gsl::span<fast_divmod> div_strides_span = output_div_pitches.CpuSpan();

  for (size_t i = 0; i < rank; ++i) {
    input_stride_span[i] = input_pitches[i];
    div_strides_span[i] = fast_divmod(gsl::narrow_cast<int>(output_pitches[i]));
  }
  size_t output_count = Y->Shape().Size();

  if (is_resize_) {
    CudaAsyncBuffer<int64_t> input_shape(this, X_dims);
    CudaAsyncBuffer<int64_t> output_shape(this, output_dims);
    CudaAsyncBuffer<float> roi_vals(this, roi);
    CudaAsyncBuffer<float> scales_vals(this, scales);

    size_t temp_buffer_size = CalcResizeBufferSize(mode_, output_dims);
    auto dims_mapping_buffer = GetScratchBuffer<unsigned char>(temp_buffer_size);
    void* dims_mapping = reinterpret_cast<void*>(dims_mapping_buffer.get());
    ResizeImpl(mode_, (int)rank, input_shape, output_shape,
               input_strides, output_div_pitches, scales_vals, roi_vals,
               reinterpret_cast<const CudaT*>(X->template Data<T>()),
               reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
               output_count, use_extrapolation_, ToCudaType<T>::FromFloat(extrapolation_value_),
               cubic_coeff_a_, exclude_outside_,
               coordinate_transform_mode_, nearest_mode_,
               dims_mapping);
  } else {
    input_strides.CopyToGpu();
    output_div_pitches.CopyToGpu();

    CudaAsyncBuffer<fast_divmod> scales_div(this, rank);
    gsl::span<fast_divmod> scales_div_span = scales_div.CpuSpan();

    for (size_t i = 0; i < rank; ++i) {
      scales_div_span[i] = fast_divmod(gsl::narrow_cast<int>(ceil(scales[i])));
    }
    scales_div.CopyToGpu();

    UpampleImpl(mode_,
                rank,
                (UpsampleMode::LINEAR == mode_) ? (rank == 2 ? X_dims[0] : X_dims[2]) : 0,
                input_strides.GpuPtr(),
                output_div_pitches.GpuPtr(),
                scales_div.GpuPtr(),
                reinterpret_cast<const CudaT*>(X->template Data<T>()),
                reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
                output_count);
  }

  return Status::OK();
}

template <typename T>
Status Upsample<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);

  std::vector<int64_t> output_dims(X->Shape().GetDims().size());
  std::vector<float> roi_array(X->Shape().GetDims().size() * 2, 0.0f);
  if (!roi_cached_) {
    if (need_roi_input_) {
      ORT_ENFORCE(roi_input_idx_ > 0, "Invalid roi input index.");
      ParseRoiData(context->Input<Tensor>(roi_input_idx_), roi_array);
    }
  }
  const std::vector<float>& roi = roi_cached_ ? roi_ : roi_array;

  if (OpKernel::Node().InputDefs().size() == 1) {
    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_, X->Shape().GetDims(), output_dims);
    return BaseCompute(context, roi, scales_, output_dims);
  }

  const auto* scales = context->Input<Tensor>(scales_input_idx_);
  const auto* sizes = context->Input<Tensor>(sizes_input_idx_);
  ORT_ENFORCE(scales != nullptr);

  if (scales_cached_) {
    ORT_ENFORCE(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
    ComputeOutputShape(scales_, X->Shape().GetDims(), output_dims);
    return BaseCompute(context, roi, scales_, output_dims);
  }

  std::vector<float> scales_array(X->Shape().GetDims().size());
  if (scales != nullptr && scales->Shape().Size() != 0) {
    // use scales input data
    ORT_ENFORCE(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
    ParseScalesData(scales, scales_array);
    ComputeOutputShape(scales_array, X->Shape().GetDims(), output_dims);
  } else {
    // When sizes input is available directly populate it into the output_dims array.
    ORT_ENFORCE(sizes != nullptr && sizes->Shape().Size() != 0,
                "Either scales or sizes MUST be provided as input.");
    ORT_ENFORCE(sizes->Shape().Size() == static_cast<int64_t>(output_dims.size()),
                "Resize: input tensor's rank does not match the output tensor's rank.");
    memcpy(output_dims.data(), sizes->template Data<int64_t>(), sizes->Shape().Size() * sizeof(int64_t));
    ParseScalesDataFromOutputSize(output_dims, X->Shape().GetDims(), scales_array);
  }

  return BaseCompute(context, roi, scales_array, output_dims);
}

}  // namespace cuda
}  // namespace onnxruntime

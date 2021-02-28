// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "upsample.h"
#include "upsample_impl.h"
#include "core/providers/cuda/tensor/resize_impl.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

#define REGISTER_VERSIONED_TYPED_KERNEL(T, start, end)            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Upsample,                                                   \
      kOnnxDomain,                                                \
      start,                                                      \
      end,                                                        \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Upsample<T>)

REGISTER_VERSIONED_TYPED_KERNEL(float, 7, 8);
REGISTER_VERSIONED_TYPED_KERNEL(double, 7, 8);
REGISTER_VERSIONED_TYPED_KERNEL(MLFloat16, 7, 8);
REGISTER_VERSIONED_TYPED_KERNEL(int32_t, 7, 8);
REGISTER_VERSIONED_TYPED_KERNEL(uint8_t, 7, 8);

// Upsample was deprecated in opset 10
REGISTER_VERSIONED_TYPED_KERNEL(float, 9, 9);
REGISTER_VERSIONED_TYPED_KERNEL(double, 9, 9);
REGISTER_VERSIONED_TYPED_KERNEL(MLFloat16, 9, 9);
REGISTER_VERSIONED_TYPED_KERNEL(int32_t, 9, 9);
REGISTER_VERSIONED_TYPED_KERNEL(uint8_t, 9, 9);

template <typename T>
Status Upsample<T>::BaseCompute(OpKernelContext* context,
                                const std::vector<float>& roi,
                                const std::vector<float>& scales,
                                const std::vector<int64_t>& output_dims) const {
  const Tensor* X = context->Input<Tensor>(0);
  const std::vector<int64_t>& X_dims = X->Shape().GetDims();
  int32_t rank = static_cast<int32_t>(X_dims.size());

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

  // Return early if the output tensor is going to be of size 0
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  typedef typename ToCudaType<T>::MappedType CudaT;

  // kernel
  TensorPitches input_pitches(X_dims);
  TArray<int64_t> input_strides(input_pitches);

  TensorPitches output_pitches(output_dims);
  TArray<fast_divmod> output_div_pitches(rank);

  for (int32_t i = 0; i < rank; ++i) {
    output_div_pitches[i] = fast_divmod(gsl::narrow_cast<int>(output_pitches[i]));
  }
  size_t output_count = Y->Shape().Size();

  if (is_resize_) {
    TArray<int64_t> input_shape(X_dims);
    TArray<int64_t> output_shape(output_dims);
    TArray<float, 10> roi_vals(roi);
    TArray<float> scales_vals(scales);

    size_t temp_buffer_size = CalcResizeBufferSize(mode_, output_dims);
    auto dims_mapping_buffer = GetScratchBuffer<unsigned char>(temp_buffer_size);
    void* dims_mapping = reinterpret_cast<void*>(dims_mapping_buffer.get());
    ResizeImpl(Stream(), mode_, (int)rank, input_shape, output_shape,
               input_strides, output_div_pitches, scales_vals, roi_vals,
               reinterpret_cast<const CudaT*>(X->template Data<T>()),
               reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
               output_count, use_extrapolation_, ToCudaType<T>::FromFloat(extrapolation_value_),
               cubic_coeff_a_, exclude_outside_,
               coordinate_transform_mode_, nearest_mode_,
               dims_mapping);
  } else {
    TArray<fast_divmod> scales_div(rank);

    for (int32_t i = 0; i < rank; ++i) {
      scales_div[i] = fast_divmod(gsl::narrow_cast<int>(ceil(scales[i])));
    }

    UpampleImpl(Stream(),
                mode_,
                rank,
                (UpsampleMode::LINEAR == mode_) ? (rank == 2 ? X_dims[0] : X_dims[2]) : 0,
                input_strides,
                output_div_pitches,
                scales_div,
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
    bool use_default_roi = true;
    if (need_roi_input_) {
      ORT_ENFORCE(roi_input_idx_ > 0, "Invalid roi input index.");
      const auto* roi = context->Input<Tensor>(roi_input_idx_);
      if (roi != nullptr) {
        ParseRoiData(roi, roi_array);
        use_default_roi = false;
      }
    }
    if (use_default_roi) {
      // default roi includes ensures all the values in that axis are included in the roi
      // normalized roi is thus : [start, end] = [0, 1]
      const auto& input_dims = X->Shape().GetDims();
      size_t input_rank = input_dims.size();
      roi_array.resize(input_rank * 2);
      for (size_t i = 0; i < input_rank; ++i) {
        roi_array[i] = 0;
        roi_array[i + input_rank] = 1;
      }
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

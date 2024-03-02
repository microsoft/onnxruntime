// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "upsample.h"

#include <utility>

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
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
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
Upsample<T>::Upsample(const OpKernelInfo& info) : UpsampleBase(info), CudaKernel(info) {
  if (UpsampleBase::antialias_) {
    // Copy the table on DEVICE
    const uint8_t* lookup_table = GetLookupTableShared();
    auto alloc = info.GetAllocator(OrtMemTypeDefault);
    shared_lookup_table_ondevice_ = IAllocator::MakeUniquePtr<uint8_t>(std::move(alloc), kLookupTableSize);
    CUDA_CALL_THROW(cudaMemcpyAsync(shared_lookup_table_ondevice_.get(), lookup_table, kLookupTableSize,
                                    cudaMemcpyHostToDevice, nullptr));
  }
}

template <typename T>
Status Upsample<T>::BaseCompute(OpKernelContext* context,
                                gsl::span<const float> roi,
                                gsl::span<const float> scales,
                                gsl::span<const int64_t> output_dims) const {
  const Tensor* X = context->Input<Tensor>(0);
  auto X_dims = X->Shape().GetDims();
  int32_t rank = static_cast<int32_t>(X_dims.size());

  ORT_ENFORCE(static_cast<int32_t>(output_dims.size()) == rank, "Rank of input and output tensor should be same.");
  if (rank == 0)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  is_resize_ ? "Resize: input tensor cannot be scalar." : "Upsample: input tensor cannot be scalar.");
  if (rank != static_cast<int32_t>(scales.size()))
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  is_resize_ ? "Resize: input tensor's dimension does not match the scales."
                             : "Upsample: input tensor's dimension does not match the scales.");
  if (roi.size() != 2 * X_dims.size())
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
    const bool is_same = std::all_of(scales.begin(), scales.end(), [](float v) { return v == 1.0f; }) &&
                         (coordinate_transform_mode_ != ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE);
    if (is_same) {
      CUDA_CALL_THROW(cudaMemcpyAsync(Y->MutableData<T>(), X->Data<T>(),
                                      output_count * sizeof(T), cudaMemcpyDeviceToDevice, Stream(context)));
      return Status::OK();
    }

    if (antialias_) {
      TempSpaceAllocateFunc allocate_temp_space = [&](size_t bytes_size) {
        return GetScratchBuffer<uint8_t>(bytes_size, context->GetComputeStream());
      };

      std::optional<float> extrapolation_value;
      if (use_extrapolation_)
        extrapolation_value.emplace(extrapolation_value_);

      switch (mode_) {
        case UpsampleMode::LINEAR: {
          if (X_dims.size() == 2 || X_dims.size() == 4) {
            const bool is_2D = X_dims.size() == 2;

            int64_t batch_size = 1;
            int64_t num_channels = 1;

            int64_t input_height;
            int64_t input_width;

            int64_t output_height;
            int64_t output_width;

            float height_scale;
            float width_scale;

            if (is_2D) {
              input_height = X_dims[0];
              input_width = X_dims[1];

              output_height = output_dims[0];
              output_width = output_dims[1];

              height_scale = scales[0];
              width_scale = scales[1];
            } else {
              if (scales[0] == 1.0f && scales[1] == 1.0f) {
                batch_size = X_dims[Channels<LAYOUT_NCHW>::N];
                num_channels = X_dims[Channels<LAYOUT_NCHW>::C];
                input_height = X_dims[Channels<LAYOUT_NCHW>::H];
                input_width = X_dims[Channels<LAYOUT_NCHW>::W];

                output_height = output_dims[Channels<LAYOUT_NCHW>::H];
                output_width = output_dims[Channels<LAYOUT_NCHW>::W];

                height_scale = scales[2];
                width_scale = scales[3];
              } else {
                return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Resize", ": NHWC is not supported yet");
              }
            }

            ResizeAntiAliasImpl(Stream(context),
                                rank,
                                mode_,
                                coordinate_transform_mode_,
                                X_dims, output_dims,
                                batch_size, num_channels,
                                std::make_tuple(0, input_height, input_width),
                                std::make_tuple(0, output_height, output_width),
                                std::make_tuple(0.f, height_scale, width_scale),
                                output_div_pitches,
                                roi,
                                extrapolation_value,
                                exclude_outside_,
                                allocate_temp_space,
                                shared_lookup_table_ondevice_.get(),
                                reinterpret_cast<const CudaT*>(X->Data<T>()),
                                reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                                output_count);

          } else if (X_dims.size() == 3 || X_dims.size() == 5) {
            const bool is_3D = X_dims.size() == 3;

            if (!is_3D) {
              if (!(scales[0] == 1.0f && scales[1] == 1.0f)) {
                return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Resize", ": NDHWC is not supported yet");
              }
            }

            const int64_t batch_size = is_3D ? 1 : X_dims[0];
            const int64_t num_channels = is_3D ? 1 : X_dims[1];
            const int64_t input_depth = is_3D ? X_dims[0] : X_dims[2];
            const int64_t input_height = is_3D ? X_dims[1] : X_dims[3];
            const int64_t input_width = is_3D ? X_dims[2] : X_dims[4];

            const int64_t output_depth = is_3D ? output_dims[0] : output_dims[2];
            const int64_t output_height = is_3D ? output_dims[1] : output_dims[3];
            const int64_t output_width = is_3D ? output_dims[2] : output_dims[4];

            const float depth_scale = is_3D ? scales[0] : scales[2];
            const float height_scale = is_3D ? scales[1] : scales[3];
            const float width_scale = is_3D ? scales[2] : scales[4];

            ResizeAntiAliasImpl(Stream(context),
                                rank,
                                mode_,
                                coordinate_transform_mode_,
                                X_dims, output_dims,
                                batch_size, num_channels,
                                std::make_tuple(input_depth, input_height, input_width),
                                std::make_tuple(output_depth, output_height, output_width),
                                std::make_tuple(depth_scale, height_scale, width_scale),
                                output_div_pitches,
                                roi,
                                extrapolation_value,
                                exclude_outside_,
                                allocate_temp_space,
                                shared_lookup_table_ondevice_.get(),
                                reinterpret_cast<const CudaT*>(X->Data<T>()),
                                reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                                output_count);
          } else {
            return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Resize",
                                   ": 'Linear' mode only support 2-D inputs or 3-D inputs ('Bilinear', 'Trilinear') "
                                   "or 4-D inputs or 5-D inputs with the corresponding outermost 2 scale values "
                                   "being 1.");
          }
        } break;
        case UpsampleMode::CUBIC: {
          if (X_dims.size() != 2 && X_dims.size() != 4) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Resize",
                                   ": 'Cubic' mode only support 2-D inputs ('Bicubic') or 4-D inputs "
                                   "with the corresponding outermost 2 scale values being 1.");
          }

          const bool is_2D = X_dims.size() == 2;
          const bool is_nchw = is_2D ? true : (scales[1] == 1.0f && scales[1] == 1.0f);

          ORT_RETURN_IF_NOT(is_nchw,
                            "Resize 'Cubic' mode only supports NCWH layout "
                            " with 2-D or 4-D with leading dims equal to 1");

          const int64_t batch_size = is_2D ? 1 : X_dims[Channels<LAYOUT_NCHW>::N];
          const int64_t num_channels = is_2D ? 1 : X_dims[Channels<LAYOUT_NCHW>::C];
          const int64_t input_height = is_2D ? X_dims[0] : X_dims[Channels<LAYOUT_NCHW>::H];
          const int64_t input_width = is_2D ? X_dims[1] : X_dims[Channels<LAYOUT_NCHW>::W];

          const int64_t output_height = is_2D ? output_dims[0] : output_dims[Channels<LAYOUT_NCHW>::H];
          const int64_t output_width = is_2D ? output_dims[1] : output_dims[Channels<LAYOUT_NCHW>::W];
          const float height_scale = is_2D ? scales[0] : scales[2];
          const float width_scale = is_2D ? scales[1] : scales[3];

          ResizeAntiAliasImpl(Stream(context), rank, mode_, coordinate_transform_mode_,
                              X_dims, output_dims,
                              batch_size, num_channels,
                              std::make_tuple(0, input_height, input_width),
                              std::make_tuple(0, output_height, output_width),
                              std::make_tuple(0.f, height_scale, width_scale),
                              output_div_pitches,
                              roi,
                              extrapolation_value,
                              exclude_outside_,
                              allocate_temp_space,
                              shared_lookup_table_ondevice_.get(),
                              reinterpret_cast<const CudaT*>(X->Data<T>()),
                              reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                              output_count);
        } break;
        default:
          return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Resize: unexpected mode");
      }
    } else {
      TArray<int64_t> input_shape(X_dims);
      TArray<int64_t> output_shape(output_dims);
      TArray<float, 10> roi_vals(roi);
      TArray<float> scales_vals(scales);

      size_t temp_buffer_size = CalcResizeBufferSize(mode_, output_dims);
      auto dims_mapping_buffer = GetScratchBuffer<unsigned char>(temp_buffer_size, context->GetComputeStream());
      void* dims_mapping = reinterpret_cast<void*>(dims_mapping_buffer.get());
      ResizeImpl(Stream(context), mode_, rank, input_shape, output_shape,
                 input_strides, output_div_pitches, scales_vals, roi_vals,
                 reinterpret_cast<const CudaT*>(X->Data<T>()),
                 reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                 output_count, use_extrapolation_, ToCudaType<T>::FromFloat(extrapolation_value_),
                 cubic_coeff_a_, exclude_outside_,
                 coordinate_transform_mode_, nearest_mode_,
                 dims_mapping);
    }
  } else {
    TArray<fast_divmod> scales_div(rank);

    for (int32_t i = 0; i < rank; ++i) {
      scales_div[i] = fast_divmod(gsl::narrow_cast<int>(ceil(scales[i])));
    }

    UpampleImpl(Stream(context),
                mode_,
                rank,
                (UpsampleMode::LINEAR == mode_) ? (rank == 2 ? X_dims[0] : X_dims[2]) : 0,
                input_strides,
                output_div_pitches,
                scales_div,
                reinterpret_cast<const CudaT*>(X->Data<T>()),
                reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                output_count);
  }

  return Status::OK();
}

template <typename T>
Status Upsample<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);
  auto input_dims = X->Shape().GetDims();

  TensorShapeVector output_dims(input_dims.size());
  InlinedVector<float> roi_array(input_dims.size() * 2, 0.0f);
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
      size_t input_rank = input_dims.size();
      roi_array.resize(input_rank * 2);
      for (size_t i = 0; i < input_rank; ++i) {
        roi_array[i] = 0;
        roi_array[i + input_rank] = 1;
      }
    }
  }

  ComputeROIWithAxes(roi_array, input_dims.size());

  InlinedVector<float> scales_array(input_dims.size());
  // opset < 10
  if (OpKernel::Node().InputDefs().size() == 1) {
    // Compute output shape from scales attributes and input dims
    scales_array = scales_;

    ComputeOutputShape(scales_array, input_dims, output_dims);
    return BaseCompute(context, roi_array, scales_, output_dims);
  }

  const Tensor* scales = context->Input<Tensor>(scales_input_idx_);
  const Tensor* sizes = context->Input<Tensor>(sizes_input_idx_);

  // This is when scales are obtained and cached from a constant initializer
  if (scales_cached_) {
    ORT_RETURN_IF_NOT(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
    scales_array = scales_;
    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_array, input_dims, output_dims);
    return BaseCompute(context, roi_array, scales_array, output_dims);
  }

  // Scales and sizes are input to the node
  if (scales != nullptr && scales->Shape().Size() != 0) {
    // use scales input data
    ORT_ENFORCE(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
    ORT_RETURN_IF_ERROR(ParseScalesData(scales, scales_array, input_dims.size()));

    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_array, input_dims, output_dims);
  } else {
    // When sizes input is available directly populate it into the output_dims array.
    ORT_ENFORCE(sizes != nullptr && sizes->Shape().Size() != 0,
                "Either scales or sizes MUST be provided as input.");
    ORT_RETURN_IF_ERROR(ParseSizesData(sizes, output_dims, input_dims));
    ORT_RETURN_IF_ERROR(ParseScalesDataAndAdjustOutputSize(output_dims, input_dims, scales_array));
  }

  return BaseCompute(context, roi_array, scales_array, output_dims);
}

}  // namespace cuda
}  // namespace onnxruntime

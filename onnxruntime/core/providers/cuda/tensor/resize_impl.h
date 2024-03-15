// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include <tuple>

#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"
#include "core/providers/cpu/tensor/upsamplebase.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
template <>
struct AccumulateType<half> {
  using type = float;
};
namespace cuda {

struct TransformCoordinate_ASYMMETRIC {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float x_scale,
                                                       float, float, float, float) const {
    return x_resized / x_scale;
  }
};

struct TransformCoordinate_HALF_PIXEL {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float x_scale,
                                                       float, float, float, float) const {
    return ((x_resized + 0.5f) / x_scale) - 0.5f;
  }
};

struct TransformCoordinate_PYTORCH_HALF_PIXEL {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float x_scale, float length_resized, float,
                                                       float, float) const {
    return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
  }
};

struct TransformCoordinate_TF_HALF_PIXEL_FOR_NN {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float x_scale,
                                                       float, float, float, float) const {
    return (x_resized + 0.5f) / x_scale;
  }
};

struct TransformCoordinate_ALIGN_CORNERS {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float, float length_resized,
                                                       float length_original, float, float) const {
    return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
  }
};

struct TransformCoordinate_TF_CROP_AND_RESIZE {
  __device__ __host__ __forceinline__ float operator()(float x_resized, float, float length_resized,
                                                       float length_original, float roi_start, float roi_end) const {
    auto orig = length_resized > 1
                    ? roi_start * (length_original - 1) +
                          (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                    : 0.5 * (roi_start + roi_end) * (length_original - 1);
    return static_cast<float>(orig);
  }
};

size_t CalcResizeBufferSize(const onnxruntime::UpsampleMode upsample_mode,
                            const gsl::span<const int64_t>& output_dims);

template <typename T>
void ResizeImpl(
    cudaStream_t stream,
    const onnxruntime::UpsampleMode upsample_mode,
    const int rank,
    TArray<int64_t>& input_shape,
    TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals,
    TArray<float, 10>& roi,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    const T extrapolation_value,
    float cubic_coeff_a,
    bool exclude_outside,
    onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
    onnxruntime::ResizeNearestMode nearest_mode,
    void* dims_mapping);

using TempSpaceAllocateFunc = std::function<onnxruntime::IAllocatorUniquePtr<uint8_t>(size_t buffer_size)>;

template <class T>
void ResizeAntiAliasImpl(
    cudaStream_t stream,
    int rank,
    const UpsampleMode upsample_mode,
    ResizeCoordinateTransformationMode coordinate_transform_mode,
    gsl::span<const int64_t> input_shape,
    gsl::span<const int64_t> output_shape,
    int64_t batch_size, int64_t num_channels,
    std::tuple<int64_t, int64_t, int64_t> inferred_input_dims,
    std::tuple<int64_t, int64_t, int64_t> inferred_output_dims,
    std::tuple<float, float, float> inferred_dim_rscales,
    const TArray<fast_divmod>& output_div_pitches,
    gsl::span<const float> roi_vals,  // CPU
    const std::optional<float>& extrapolation_value,
    bool exclude_outside,
    TempSpaceAllocateFunc allocate_temp_space,
    const uint8_t* clip8_lookups,
    const T* input_data,
    T* output_data,
    const size_t N);

/// <summary>
/// Compute scaled support value for a given dimension inverse scale
/// </summary>
/// <param name="support_value">Support value from parameters</param>
/// <param name="inv_scale">inverse scale value comes from input/attr for</param>
/// <returns></returns>
inline float ComputeScaledSupportValue(float support_value, float rscale) {
  const float scale = 1.0f / rscale;
  float scaled_support = (scale >= 1.0f) ? (support_value * 0.5f) * scale : support_value * 0.5f;
  return scaled_support;
}

/// <summary>
/// Compute window size for a given dimension scaled support value.
/// </summary>
/// <param name="scaled_support"></param>
/// <returns></returns>
inline int32_t ComputeWindowSize(float scaled_support) {
  SafeInt<int32_t> window_size(ceilf(scaled_support));
  return window_size * 2 + 1;
}

/// <summary>
/// Computes scale buffer size in number of elements for allocation purposes.
/// </summary>
/// <param name="output_size"></param>
/// <param name="window_size"></param>
/// <returns>Number of elements to fit in the buffer</returns>
inline SafeInt<int64_t> ComputeWeightedCoeffBufferSize(int64_t output_size, int32_t window_size) {
  SafeInt<int64_t> buffer_size(output_size);
  return buffer_size * window_size;
}

}  // namespace cuda
}  // namespace onnxruntime

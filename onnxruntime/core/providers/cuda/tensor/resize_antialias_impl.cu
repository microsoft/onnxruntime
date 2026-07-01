// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/tensor/resize_impl.h"

#define FUNC_DEF __device__

namespace onnxruntime {
namespace cuda {

using onnxruntime::ResizeCoordinateTransformationMode;
using onnxruntime::UpsampleMode;

/// <summary>
/// Compute a buffer for bilinear data for CUDA antialias resizing.
/// </summary>
static std::tuple<int64_t, int64_t> ComputeBilinearScaleBufferSize(
    int64_t output_height, int64_t output_width,
    float height_rscale, float width_rscale,
    float support_value,
    float& scaled_support_height, float& scaled_support_width,
    int32_t& window_size_height, int32_t& window_size_width) {
  scaled_support_height = ComputeScaledSupportValue(support_value, height_rscale);
  scaled_support_width = ComputeScaledSupportValue(support_value, width_rscale);
  window_size_height = ComputeWindowSize(scaled_support_height);
  window_size_width = ComputeWindowSize(scaled_support_width);

  auto height_buffer_size = ComputeWeightedCoeffBufferSize(output_height, window_size_height);
  auto width_buffer_size = ComputeWeightedCoeffBufferSize(output_width, window_size_width);

  return std::make_tuple(height_buffer_size, width_buffer_size);
}

/// <summary>
/// Compute a buffer for btrilinear data for CUDA antialias resizing.
/// </summary>
static std::tuple<int64_t, int64_t, int64_t> ComputeTrilinearScaleBufferSize(
    int64_t output_depth, int64_t output_height, int64_t output_width,
    float depth_rscale, float height_rscale, float width_rscale,
    float support_value,
    float& scaled_support_depth, float& scaled_support_height,
    float& scaled_support_width, int32_t& window_size_depth,
    int32_t& window_size_height, int32_t& window_size_width) {
  scaled_support_depth = ComputeScaledSupportValue(support_value, depth_rscale);
  window_size_depth = ComputeWindowSize(scaled_support_depth);
  auto depth_buffer_size = ComputeWeightedCoeffBufferSize(output_depth, window_size_depth);

  const auto [y_buffer_size, w_buffer_size] = ComputeBilinearScaleBufferSize(output_height,
                                                                             output_width, height_rscale,
                                                                             width_rscale, support_value,
                                                                             scaled_support_height,
                                                                             scaled_support_width,
                                                                             window_size_height, window_size_width);
  return std::make_tuple(depth_buffer_size, y_buffer_size, w_buffer_size);
}

// Antialiasing filters
struct BilinearFilter {
  __device__ __host__ float operator()(float x, float /* cubic_coeff_a */) const {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

struct BiCubicFilter {
  __device__ __host__ float operator()(float x, float cubic_coeff_a) const {
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
     */
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return ((cubic_coeff_a + 2.0f) * x - (cubic_coeff_a + 3.0f)) * x * x + 1;
    }
    if (x < 2.0f) {
      return (((x - 5.0f) * x + 8.f) * x - 4.f) * cubic_coeff_a;
    }
    return 0.0f;
  }
};

struct TriLinearFilter {
  __device__ __host__ float operator()(float x, float /* cubic_coeff_a */) const {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

template <typename AccumType>
struct AccumTypeCaster {
  static __device__ __host__ AccumType* cast(AccumType* p) {
    return p;
  }
};

template <>
struct AccumTypeCaster<int32_t> {
  static __device__ __host__ float* cast(int32_t* p) {
    return reinterpret_cast<float*>(p);
  }
};

template <typename T, typename AccumType>
__global__ void _ComputeInterpolationAtLevel1(
    int64_t num_channels,
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    const fast_divmod div_output_width,
    const fast_divmod div_output_image,
    int32_t window_size,
    const uint8_t* clip8_table,
    const int64_t* bound_data,
    std::tuple<int64_t*, int64_t*> outof_bounds_buffers,
    const AccumType* weight_coefficients,
    const T* Xdata, T* Ydata,
    const int N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // No need to do scale
  if (output_width == input_width) {
    Ydata[id] = Xdata[id];
    return;
  }

  int bxc, output_image_index;
  div_output_image.divmod(id, bxc, output_image_index);

  int output_y, output_x;
  div_output_width.divmod(output_image_index, output_y, output_x);

  CUDA_LONG input_index = static_cast<CUDA_LONG>(bxc * num_channels * input_height * input_width);
  CUDA_LONG output_index = static_cast<CUDA_LONG>(bxc * num_channels * output_height * output_width);

  auto* Ydata_offset = Ydata + output_index + output_width * output_y + output_x;
  const auto* bound = bound_data;

  AccumType output = onnxruntime::is_8bit_v<T> ? ConstValue::mag_factor : 0;

  const auto* weight_coeff = weight_coefficients + window_size * output_x;
  int64_t xmin = bound[static_cast<ptrdiff_t>(output_x) * 2];
  int64_t xmax = bound[static_cast<ptrdiff_t>(output_x) * 2 + 1];

  // Input window
  const auto* Xdata_offset = Xdata + input_index + input_width * output_y + xmin;

  for (; xmin < xmax; ++xmin) {
    if constexpr (std::is_same<T, half>::value) {
      // This cast is needed when we deal with half
      output += static_cast<AccumType>((*Xdata_offset++)) * (*weight_coeff++);
    } else {
      output += (*Xdata_offset++) * (*weight_coeff++);
    }
  }

  if constexpr (onnxruntime::is_8bit_v<T>) {
    const uint8_t* clip8_lookups = &clip8_table[640];
    *Ydata_offset = static_cast<T>(clip8_lookups[output >> 22]);
  } else if constexpr (std::is_same<T, int32_t>::value) {
    *Ydata_offset = static_cast<int32_t>(std::round(output));
  } else {
    *Ydata_offset = static_cast<T>(output);
  }
}

template <typename T, typename AccumType>
__global__ void _ComputeInterpolationAtLevel2(
    int64_t num_channels,
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    const fast_divmod div_output_height,
    const fast_divmod div_output_width,
    const fast_divmod div_output_image,
    int32_t window_size,
    bool use_extrapolation, float extrapolation_value,
    const uint8_t* clip8_table,
    const int64_t* bound_data,
    std::tuple<int64_t*, int64_t*> outof_bounds_buffers,
    const AccumType* weight_coefficients,
    const T* Xdata, T* Ydata, int N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // No need to do scale
  if (output_height == input_height) {
    Ydata[id] = Xdata[id];
    return;
  }

  int bxc, output_image_index;
  div_output_image.divmod(id, bxc, output_image_index);

  int output_z, output_y, output_x, temp;
  div_output_height.divmod(output_image_index, output_z, temp);
  div_output_width.divmod(temp, output_y, output_x);

  CUDA_LONG input_index = static_cast<CUDA_LONG>(bxc * num_channels * input_height * input_width +
                                                 output_z * input_height * input_width);
  CUDA_LONG output_index = static_cast<CUDA_LONG>(bxc * num_channels * output_height * output_width +
                                                  output_z * output_height * output_width);

  auto* Ydata_offset = Ydata + output_index + output_width * output_y + output_x;

  if (use_extrapolation) {
    const auto* w_outof_bounds = std::get<1>(outof_bounds_buffers);
    // Extrapolate along the w dimension
    if (w_outof_bounds[static_cast<ptrdiff_t>(output_x)] != -1) {
      *Ydata_offset = static_cast<T>(extrapolation_value);
      return;
    }

    // Extrapolate along the y dimension
    const auto* y_outof_bounds = std::get<0>(outof_bounds_buffers);
    if (y_outof_bounds[static_cast<ptrdiff_t>(output_y)] != -1) {
      *Ydata_offset = static_cast<T>(extrapolation_value);
      return;
    }
  }

  const auto* bound = bound_data;

  AccumType output = onnxruntime::is_8bit_v<T> ? ConstValue::mag_factor : 0;

  const auto* weight_coeff = weight_coefficients + window_size * output_y;
  int64_t ymin = bound[static_cast<ptrdiff_t>(output_y) * 2];
  int64_t ymax = bound[static_cast<ptrdiff_t>(output_y) * 2 + 1];

  const auto* Xdata_offset = Xdata + input_index + ymin * output_width + output_x;

  for (; ymin < ymax; ++ymin) {
    if constexpr (std::is_same<T, half>::value) {
      // We cast to AccumType to resolve ambiguous call to operator* for half in CUDA
      output += static_cast<AccumType>((*Xdata_offset)) * (*weight_coeff++);
    } else {
      output += (*Xdata_offset) * (*weight_coeff++);
    }
    Xdata_offset += input_width;
  }

  if constexpr (onnxruntime::is_8bit_v<T>) {
    const uint8_t* clip8_lookups = &clip8_table[640];
    *Ydata_offset = static_cast<T>(clip8_lookups[output >> 22]);
  } else if constexpr (std::is_same<T, int32_t>::value) {
    *Ydata_offset = static_cast<int32_t>(std::round(output));
  } else {
    *Ydata_offset = output;
  }
}

/// Fused 2D antialias interpolation kernel.
/// Applies separable H×W filter in a single pass: each thread computes one output pixel
/// by iterating over the 2D filter window directly from the input tensor.
/// Eliminates the intermediate buffer and second kernel launch of the separable approach.
///
/// Requirements:
///   - AccumType must be a floating-point type (float or double). For int32_t (8-bit
///     quantized weights), use the existing two-pass separable kernels instead, because
///     fusing fixed-point h_weight * w_weight would require different quantization.
///
/// Filter separability: weight(y,x) = h_weight[y - ymin] * w_weight[x - xmin]
template <typename T, typename AccumType>
__global__ void _ComputeFusedInterpolation2D(
    int64_t num_channels,
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    const fast_divmod div_output_hw,
    const fast_divmod div_output_width,
    const fast_divmod div_output_image,
    int32_t h_window_size, int32_t w_window_size,
    bool use_extrapolation, float extrapolation_value,
    const int64_t* h_bound_data,
    const int64_t* w_bound_data,
    const int64_t* h_outof_bounds,
    const int64_t* w_outof_bounds,
    const AccumType* h_weight_coefficients,
    const AccumType* w_weight_coefficients,
    const T* Xdata, T* Ydata,
    const int N) {
  static_assert(!std::is_same<AccumType, int32_t>::value,
                "Fused 2D kernel does not support int32_t accumulation (8-bit quantized weights)");

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // Decompose flat index → (batch, channel, y, x)
  int batch_idx, image_index;
  div_output_image.divmod(id, batch_idx, image_index);

  int channel_idx, spatial_index;
  div_output_hw.divmod(image_index, channel_idx, spatial_index);

  int output_y, output_x;
  div_output_width.divmod(spatial_index, output_y, output_x);

  auto* Ydata_out = Ydata + id;

  // Extrapolation: check if this output pixel maps outside the input
  if (use_extrapolation) {
    if (w_outof_bounds[output_x] != -1 || h_outof_bounds[output_y] != -1) {
      *Ydata_out = static_cast<T>(extrapolation_value);
      return;
    }
  }

  // Look up clamped input bounds for this output pixel
  const int64_t xmin = w_bound_data[static_cast<ptrdiff_t>(output_x) * 2];
  const int64_t xmax = w_bound_data[static_cast<ptrdiff_t>(output_x) * 2 + 1];
  const int64_t ymin = h_bound_data[static_cast<ptrdiff_t>(output_y) * 2];
  const int64_t ymax = h_bound_data[static_cast<ptrdiff_t>(output_y) * 2 + 1];

  const auto* w_weights = w_weight_coefficients + w_window_size * output_x;
  const auto* h_weights = h_weight_coefficients + h_window_size * output_y;

  const CUDA_LONG input_base = static_cast<CUDA_LONG>(
      (batch_idx * num_channels + channel_idx) * input_height * input_width);

  // Fused separable 2D filter: sum_{yi,xi} input[yi][xi] * h_w[yi-ymin] * w_w[xi-xmin]
  AccumType result = static_cast<AccumType>(0);
  for (int64_t yi = ymin; yi < ymax; ++yi) {
    const AccumType h_w = h_weights[yi - ymin];
    const auto* input_row = Xdata + input_base + yi * input_width;
    for (int64_t xi = xmin; xi < xmax; ++xi) {
      result += static_cast<AccumType>(input_row[xi]) * h_w * w_weights[xi - xmin];
    }
  }

  if constexpr (std::is_same<T, int32_t>::value) {
    *Ydata_out = static_cast<int32_t>(roundf(result));
  } else {
    *Ydata_out = static_cast<T>(result);
  }
}

/// Fused 2D antialias interpolation kernel for NHWC layout.
/// Each thread computes one output element (one channel of one spatial position).
/// Data layout: [batch, height, width, channels]
/// Flat index decomposition: id → (batch, y, x, channel)
///
/// The filter weights are shared across channels (same spatial interpolation),
/// so the H/W weight lookup is identical to the NCHW kernel — only the
/// input/output addressing changes.
template <typename T, typename AccumType>
__global__ void _ComputeFusedInterpolation2D_NHWC(
    int64_t num_channels,
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    const fast_divmod div_output_wc,       // output_width * num_channels
    const fast_divmod div_output_channel,  // num_channels
    const fast_divmod div_output_image,    // output_height * output_width * num_channels
    int32_t h_window_size, int32_t w_window_size,
    bool use_extrapolation, float extrapolation_value,
    const int64_t* h_bound_data,
    const int64_t* w_bound_data,
    const int64_t* h_outof_bounds,
    const int64_t* w_outof_bounds,
    const AccumType* h_weight_coefficients,
    const AccumType* w_weight_coefficients,
    const T* Xdata, T* Ydata,
    const int N) {
  static_assert(!std::is_same<AccumType, int32_t>::value,
                "Fused 2D NHWC kernel does not support int32_t accumulation");

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // Decompose flat index → (batch, y, x, channel) for NHWC layout
  int batch_idx, image_index;
  div_output_image.divmod(id, batch_idx, image_index);

  int output_y, wc_index;
  div_output_wc.divmod(image_index, output_y, wc_index);

  int output_x, channel_idx;
  div_output_channel.divmod(wc_index, output_x, channel_idx);

  auto* Ydata_out = Ydata + id;

  // Extrapolation: check if this output pixel maps outside the input
  if (use_extrapolation) {
    if (w_outof_bounds[output_x] != -1 || h_outof_bounds[output_y] != -1) {
      *Ydata_out = static_cast<T>(extrapolation_value);
      return;
    }
  }

  // Look up clamped input bounds for this output pixel
  const int64_t xmin = w_bound_data[static_cast<ptrdiff_t>(output_x) * 2];
  const int64_t xmax = w_bound_data[static_cast<ptrdiff_t>(output_x) * 2 + 1];
  const int64_t ymin = h_bound_data[static_cast<ptrdiff_t>(output_y) * 2];
  const int64_t ymax = h_bound_data[static_cast<ptrdiff_t>(output_y) * 2 + 1];

  const auto* w_weights = w_weight_coefficients + w_window_size * output_x;
  const auto* h_weights = h_weight_coefficients + h_window_size * output_y;

  // NHWC input base: batch * H_in * W_in * C
  const CUDA_LONG input_batch_base = static_cast<CUDA_LONG>(batch_idx) * input_height * input_width * num_channels;

  // Fused separable 2D filter over the spatial window, reading one channel
  // NHWC addressing: input[batch][yi][xi][channel] = input_batch_base + yi * W_in * C + xi * C + channel_idx
  AccumType result = static_cast<AccumType>(0);
  for (int64_t yi = ymin; yi < ymax; ++yi) {
    const AccumType h_w = h_weights[yi - ymin];
    const CUDA_LONG row_base = input_batch_base + static_cast<CUDA_LONG>(yi) * input_width * num_channels;
    for (int64_t xi = xmin; xi < xmax; ++xi) {
      const CUDA_LONG input_idx = row_base + static_cast<CUDA_LONG>(xi) * num_channels + channel_idx;
      result += static_cast<AccumType>(Xdata[input_idx]) * h_w * w_weights[xi - xmin];
    }
  }

  if constexpr (std::is_same<T, int32_t>::value) {
    *Ydata_out = static_cast<int32_t>(roundf(result));
  } else {
    *Ydata_out = static_cast<T>(result);
  }
}

template <typename T, typename AccumType>
__global__ void _ComputeInterpolationAtLevel3(
    int64_t input_depth,
    int64_t input_height, int64_t input_width,
    int64_t output_depth,
    int64_t output_height, int64_t output_width,
    const fast_divmod div_output_height,
    const fast_divmod div_output_width,
    const fast_divmod div_output_image,
    int32_t window_size,
    bool use_extrapolation, float extrapolation_value,
    const uint8_t* clip8_table,
    const int64_t* bound_data,
    std::tuple<int64_t*, int64_t*, int64_t*> outof_bounds_buffers,
    const AccumType* weight_coefficients,
    const T* Xdata, T* Ydata, int N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // No need to do scale
  if (input_depth == output_depth) {
    Ydata[id] = Xdata[id];
    return;
  }

  int bxc, output_image_index;
  div_output_image.divmod(id, bxc, output_image_index);

  int output_z, output_y, output_x, temp;
  div_output_height.divmod(output_image_index, output_z, temp);
  div_output_width.divmod(temp, output_y, output_x);

  CUDA_LONG input_index = static_cast<CUDA_LONG>(bxc * input_depth * input_height * input_width);

  auto* Ydata_offset = Ydata + id;

  if (use_extrapolation) {
    const auto* w_outof_bounds = std::get<2>(outof_bounds_buffers);
    // Extrapolate along the w dimension
    if (w_outof_bounds[static_cast<ptrdiff_t>(output_x)] != -1) {
      *Ydata_offset = static_cast<T>(extrapolation_value);
      return;
    }

    // Extrapolate along the y dimension
    const auto* y_outof_bounds = std::get<1>(outof_bounds_buffers);
    if (y_outof_bounds[static_cast<ptrdiff_t>(output_y)] != -1) {
      *Ydata_offset = static_cast<T>(extrapolation_value);
      return;
    }

    // Extrapolate along the z dimension
    const int64_t* z_outof_bounds = std::get<0>(outof_bounds_buffers);
    if (z_outof_bounds != nullptr && z_outof_bounds[static_cast<ptrdiff_t>(output_z)] != -1) {
      *Ydata_offset = static_cast<T>(extrapolation_value);
      return;
    }
  }

  const auto* bound = bound_data;

  AccumType output = onnxruntime::is_8bit_v<T> ? ConstValue::mag_factor : 0;

  const auto* weight_coeff = weight_coefficients + window_size * output_z;
  int64_t zmin = bound[static_cast<ptrdiff_t>(output_z) * 2];
  int64_t zmax = bound[static_cast<ptrdiff_t>(output_z) * 2 + 1];

  const auto z_step = input_height * input_width;
  const auto* Xdata_offset = Xdata + input_index + zmin * z_step + output_y * output_width + output_x;

  for (; zmin < zmax; ++zmin) {
    if constexpr (std::is_same<T, half>::value) {
      // We cast to AccumType to resolve ambiguous call to operator* for half in CUDA
      output += static_cast<AccumType>((*Xdata_offset)) * (*weight_coeff++);
    } else {
      output += (*Xdata_offset) * (*weight_coeff++);
    }
    Xdata_offset += z_step;
  }

  if constexpr (onnxruntime::is_8bit_v<T>) {
    const uint8_t* clip8_lookups = &clip8_table[640];
    *Ydata_offset = static_cast<T>(clip8_lookups[output >> 22]);
  } else if constexpr (std::is_same<T, int32_t>::value) {
    *Ydata_offset = static_cast<int32_t>(std::round(output));
  } else {
    *Ydata_offset = output;
  }
}

/// <summary>
/// This function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2]
/// 2. out_of_bounds: int64_t[output_size]
/// 3. scale_data: T[output_size * window_size]
///
/// Template parameter AccumType
/// </summary>
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
FUNC_DEF void SetupUpsampleFilterAnitAliasImpl(
    int64_t i,
    int64_t input_size, int64_t output_size,
    float rscale,
    float roi_start, float roi_end,
    float scaled_support, int32_t window_size, bool exclude_outside,
    float cubic_coeff_a,
    int64_t* bounds,
    int64_t* out_of_bounds,
    AccumType* scale_data) {
  Filter filter{};
  CudaFunctionOriginalCoordinate get_original_coordinate{};

  const auto scale = 1.f / rscale;
  const float inv_scale = (scale >= 1.0f) ? 1.0f / scale : 1.0f;

  const float id = static_cast<float>(i);
  float center = 0.5f;
  if (scale == 1.0f) {
    center += id;
  } else {
    center += get_original_coordinate(id, rscale,
                                      static_cast<float>(output_size),
                                      static_cast<float>(input_size),
                                      roi_start, roi_end);
  }

  if (center - 0.5f < 0 || center - 0.5f > static_cast<float>(input_size - 1)) {
    out_of_bounds[i] = i;
  } else {
    out_of_bounds[i] = -1;
  }

  float total_weight{0};

  auto fmin = _Floor(center - scaled_support + 0.5f);
  auto fmax = _Floor(center + scaled_support + 0.5f);

  int64_t min_real = static_cast<int64_t>(fmin);
  int64_t max_real = static_cast<int64_t>(fmax);
  int64_t min_cut = std::max<int64_t>(min_real, 0);
  int64_t max_cut = std::min(max_real, input_size);

  int64_t min_val = exclude_outside ? min_cut : min_real;
  int64_t max_val = exclude_outside ? max_cut : max_real;
  bounds[i * 2] = min_cut;
  bounds[i * 2 + 1] = max_cut;

  // This is done for int32_t case, when the final result is in int32_t, but
  // we perform calculations in float. All other types as is.
  auto* scale_buffer = AccumTypeCaster<AccumType>::cast(&scale_data[i * window_size]);

  max_val -= min_val;
  for (int64_t x = 0; x < max_val; x++) {
    const float arg = (x + min_val - center + 0.5f) * inv_scale;
    const auto w = filter(arg, cubic_coeff_a);
    scale_buffer[x] = w;
    total_weight += w;
  }

  if (!exclude_outside) {
    int64_t neg_xsize = min_val < 0 ? -min_val : 0;
    for (int64_t x = 0; x < neg_xsize; x++) {
      scale_buffer[neg_xsize] += scale_buffer[x];
    }

    int64_t bound_size =
        max_val + min_val > input_size ? max_val + min_val - input_size : 0;
    for (int64_t x = max_val - bound_size; x < max_val; x++) {
      scale_buffer[max_val - bound_size - 1] +=
          scale_buffer[x];
    }

    for (int64_t x = 0; (neg_xsize | bound_size) > 0 && x < max_cut - min_cut; x++) {
      scale_buffer[x] = scale_buffer[x + neg_xsize];
    }
  }

  const float total_weight_inv = (total_weight == 0) ? 1.f : (1.f / total_weight);
  if constexpr (std::is_same<AccumType, int32_t>::value) {
    auto* scale_buffer_int = reinterpret_cast<int32_t*>(scale_buffer);
    for (int64_t x = 0; x < max_cut - min_cut; x++) {
      scale_buffer[x] *= total_weight_inv;
      // normalize the scale to 1 << 22 for int8/uint8
      scale_buffer_int[x] = static_cast<int32_t>(_Round(scale_buffer[x] * ConstValue::mag_factor_x_2));
    }
  } else {
    for (int64_t x = 0; x < max_cut - min_cut; x++) {
      scale_buffer[x] *= total_weight_inv;
    }
  }
}

/// This kernel computes antialias filter for bilinear or bicubic upsampling.
/// The function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2] for each of the two dimensions
/// 2. out_of_bounds: int64_t[output_size] for each of the two dimensions
/// 3. scale_data: AccumType[output_size * window_size] for each of the two dimensions
/// Buffers layout [h_data, w_data]
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
__global__ void _SetupBilinearUpsampleFilterAntiAlias(
    std::tuple<int64_t, int64_t> input_dims,       // h, w
    std::tuple<int64_t, int64_t> output_dims,      // h, w
    std::tuple<float, float> inv_scale_vals,       // h, w
    std::tuple<float, float> roi_start_vals,       // h, w
    std::tuple<float, float> roi_end_vals,         // h, w
    std::tuple<float, float> dim_scaled_support,   // Pre-computed scaled support values h, w
    std::tuple<int32_t, int32_t> dim_window_size,  // Pre-computed windows sizes h, w
    float cubic_coeff_a,
    bool exclude_outside,
    int64_t* bounds,
    int64_t* out_of_bounds,
    std::tuple<AccumType*, AccumType*> weighted_coefficients  // y, h buffers
) {
  const auto N = std::get<0>(output_dims) + std::get<1>(output_dims);

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  if (id < std::get<0>(output_dims)) {
    // Setup for y
    int64_t input_size = std::get<0>(input_dims);
    int64_t output_size = std::get<0>(output_dims);
    float inv_scale = std::get<0>(inv_scale_vals);
    float roi_start = std::get<0>(roi_start_vals);
    float roi_end = std::get<0>(roi_end_vals);
    float scaled_support = std::get<0>(dim_scaled_support);
    int32_t window_size = std::get<0>(dim_window_size);

    SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
        id,
        input_size, output_size,
        inv_scale,
        roi_start, roi_end,
        scaled_support, window_size,
        exclude_outside,
        cubic_coeff_a,
        bounds,
        out_of_bounds,
        std::get<0>(weighted_coefficients));

  } else {
    // Setup for w
    // w = id - output_height

    int64_t input_size = std::get<1>(input_dims);
    int64_t output_size = std::get<1>(output_dims);
    float inv_scale = std::get<1>(inv_scale_vals);
    float roi_start = std::get<1>(roi_start_vals);
    float roi_end = std::get<1>(roi_end_vals);

    float scaled_support = std::get<1>(dim_scaled_support);
    int32_t window_size = std::get<1>(dim_window_size);

    // Adjust buffer positions
    const auto y_output_size = std::get<0>(output_dims);

    auto i = id - y_output_size;
    bounds += (y_output_size * 2);
    out_of_bounds += y_output_size;

    SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
        i,
        input_size, output_size,
        inv_scale,
        roi_start, roi_end,
        scaled_support, window_size,
        exclude_outside,
        cubic_coeff_a,
        bounds,
        out_of_bounds,
        std::get<1>(weighted_coefficients));
  }
}

/// <summary>
/// Compute AntiAlias filter for trilinear upsampling, all in one go
/// The function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2] for each of the three dimensions
/// 2. out_of_bounds: int64_t[output_size] for each of the three dimensions
/// 3. scale_data: AccumType[output_size * window_size] for each of the three dimensions
/// Each kind of buffer contains data for all 3 dims.
/// Buffers layout [d_data, h_data, w_data]
/// </summary>
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
__global__ void _SetupTrilinerarUpsampleFilterAntiAlias(
    std::tuple<int64_t, int64_t, int64_t> input_dims,       // d, h, w
    std::tuple<int64_t, int64_t, int64_t> output_dims,      // d, h, w
    std::tuple<float, float, float> inv_scale_vals,         // d, h, w
    std::tuple<float, float, float> roi_start_vals,         // d, h, w
    std::tuple<float, float, float> roi_end_vals,           // d, h, w
    std::tuple<float, float, float> dim_scaled_support,     // Pre-computed scaled support values d, h, w
    std::tuple<int32_t, int32_t, int32_t> dim_window_size,  // Pre-computed windows sizes d, h, w
    bool exclude_outisde,
    int64_t* bounds,
    int64_t* out_of_bounds,
    std::tuple<AccumType*, AccumType*, AccumType*> weighted_coefficients) {
  const auto N = std::get<0>(output_dims) + std::get<1>(output_dims) + std::get<2>(output_dims);

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  if (id < std::get<0>(output_dims)) {
    // Setup for d by default (id < output_depth)
    int64_t input_size = std::get<0>(input_dims);
    int64_t output_size = std::get<0>(output_dims);
    float inv_scale = std::get<0>(inv_scale_vals);
    float roi_start = std::get<0>(roi_start_vals);
    float roi_end = std::get<0>(roi_end_vals);
    float scaled_support = std::get<0>(dim_scaled_support);
    int32_t window_size = std::get<0>(dim_window_size);

    SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
        id,
        input_size, output_size,
        inv_scale,
        roi_start, roi_end,
        scaled_support, window_size,
        exclude_outisde,
        onnxruntime::antialias_constants::kCubicCoeffA,  // Default value for trilinear
        bounds,
        out_of_bounds,
        std::get<0>(weighted_coefficients));

  } else if (id >= std::get<0>(output_dims) && id < (std::get<0>(output_dims) + std::get<1>(output_dims))) {
    int64_t input_size = std::get<1>(input_dims);
    int64_t output_size = std::get<1>(output_dims);
    float inv_scale = std::get<1>(inv_scale_vals);
    float roi_start = std::get<1>(roi_start_vals);
    float roi_end = std::get<1>(roi_end_vals);

    float scaled_support = std::get<1>(dim_scaled_support);
    int32_t window_size = std::get<1>(dim_window_size);

    // Adjust buffer positions
    const auto d_output_size = std::get<0>(output_dims);

    auto i = id - d_output_size;
    bounds += d_output_size * 2;
    out_of_bounds += d_output_size;

    SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
        i,
        input_size, output_size,
        inv_scale,
        roi_start, roi_end,
        scaled_support, window_size,
        exclude_outisde,
        onnxruntime::antialias_constants::kCubicCoeffA,  // Default value for trilinear
        bounds,
        out_of_bounds,
        std::get<1>(weighted_coefficients));
  } else {
    int64_t input_size = std::get<2>(input_dims);
    int64_t output_size = std::get<2>(output_dims);
    float inv_scale = std::get<2>(inv_scale_vals);
    float roi_start = std::get<2>(roi_start_vals);
    float roi_end = std::get<2>(roi_end_vals);
    float scaled_support = std::get<2>(dim_scaled_support);
    int32_t window_size = std::get<2>(dim_window_size);

    // Adjust buffer positions
    const auto d_y_output_size = std::get<0>(output_dims) + std::get<1>(output_dims);

    auto i = id - d_y_output_size;
    bounds += (d_y_output_size * 2);
    out_of_bounds += d_y_output_size;

    SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
        i,
        input_size, output_size,
        inv_scale,
        roi_start, roi_end,
        scaled_support, window_size,
        exclude_outisde,
        onnxruntime::antialias_constants::kCubicCoeffA,  // Default value for trilinear
        bounds,
        out_of_bounds,
        std::get<2>(weighted_coefficients));
  }
}

#define CASEA_COORD_ANTIALIAS(coordinate_mode, TransformCoordType, ...) \
  case coordinate_mode: {                                               \
    using coord_t = TransformCoordType;                                 \
    return __VA_ARGS__();                                               \
    break;                                                              \
  }

#define DISPATCH_ANTIALIAS_FILTER_SETUP(coord_enum, ...)                              \
  [&] {                                                                               \
    const auto the_type = coord_enum;                                                 \
    switch (the_type) {                                                               \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::HALF_PIXEL,           \
                            TransformCoordinate_HALF_PIXEL, __VA_ARGS__)              \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::ASYMMETRIC,           \
                            TransformCoordinate_ASYMMETRIC, __VA_ARGS__)              \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL,   \
                            TransformCoordinate_PYTORCH_HALF_PIXEL, __VA_ARGS__)      \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::ALIGN_CORNERS,        \
                            TransformCoordinate_ALIGN_CORNERS, __VA_ARGS__)           \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN, \
                            TransformCoordinate_TF_HALF_PIXEL_FOR_NN, __VA_ARGS__)    \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE,   \
                            TransformCoordinate_TF_CROP_AND_RESIZE, __VA_ARGS__)      \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::HALF_PIXEL_SYMMETRIC, \
                            TransformCoordinate_HALF_PIXEL_SYMMETRIC, __VA_ARGS__)    \
      default:                                                                        \
        ORT_THROW("unknown ResizeCoordinateTransformationMode");                      \
    }                                                                                 \
  }()

namespace {
template <typename T>
IAllocatorUniquePtr<uint8_t> AllocateTyped(
    const TempSpaceAllocateFunc& alloc,
    size_t elements) {
  return alloc(elements * sizeof(T));
}

template <typename T>
T* GetTyped(IAllocatorUniquePtr<uint8_t>& bytes) {
  return reinterpret_cast<T*>(bytes.get());
}
}  // namespace

template <typename T>
void ResizeTrilinearUpsample(
    cudaStream_t stream,
    int rank,
    const UpsampleMode /*upsample_mode*/,
    ResizeCoordinateTransformationMode coordinate_transform_mode,
    gsl::span<const int64_t> /*input_shape*/,
    gsl::span<const int64_t> /*output_shape*/,
    int64_t batch_size, int64_t num_channels,
    std::tuple<int64_t, int64_t, int64_t> inferred_input_dims,
    std::tuple<int64_t, int64_t, int64_t> inferred_output_dims,
    std::tuple<float, float, float> inferred_dim_rscales,
    const TArray<fast_divmod>& output_div_pitches,
    gsl::span<const float> roi_vals,
    const std::optional<float>& extrapolation,
    bool exclude_outside,
    const TempSpaceAllocateFunc& allocate_temp_space,
    const uint8_t* clip8_lookups,
    const T* input_data,
    T* output_data,
    const size_t N) {
  using AccumType = typename onnxruntime::AccumulateType<T>::type;

  const bool use_extrapolation = extrapolation.has_value();
  const float extrapolation_value = use_extrapolation ? *extrapolation : 0.f;

  int64_t input_depth, input_height, input_width;
  std::tie(input_depth, input_height, input_width) = inferred_input_dims;

  int64_t output_depth, output_height, output_width;
  std::tie(output_depth, output_height, output_width) = inferred_output_dims;

  int blocksPerDimsMappingGrid =
      static_cast<int>(ceil((output_depth + output_height + output_width) / 32.0));

  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  constexpr float support_value = antialias_constants::kSupportSize;
  float z_scale, h_scale, w_scale;
  std::tie(z_scale, h_scale, w_scale) = inferred_dim_rscales;

  const auto& div_output_width = output_div_pitches[rank - 2];

  SafeInt<int64_t> bounds_buffer_size = (SafeInt<int64_t>(output_depth) + output_height + output_width) * 2;
  SafeInt<int64_t> out_of_bounds_buffer_size = (SafeInt<int64_t>(output_depth) + output_height + output_width);

  auto bounds_buffer_ptr = AllocateTyped<int64_t>(allocate_temp_space, bounds_buffer_size);
  auto out_of_bounds_buffer_ptr = AllocateTyped<int64_t>(allocate_temp_space, out_of_bounds_buffer_size);

  int64_t* z_bounds_buffer = GetTyped<int64_t>(bounds_buffer_ptr);
  int64_t* y_bounds_buffer = z_bounds_buffer + output_depth * 2;
  int64_t* w_bounds_buffer = y_bounds_buffer + output_height * 2;

  int64_t* z_outof_bounds_buffer = GetTyped<int64_t>(out_of_bounds_buffer_ptr);
  int64_t* y_outof_bounds_buffer = z_outof_bounds_buffer + output_depth;
  int64_t* w_outof_bounds_buffer = y_outof_bounds_buffer + output_height;

  float z_scaled_support, h_scaled_support, w_scaled_support;
  int32_t z_window_size, h_window_size, w_window_size;
  const auto [z_buffer_size, y_buffer_size, w_buffer_size] = ComputeTrilinearScaleBufferSize(
      output_depth, output_height, output_width,
      z_scale, h_scale, w_scale, support_value,
      z_scaled_support, h_scaled_support, w_scaled_support,
      z_window_size, h_window_size, w_window_size);

  const int64_t weighted_buffer_size = SafeInt<int64_t>(z_buffer_size) + y_buffer_size + w_buffer_size;

  auto weighted_buffer_ptr = AllocateTyped<AccumType>(allocate_temp_space, weighted_buffer_size);
  AccumType* z_weighted_buffer = GetTyped<AccumType>(weighted_buffer_ptr);
  AccumType* y_weighted_buffer = z_weighted_buffer + z_buffer_size;
  AccumType* w_weighted_buffer = y_weighted_buffer + y_buffer_size;

  const auto h_w_interpolate_temp_buf_size = SafeInt<int64_t>(batch_size) * num_channels *
                                             input_depth * input_height * output_width;
  auto h_w_interpolate_temp_buffer_ptr = AllocateTyped<T>(allocate_temp_space,
                                                          narrow<size_t>(h_w_interpolate_temp_buf_size));

  const auto h_w_interpolate_result_buffer_size = SafeInt<int64_t>(batch_size) * num_channels *
                                                  input_depth * output_height * output_width;
  auto h_w_interpolate_result_buffer_ptr = AllocateTyped<T>(allocate_temp_space, h_w_interpolate_result_buffer_size);

  // clang-format off
  DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
    _SetupTrilinerarUpsampleFilterAntiAlias<AccumType,
                                            TriLinearFilter,
                                            coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
        inferred_input_dims,
        inferred_output_dims,
        inferred_dim_rscales,
        std::make_tuple(roi_vals[rank - 3], roi_vals[rank - 2], roi_vals[rank - 1]),  // roi starts d, h, w
        std::make_tuple(roi_vals[rank - 3 + rank], roi_vals[rank - 2 + rank],         // roi ends d, h, w
                        roi_vals[rank - 1 + rank]),
        std::make_tuple(z_scaled_support, h_scaled_support, w_scaled_support),
        std::make_tuple(z_window_size, h_window_size, w_window_size),
        exclude_outside,
        GetTyped<int64_t>(bounds_buffer_ptr),
        GetTyped<int64_t>(out_of_bounds_buffer_ptr),
        std::make_tuple(z_weighted_buffer, y_weighted_buffer, w_weighted_buffer));
  });

  // clang-format on
  const fast_divmod div_w_image(narrow<int>(num_channels * input_depth * input_height * output_width));
  // clang-format off
  _ComputeInterpolationAtLevel1<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      num_channels * input_depth, input_height, input_width, input_height, output_width,
      div_output_width,
      div_w_image,
      w_window_size,
      clip8_lookups,
      w_bounds_buffer,
      std::make_tuple(y_outof_bounds_buffer, w_outof_bounds_buffer),
      w_weighted_buffer, input_data,
      GetTyped<T>(h_w_interpolate_temp_buffer_ptr),
      narrow<int>(h_w_interpolate_temp_buf_size));

  // clang-format on
  const fast_divmod div_output_height{narrow<int>(output_height * output_width)};
  const fast_divmod div_h_w_image(narrow<int>(num_channels * input_depth * output_height * output_width));
  // clang-format off
  _ComputeInterpolationAtLevel2<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      num_channels * input_depth, input_height, output_width, output_height, output_width,
      div_output_height,
      div_output_width,
      div_h_w_image,
      h_window_size,
      false, 0.f,  // No extrapolation
      clip8_lookups,
      y_bounds_buffer,
      std::make_tuple(y_outof_bounds_buffer, w_outof_bounds_buffer),
      y_weighted_buffer, GetTyped<T>(h_w_interpolate_temp_buffer_ptr),
      GetTyped<T>(h_w_interpolate_result_buffer_ptr),
      narrow<int>(h_w_interpolate_result_buffer_size));

  // clang-format on
  const fast_divmod div_z_h_w_image(narrow<int>(input_depth * output_height * output_width));
  // clang-format off
  _ComputeInterpolationAtLevel3<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_depth, output_height, output_width,
      output_depth, output_height, output_width,
      div_output_height,
      div_output_width,
      div_z_h_w_image,
      z_window_size,
      use_extrapolation, extrapolation_value,
      clip8_lookups,
      z_bounds_buffer,
      std::make_tuple(z_outof_bounds_buffer, y_outof_bounds_buffer, w_outof_bounds_buffer),
      z_weighted_buffer, GetTyped<T>(h_w_interpolate_result_buffer_ptr),
      output_data,
      narrow<int>(N));
  // clang-format on
}

template <class T>
void ResizeBiLinearUpsample(cudaStream_t stream,
                            int rank,
                            const UpsampleMode /*upsample_mode*/,
                            ResizeCoordinateTransformationMode coordinate_transform_mode,
                            gsl::span<const int64_t> /*input_shape*/,
                            gsl::span<const int64_t> /*output_shape*/,
                            int64_t /*batch_size*/, int64_t num_channels,
                            std::tuple<int64_t, int64_t, int64_t> inferred_input_dims,
                            std::tuple<int64_t, int64_t, int64_t> inferred_output_dims,
                            std::tuple<float, float, float> inferred_dim_rscales,
                            const TArray<fast_divmod>& output_div_pitches,
                            gsl::span<const float> roi_vals,
                            const std::optional<float>& extrapolation,
                            bool exclude_outside,
                            const TempSpaceAllocateFunc& allocate_temp_space,
                            const uint8_t* clip8_lookups,
                            const T* input_data,
                            T* output_data,
                            const size_t N) {
  using AccumType = typename onnxruntime::AccumulateType<T>::type;

  const bool use_extrapolation = extrapolation.has_value();
  const float extrapolation_value = use_extrapolation ? *extrapolation : 0.f;

  int64_t input_depth, input_height, input_width;
  std::tie(input_depth, input_height, input_width) = inferred_input_dims;

  int64_t output_depth, output_height, output_width;
  std::tie(output_depth, output_height, output_width) = inferred_output_dims;

  int blocksPerDimsMappingGrid =
      narrow<int>(CeilDiv((output_depth + output_height + output_width), 32));

  // rank 2 or 4
  const fast_divmod div_output_image = (rank > 2) ? output_div_pitches[rank - 4]
                                                  : fast_divmod(gsl::narrow_cast<int>(N));
  const fast_divmod& div_output_width = output_div_pitches[rank - 2];

  constexpr float support_value = antialias_constants::kSupportSize;

  float h_scale, w_scale;
  std::tie(std::ignore, h_scale, w_scale) = inferred_dim_rscales;

  int blocksPerGrid = narrow<int>(CeilDiv(N, GridDim::maxThreadsPerBlock));

  SafeInt<int64_t> bounds_buffer_size = (SafeInt<int64_t>(output_height) + output_width) * 2;
  SafeInt<int64_t> out_of_bounds_buffer_size = (SafeInt<int64_t>(output_height) + output_width);

  float h_scaled_support, w_scaled_support;
  int32_t h_window_size, w_window_size;
  const auto [weighted_y_size, weighted_w_size] =
      ComputeBilinearScaleBufferSize(output_height, output_width,
                                     h_scale, w_scale, support_value,
                                     h_scaled_support, w_scaled_support, h_window_size, w_window_size);

  auto bounds_buffer_ptr = AllocateTyped<int64_t>(allocate_temp_space, bounds_buffer_size);
  auto out_of_bounds_buffer_ptr = AllocateTyped<int64_t>(allocate_temp_space, out_of_bounds_buffer_size);

  int64_t* y_bounds_buffer = GetTyped<int64_t>(bounds_buffer_ptr);
  int64_t* w_bounds_buffer = y_bounds_buffer + output_height * 2;

  int64_t* y_outof_bounds_buffer = GetTyped<int64_t>(out_of_bounds_buffer_ptr);
  int64_t* w_outof_bounds_buffer = y_outof_bounds_buffer + output_height;

  const int64_t weighted_buffer_size = SafeInt<int64_t>(weighted_y_size) + weighted_w_size;
  auto weighted_buffer_ptr = AllocateTyped<AccumType>(allocate_temp_space, narrow<size_t>(weighted_buffer_size));

  AccumType* y_weighted_buffer = GetTyped<AccumType>(weighted_buffer_ptr);
  AccumType* w_weighted_buffer = y_weighted_buffer + weighted_y_size;

  // clang-format off
  DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
    _SetupBilinearUpsampleFilterAntiAlias<AccumType,
                                          BilinearFilter,
                                          coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
        std::make_tuple(input_height, input_width),
        std::make_tuple(output_height, output_width),
        std::make_tuple(h_scale, w_scale),
        std::make_tuple(roi_vals[rank - 2], roi_vals[rank - 1]),                // roi starts h, w
        std::make_tuple(roi_vals[rank - 2 + rank], roi_vals[rank - 1 + rank]),  // roi ends h, w
        std::make_tuple(h_scaled_support, w_scaled_support),
        std::make_tuple(h_window_size, w_window_size),
        onnxruntime::antialias_constants::kCubicCoeffA, exclude_outside,
        GetTyped<int64_t>(bounds_buffer_ptr),
        GetTyped<int64_t>(out_of_bounds_buffer_ptr),
        std::make_tuple(y_weighted_buffer, w_weighted_buffer));
  });
  // clang-format on

  if constexpr (!std::is_same_v<AccumType, int32_t> && !std::is_same_v<T, int32_t>) {
    // Fused 2D kernel: single pass reads input and writes output directly.
    // Eliminates the intermediate buffer and second kernel launch.
    // Not available for int32_t accumulation (8-bit quantized weights) because
    // fusing h_weight * w_weight in fixed-point requires different quantization.
    // Also excluded for int32_t output because the different operation order
    // causes floating-point precision differences near 0.5 that affect rounding.
    const fast_divmod div_output_hw{narrow<int>(output_height * output_width)};
    // clang-format off
    _ComputeFusedInterpolation2D<T, AccumType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        num_channels,
        input_height, input_width, output_height, output_width,
        div_output_hw,
        div_output_width,
        div_output_image,
        h_window_size, w_window_size,
        use_extrapolation, extrapolation_value,
        y_bounds_buffer, w_bounds_buffer,
        y_outof_bounds_buffer, w_outof_bounds_buffer,
        y_weighted_buffer, w_weighted_buffer,
        input_data, output_data,
        narrow<int>(N));
    // clang-format on
  } else {
    // Two-pass separable approach for integer types.
    // Level1: resize W dimension (input → temp buffer)
    // Level2: resize H dimension (temp buffer → output)
    const auto temp_buf_size = num_channels * input_height * output_width;
    auto image_temp_buffer = AllocateTyped<T>(allocate_temp_space, narrow<size_t>(temp_buf_size));

    const fast_divmod div_step_image{narrow<int>(num_channels * input_height * output_width)};
    // clang-format off
    _ComputeInterpolationAtLevel1<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        num_channels, input_height, input_width, input_height, output_width,
        div_output_width,
        div_step_image,
        w_window_size,
        clip8_lookups,
        w_bounds_buffer,
        std::make_tuple(y_outof_bounds_buffer, w_outof_bounds_buffer),
        w_weighted_buffer, input_data, GetTyped<T>(image_temp_buffer),
        narrow<int>(temp_buf_size));
    // clang-format on

    const fast_divmod div_output_height{narrow<int>(output_height * output_width)};
    // clang-format off
    _ComputeInterpolationAtLevel2<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        num_channels, input_height, output_width, output_height, output_width,
        div_output_height,
        div_output_width,
        div_output_image,
        h_window_size,
        use_extrapolation, extrapolation_value,
        clip8_lookups,
        y_bounds_buffer,
        std::make_tuple(y_outof_bounds_buffer, w_outof_bounds_buffer),
        y_weighted_buffer, GetTyped<T>(image_temp_buffer), output_data,
        narrow<int>(N));
    // clang-format on
  }
}

template <typename T>
void ResizeBicubicUpsample(cudaStream_t stream,
                           int rank,
                           const UpsampleMode /*upsample_mode*/,
                           ResizeCoordinateTransformationMode coordinate_transform_mode,
                           const float cubic_coeff_a,
                           gsl::span<const int64_t> /*input_shape*/,
                           gsl::span<const int64_t> /*output_shape*/,
                           int64_t batch_size, int64_t num_channels,
                           std::tuple<int64_t, int64_t, int64_t> inferred_input_dims,
                           std::tuple<int64_t, int64_t, int64_t> inferred_output_dims,
                           std::tuple<float, float, float> inferred_dim_rscales,
                           const TArray<fast_divmod>& output_div_pitches,
                           gsl::span<const float> roi_vals,
                           const std::optional<float>& extrapolation,
                           bool exclude_outside,
                           const TempSpaceAllocateFunc& allocate_temp_space,
                           const uint8_t* clip8_lookups,
                           const T* input_data,
                           T* output_data,
                           const size_t N) {
  using AccumType = typename onnxruntime::AccumulateType<T>::type;

  const bool use_extrapolation = extrapolation.has_value();
  const float extrapolation_value = use_extrapolation ? *extrapolation : 0.f;

  int64_t input_depth, input_height, input_width;
  std::tie(input_depth, input_height, input_width) = inferred_input_dims;

  int64_t output_depth, output_height, output_width;
  std::tie(output_depth, output_height, output_width) = inferred_output_dims;

  int blocksPerGrid = narrow<int>(CeilDiv(N, GridDim::maxThreadsPerBlock));
  const fast_divmod div_output_image = (rank > 2) ? output_div_pitches[rank - 4]
                                                  : fast_divmod(gsl::narrow_cast<int>(N));
  const fast_divmod& div_output_width = output_div_pitches[rank - 2];

  constexpr float support_value = antialias_constants::kBiCubicSupportSize;

  int blocksPerDimsMappingGrid =
      narrow<int>(CeilDiv((output_depth + output_height + output_width), 32));

  float h_scale, w_scale;
  std::tie(std::ignore, h_scale, w_scale) = inferred_dim_rscales;

  SafeInt<int64_t> bounds_buffer_size = (SafeInt<int64_t>(output_height) + output_width) * 2;
  SafeInt<int64_t> out_of_bounds_buffer_size = (SafeInt<int64_t>(output_height) + output_width);

  float h_scaled_support, w_scaled_support;
  int32_t h_window_size, w_window_size;
  const auto [weighted_y_size, weighted_w_size] =
      ComputeBilinearScaleBufferSize(output_height, output_width,
                                     h_scale, w_scale, support_value,
                                     h_scaled_support, w_scaled_support, h_window_size, w_window_size);

  auto bounds_buffer_ptr = AllocateTyped<int64_t>(allocate_temp_space, bounds_buffer_size);
  auto out_of_bounds_buffer_ptr = AllocateTyped<int64_t>(allocate_temp_space, out_of_bounds_buffer_size);

  int64_t* y_bounds_buffer = GetTyped<int64_t>(bounds_buffer_ptr);
  int64_t* w_bounds_buffer = y_bounds_buffer + output_height * 2;

  int64_t* y_outof_bounds_buffer = GetTyped<int64_t>(out_of_bounds_buffer_ptr);
  int64_t* w_outof_bounds_buffer = y_outof_bounds_buffer + output_height;

  const int64_t weighted_buffer_size = SafeInt<int64_t>(weighted_y_size) + weighted_w_size;
  auto weighted_buffer_ptr = AllocateTyped<AccumType>(allocate_temp_space, weighted_buffer_size);

  AccumType* y_weighted_buffer = GetTyped<AccumType>(weighted_buffer_ptr);
  AccumType* w_weighted_buffer = y_weighted_buffer + weighted_y_size;

  // clang-format off
  DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
    _SetupBilinearUpsampleFilterAntiAlias<AccumType,
                                          BiCubicFilter,
                                          coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
        std::make_tuple(input_height, input_width),
        std::make_tuple(output_height, output_width),
        std::make_tuple(h_scale, w_scale),
        std::make_tuple(roi_vals[rank - 2], roi_vals[rank - 1]),                // roi starts h, w
        std::make_tuple(roi_vals[rank - 2 + rank], roi_vals[rank - 1 + rank]),  // roi ends h, w
        std::make_tuple(h_scaled_support, w_scaled_support),
        std::make_tuple(h_window_size, w_window_size),
        cubic_coeff_a, exclude_outside,
        GetTyped<int64_t>(bounds_buffer_ptr),
        GetTyped<int64_t>(out_of_bounds_buffer_ptr),
        std::make_tuple(y_weighted_buffer, w_weighted_buffer));
  });
  // clang-format on

  if constexpr (!std::is_same_v<AccumType, int32_t> && !std::is_same_v<T, int32_t>) {
    // Fused 2D kernel: single pass reads input and writes output directly.
    const fast_divmod div_output_hw{narrow<int>(output_height * output_width)};
    // clang-format off
    _ComputeFusedInterpolation2D<T, AccumType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        num_channels,
        input_height, input_width, output_height, output_width,
        div_output_hw,
        div_output_width,
        div_output_image,
        h_window_size, w_window_size,
        use_extrapolation, extrapolation_value,
        y_bounds_buffer, w_bounds_buffer,
        y_outof_bounds_buffer, w_outof_bounds_buffer,
        y_weighted_buffer, w_weighted_buffer,
        input_data, output_data,
        narrow<int>(N));
    // clang-format on
  } else {
    // Two-pass separable approach for integer types (8-bit with int32_t accumulation,
    // or int32_t output where rounding is sensitive to floating-point operation order).
    const auto temp_buf_size = SafeInt<int64_t>(batch_size) * num_channels * input_height * output_width;
    auto image_temp_buffer = AllocateTyped<T>(allocate_temp_space, narrow<size_t>(temp_buf_size));

    int blocksPerGridL1 = narrow<int>(CeilDiv(temp_buf_size, GridDim::maxThreadsPerBlock));
    const fast_divmod div_step_image(narrow<int>(num_channels * input_height * output_width));
    // clang-format off
    _ComputeInterpolationAtLevel1<T><<<blocksPerGridL1, GridDim::maxThreadsPerBlock, 0, stream>>>(
        num_channels, input_height, input_width, input_height, output_width,
        div_output_width,
        div_step_image,
        w_window_size,
        clip8_lookups,
        w_bounds_buffer,
        std::make_tuple(y_outof_bounds_buffer, w_outof_bounds_buffer),
        w_weighted_buffer, input_data, GetTyped<T>(image_temp_buffer),
        narrow<int>(temp_buf_size));
    // clang-format on

    const fast_divmod div_output_height{narrow<int>(output_height * output_width)};
    // clang-format off
    _ComputeInterpolationAtLevel2<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        num_channels, input_height, output_width, output_height, output_width,
        div_output_height,
        div_output_width,
        div_output_image,
        h_window_size,
        use_extrapolation, extrapolation_value,
        clip8_lookups,
        y_bounds_buffer,
        std::make_tuple(y_outof_bounds_buffer, w_outof_bounds_buffer),
        y_weighted_buffer, GetTyped<T>(image_temp_buffer), output_data,
        narrow<int>(N));
    // clang-format on
  }
}

/// NHWC 2D antialias bilinear/bicubic launcher (fused kernel only).
/// Handles both bilinear and bicubic — the filter type is encoded in the precomputed weights.
/// For 8-bit types (int32_t accumulation), falls back to the NCHW separable path
/// after transposing, which is not implemented here — the caller should reject those types.
template <class T, typename FilterType>
void ResizeNhwcBilinearBicubicUpsample(cudaStream_t stream,
                                       ResizeCoordinateTransformationMode coordinate_transform_mode,
                                       float cubic_coeff_a,
                                       int64_t batch_size, int64_t num_channels,
                                       int64_t input_height, int64_t input_width,
                                       int64_t output_height, int64_t output_width,
                                       float height_scale, float width_scale,
                                       float support_value,
                                       gsl::span<const float> roi_vals,
                                       const std::optional<float>& extrapolation,
                                       bool exclude_outside,
                                       const TempSpaceAllocateFunc& allocate_temp_space,
                                       const T* input_data,
                                       T* output_data,
                                       const size_t N) {
  ORT_UNUSED_PARAMETER(batch_size);  // Used implicitly via N = batch_size * H * W * C
  using AccumType = typename onnxruntime::AccumulateType<T>::type;

  const bool use_extrapolation = extrapolation.has_value();
  const float extrapolation_value = use_extrapolation ? *extrapolation : 0.f;

  int blocksPerDimsMappingGrid =
      narrow<int>(CeilDiv((output_height + output_width), 32));

  int blocksPerGrid = narrow<int>(CeilDiv(N, GridDim::maxThreadsPerBlock));

  // NHWC divmods: flat index = batch * (H_out * W_out * C) + y * (W_out * C) + x * C + c
  const fast_divmod div_output_image{narrow<int>(output_height * output_width * num_channels)};
  const fast_divmod div_output_wc{narrow<int>(output_width * num_channels)};
  const fast_divmod div_output_channel{narrow<int>(num_channels)};

  float h_scaled_support, w_scaled_support;
  int32_t h_window_size, w_window_size;
  const auto [weighted_y_size, weighted_w_size] =
      ComputeBilinearScaleBufferSize(output_height, output_width,
                                     height_scale, width_scale, support_value,
                                     h_scaled_support, w_scaled_support, h_window_size, w_window_size);

  SafeInt<int64_t> bounds_buffer_size = (SafeInt<int64_t>(output_height) + output_width) * 2;
  SafeInt<int64_t> out_of_bounds_buffer_size = (SafeInt<int64_t>(output_height) + output_width);

  auto bounds_buffer_ptr = AllocateTyped<int64_t>(allocate_temp_space, bounds_buffer_size);
  auto out_of_bounds_buffer_ptr = AllocateTyped<int64_t>(allocate_temp_space, out_of_bounds_buffer_size);

  int64_t* y_bounds_buffer = GetTyped<int64_t>(bounds_buffer_ptr);
  int64_t* w_bounds_buffer = y_bounds_buffer + output_height * 2;

  int64_t* y_outof_bounds_buffer = GetTyped<int64_t>(out_of_bounds_buffer_ptr);
  int64_t* w_outof_bounds_buffer = y_outof_bounds_buffer + output_height;

  const int64_t weighted_buffer_size = SafeInt<int64_t>(weighted_y_size) + weighted_w_size;
  auto weighted_buffer_ptr = AllocateTyped<AccumType>(allocate_temp_space, narrow<size_t>(weighted_buffer_size));

  AccumType* y_weighted_buffer = GetTyped<AccumType>(weighted_buffer_ptr);
  AccumType* w_weighted_buffer = y_weighted_buffer + weighted_y_size;

  // roi_vals layout for NHWC 4D: [roi_start_h, roi_start_w, ..., roi_end_h, roi_end_w]
  // The caller already extracted H/W roi values at positions matching the spatial dims.
  // For the setup kernel, we pass roi_h and roi_w starts/ends.
  // roi_vals is always arranged as [start_dims..., end_dims...] with rank entries each.
  // For NHWC, spatial dims are at index 1 (H) and 2 (W) in the 4-element roi.

  // clang-format off
  DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
    _SetupBilinearUpsampleFilterAntiAlias<AccumType,
                                          FilterType,
                                          coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
        std::make_tuple(input_height, input_width),
        std::make_tuple(output_height, output_width),
        std::make_tuple(height_scale, width_scale),
        std::make_tuple(roi_vals[1], roi_vals[2]),                    // roi starts h, w (NHWC indices)
        std::make_tuple(roi_vals[1 + 4], roi_vals[2 + 4]),            // roi ends h, w
        std::make_tuple(h_scaled_support, w_scaled_support),
        std::make_tuple(h_window_size, w_window_size),
        cubic_coeff_a, exclude_outside,
        GetTyped<int64_t>(bounds_buffer_ptr),
        GetTyped<int64_t>(out_of_bounds_buffer_ptr),
        std::make_tuple(y_weighted_buffer, w_weighted_buffer));
  });
  // clang-format on

  // clang-format off
  _ComputeFusedInterpolation2D_NHWC<T, AccumType><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      num_channels,
      input_height, input_width, output_height, output_width,
      div_output_wc,
      div_output_channel,
      div_output_image,
      h_window_size, w_window_size,
      use_extrapolation, extrapolation_value,
      y_bounds_buffer, w_bounds_buffer,
      y_outof_bounds_buffer, w_outof_bounds_buffer,
      y_weighted_buffer, w_weighted_buffer,
      input_data, output_data,
      narrow<int>(N));
  // clang-format on
}

template <class T>
void ResizeAntiAliasImpl(
    cudaStream_t stream,
    int rank,
    const UpsampleMode upsample_mode,
    ResizeCoordinateTransformationMode coordinate_transform_mode,
    const float cubic_coeff_a,
    gsl::span<const int64_t> input_shape,
    gsl::span<const int64_t> output_shape,
    int64_t batch_size, int64_t num_channels,
    std::tuple<int64_t, int64_t, int64_t> inferred_input_dims,
    std::tuple<int64_t, int64_t, int64_t> inferred_output_dims,
    std::tuple<float, float, float> inferred_dim_rscales,
    const TArray<fast_divmod>& output_div_pitches,
    gsl::span<const float> roi_vals,
    const std::optional<float>& extrapolation,
    bool exclude_outside,
    bool is_nhwc,
    TempSpaceAllocateFunc allocate_temp_space,
    const uint8_t* clip8_lookups,
    const T* input_data,
    T* output_data,
    const size_t N) {
  // We support a special case of bilinear or bicubic if the input data is 4D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  const bool is_2D = (rank == 2 || rank == 4);

  // We support a special case of trilinear or tricubic if the input data is 5D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  const bool is_3D = (rank == 3 || rank == 5);

  // Should not hit this as we have already validated input rank/scales and we provide verbose error messages
  // to the user.
  ORT_ENFORCE(is_2D || is_3D, "Only bilinear/trilinear and bicubic modes are supported in Resize anti-alias mode");

  if (is_nhwc && is_2D) {
    using AccumType = typename onnxruntime::AccumulateType<T>::type;
    if constexpr (std::is_same_v<AccumType, int32_t>) {
      // 8-bit types (uint8_t, int8_t) use int32_t accumulation which is not supported
      // by the fused NHWC kernel. Reject at runtime.
      ORT_NOT_IMPLEMENTED("NHWC antialias resize is not supported for 8-bit integer types on CUDA");
    } else {
      // NHWC path: use the fused NHWC kernel for both bilinear and bicubic
      int64_t input_height, input_width;
      std::tie(std::ignore, input_height, input_width) = inferred_input_dims;

      int64_t output_height, output_width;
      std::tie(std::ignore, output_height, output_width) = inferred_output_dims;

      float height_scale, width_scale;
      std::tie(std::ignore, height_scale, width_scale) = inferred_dim_rscales;

      switch (upsample_mode) {
        case UpsampleMode::LINEAR: {
          constexpr float support_value = antialias_constants::kSupportSize;
          ResizeNhwcBilinearBicubicUpsample<T, BilinearFilter>(
              stream, coordinate_transform_mode, cubic_coeff_a,
              batch_size, num_channels,
              input_height, input_width, output_height, output_width,
              height_scale, width_scale, support_value,
              roi_vals, extrapolation, exclude_outside,
              allocate_temp_space, input_data, output_data, N);
        } break;
        case CUBIC: {
          constexpr float support_value = antialias_constants::kBiCubicSupportSize;
          ResizeNhwcBilinearBicubicUpsample<T, BiCubicFilter>(
              stream, coordinate_transform_mode, cubic_coeff_a,
              batch_size, num_channels,
              input_height, input_width, output_height, output_width,
              height_scale, width_scale, support_value,
              roi_vals, extrapolation, exclude_outside,
              allocate_temp_space, input_data, output_data, N);
        } break;
        default:
          ORT_NOT_IMPLEMENTED("Only bilinear and bicubic modes are supported for NHWC Resize anti-alias mode");
          break;
      }
    }
    return;
  }

  switch (upsample_mode) {
    case UpsampleMode::LINEAR: {
      if (is_2D) {
        ResizeBiLinearUpsample<T>(stream, rank, upsample_mode, coordinate_transform_mode,
                                  input_shape, output_shape, batch_size, num_channels,
                                  inferred_input_dims, inferred_output_dims, inferred_dim_rscales,
                                  output_div_pitches, roi_vals, extrapolation, exclude_outside,
                                  allocate_temp_space, clip8_lookups, input_data, output_data, N);
      } else if (is_3D) {
        ResizeTrilinearUpsample<T>(stream, rank, upsample_mode, coordinate_transform_mode,
                                   input_shape, output_shape, batch_size, num_channels,
                                   inferred_input_dims, inferred_output_dims, inferred_dim_rscales,
                                   output_div_pitches, roi_vals, extrapolation, exclude_outside,
                                   allocate_temp_space, clip8_lookups, input_data, output_data, N);
      } else {
        ORT_NOT_IMPLEMENTED("Resize supports only 2-D or 3-D in LINEAR mode.");
      }
    } break;
    case CUBIC: {
      if (is_2D) {
        ResizeBicubicUpsample<T>(stream, rank, upsample_mode, coordinate_transform_mode, cubic_coeff_a,
                                 input_shape, output_shape, batch_size, num_channels,
                                 inferred_input_dims, inferred_output_dims, inferred_dim_rscales,
                                 output_div_pitches, roi_vals, extrapolation, exclude_outside,
                                 allocate_temp_space, clip8_lookups, input_data, output_data, N);
      } else {
        ORT_NOT_IMPLEMENTED("Resize supports only 2-D in CUBIC mode.");
      }
    } break;
    default:
      ORT_NOT_IMPLEMENTED("Only bilinear/trilinear and bicubic modes are supported in Resize anti-alias mode");
      break;
  }
}

#define SPECIALIZED_ANTIALIAS_IMPL(T)                               \
  template void ResizeAntiAliasImpl<T>(                             \
      cudaStream_t stream,                                          \
      int rank,                                                     \
      const UpsampleMode upsample_mode,                             \
      ResizeCoordinateTransformationMode coordinate_transform_mode, \
      float cubic_coeff_a,                                          \
      gsl::span<const int64_t> input_shape,                         \
      gsl::span<const int64_t> output_shape,                        \
      int64_t batch_size, int64_t num_channels,                     \
      std::tuple<int64_t, int64_t, int64_t> inferred_input_dims,    \
      std::tuple<int64_t, int64_t, int64_t> inferred_output_dims,   \
      std::tuple<float, float, float> inferred_dim_rscales,         \
      const TArray<fast_divmod>& output_div_pitches,                \
      gsl::span<const float> roi_vals,                              \
      const std::optional<float>& extrapolation_value,              \
      bool exclude_outside,                                         \
      bool is_nhwc,                                                 \
      TempSpaceAllocateFunc allocate_temp_space,                    \
      const uint8_t* clip8_lookups,                                 \
      const T* input_data,                                          \
      T* output_data,                                               \
      const size_t N);

SPECIALIZED_ANTIALIAS_IMPL(float)
SPECIALIZED_ANTIALIAS_IMPL(double)
SPECIALIZED_ANTIALIAS_IMPL(half)
SPECIALIZED_ANTIALIAS_IMPL(int32_t)
SPECIALIZED_ANTIALIAS_IMPL(uint8_t)

}  // namespace cuda
}  // namespace onnxruntime

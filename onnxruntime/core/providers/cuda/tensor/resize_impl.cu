#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/tensor/resize_impl.h"

namespace onnxruntime {
namespace cuda {

using onnxruntime::ResizeCoordinateTransformationMode;
using onnxruntime::ResizeNearestMode;
using onnxruntime::UpsampleMode;

__device__ int NearestPixel_SIMPLE(float x_original, bool is_down_sampling) {
  if (is_down_sampling) {
    return static_cast<int>(ceil(x_original));
  } else {
    return static_cast<int>(x_original);
  }
}

__device__ int NearestPixel_ROUND_PREFER_FLOOR(float x_original, bool) {
  if (x_original == static_cast<int>(x_original) + 0.5f) {
    return static_cast<int>(floor(x_original));
  }
  return static_cast<int>(round(x_original));
}

__device__ int NearestPixel_ROUND_PREFER_CEIL(float x_original, bool) {
  return static_cast<int>(round(x_original));
}

__device__ int NearestPixel_FLOOR(float x_original, bool) {
  return static_cast<int>(floor(x_original));
}

__device__ int NearestPixel_CEIL(float x_original, bool) {
  return static_cast<int>(ceil(x_original));
}

using CudaFunctionNearestPixel = int (*)(float, bool);
__device__ CudaFunctionNearestPixel func_NearestPixel_SIMPLE = NearestPixel_SIMPLE;
__device__ CudaFunctionNearestPixel func_NearestPixel_ROUND_PREFER_FLOOR = NearestPixel_ROUND_PREFER_FLOOR;
__device__ CudaFunctionNearestPixel func_NearestPixel_ROUND_PREFER_CEIL = NearestPixel_ROUND_PREFER_CEIL;
__device__ CudaFunctionNearestPixel func_NearestPixel_FLOOR = NearestPixel_FLOOR;
__device__ CudaFunctionNearestPixel func_NearestPixel_CEIL = NearestPixel_CEIL;

CudaFunctionNearestPixel GetDeviceNearstPixelFunction(ResizeNearestMode nearest_mode) {
  static bool already_copied = false;
  static std::mutex s_mutext;
  static CudaFunctionNearestPixel s_nearest_pixel[ResizeNearestMode::NearestModeCount];
  if (!already_copied) {
    std::lock_guard<std::mutex> lock(s_mutext);
    if (!already_copied) {
      CUDA_CALL(cudaMemcpyFromSymbol(&s_nearest_pixel[ResizeNearestMode::SIMPLE],
                                     func_NearestPixel_SIMPLE, sizeof(CudaFunctionNearestPixel)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_nearest_pixel[ResizeNearestMode::ROUND_PREFER_FLOOR],
                                     func_NearestPixel_ROUND_PREFER_FLOOR, sizeof(CudaFunctionNearestPixel)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_nearest_pixel[ResizeNearestMode::ROUND_PREFER_CEIL],
                                     func_NearestPixel_ROUND_PREFER_CEIL, sizeof(CudaFunctionNearestPixel)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_nearest_pixel[ResizeNearestMode::FLOOR],
                                     func_NearestPixel_FLOOR, sizeof(CudaFunctionNearestPixel)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_nearest_pixel[ResizeNearestMode::CEIL],
                                     func_NearestPixel_CEIL, sizeof(CudaFunctionNearestPixel)));
      already_copied = true;
    }
  }
  return s_nearest_pixel[nearest_mode];
}

__device__ float TransformCoordinate_ASYMMETRIC(float x_resized, float x_scale, float, float, float, float) {
  return x_resized / x_scale;
}

__device__ float TransformCoordinate_HALF_PIXEL(float x_resized, float x_scale, float, float, float, float) {
  return ((x_resized + 0.5f) / x_scale) - 0.5f;
}

__device__ float TransformCoordinate_PYTORCH_HALF_PIXEL(
    float x_resized, float x_scale, float length_resized, float, float, float) {
  return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
}

__device__ float TransformCoordinate_TF_HALF_PIXEL_FOR_NN(
    float x_resized, float x_scale, float, float, float, float) {
  return (x_resized + 0.5f) / x_scale;
}

__device__ float TransformCoordinate_ALIGN_CORNERS(
    float x_resized, float, float length_resized, float length_original, float, float) {
  return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
}

__device__ float TransformCoordinate_TF_CROP_AND_RESIZE(
    float x_resized, float, float length_resized, float length_original, float roi_start, float roi_end) {
  auto orig = length_resized > 1
                  ? roi_start * (length_original - 1) + (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                  : 0.5 * (roi_start + roi_end) * (length_original - 1);
  return static_cast<float>(orig);
}

using CudaFunctionOriginalCoordinate = float (*)(float, float, float, float, float, float);

__device__ CudaFunctionOriginalCoordinate func_TransformCoordinate_ASYMMETRIC = TransformCoordinate_ASYMMETRIC;
__device__ CudaFunctionOriginalCoordinate func_TransformCoordinate_HALF_PIXEL = TransformCoordinate_HALF_PIXEL;
__device__ CudaFunctionOriginalCoordinate func_TransformCoordinate_PYTORCH_HALF_PIXEL = TransformCoordinate_PYTORCH_HALF_PIXEL;
__device__ CudaFunctionOriginalCoordinate func_TransformCoordinate_ALIGN_CORNERS = TransformCoordinate_ALIGN_CORNERS;
__device__ CudaFunctionOriginalCoordinate func_TransformCoordinate_TF_HALF_PIXEL_FOR_NN = TransformCoordinate_TF_HALF_PIXEL_FOR_NN;
__device__ CudaFunctionOriginalCoordinate func_TransformCoordinate_TF_CROP_AND_RESIZE = TransformCoordinate_TF_CROP_AND_RESIZE;

CudaFunctionOriginalCoordinate GetDeviceOriginalCoordinateFunc(ResizeCoordinateTransformationMode coordinate_transform_mode) {
  static bool already_copied = false;
  static std::mutex s_mutext;
  static CudaFunctionOriginalCoordinate s_coordinate_tranforms[ResizeCoordinateTransformationMode::CoordinateTransformationModeCount];
  if (!already_copied) {
    std::lock_guard<std::mutex> lock(s_mutext);
    if (!already_copied) {
      CUDA_CALL(cudaMemcpyFromSymbol(&s_coordinate_tranforms[ResizeCoordinateTransformationMode::HALF_PIXEL],
                                     func_TransformCoordinate_HALF_PIXEL, sizeof(CudaFunctionOriginalCoordinate)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_coordinate_tranforms[ResizeCoordinateTransformationMode::ASYMMETRIC],
                                     func_TransformCoordinate_ASYMMETRIC, sizeof(CudaFunctionOriginalCoordinate)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_coordinate_tranforms[ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL],
                                     func_TransformCoordinate_PYTORCH_HALF_PIXEL, sizeof(CudaFunctionOriginalCoordinate)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_coordinate_tranforms[ResizeCoordinateTransformationMode::ALIGN_CORNERS],
                                     func_TransformCoordinate_ALIGN_CORNERS, sizeof(CudaFunctionOriginalCoordinate)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_coordinate_tranforms[ResizeCoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN],
                                     func_TransformCoordinate_TF_HALF_PIXEL_FOR_NN, sizeof(CudaFunctionOriginalCoordinate)));
      CUDA_CALL(cudaMemcpyFromSymbol(&s_coordinate_tranforms[ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE],
                                     func_TransformCoordinate_TF_CROP_AND_RESIZE, sizeof(CudaFunctionOriginalCoordinate)));
      already_copied = true;
    }
  }
  return s_coordinate_tranforms[coordinate_transform_mode];
}

template <typename T>
__global__ void _ResizeNearestKernel(
    const size_t rank,
    const int64_t* input_shape,
    const int64_t* output_shape,
    const int64_t* input_pitches,
    const fast_divmod* output_div_pitches,
    const float* scales,
    const float* roi,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    float extrapolation_value,
    CudaFunctionOriginalCoordinate transform_coordinate,
    CudaFunctionNearestPixel calc_nearest_pixel) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;

  int div, mod;
  bool extrapolation_occured = false;
  for (int dim = 0; dim < rank; ++dim) {
    output_div_pitches[dim].divmod(output_index, div, mod);
    output_index = mod;
    float orig_coord = transform_coordinate(static_cast<float>(div), scales[dim], static_cast<float>(output_shape[dim]),
                                            static_cast<float>(input_shape[dim]), roi[dim], roi[dim + rank]);
    if (extrapolation_enabled && !extrapolation_occured) {
      extrapolation_occured = (orig_coord < 0.f || orig_coord > static_cast<float>(input_shape[dim] - 1));
    }
    div = calc_nearest_pixel(orig_coord, scales[dim] < 1);
    if (div >= input_shape[dim]) div = input_shape[dim] - 1;
    if (div < 0) div = 0;
    input_index += input_pitches[dim] * div;
  }
  output_data[id] = extrapolation_occured ? static_cast<T>(extrapolation_value) : input_data[input_index];
}

struct BilinearMappingInfo {
  int origin_;
  float weight_;
  int extrapolate_;
};

template <typename T>
__global__ void _ResizeBilinearCoordinateMapping(
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    float scale_height, float scale_width,
    float roi_height_start, float roi_height_end,
    float roi_width_start, float roi_width_end,
    const size_t SumHW, bool extrapolation_enabled,
    CudaFunctionOriginalCoordinate transform_coordinate,
    BilinearMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumHW);
  if (id < output_height) {  //  y = id
    float input_y = transform_coordinate(static_cast<float>(id), scale_height,
                                         static_cast<float>(output_height), static_cast<float>(input_height),
                                         roi_height_start, roi_height_end);
    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (input_y < 0 || input_y > static_cast<float>(input_height - 1)));
    input_y = max(0.0f, min(input_y, static_cast<float>(input_height - 1)));
    int y_int = static_cast<int>(input_y);
    dims_mapping[id].origin_ = y_int;
    dims_mapping[id].weight_ = (y_int >= input_height - 1) ? 0.5f : input_y - y_int;
  } else {  //x = id - output_height
    float input_x = transform_coordinate(static_cast<float>(id - output_height), scale_width,
                                         static_cast<float>(output_width), static_cast<float>(input_width),
                                         roi_width_start, roi_width_end);
    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (input_x < 0 || input_x > static_cast<float>(input_width - 1)));
    input_x = max(0.0f, min(input_x, static_cast<float>(input_width - 1)));
    int x_int = static_cast<int>(input_x);
    dims_mapping[id].origin_ = x_int;
    dims_mapping[id].weight_ = (x_int >= input_width - 1) ? 0.5f : input_x - x_int;
  }
}

// The following method supports a N-D input in 'Linear mode'. Last two dimension is [H, W].
// the scale values for the outer dimensions except last two are 1.
template <typename T>
__global__ void _ResizeBilinearKernel(
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    fast_divmod div_output_width, fast_divmod div_output_image,
    const T* input_data, T* output_data, const size_t N,
    float extrapolation_value,
    BilinearMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int bxc, output_image_index;
  div_output_image.divmod(id, bxc, output_image_index);
  CUDA_LONG input_index = bxc * input_height * input_width;
  int output_y, output_x;
  div_output_width.divmod(output_image_index, output_y, output_x);

  if (dims_mapping[output_y].extrapolate_ || dims_mapping[output_x + output_height].extrapolate_) {
    output_data[id] = extrapolation_value;
    return;
  }
  float y_offset_0 = dims_mapping[output_y].weight_;
  int y_int = dims_mapping[output_y].origin_;
  float x_offset_0 = dims_mapping[output_x + output_height].weight_;
  int x_int = dims_mapping[output_x + output_height].origin_;
  input_index += y_int * input_width + x_int;

  T x00 = input_data[input_index];
  bool end_of_h = (y_int >= input_height - 1);
  bool end_of_w = (x_int >= input_width - 1);
  T x10 = end_of_w ? x00 : input_data[input_index + 1];
  T x01 = end_of_h ? x00 : input_data[input_index + input_width];
  T x11 = end_of_w ? x01 : (end_of_h ? x10 : input_data[input_index + input_width + 1]);

  float y_offset_1 = 1.0f - y_offset_0;
  float x_offset_1 = 1.0f - x_offset_0;
  output_data[id] =
      x00 * static_cast<T>(y_offset_1 * x_offset_1) +
      x01 * static_cast<T>(y_offset_0 * x_offset_1) +
      x10 * static_cast<T>(y_offset_1 * x_offset_0) +
      x11 * static_cast<T>(y_offset_0 * x_offset_0);
}

template <typename T>
__device__ __forceinline__ float CubicInterpolationRowwise(
    const T* image, int x, int y, int input_height, int input_width,
    float coeff0, float coeff1, float coeff2, float coeff3) {
  int row_index = max(0, min(y, input_height - 1)) * input_width;
  return coeff0 * static_cast<float>(image[row_index + max(0, min(x - 1, input_width - 1))])
      + coeff1 * static_cast<float>(image[row_index + max(0, min(x, input_width - 1))])
      + coeff2 * static_cast<float>(image[row_index + max(0, min(x + 1, input_width - 1))])
      + coeff3 * static_cast<float>(image[row_index + max(0, min(x + 2, input_width - 1))]);
}

struct CubicMappingInfo {
  int origin_;
  int extrapolate_;
  float coeff0_;
  float coeff1_;
  float coeff2_;
  float coeff3_;
};

template <typename T>
__global__ void _ResizeCubicCoordinateMapping(
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    float scale_height, float scale_width,
    float roi_height_start, float roi_height_end,
    float roi_width_start, float roi_width_end,
    const size_t SumHW, bool extrapolation_enabled,
    float cubic_coeff_a, bool exclude_outside,
    CudaFunctionOriginalCoordinate transform_coordinate,
    CubicMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumHW);
  auto& dm = dims_mapping[id];
  bool is_y_axis = (id < output_height);
  int max_input_coord = static_cast<int>(is_y_axis ? input_height : input_width);

  float input_coordinat = transform_coordinate(
      static_cast<float>(is_y_axis ? id : id - output_height),
      (is_y_axis ? scale_height : scale_width),
      static_cast<float>(is_y_axis ? output_height : output_width),
      static_cast<float>(max_input_coord),
      (is_y_axis ? roi_height_start : roi_width_start),
      (is_y_axis ? roi_height_end : roi_width_end));
  int coord_int = static_cast<int>(floor(input_coordinat));
  float s_coord = abs(input_coordinat - coord_int);
  float coeff_sum = 1.0f;
  float coeff_0 = static_cast<float>(((cubic_coeff_a * (s_coord + 1) - 5 * cubic_coeff_a) * (s_coord + 1) + 8 * cubic_coeff_a) * (s_coord + 1) - 4 * cubic_coeff_a);
  float coeff_1 = static_cast<float>(((cubic_coeff_a + 2) * s_coord - (cubic_coeff_a + 3)) * s_coord * s_coord + 1);
  float coeff_2 = static_cast<float>(((cubic_coeff_a + 2) * (1 - s_coord) - (cubic_coeff_a + 3)) * (1 - s_coord) * (1 - s_coord) + 1);
  float coeff_3 = static_cast<float>(((cubic_coeff_a * (2 - s_coord) - 5 * cubic_coeff_a) * (2 - s_coord) + 8 * cubic_coeff_a) * (2 - s_coord) - 4 * cubic_coeff_a);
  if (exclude_outside) {
    coeff_0 = (coord_int - 1 < 0 || coord_int - 1 >= max_input_coord) ? 0.0 : coeff_0;
    coeff_1 = (coord_int + 0 < 0 || coord_int + 0 >= max_input_coord) ? 0.0 : coeff_1;
    coeff_2 = (coord_int + 1 < 0 || coord_int + 1 >= max_input_coord) ? 0.0 : coeff_2;
    coeff_3 = (coord_int + 2 < 0 || coord_int + 2 >= max_input_coord) ? 0.0 : coeff_3;
    coeff_sum = coeff_0 + coeff_1 + coeff_2 + coeff_3;
  }
  dm.origin_ = coord_int;
  dm.coeff0_ = coeff_0 / coeff_sum;
  dm.coeff1_ = coeff_1 / coeff_sum;
  dm.coeff2_ = coeff_2 / coeff_sum;
  dm.coeff3_ = coeff_3 / coeff_sum;
  dm.extrapolate_ = (int)(extrapolation_enabled && (input_coordinat < 0 || input_coordinat > static_cast<float>(max_input_coord - 1)));
}

template <typename T>
__global__ void _ResizeBiCubicKernel(
    int64_t input_height, int64_t input_width, int64_t output_height, int64_t output_width,
    fast_divmod div_output_width, fast_divmod div_output_image,
    const T* input_data, T* output_data, const size_t N, float extrapolation_value,
    CubicMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int bxc, output_image_index, output_x, output_y;
  div_output_image.divmod(id, bxc, output_image_index);
  CUDA_LONG input_index = bxc * input_height * input_width;
  div_output_width.divmod(output_image_index, output_y, output_x);

  CubicMappingInfo& y_info = dims_mapping[output_y];
  CubicMappingInfo& x_info = dims_mapping[output_x + output_height];
  if (y_info.extrapolate_ || x_info.extrapolate_) {
    output_data[id] = extrapolation_value;
    return;
  }

  float w0 = x_info.coeff0_;
  float w1 = x_info.coeff1_;
  float w2 = x_info.coeff2_;
  float w3 = x_info.coeff3_;
  int x_int = x_info.origin_;
  int y_int = y_info.origin_;
  const T* image = input_data + input_index;
  output_data[id] = y_info.coeff0_ * CubicInterpolationRowwise(image, x_int, y_int - 1, input_height, input_width, w0, w1, w2, w3)
      + y_info.coeff1_ * CubicInterpolationRowwise(image, x_int, y_int, input_height, input_width, w0, w1, w2, w3)
      + y_info.coeff2_ * CubicInterpolationRowwise(image, x_int, y_int + 1, input_height, input_width, w0, w1, w2, w3)
      + y_info.coeff3_ * CubicInterpolationRowwise(image, x_int, y_int + 2, input_height, input_width, w0, w1, w2, w3);
}

size_t CalcResizeBufferSize(const onnxruntime::UpsampleMode upsample_mode,
                            const std::vector<int64_t>& output_dims) {
  switch (upsample_mode) {
    case UpsampleMode::NN:
      return 0;
    case UpsampleMode::LINEAR:
      return sizeof(BilinearMappingInfo) * std::accumulate(output_dims.rbegin(), output_dims.rbegin() + 2, 0);
    case UpsampleMode::CUBIC:
      return sizeof(CubicMappingInfo) * std::accumulate(output_dims.rbegin(), output_dims.rbegin() + 2, 0);
  }
  return 0;
}

template <typename T>
void ResizeImpl(
    const UpsampleMode upsample_mode,
    const int rank,
    CudaKernel::CudaAsyncBuffer<int64_t>& input_shape,
    CudaKernel::CudaAsyncBuffer<int64_t>& output_shape,
    CudaKernel::CudaAsyncBuffer<int64_t>& input_strides,
    CudaKernel::CudaAsyncBuffer<fast_divmod>& output_div_pitches,
    CudaKernel::CudaAsyncBuffer<float>& scales_vals,
    CudaKernel::CudaAsyncBuffer<float>& roi_vals,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    float extrapolation_value,
    float cubic_coeff_a,
    bool exclude_outside,
    ResizeCoordinateTransformationMode coordinate_transform_mode,
    ResizeNearestMode nearest_mode,
    void* dims_mapping) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  CudaFunctionOriginalCoordinate transform_coordinate = GetDeviceOriginalCoordinateFunc(coordinate_transform_mode);
  CudaFunctionNearestPixel calc_nearest_pixel = GetDeviceNearstPixelFunction(nearest_mode);
  fast_divmod div_output_image = (rank > 2) ? output_div_pitches.CpuPtr()[rank - 3] : fast_divmod(gsl::narrow_cast<int>(N));
  int64_t output_height = output_shape.CpuPtr()[rank - 2];
  int64_t output_width = output_shape.CpuPtr()[rank - 1];
  int blocksPerDimsMappingGrid = (int)(ceil(static_cast<float>(output_height + output_width) / 32));

  switch (upsample_mode) {
    case UpsampleMode::NN:
      input_shape.CopyToGpu();
      output_shape.CopyToGpu();
      roi_vals.CopyToGpu();
      scales_vals.CopyToGpu();
      input_strides.CopyToGpu();
      output_div_pitches.CopyToGpu();
      _ResizeNearestKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          rank, input_shape.GpuPtr(), output_shape.GpuPtr(),
          input_strides.GpuPtr(), output_div_pitches.GpuPtr(),
          scales_vals.GpuPtr(), roi_vals.GpuPtr(),
          input_data, output_data, N,
          extrapolation_enabled, extrapolation_value,
          transform_coordinate, calc_nearest_pixel);
      return;
    case UpsampleMode::LINEAR:
      _ResizeBilinearCoordinateMapping<T><<<blocksPerDimsMappingGrid, 32, 0>>>(
          input_shape.CpuPtr()[rank - 2], input_shape.CpuPtr()[rank - 1],
          output_height, output_width,
          scales_vals.CpuPtr()[rank - 2], scales_vals.CpuPtr()[rank - 1],
          roi_vals.CpuPtr()[rank - 2], roi_vals.CpuPtr()[rank - 2 + rank],
          roi_vals.CpuPtr()[rank - 1], roi_vals.CpuPtr()[rank - 1 + rank],
          output_height + output_width, extrapolation_enabled, transform_coordinate,
          reinterpret_cast<BilinearMappingInfo*>(dims_mapping));
      _ResizeBilinearKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          input_shape.CpuPtr()[rank - 2], input_shape.CpuPtr()[rank - 1],
          output_height, output_width,
          output_div_pitches.CpuPtr()[rank - 2], div_output_image,
          input_data, output_data, N, extrapolation_value,
          reinterpret_cast<BilinearMappingInfo*>(dims_mapping));
      return;
    case UpsampleMode::CUBIC:
      _ResizeCubicCoordinateMapping<T><<<blocksPerDimsMappingGrid, 32, 0>>>(
          input_shape.CpuPtr()[rank - 2], input_shape.CpuPtr()[rank - 1],
          output_height, output_width,
          scales_vals.CpuPtr()[rank - 2], scales_vals.CpuPtr()[rank - 1],
          roi_vals.CpuPtr()[rank - 2], roi_vals.CpuPtr()[rank - 2 + rank],
          roi_vals.CpuPtr()[rank - 1], roi_vals.CpuPtr()[rank - 1 + rank],
          output_height + output_width, extrapolation_enabled,
          cubic_coeff_a, exclude_outside, transform_coordinate,
          reinterpret_cast<CubicMappingInfo*>(dims_mapping));
      _ResizeBiCubicKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          input_shape.CpuPtr()[rank - 2], input_shape.CpuPtr()[rank - 1],
          output_height, output_width,
          output_div_pitches.CpuPtr()[rank - 2], div_output_image,
          input_data, output_data, N, extrapolation_value,
          reinterpret_cast<CubicMappingInfo*>(dims_mapping));
      // CUDA_CALL(cudaGetLastError());
      return;
  }
}

#define SPECIALIZED_IMPL(T)                                         \
  template void ResizeImpl<T>(                                      \
      const UpsampleMode upsample_mode,                             \
      const int rank,                                               \
      CudaKernel::CudaAsyncBuffer<int64_t>& input_shape,            \
      CudaKernel::CudaAsyncBuffer<int64_t>& output_shape,           \
      CudaKernel::CudaAsyncBuffer<int64_t>& input_strides,          \
      CudaKernel::CudaAsyncBuffer<fast_divmod>& output_div_pitches, \
      CudaKernel::CudaAsyncBuffer<float>& scales_vals,              \
      CudaKernel::CudaAsyncBuffer<float>& roi_vals,                 \
      const T* input_data,                                          \
      T* output_data,                                               \
      const size_t N,                                               \
      bool extrapolation_enabled,                                   \
      float extrapolation_value,                                    \
      float cubic_coeff_a,                                          \
      bool exclude_outside,                                         \
      ResizeCoordinateTransformationMode coordinate_transform_mode, \
      ResizeNearestMode nearest_mode,                               \
      void* dims_mapping);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(uint8_t)

}  // namespace cuda
}  // namespace onnxruntime

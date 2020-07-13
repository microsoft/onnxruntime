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

struct NearestMappingInfo {
  int origin_;
  int extrapolate_;
};

template <typename T>
__global__ void _ResizeNearestMappingKernel2D(
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const float scales_height, const float scales_width,
    const float roi_start_height, const float roi_end_height,
    const float roi_start_width, const float roi_end_width,
    const bool extrapolation_enabled,
    CudaFunctionOriginalCoordinate transform_coordinate,
    CudaFunctionNearestPixel calc_nearest_pixel,
    NearestMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, output_height + output_width);
  if (id >= 0 && id < output_height) {  // for Height
    int dim = id;
    float orig_coord = transform_coordinate(static_cast<float>(dim), scales_height, static_cast<float>(output_height),
                                            static_cast<float>(input_height), roi_start_height, roi_end_height);
    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (orig_coord < 0.f || orig_coord > static_cast<float>(input_height - 1)));
    dim = calc_nearest_pixel(orig_coord, scales_height < 1);
    if (dim >= input_height) dim = input_height - 1;
    if (dim < 0) dim = 0;
    dims_mapping[id].origin_ = dim;
  } else {
    int dim = id - output_height;
    float orig_coord = transform_coordinate(static_cast<float>(dim), scales_width, static_cast<float>(output_width),
                                            static_cast<float>(input_width), roi_start_width, roi_end_width);
    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (orig_coord < 0.f || orig_coord > static_cast<float>(input_width - 1)));
    dim = calc_nearest_pixel(orig_coord, scales_width < 1);
    if (dim >= input_width) dim = input_width - 1;
    if (dim < 0) dim = 0;
    dims_mapping[id].origin_ = dim;
    return;
  }
}

template <typename T>
__global__ void _ResizeNearestMappingKernel(
    const size_t rank,
    const TArray<int64_t> input_shape,
    const TArray<int64_t> output_shape,
    const TArray<float> scales,
    const TArray<float> roi,
    const size_t total_dim_sum,
    bool extrapolation_enabled,
    CudaFunctionOriginalCoordinate transform_coordinate,
    CudaFunctionNearestPixel calc_nearest_pixel,
    int64_t* prefix_dim_sum,
    NearestMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, total_dim_sum);
  int64_t dim_sum = 0;
  for (int axis = 0; axis < rank; ++axis) {
    if (id == dim_sum) {
      prefix_dim_sum[axis] = dim_sum;
    }
    if (id >= dim_sum && id < dim_sum + output_shape[axis]) {
      int dim = id - dim_sum;
      float orig_coord = transform_coordinate(static_cast<float>(dim), scales[axis], static_cast<float>(output_shape[axis]),
                                              static_cast<float>(input_shape[axis]), roi[axis], roi[axis + rank]);
      dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (orig_coord < 0.f || orig_coord > static_cast<float>(input_shape[axis] - 1)));
      dim = calc_nearest_pixel(orig_coord, scales[axis] < 1);
      if (dim >= input_shape[axis]) dim = input_shape[axis] - 1;
      if (dim < 0) dim = 0;
      dims_mapping[id].origin_ = dim;
      return;
    }
    dim_sum += output_shape[axis];
  }
}

template <typename T, bool UseExtrapolation>
__global__ void _ResizeNearestKernel2D(
    const int64_t output_height, const int64_t output_width,
    const int64_t input_stride_image, const int input_stride_row,
    const fast_divmod output_stride_image, const fast_divmod output_stride_row,
    const T* input_data, T* output_data, const size_t N,
    const T extrapolation_value, const NearestMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int imageid, h, w, output_index;
  output_stride_image.divmod(static_cast<int>(id), imageid, output_index);
  output_stride_row.divmod(output_index, h, w);
  if (UseExtrapolation) {
    if (dims_mapping[h].extrapolate_ + dims_mapping[output_height + w].extrapolate_) {
      output_data[id] = extrapolation_value;
      return;
    }
  }
  int input_index = input_stride_image * imageid +
                    input_stride_row * dims_mapping[h].origin_ +
                    dims_mapping[output_height + w].origin_;
  output_data[id] = input_data[input_index];
}

template <typename T>
__global__ void _ResizeNearestKernel(
    const int rank,
    const TArray<int64_t> input_strides,
    const TArray<fast_divmod> output_div_pitches,
    const T* input_data,
    T* output_data,
    const size_t N,
    const T extrapolation_value,
    const int64_t* prefix_dim_sum,
    const NearestMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int output_index = static_cast<int>(id);
  int input_index = 0;
  int extrapolation_occured = 0;
  for (int axis = 0; axis < rank; ++axis) {
    int dim = 0;
    output_div_pitches[axis].divmod(output_index, dim, output_index);
    const NearestMappingInfo& mi = dims_mapping[prefix_dim_sum[axis] + dim];
    extrapolation_occured += mi.extrapolate_;
    input_index += input_strides[axis] * mi.origin_;
  }
  output_data[id] = extrapolation_occured ? extrapolation_value : input_data[input_index];
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
    const T extrapolation_value,
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
  return coeff0 * static_cast<float>(image[row_index + max(0, min(x - 1, input_width - 1))]) +
         coeff1 * static_cast<float>(image[row_index + max(0, min(x, input_width - 1))]) +
         coeff2 * static_cast<float>(image[row_index + max(0, min(x + 1, input_width - 1))]) +
         coeff3 * static_cast<float>(image[row_index + max(0, min(x + 2, input_width - 1))]);
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
    const T* input_data, T* output_data, const size_t N, const T extrapolation_value,
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
  output_data[id] = y_info.coeff0_ * CubicInterpolationRowwise(image, x_int, y_int - 1, input_height, input_width, w0, w1, w2, w3) +
                    y_info.coeff1_ * CubicInterpolationRowwise(image, x_int, y_int, input_height, input_width, w0, w1, w2, w3) +
                    y_info.coeff2_ * CubicInterpolationRowwise(image, x_int, y_int + 1, input_height, input_width, w0, w1, w2, w3) +
                    y_info.coeff3_ * CubicInterpolationRowwise(image, x_int, y_int + 2, input_height, input_width, w0, w1, w2, w3);
}

size_t CalcResizeBufferSize(const onnxruntime::UpsampleMode upsample_mode,
                            const std::vector<int64_t>& output_dims) {
  switch (upsample_mode) {
    case UpsampleMode::NN:
      return sizeof(int64_t) * output_dims.size() + sizeof(NearestMappingInfo) * std::accumulate(output_dims.begin(), output_dims.end(), 0);
    case UpsampleMode::LINEAR:
      return sizeof(BilinearMappingInfo) * std::accumulate(output_dims.rbegin(), output_dims.rbegin() + 2, 0);
    case UpsampleMode::CUBIC:
      return sizeof(CubicMappingInfo) * std::accumulate(output_dims.rbegin(), output_dims.rbegin() + 2, 0);
  }
  return 0;
}

template <typename T>
void ResizeNearestImpl(
    const int rank,
    TArray<int64_t>& input_shape,
    TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals,
    TArray<float>& roi_vals,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    const T extrapolation_value,
    float cubic_coeff_a,
    CudaFunctionOriginalCoordinate transform_coordinate,
    CudaFunctionNearestPixel calc_nearest_pixel,
    int64_t* /* prefix_dim_sum */,
    NearestMappingInfo* dims_mapping) {
  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  bool could2d = rank >= 2 &&
                 transform_coordinate != GetDeviceOriginalCoordinateFunc(ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE) &&
                 std::all_of(scales_vals.Data(), scales_vals.Data() + (rank - 2), [](float v) { return v == 1.0; });
  if (could2d) {
    int64_t output_height = output_shape[rank - 2];
    int64_t output_width = output_shape[rank - 1];
    fast_divmod div_output_image = (rank > 2) ? output_div_pitches[rank - 3] : fast_divmod(output_height * output_width);
    int blocksPerDimsMappingGrid = static_cast<int>(ceil((output_height + output_width) / 32.0));

    _ResizeNearestMappingKernel2D<T><<<blocksPerDimsMappingGrid, 32, 0>>>(
        input_shape[rank - 2], input_shape[rank - 1],
        output_height, output_width,
        scales_vals[rank - 2], scales_vals[rank - 1],
        roi_vals[rank - 2], roi_vals[rank - 2 + rank],
        roi_vals[rank - 1], roi_vals[rank - 1 + rank],
        extrapolation_enabled, transform_coordinate, calc_nearest_pixel,
        dims_mapping);
    if (extrapolation_enabled) {
      _ResizeNearestKernel2D<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_height, output_width,
          input_shape[rank - 2] * input_shape[rank - 1], input_shape[rank - 1],
          div_output_image, output_div_pitches[rank - 2],
          input_data, output_data, N,
          extrapolation_value,
          dims_mapping);
    } else {
      _ResizeNearestKernel2D<T, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          output_height, output_width,
          input_shape[rank - 2] * input_shape[rank - 1], input_shape[rank - 1],
          div_output_image, output_div_pitches[rank - 2],
          input_data, output_data, N,
          extrapolation_value,
          dims_mapping);
    }
    return;
  }

  int64_t total_dim_sum = std::accumulate(output_shape.Data(), output_shape.Data() + rank, 0);
  int blocksPerDimsMappingGrid = (int)(ceil(static_cast<double>(total_dim_sum) / 32));
  _ResizeNearestMappingKernel<T><<<blocksPerDimsMappingGrid, 32, 0>>>(
      rank, input_shape, output_shape,
      scales_vals, roi_vals,
      total_dim_sum, extrapolation_enabled,
      transform_coordinate, calc_nearest_pixel,
      reinterpret_cast<int64_t*>(dims_mapping),
      reinterpret_cast<NearestMappingInfo*>(reinterpret_cast<int64_t*>(dims_mapping) + rank));
  _ResizeNearestKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      rank, input_strides, output_div_pitches,
      input_data, output_data, N,
      extrapolation_value,
      reinterpret_cast<const int64_t*>(dims_mapping),
      reinterpret_cast<const NearestMappingInfo*>(reinterpret_cast<int64_t*>(dims_mapping) + rank));
  return;
}

template <typename T>
void ResizeImpl(
    const UpsampleMode upsample_mode,
    const int rank,
    TArray<int64_t>& input_shape,
    TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals,
    TArray<float>& roi_vals,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    const T extrapolation_value,
    float cubic_coeff_a,
    bool exclude_outside,
    ResizeCoordinateTransformationMode coordinate_transform_mode,
    ResizeNearestMode nearest_mode,
    void* dims_mapping) {
  bool isSame = std::all_of(scales_vals.Data(), scales_vals.Data() + rank, [](float v) { return v == 1.0f; }) &&
                (coordinate_transform_mode != ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE);
  if (isSame) {
    cudaMemcpyAsync(output_data, input_data, N * sizeof(T), cudaMemcpyDeviceToDevice);
    return;
  }

  CudaFunctionOriginalCoordinate transform_coordinate = GetDeviceOriginalCoordinateFunc(coordinate_transform_mode);
  CudaFunctionNearestPixel calc_nearest_pixel = GetDeviceNearstPixelFunction(nearest_mode);
  if (upsample_mode == UpsampleMode::NN) {
    ResizeNearestImpl(
        rank, input_shape, output_shape, input_strides, output_div_pitches,
        scales_vals, roi_vals, input_data, output_data, N,
        extrapolation_enabled, extrapolation_value, cubic_coeff_a,
        transform_coordinate, calc_nearest_pixel,
        reinterpret_cast<int64_t*>(dims_mapping),
        reinterpret_cast<NearestMappingInfo*>(reinterpret_cast<int64_t*>(dims_mapping) + rank));
    return;
  }

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  fast_divmod div_output_image = (rank > 2) ? output_div_pitches[rank - 3] : fast_divmod(gsl::narrow_cast<int>(N));
  int64_t output_height = output_shape[rank - 2];
  int64_t output_width = output_shape[rank - 1];
  int blocksPerDimsMappingGrid = (int)(ceil((output_height + output_width) / 32.0));
  switch (upsample_mode) {
    case UpsampleMode::LINEAR:
      _ResizeBilinearCoordinateMapping<T><<<blocksPerDimsMappingGrid, 32, 0>>>(
          input_shape[rank - 2], input_shape[rank - 1],
          output_height, output_width,
          scales_vals[rank - 2], scales_vals[rank - 1],
          roi_vals[rank - 2], roi_vals[rank - 2 + rank],
          roi_vals[rank - 1], roi_vals[rank - 1 + rank],
          output_height + output_width, extrapolation_enabled, transform_coordinate,
          reinterpret_cast<BilinearMappingInfo*>(dims_mapping));
      _ResizeBilinearKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          input_shape[rank - 2], input_shape[rank - 1],
          output_height, output_width,
          output_div_pitches[rank - 2], div_output_image,
          input_data, output_data, N, extrapolation_value,
          reinterpret_cast<BilinearMappingInfo*>(dims_mapping));
      return;
    case UpsampleMode::CUBIC:
      _ResizeCubicCoordinateMapping<T><<<blocksPerDimsMappingGrid, 32, 0>>>(
          input_shape[rank - 2], input_shape[rank - 1],
          output_height, output_width,
          scales_vals[rank - 2], scales_vals[rank - 1],
          roi_vals[rank - 2], roi_vals[rank - 2 + rank],
          roi_vals[rank - 1], roi_vals[rank - 1 + rank],
          output_height + output_width, extrapolation_enabled,
          cubic_coeff_a, exclude_outside, transform_coordinate,
          reinterpret_cast<CubicMappingInfo*>(dims_mapping));
      _ResizeBiCubicKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          input_shape[rank - 2], input_shape[rank - 1],
          output_height, output_width,
          output_div_pitches[rank - 2], div_output_image,
          input_data, output_data, N, extrapolation_value,
          reinterpret_cast<CubicMappingInfo*>(dims_mapping));
      return;
  }
}

#define SPECIALIZED_IMPL(T)                                         \
  template void ResizeImpl<T>(                                      \
      const UpsampleMode upsample_mode,                             \
      const int rank,                                               \
      TArray<int64_t>& input_shape,                                 \
      TArray<int64_t>& output_shape,                                \
      TArray<int64_t>& input_strides,                               \
      TArray<fast_divmod>& output_div_pitches,                      \
      TArray<float>& scales_vals,                                   \
      TArray<float>& roi_vals,                                      \
      const T* input_data,                                          \
      T* output_data,                                               \
      const size_t N,                                               \
      bool extrapolation_enabled,                                   \
      const T extrapolation_value,                                  \
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

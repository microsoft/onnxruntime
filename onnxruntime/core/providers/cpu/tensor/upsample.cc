// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/upsample.h"
#include <sstream>

using namespace onnxruntime::common;
using namespace std;
namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Upsample<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Upsample<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    uint8_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    Upsample<uint8_t>);

template <typename T>
void UpsampleNearest2x(int64_t batch_size,
                       int64_t num_channels,
                       int64_t input_height,
                       int64_t input_width,
                       const T* input,
                       T* output) {
  const int64_t output_height = input_height * 2;
  const int64_t output_width = input_width * 2;
  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        const int64_t in_y = y / 2;
        for (int64_t x = 0; x < input_width; ++x) {
          const T v = input[in_y * input_width + x];
          const int64_t oidx = output_width * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
      }
      input += input_height * input_width;
      output += output_height * output_width;
    }
  }
}

template <typename T>
Status UpsampleNearest(const T* input,
                       T* output,
                       const TensorShape& input_shape,
                       const TensorShape& output_shape,
                       const vector<float>& scales,
                       const vector<float>& roi,
                       bool is_resize,
                       bool extrapolation_enabled,
                       float extrapolation_value,
                       bool use_nearest2x_optimization,
                       GetOriginalCoordinateFunc get_original_coordinate,
                       GetNearestPixelFunc get_nearest_pixel) {
  if (!input || !output)
    return Status(ONNXRUNTIME, FAIL,
                  is_resize ? "Resize: input/output value is nullptr"
                            : "Upsample: input/output value is nullptr");
  if (input_shape.NumDimensions() != output_shape.NumDimensions())
    return Status(ONNXRUNTIME, FAIL,
                  is_resize ? "Resize: input/output value's dimension mismatch"
                            : "Upsample: input/output value's dimension mismatch");
  if (input_shape.NumDimensions() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  is_resize ? "Resize: input shape needs to be at least a single dimension"
                            : "Upsample: input shape needs to be at least a single dimension.");
  }

  int64_t n_dim = static_cast<int64_t>(input_shape.NumDimensions());

  std::vector<int64_t> input_dim_counters(n_dim);
  std::vector<int64_t> input_dim_factor(n_dim);
  std::vector<bool> use_extrapolation_value(n_dim);
  input_dim_factor[n_dim - 1] = 1;  // initialize dimension factor
  for (int64_t dim_idx = n_dim - 2; dim_idx >= 0; dim_idx--) {
    input_dim_factor[dim_idx] = input_dim_factor[dim_idx + 1] * input_shape[dim_idx + 1];
  }

  int64_t output_idx = 0;
  int64_t input_idx = 0;

#define OneDimensionProcessor(dim_inx)                                                                                                                                                                                                               \
  use_extrapolation_value[dim_inx] = false;                                                                                                                                                                                                          \
  float original_##dim_inx##_idx = get_original_coordinate(static_cast<float>(output_dim##dim_inx##_inx), scales[dim_inx], static_cast<float>(output_shape[dim_inx]), static_cast<float>(input_shape[dim_inx]), roi[dim_inx], roi[n_dim + dim_inx]); \
  if (extrapolation_enabled && (original_##dim_inx##_idx < 0 || original_##dim_inx##_idx > input_shape[dim_inx] - 1)) use_extrapolation_value[dim_inx] = true;                                                                                       \
  int64_t input_dim##dim_inx##_inx = get_nearest_pixel(original_##dim_inx##_idx, scales[dim_inx] < 1);                                                                                                                                               \
  if (input_dim##dim_inx##_inx > input_shape[dim_inx] - 1) input_dim##dim_inx##_inx = input_shape[dim_inx] - 1;                                                                                                                                      \
  if (input_dim##dim_inx##_inx < 0) input_dim##dim_inx##_inx = 0;                                                                                                                                                                                    \
  if (input_dim##dim_inx##_inx != input_dim_counters[dim_inx]) {                                                                                                                                                                                     \
    input_idx += (input_dim##dim_inx##_inx - input_dim_counters[dim_inx]) * input_dim_factor[dim_inx];                                                                                                                                               \
    input_dim_counters[dim_inx] = input_dim##dim_inx##_inx;                                                                                                                                                                                          \
  }

  if (n_dim == 1) {
    for (int64_t output_dim0_inx = 0; output_dim0_inx < output_shape[0]; output_dim0_inx++) {
      OneDimensionProcessor(0);
      output[output_idx++] = use_extrapolation_value[0] ? static_cast<T>(extrapolation_value) : input[input_idx];
    }
    return Status::OK();
  }

  if (n_dim == 2) {
    for (int64_t output_dim0_inx = 0; output_dim0_inx < output_shape[0]; output_dim0_inx++) {
      OneDimensionProcessor(0);
      for (int64_t output_dim1_inx = 0; output_dim1_inx < output_shape[1]; output_dim1_inx++) {
        OneDimensionProcessor(1);
        output[output_idx++] = (use_extrapolation_value[0] || use_extrapolation_value[1])
                                   ? static_cast<T>(extrapolation_value)
                                   : input[input_idx];
      }
    }
    return Status::OK();
  }

  if (n_dim == 3) {
    for (int64_t output_dim0_inx = 0; output_dim0_inx < output_shape[0]; output_dim0_inx++) {
      OneDimensionProcessor(0);
      for (int64_t output_dim1_inx = 0; output_dim1_inx < output_shape[1]; output_dim1_inx++) {
        OneDimensionProcessor(1);
        for (int64_t output_dim2_inx = 0; output_dim2_inx < output_shape[2]; output_dim2_inx++) {
          OneDimensionProcessor(2);
          bool use_extrapolation = std::any_of(use_extrapolation_value.begin(), use_extrapolation_value.end(),
                                               [](bool use_extrapolation) {
                                                 return use_extrapolation == true;
                                               });

          output[output_idx++] = use_extrapolation ? static_cast<T>(extrapolation_value) : input[input_idx];
        }
      }
    }
    return Status::OK();
  }

  if (n_dim == 4) {
    if (use_nearest2x_optimization && scales[0] == 1 && scales[1] == 1 && scales[2] == 2 && scales[3] == 2) {
      UpsampleNearest2x<T>(input_shape[0], input_shape[1], input_shape[2], input_shape[3], input, output);
      return Status::OK();
    }
    for (int64_t output_dim0_inx = 0; output_dim0_inx < output_shape[0]; output_dim0_inx++) {
      OneDimensionProcessor(0);
      for (int64_t output_dim1_inx = 0; output_dim1_inx < output_shape[1]; output_dim1_inx++) {
        OneDimensionProcessor(1);
        for (int64_t output_dim2_inx = 0; output_dim2_inx < output_shape[2]; output_dim2_inx++) {
          OneDimensionProcessor(2);
          for (int64_t output_dim3_inx = 0; output_dim3_inx < output_shape[3]; output_dim3_inx++) {
            OneDimensionProcessor(3);
            bool use_extrapolation = std::any_of(use_extrapolation_value.begin(), use_extrapolation_value.end(),
                                                 [](bool use_extrapolation) {
                                                   return use_extrapolation == true;
                                                 });

            output[output_idx++] = use_extrapolation ? static_cast<T>(extrapolation_value) : input[input_idx];
          }
        }
      }
    }
    return Status::OK();
  }

#undef OneDimensionProcessor

  std::vector<int64_t> output_dim_counter(n_dim);
  output_dim_counter[n_dim - 1] = -1;  // initialize dimension counter

  for (; output_idx < output_shape.Size(); output_idx++) {
    for (int64_t dim_idx = n_dim - 1; dim_idx >= 0; dim_idx--) {
      if (++output_dim_counter[dim_idx] < output_shape[dim_idx]) {
        int64_t current_input_dim_counter = 0;
        auto original_idx = get_original_coordinate(static_cast<float>(output_dim_counter[dim_idx]), scales[dim_idx],
                                                    static_cast<float>(output_shape[dim_idx]), static_cast<float>(input_shape[dim_idx]),
                                                    roi[dim_idx], roi[n_dim + dim_idx]);
        current_input_dim_counter = get_nearest_pixel(original_idx, scales[dim_idx] < 1);
        current_input_dim_counter = std::max((int64_t)0,
                                             std::min(current_input_dim_counter, (input_shape[dim_idx] - 1)));

        if (current_input_dim_counter != input_dim_counters[dim_idx]) {
          input_idx += (current_input_dim_counter - input_dim_counters[dim_idx]) * input_dim_factor[dim_idx];
          input_dim_counters[dim_idx] = current_input_dim_counter;
        }
        break;
      } else {
        output_dim_counter[dim_idx] = 0;
        input_idx += (0 - input_dim_counters[dim_idx]) * input_dim_factor[dim_idx];
        input_dim_counters[dim_idx] = 0;
      }
    }

    output[output_idx] = input[input_idx];
  }

  return Status::OK();
}

//This is a generic upsample in linear mode for N-D tensor.
//But what's the correct behavior for linear mode is not clear right now.
//this function is not enabled yet.
//this function is not tested for opset 11 changes yet
template <typename T>
Status UpsampleLinear(const T* input,
                      T* output,
                      const TensorShape& input_shape,
                      const TensorShape& output_shape,
                      const vector<float>& scales,
                      bool is_resize,
                      const std::vector<float>& roi,
                      GetOriginalCoordinateFunc get_original_coordinate) {
  if (!input || !output)
    return Status(ONNXRUNTIME, FAIL,
                  is_resize ? "Resize: input / output value is nullptr"
                            : "Upsample: input / output value is nullptr");
  if (input_shape.NumDimensions() != output_shape.NumDimensions())
    return Status(ONNXRUNTIME, FAIL,
                  is_resize ? "Resize: input/output value's dimension mismatch"
                            : "Upsample: input/output value's dimension mismatch");
  auto n_dim = input_shape.NumDimensions();
  for (size_t i = 0, size = output_shape.Size(); i < size; i++) {
    std::vector<int64_t> val1;
    std::vector<int64_t> val2;
    std::vector<float> d1;
    std::vector<float> d2;
    size_t cur_idx = i;
    //val1, vla2, d1, d2 are in reverse order
    for (auto j = static_cast<int64_t>(n_dim - 1); j >= 0; j--) {
      float resized_index = cur_idx % output_shape[j];
      float original_index = get_original_coordinate(static_cast<float>(resized_index), scales[j],
                                                     static_cast<float>(output_shape[j]), static_cast<float>(input_shape[j]),
                                                     roi[j], roi[n_dim + j]);
      float v = std::max(0.0f, std::min(original_index, static_cast<float>(input_shape[j] - 1)));
      auto v1 = std::min(static_cast<int64_t>(v), input_shape[j] - 1);
      auto v2 = std::min(v1 + 1, input_shape[j] - 1);
      if (v1 == v2) {
        d1.push_back(0.5f);
        d2.push_back(0.5f);
      } else {
        d1.push_back(std::abs(v - v1));
        d2.push_back(std::abs(v - v2));
      }
      val1.push_back(v1);
      val2.push_back(v2);
      cur_idx /= output_shape[j];
    }

    output[i] = 0;
    int64_t step = static_cast<int64_t>(1 << n_dim) - 1;
    while (step >= 0) {
      auto cur = step;
      float w = 1.0f;
      size_t old_idx = 0;
      size_t base = 1;
      for (auto j = static_cast<int64_t>(n_dim - 1); j >= 0; j--) {
        int64_t reverse_idx = static_cast<int64_t>(n_dim - 1) - j;
        w *= (cur % 2) ? d1[reverse_idx] : d2[reverse_idx];
        old_idx += ((cur % 2) ? val2[reverse_idx] : val1[reverse_idx]) * base;
        base *= input_shape[j];
        cur >>= 1;
      }
      output[i] += input[old_idx] * w;
      step--;
    }
  }
  return Status::OK();
}

// The following method supports a 4-D input in 'Linear mode'
// that amounts to 'Bilinear' Upsampling/Resizing in the sense that it assumes
// the scale values for the outermost 2 dimensions are 1.
// This is the common use-case where the 4-D input (batched multi-channel images)
// is usually of shape [N, C, H, W] and the scales are [1.0, 1.0, height_scale, width_scale]
template <typename T>
void UpsampleBilinear(int64_t batch_size,
                      int64_t num_channels,
                      int64_t input_height,
                      int64_t input_width,
                      int64_t output_height,
                      int64_t output_width,
                      float height_scale,
                      float width_scale,
                      const std::vector<float>& roi,
                      bool use_extrapolation,
                      float extrapolation_value,
                      const T* Xdata,
                      T* Ydata,
                      AllocatorPtr& alloc,
                      GetOriginalCoordinateFunc get_original_coordinate) {
  std::vector<float> y_original;
  std::vector<float> x_original;

  size_t idx_buffer_size = 2 * sizeof(int64_t) * (output_height + output_width);
  size_t scale_buffer_size = 2 * sizeof(float_t) * (output_height + output_width);
  auto inx_scale_data_buffer = alloc->Alloc(idx_buffer_size + scale_buffer_size);
  BufferUniquePtr idx_scale_data_buffer_holder(inx_scale_data_buffer, BufferDeleter(alloc));
  auto* idx_data = static_cast<int64_t*>(idx_scale_data_buffer_holder.get());
  int64_t* input_width_mul_y1 = idx_data;
  int64_t* input_width_mul_y2 = idx_data + output_height;
  int64_t* in_x1 = idx_data + 2 * output_height;
  int64_t* in_x2 = idx_data + 2 * output_height + output_width;

  auto* scale_data = reinterpret_cast<float*>(in_x2 + output_width);
  float* dy1 = scale_data;
  float* dy2 = scale_data + output_height;
  float* dx1 = scale_data + 2 * output_height;
  float* dx2 = scale_data + 2 * output_height + output_width;

  auto roi_y_start = roi.size() / 2 - 2;
  auto roi_y_end = roi.size() - 2;
  for (int64_t y = 0; y < output_height; ++y) {
    float in_y = get_original_coordinate(static_cast<float>(y), height_scale,
                                         static_cast<float>(output_height), static_cast<float>(input_height),
                                         roi[roi_y_start], roi[roi_y_end]);
    y_original.emplace_back(in_y);
    in_y = std::max(0.0f, std::min(in_y, static_cast<float>(input_height - 1)));

    const int64_t in_y1 = std::min(static_cast<int64_t>(in_y), input_height - 1);
    const int64_t in_y2 = std::min(in_y1 + 1, input_height - 1);
    dy1[y] = std::fabs(in_y - in_y1);
    dy2[y] = std::fabs(in_y - in_y2);

    if (in_y1 == in_y2) {
      dy1[y] = 0.5f;
      dy2[y] = 0.5f;
    }

    input_width_mul_y1[y] = input_width * in_y1;
    input_width_mul_y2[y] = input_width * in_y2;
  }

  auto roi_x_start = roi.size() / 2 - 1;
  auto roi_x_end = roi.size() - 1;
  for (int64_t x = 0; x < output_width; ++x) {
    float in_x = get_original_coordinate(static_cast<float>(x), width_scale,
                                         static_cast<float>(output_width), static_cast<float>(input_width),
                                         roi[roi_x_start], roi[roi_x_end]);
    x_original.emplace_back(in_x);
    in_x = std::max(0.0f, std::min(in_x, static_cast<float>(input_width - 1)));

    in_x1[x] = std::min(static_cast<int64_t>(in_x), input_width - 1);
    in_x2[x] = std::min(in_x1[x] + 1, input_width - 1);

    dx1[x] = std::abs(in_x - in_x1[x]);
    dx2[x] = std::abs(in_x - in_x2[x]);
    if (in_x1[x] == in_x2[x]) {
      dx1[x] = 0.5f;
      dx2[x] = 0.5f;
    }
  }

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        for (int64_t x = 0; x < output_width; ++x) {
          // when use_extrapolation is set and original index of x or y is out of the dim range
          // then use extrapolation_value as the output value.
          if (use_extrapolation &&
              ((y_original[y] < 0 || y_original[y] > static_cast<float>(input_height - 1)) ||
               (x_original[x] < 0 || x_original[x] > static_cast<float>(input_width - 1)))) {
            Ydata[output_width * y + x] = static_cast<T>(extrapolation_value);
            continue;
          }

          T X11 = Xdata[input_width_mul_y1[y] + in_x1[x]];
          T X21 = Xdata[input_width_mul_y1[y] + in_x2[x]];
          T X12 = Xdata[input_width_mul_y2[y] + in_x1[x]];
          T X22 = Xdata[input_width_mul_y2[y] + in_x2[x]];

          Ydata[output_width * y + x] = static_cast<T>(dx2[x] * dy2[y] * X11 +
                                                       dx1[x] * dy2[y] * X21 +
                                                       dx2[x] * dy1[y] * X12 +
                                                       dx1[x] * dy1[y] * X22);
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }
}

// Calculates cubic coeff based on Robert Keys approach
// https://ieeexplore.ieee.org/document/1163711
std::array<float, CubicModeGridLength> GetCubicCoeffs(float s, float cubic_coeff_a = -0.75) {
  auto abs_s = std::abs(s);
  std::array<float, CubicModeGridLength> coeffs;
  coeffs[0] = static_cast<float>(((cubic_coeff_a * (abs_s + 1) - 5 * cubic_coeff_a) * (abs_s + 1) + 8 * cubic_coeff_a) * (abs_s + 1) - 4 * cubic_coeff_a);
  coeffs[1] = static_cast<float>(((cubic_coeff_a + 2) * abs_s - (cubic_coeff_a + 3)) * abs_s * abs_s + 1);
  coeffs[2] = static_cast<float>(((cubic_coeff_a + 2) * (1 - abs_s) - (cubic_coeff_a + 3)) * (1 - abs_s) * (1 - abs_s) + 1);
  coeffs[3] = static_cast<float>(((cubic_coeff_a * (2 - abs_s) - 5 * cubic_coeff_a) * (2 - abs_s) + 8 * cubic_coeff_a) * (2 - abs_s) - 4 * cubic_coeff_a);
  return coeffs;
}

// Get the tensor data at the requested coordinate
template <typename T>
T GetDataForCoordinate(const T* Xdata,
                       int64_t x, int64_t y,
                       int64_t input_height, int64_t input_width) {
  x = std::max(static_cast<int64_t>(0), std::min(x, input_width - 1));
  y = std::max(static_cast<int64_t>(0), std::min(y, input_height - 1));
  return Xdata[y * input_width + x];
}

// Computes cubic convolution interpolation in 1D
template <typename T>
float CubicInterpolation1D(const T* Xdata,
                           int64_t x,
                           int64_t y,
                           int64_t input_height,
                           int64_t input_width,
                           std::array<float, CubicModeGridLength>& coeff_array,
                           float coeff_sum,
                           std::unordered_map<int64_t, float>& cache) {
  // When calculating cubic interpolation we move the 4*4 grid across the original data and therefore there is
  // opportunity to cache the results for previously seen combinations.
  // Check if the result is already available in the cache
  auto grid_start_pos = (y)*input_width + (x - 1);
  if (cache.find(grid_start_pos) != cache.end()) {
    return cache[grid_start_pos];
  }

  // get the neighbors in 1D and find interpolation for this dimension
  // for 1D cubic interpolation 4 samples are used. 2 on the left and 2 on the right of x
  float result = 0;
  for (int i = 0, j = -1; i < static_cast<int>(CubicModeGridLength); i++, j++) {
    auto orig_data = GetDataForCoordinate(Xdata, x + j, y, input_height, input_width);
    result += coeff_array[i] / coeff_sum * orig_data;
  }
  cache[grid_start_pos] = result;

  return result;
}

template <typename T>
void ResizeBiCubic(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    float height_scale,
    float width_scale,
    float cubic_coeff_a,
    bool use_extrapolation,
    float extrapolation_value,
    bool exclude_outside,
    const std::vector<float>& roi,
    const T* Xdata,
    T* Ydata,
    GetOriginalCoordinateFunc get_original_coordinate) {
  std::vector<float> y_original;
  std::vector<float> x_original;
  std::unordered_map<float, std::array<float, CubicModeGridLength>> cubic_coeffs;
  std::unordered_map<float, std::unordered_map<int64_t, float>> coeff_to_1Dinterpolation_map;
  auto roi_y_start = roi.size() / 2 - 2;
  auto roi_y_end = roi.size() - 2;
  auto roi_x_start = roi.size() / 2 - 1;
  auto roi_x_end = roi.size() - 1;

  // generate coefficients in y direction
  for (int64_t y = 0; y < output_height; ++y) {
    float in_y = get_original_coordinate(static_cast<float>(y), height_scale,
                                         static_cast<float>(output_height), static_cast<float>(input_height),
                                         roi[roi_y_start], roi[roi_y_end]);
    y_original.emplace_back(in_y);
    auto s = y_original[y] - std::floor(y_original[y]);
    if (cubic_coeffs.find(s) == cubic_coeffs.end()) {
      cubic_coeffs[s] = GetCubicCoeffs(s, cubic_coeff_a);
      coeff_to_1Dinterpolation_map[s] = {};
    }
  }

  // generate coefficients in x direction
  for (int64_t x = 0; x < output_width; ++x) {
    float in_x = get_original_coordinate(static_cast<float>(x), width_scale,
                                         static_cast<float>(output_width), static_cast<float>(input_width),
                                         roi[roi_x_start], roi[roi_x_end]);
    x_original.emplace_back(in_x);
    auto s = x_original[x] - std::floor(x_original[x]);
    if (cubic_coeffs.find(s) == cubic_coeffs.end()) {
      cubic_coeffs[s] = GetCubicCoeffs(s, cubic_coeff_a);
      coeff_to_1Dinterpolation_map[s] = {};
    }
  }

  // setup up temp arrays to hold coefficients when exclude_outside is set to true
  std::array<float, CubicModeGridLength> y_coeff_holder;
  std::array<float, CubicModeGridLength> x_coeff_holder;
  float y_coeff_sum = 1;
  float x_coeff_sum = 1;

  for (int64_t n = 0; n < batch_size; n++) {
    for (int64_t c = 0; c < num_channels; c++) {
      for (int64_t y = 0; y < output_height; ++y) {
        auto in_y = y_original[y];

        // when use_extrapolation is set and original index is out of the dim range
        // then use extrapolation_value as the output value.
        if (use_extrapolation && (in_y < 0 || in_y > static_cast<float>(input_height - 1))) {
          for (int64_t x = 0; x < output_width; ++x) {
            Ydata[y * output_width + x] = extrapolation_value;
          }
          continue;
        }

        auto y_int = static_cast<int64_t>(std::floor(in_y));
        auto& coeff_y = exclude_outside ? y_coeff_holder : cubic_coeffs[in_y - y_int];
        y_coeff_sum = 1;

        if (exclude_outside) {
          // When true, the weight of sampling locations outside the grid will be set to 0
          // and the weight will be renormalized so that their sum is 1.0
          y_coeff_sum = 0;
          auto& orig_y_coeffs = cubic_coeffs[in_y - y_int];
          for (int64_t i = 0, y_val = y_int - 1; y_val <= y_int + 2; y_val++, i++) {
            y_coeff_holder[i] = (y_val < 0 || y_val >= static_cast<float>(input_height)) ? 0.0f : orig_y_coeffs[i];
            y_coeff_sum += y_coeff_holder[i];
          }
        }

        for (int64_t x = 0; x < output_width; ++x) {
          auto in_x = x_original[x];

          // when use_extrapolation is set and original index is out of the dim range
          // then use extrapolation_value as the output value.
          if (use_extrapolation && (in_x < 0 || in_x > static_cast<float>(input_width - 1))) {
            Ydata[y * output_width + x] = extrapolation_value;
            continue;
          }

          auto x_int = static_cast<int64_t>(std::floor(in_x));
          auto s_x = static_cast<float>(in_x - x_int);
          auto& coeff_x = exclude_outside ? x_coeff_holder : cubic_coeffs[s_x];
          x_coeff_sum = 1;

          if (exclude_outside) {
            // When true, the weight of sampling locations outside the grid will be set to 0
            // and the weight will be renormalized so that their sum is 1.0
            x_coeff_sum = 0;
            auto& orig_x_coeff = cubic_coeffs[s_x];
            for (int64_t i = 0, x_val = x_int - 1; x_val <= x_int + 2; x_val++, i++) {
              x_coeff_holder[i] = (x_val < 0 || x_val >= static_cast<float>(input_width)) ? 0.0f : orig_x_coeff[i];
              x_coeff_sum += x_coeff_holder[i];
            }
          }

          // Compute cubic interpolation in x dimension using the x coefficients.
          // From the result of cubic interpolation in x dim, compute cubic interpolation in y dimension
          auto& interpolation_result_cache = coeff_to_1Dinterpolation_map[s_x];
          float result = 0;
          for (int64_t y_val = y_int - 1, i = 0; y_val <= y_int + 2; y_val++, i++) {
            auto x_interpolation_result = CubicInterpolation1D(Xdata, x_int, y_val,
                                                               input_height, input_width, coeff_x, x_coeff_sum,
                                                               interpolation_result_cache);
            result += x_interpolation_result * coeff_y[i] / y_coeff_sum;
          }

          Ydata[y * output_width + x] = static_cast<T>(result);
        }
      }

      Xdata += input_height * input_width;
      Ydata += output_height * output_width;

      // clear the cache when moving to the next channel
      coeff_to_1Dinterpolation_map.clear();
    }
  }
}

template <typename T>
Status Upsample<T>::BaseCompute(OpKernelContext* context,
                                const std::vector<float>& roi,
                                const std::vector<float>& scales,
                                const std::vector<int64_t>& output_dims) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);
  const std::vector<int64_t>& dims = X->Shape().GetDims();
  ORT_ENFORCE(output_dims.size() == dims.size(), "Rank of input and output tensor should be same.");

  Tensor* Y = context->Output(0, output_dims);
    
  if (dims.size() != scales.size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  is_resize ? "Resize: input tensor's dimension does not match the scales."
                            : "Upsample: input tensor's dimension does not match the scales.");

  if (roi.size() != 2 * X->Shape().GetDims().size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Resize: size of roi array should be 2 * N where N is the rank of input tensor X.");

  bool no_scale = true;
  for (std::size_t i = 0, end = output_dims.size(); i < end; i++) {
    if (no_scale && output_dims[i] != dims[i]) no_scale = false;
  }

  if (no_scale) {
    memcpy(Y->MutableDataRaw(), X->DataRaw(), Y->SizeInBytes());
    return Status::OK();
  }

  switch (mode_) {
    case UpsampleMode::NN:
      return UpsampleNearest<T>(X->template Data<T>(), Y->template MutableData<T>(), X->Shape(), Y->Shape(), scales, roi,
                                is_resize, use_extrapolation_, extrapolation_value_, use_nearest2x_optimization,
                                get_original_coordinate_, get_nearest_pixel_);
    case UpsampleMode::LINEAR: {
      //The correct behavior of 'linear' mode for an N-D input is not clear right now,
      //so only support 'bilinear' with 2-D or 4-D input tensor with outermost 2 scales as 1 in the 4-D case
      if (dims.size() != 2 && dims.size() != 4) {
        std::ostringstream oss;
        oss << "'Linear' mode only support 2-D inputs ('Bilinear') or 4-D inputs "
               "with the corresponding outermost 2 scale values being 1 in the ";
        oss << (is_resize ? "Resize operator" : "Upsample operator");
        return Status(ONNXRUNTIME, FAIL, oss.str());
      }

      bool is_2D = dims.size() == 2;
      const int64_t batch_size = is_2D ? 1 : dims[0];
      const int64_t num_channels = is_2D ? 1 : dims[1];
      const int64_t input_height = is_2D ? dims[0] : dims[2];
      const int64_t input_width = is_2D ? dims[1] : dims[3];
      const int64_t output_height = is_2D ? output_dims[0] : output_dims[2];
      const int64_t output_width = is_2D ? output_dims[1] : output_dims[3];

      AllocatorPtr alloc;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
      UpsampleBilinear(batch_size, num_channels, input_height, input_width, output_height, output_width,
                       is_2D ? scales[0] : scales[2], is_2D ? scales[1] : scales[3], roi,
                       use_extrapolation_, extrapolation_value_, X->template Data<T>(),
                       Y->template MutableData<T>(), alloc, get_original_coordinate_);
      return Status::OK();
    }
    case UpsampleMode::CUBIC: {
      bool is_2D = dims.size() == 2;
      const int64_t batch_size = is_2D ? 1 : dims[0];
      const int64_t num_channels = is_2D ? 1 : dims[1];
      const int64_t input_height = is_2D ? dims[0] : dims[2];
      const int64_t input_width = is_2D ? dims[1] : dims[3];
      const int64_t output_height = is_2D ? output_dims[0] : output_dims[2];
      const int64_t output_width = is_2D ? output_dims[1] : output_dims[3];

      ResizeBiCubic(batch_size, num_channels, input_height, input_width, output_height, output_width,
                    is_2D ? scales[0] : scales[2], is_2D ? scales[1] : scales[3], cubic_coeff_a_, use_extrapolation_,
                    extrapolation_value_, exclude_outside_, roi, X->template Data<float>(), Y->template MutableData<float>(),
                    get_original_coordinate_);
      return Status::OK();
    }
    default:
      return Status(ONNXRUNTIME, FAIL, is_resize ? "Resize: unexpected mode" : "Upsample: unexpected mode");
  }
}

template <typename T>
Status Upsample<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);

  std::vector<int64_t> output_dims(X->Shape().GetDims().size());

  // Get roi data
  // Initialize the roi array to all zeros as this will be the most common case
  // Roi data is needed only when coordinate transformation mode is set to tf_crop_and_resize
  // for all other cases we need a 0 initialized roi array
  std::vector<float> roi_array;
  const std::vector<float>* roi_ptr = roi_cached_ ? &roi_ : &roi_array;

  if (!roi_cached_) {
    if (need_roi_input_) {
      ORT_ENFORCE(roi_input_idx_ > 0, "Invalid roi input index.");

      const auto* roi = context->Input<Tensor>(roi_input_idx_);
      ParseRoiData(roi, roi_array);
    } else {
      roi_array.resize(X->Shape().GetDims().size() * 2);
      std::fill(roi_array.begin(), roi_array.end(), 0.0f);
    }
  }

  if (OpKernel::Node().InputDefs().size() == 1) {
    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_, X->Shape().GetDims(), output_dims);
    return BaseCompute(context, *roi_ptr, scales_, output_dims);
  }

  const auto* scales = context->Input<Tensor>(scales_input_idx_);
  const auto* sizes = context->Input<Tensor>(sizes_input_idx_);
  ORT_ENFORCE(scales != nullptr);

  if (scales_cached_) {
    ORT_ENFORCE(sizes == nullptr, "Only one of scales or sizes must be provided as input.");

    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_, X->Shape().GetDims(), output_dims);
    return BaseCompute(context, *roi_ptr, scales_, output_dims);
  }

  // Get scales data
  std::vector<float> scales_array(X->Shape().GetDims().size());

  if (scales != nullptr && scales->Shape().Size() != 0) {
    ORT_ENFORCE(sizes == nullptr,
                "Only one of scales or sizes must be provided as input.");
    ParseScalesData(scales, scales_array);

    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_array, X->Shape().GetDims(), output_dims);
  } else {
    ORT_ENFORCE(sizes != nullptr && sizes->Shape().Size() != 0,
                "Either scales or sizes MUST be provided as input.");

    // When sizes input is available directly populate it into the output_dims array.
    memcpy(output_dims.data(), sizes->template Data<int64_t>(), sizes->Shape().Size() * sizeof(int64_t));

    ORT_ENFORCE(X->Shape().GetDims().size() == output_dims.size(),
                "Resize: input tensor's rank does not match the output tensor's rank.");

    ParseScalesDataFromOutputSize(output_dims, X->Shape().GetDims(), scales_array);
  }

  return BaseCompute(context, *roi_ptr, scales_array, output_dims);
}
}  // namespace onnxruntime

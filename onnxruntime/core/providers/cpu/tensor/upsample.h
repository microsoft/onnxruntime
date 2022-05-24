// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif
#include <cmath>
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {

constexpr const char* UpsampleModeNN = "nearest";
constexpr const char* UpsampleModeLinear = "linear";
constexpr const char* UpsampleModeCubic = "cubic";

// In case of cubic mode the grid used to calculate the interpolation value
// is a 4x4 matrix
constexpr size_t CubicModeGridLength = 4;

using GetNearestPixelFunc = int64_t (*)(float, bool);
using GetOriginalCoordinateFunc = float (*)(float, float, float, float, float, float);

enum UpsampleMode {
  NN = 0,      // nearest neighbour
  LINEAR = 1,  // linear interpolation
  CUBIC = 2,   // cubic interpolation
};

enum ResizeCoordinateTransformationMode {
  HALF_PIXEL = 0,
  ASYMMETRIC = 1,
  PYTORCH_HALF_PIXEL = 2,
  TF_HALF_PIXEL_FOR_NN = 3,
  ALIGN_CORNERS = 4,
  TF_CROP_AND_RESIZE = 5,
  CoordinateTransformationModeCount = 6,
};

enum ResizeNearestMode {
  SIMPLE = 0,  // For resize op 10
  ROUND_PREFER_FLOOR = 1,
  ROUND_PREFER_CEIL = 2,
  FLOOR = 3,
  CEIL = 4,
  NearestModeCount = 5,
};

struct BilinearParams {
  std::vector<float> x_original;
  std::vector<float> y_original;

  BufferUniquePtr idx_scale_data_buffer_holder;

  int64_t* input_width_mul_y1;
  int64_t* input_width_mul_y2;

  int64_t* in_x1;
  int64_t* in_x2;

  float* dx1;
  float* dx2;

  float* dy1;
  float* dy2;
};

class UpsampleBase {
 protected:
  UpsampleBase(const OpKernelInfo& info) : scales_cached_(false), roi_cached_(false), use_extrapolation_(false) {
    const auto& node = info.node();
    auto opset = node.SinceVersion();
    is_resize_ = (opset >= 10);

    std::string mode;
    ORT_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());
    mode_ = StringToUpsampleMode(mode);

    auto input_count = info.GetInputCount();
    if (input_count == 1) {  // opset < 10
      ORT_ENFORCE(info.GetAttrs<float>("scales", scales_).IsOK());
      ScalesValidation(scales_, mode_);
      scales_cached_ = true;
    }

    extrapolation_value_ = info.GetAttrOrDefault<float>("extrapolation_value", 0.0f);

    // Coordinate transformation mode attr was introduced in version 11.
    // before that asymmetric mode was the only available transformation mode
    std::string coordinate_transform_mode_name =
        opset > 10
            ? info.GetAttrOrDefault<std::string>("coordinate_transformation_mode", "half_pixel")
            : "asymmetric";

    coordinate_transform_mode_ = StringToCoordinateTransformationMode(coordinate_transform_mode_name);
    if (opset >= 13 && coordinate_transform_mode_ == TF_HALF_PIXEL_FOR_NN) {
      LOGS_DEFAULT(WARNING)
          << "`tf_half_pixel_for_nn` is deprecated since opset 13, "
          << "yet this opset " << opset << " model uses the deprecated attribute";
    }

    get_original_coordinate_ = GetOriginalCoordinateFromResizedCoordinate(coordinate_transform_mode_);
    use_extrapolation_ = need_roi_input_ = (coordinate_transform_mode_ == TF_CROP_AND_RESIZE);

    std::string nearest_mode_name = (mode_ == NN && opset >= 11)
                                        ? info.GetAttrOrDefault<std::string>("nearest_mode", "round_prefer_floor")
                                        : "";
    nearest_mode_ = StringToNearestMode(nearest_mode_name);
    get_nearest_pixel_ = GetNearestPixelFromOriginal(nearest_mode_);

    cubic_coeff_a_ = info.GetAttrOrDefault<float>("cubic_coeff_a", -0.75f);
    exclude_outside_ = info.GetAttrOrDefault<int64_t>("exclude_outside", 0) == 0 ? false : true;

    if (exclude_outside_ == 1 && mode_ != CUBIC) {
      ORT_THROW("exclude_outside can be set to 1 only when mode is CUBIC. Current mode is set to " + mode);
    }

    // see if we can potentially use the nearest2x optimization. scales are checked at runtime to be {1,1,2,2}
    use_nearest2x_optimization_ =
        (opset < 11) ? true
                     : (mode_ == UpsampleMode::NN &&
                        coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ASYMMETRIC &&
                        nearest_mode_ == ResizeNearestMode::FLOOR);

    if (opset > 10) {
      roi_input_idx_ = 1;
      scales_input_idx_ = 2;
      sizes_input_idx_ = 3;
    } else if (opset <= 10 && input_count > 1) {
      scales_input_idx_ = 1;
    }

    if (scales_input_idx_ > 0) {
      const Tensor* scale;
      bool get_scale = info.TryGetConstantInput(scales_input_idx_, &scale);

      if (get_scale && scale->Shape().Size() > 0) {
        ParseScalesData(scale, scales_);
        scales_cached_ = true;
      }
    }

    // roi is only needed when coordinate transformation mode is tf_crop_and_resize
    // for all other modes no need to read roi input
    if (roi_input_idx_ > 0 && need_roi_input_) {
      const Tensor* roi;
      bool get_roi = info.TryGetConstantInput(roi_input_idx_, &roi);

      if (get_roi) {
        ParseRoiData(roi, roi_);
        roi_cached_ = true;
      }
    }
  }

  UpsampleMode mode_;
  ResizeCoordinateTransformationMode coordinate_transform_mode_;
  GetOriginalCoordinateFunc get_original_coordinate_;
  ResizeNearestMode nearest_mode_;
  GetNearestPixelFunc get_nearest_pixel_;
  float cubic_coeff_a_;
  bool exclude_outside_;
  float extrapolation_value_;
  bool use_nearest2x_optimization_ = false;

  std::vector<float> scales_;
  std::vector<float> roi_;
  bool scales_cached_;
  bool roi_cached_;
  bool need_roi_input_;
  bool use_extrapolation_;
  bool is_resize_ = false;

  int roi_input_idx_ = -1;
  int scales_input_idx_ = -1;
  int sizes_input_idx_ = -1;

  UpsampleMode StringToUpsampleMode(const std::string& mode) {
    if (mode == UpsampleModeNN) {
      return UpsampleMode::NN;
    }
    if (mode == UpsampleModeLinear) {
      return UpsampleMode::LINEAR;
    }
    if (mode == UpsampleModeCubic) {
      return UpsampleMode::CUBIC;
    }
    ORT_THROW("mode attribute is " + mode + ". It can only be " +
              UpsampleModeNN + "(default) or " + UpsampleModeLinear + " or " + UpsampleModeCubic + ".");
  }

  ResizeCoordinateTransformationMode StringToCoordinateTransformationMode(
      const std::string& coordinate_transform_mode_name) {
    if (coordinate_transform_mode_name == "asymmetric") {
      return ASYMMETRIC;
    }
    if (coordinate_transform_mode_name == "pytorch_half_pixel") {
      return PYTORCH_HALF_PIXEL;
    }
    if (coordinate_transform_mode_name == "tf_half_pixel_for_nn") {
      return TF_HALF_PIXEL_FOR_NN;
    }
    if (coordinate_transform_mode_name == "align_corners") {
      return ALIGN_CORNERS;
    }
    if (coordinate_transform_mode_name == "tf_crop_and_resize") {
      return TF_CROP_AND_RESIZE;
    }
    if (coordinate_transform_mode_name == "half_pixel") {
      return HALF_PIXEL;
    }
    ORT_THROW("coordinate_transform_mode:[" + coordinate_transform_mode_name + "] is not supportted!");
  }

  GetOriginalCoordinateFunc GetOriginalCoordinateFromResizedCoordinate(
      ResizeCoordinateTransformationMode coordinate_transform_mode) {
    switch (coordinate_transform_mode) {
      case ASYMMETRIC:
        return [](float x_resized, float x_scale, float, float, float, float) {
          return x_resized / x_scale;
        };
      case PYTORCH_HALF_PIXEL:
        return [](float x_resized, float x_scale, float length_resized, float, float, float) {
          return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
        };
      case TF_HALF_PIXEL_FOR_NN:
        return [](float x_resized, float x_scale, float, float, float, float) {
          return (x_resized + 0.5f) / x_scale;
        };
      case ALIGN_CORNERS:
        return [](float x_resized, float, float length_resized, float length_original, float, float) {
          return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
        };
      case TF_CROP_AND_RESIZE:
        return [](float x_resized, float, float length_resized, float length_original, float roi_start, float roi_end) {
          auto orig = length_resized > 1
                          ? roi_start * (length_original - 1) + (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                          : 0.5 * (roi_start + roi_end) * (length_original - 1);
          return static_cast<float>(orig);
        };
      default:  // "half_pixel"
        return [](float x_resized, float x_scale, float, float, float, float) {
          return ((x_resized + 0.5f) / x_scale) - 0.5f;
        };
    }
  }

  ResizeNearestMode StringToNearestMode(const std::string& nearest_mode_name) {
    if (nearest_mode_name == "round_prefer_floor") {
      return ROUND_PREFER_FLOOR;
    } else if (nearest_mode_name == "round_prefer_ceil") {
      return ROUND_PREFER_CEIL;
    } else if (nearest_mode_name == "floor") {
      return FLOOR;
    } else if (nearest_mode_name == "ceil") {
      return CEIL;
    } else if (nearest_mode_name == "") {
      return SIMPLE;
    }
    ORT_THROW("nearest_mode:[" + nearest_mode_name + "] is not supported!");
  }

  GetNearestPixelFunc GetNearestPixelFromOriginal(ResizeNearestMode nearest_mode) {
    switch (nearest_mode) {
      case SIMPLE:
        // versions older than 11 did not have nearest_mode attr. Use the original logic in this case
        // to maintain backward compatibility
        return [](float x_original, bool isDownSample) {
          if (isDownSample) {
            return static_cast<int64_t>(std::ceil(x_original));
          } else {
            return static_cast<int64_t>(x_original);
          }
        };
      case ROUND_PREFER_CEIL:
        return [](float x_original, bool) {
          return static_cast<int64_t>(std::round(x_original));
        };
      case FLOOR:
        return [](float x_original, bool) {
          return static_cast<int64_t>(std::floor(x_original));
        };
      case CEIL:
        return [](float x_original, bool) {
          return static_cast<int64_t>(std::ceil(x_original));
        };
      default:  // default is round_prefer_floor
        return [](float x_original, bool) {
          // for half way cases prefer floor
          if (x_original == static_cast<int64_t>(x_original) + 0.5f) {
            return static_cast<int64_t>(std::floor(x_original));
          }
          return static_cast<int64_t>(std::round(x_original));
        };
    }
  }

  void ScalesValidation(const std::vector<float>& scales, const UpsampleMode mode) const {
    if (!is_resize_) {
      for (auto& scale : scales) {
        ORT_ENFORCE(scale >= 1, "Scale value should be greater than or equal to 1.");
      }
    } else {
      for (auto& scale : scales) {
        ORT_ENFORCE(scale > 0, "Scale value should be greater than 0.");
      }
    }

    if (UpsampleMode::LINEAR == mode) {
      ORT_ENFORCE(scales.size() == 2 ||
                      (scales.size() == 4 && scales[0] == 1 && scales[1] == 1) ||
                      (scales.size() == 4 && scales[0] == 1 && scales[3] == 1) ||
                      scales.size() == 3 ||
                      (scales.size() == 5 && scales[0] == 1 && scales[1] == 1),
                  "'Linear' mode only support:\n"
                  "  * 2-D inputs or\n"
                  "  * 3-D inputs ('Bilinear', 'Trilinear') or\n"
                  "  * 4-D inputs with the corresponding outermost 2 scale values being 1"
                  " or the corresponding outermost and innermost scale values being 1 or\n"
                  "  * 5-D inputs with the corresponding outermost 2 scale values being 1"
                  "in the ",
                  is_resize_ ? "Resize operator" : "Upsample operator");
    }

    else if (UpsampleMode::CUBIC == mode) {
      ORT_ENFORCE(scales.size() == 2 || (scales.size() == 4 && scales[0] == 1 && scales[1] == 1),
                  "'Cubic' mode only support 2-D inputs ('Bicubic') or 4-D inputs "
                  "with the corresponding outermost 2 scale values being 1 in the ",
                  is_resize_ ? "Resize operator" : "Upsample operator");
    }
  }

  void
  ParseScalesData(const Tensor* scale, std::vector<float>& scales) const {
    const auto* scale_data = scale->template Data<float>();
    int64_t scales_size = scale->Shape().Size();
    ORT_ENFORCE(scales_size > 0, "scales size should be greater than 0.");
    if (scales.empty()) {
      scales.resize(scales_size);
    }
    memcpy(scales.data(), scale_data, scales_size * sizeof(float));
    ScalesValidation(scales, mode_);
  }

  void ParseRoiData(const Tensor* roi, std::vector<float>& roi_array) const {
    int64_t roi_size = roi->Shape().Size();
    if (roi_size > 0) {
      roi_array.resize(roi_size);
      memcpy(roi_array.data(), roi->template Data<float>(), roi_size * sizeof(float));
    }
  }

  void ParseScalesDataFromOutputSize(gsl::span<const int64_t> output_dims,
                                     gsl::span<const int64_t> input_dims,
                                     std::vector<float>& scales) const {
    for (size_t i = 0, end = input_dims.size(); i < end; ++i) {
      // Handle corner case to avoid dividing by zero in the next step
      if (input_dims[i] == 0) {
        // Enforce that output_dim is 0, given that we cannot scale 0 by any factor to
        // result in any non-zero value
        ORT_ENFORCE(output_dims[i] == 0,
                    "Input dim is zero but required output dim is non-zero. ",
                    "Cannot scale 0 by any factor to generate a non-zero value. ",
                    "Dimension: ", i, " Input dim value: ", input_dims[i], " Output dim value: ", output_dims[i]);
        // Scale can be any arbitrary value as technically scaling 0 by any factor
        // results in 0. Keeping scale as 1 is more intuitive given that input_dim == output_dim.
        scales[i] = 1.f;
      } else {
        scales[i] = static_cast<float>(output_dims[i]) / static_cast<float>(input_dims[i]);
      }
    }
    ScalesValidation(scales, mode_);
  }

  void ComputeOutputShape(const std::vector<float>& scales,
                          gsl::span<const int64_t> input_dims,
                          TensorShapeVector& output_dims) const {
    for (std::size_t i = 0; i < input_dims.size(); i++) {
      output_dims[i] = static_cast<int64_t>(scales[i] * input_dims[i]);
    }
  }
};  // UpsampleBase

template <typename T>
class Upsample : public UpsampleBase, public OpKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  Status BaseCompute(OpKernelContext* context, const std::vector<float>& roi, const std::vector<float>& scales,
                     const gsl::span<const int64_t>& output_dims) const;
};

BilinearParams SetupUpsampleBilinear(const int64_t input_height,
                                     const int64_t input_width,
                                     const int64_t output_height,
                                     const int64_t output_width,
                                     const float height_scale,
                                     const float width_scale,
                                     const std::vector<float>& roi,
                                     AllocatorPtr& alloc,
                                     const GetOriginalCoordinateFunc& get_original_coordinate,
                                     bool is_nchw);

template <typename T>
void UpsampleBilinear(const int64_t batch_size,
                      const int64_t num_channels,
                      const int64_t input_height,
                      const int64_t input_width,
                      const int64_t output_height,
                      const int64_t output_width,
                      const float height_scale,
                      const float width_scale,
                      const std::vector<float>& roi,
                      const bool use_extrapolation,
                      const float extrapolation_value,
                      const T* const XdataBase,
                      T* const YdataBase,
                      AllocatorPtr& alloc,
                      const GetOriginalCoordinateFunc& get_original_coordinate,
                      concurrency::ThreadPool* tp) {
  BilinearParams p = SetupUpsampleBilinear(input_height, input_width, output_height, output_width,
                                           height_scale, width_scale, roi,
                                           alloc, get_original_coordinate, true);
  for (int64_t n = 0; n < batch_size; ++n) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_channels,
        [&](std::ptrdiff_t c) {
          const T* Xdata = XdataBase + (n * num_channels + c) * (input_height * input_width);
          T* Ydata = YdataBase + (n * num_channels + c) * (output_height * output_width);
          for (int64_t y = 0; y < output_height; ++y) {
            for (int64_t x = 0; x < output_width; ++x) {
              // when use_extrapolation is set and original index of x or y is out of the dim range
              // then use extrapolation_value as the output value.
              if (use_extrapolation &&
                  ((p.y_original[y] < 0 || p.y_original[y] > static_cast<float>(input_height - 1)) ||
                   (p.x_original[x] < 0 || p.x_original[x] > static_cast<float>(input_width - 1)))) {
                Ydata[output_width * y + x] = static_cast<T>(extrapolation_value);
                continue;
              }

              T X11 = Xdata[p.input_width_mul_y1[y] + p.in_x1[x]];
              T X21 = Xdata[p.input_width_mul_y1[y] + p.in_x2[x]];
              T X12 = Xdata[p.input_width_mul_y2[y] + p.in_x1[x]];
              T X22 = Xdata[p.input_width_mul_y2[y] + p.in_x2[x]];

              Ydata[output_width * y + x] = static_cast<T>(p.dx2[x] * p.dy2[y] * X11 +
                                                           p.dx1[x] * p.dy2[y] * X21 +
                                                           p.dx2[x] * p.dy1[y] * X12 +
                                                           p.dx1[x] * p.dy1[y] * X22);
            }
          }
        });
  }
}

template <typename T>
void NhwcUpsampleBilinear(const int64_t batch_size,
                          const int64_t num_channels,
                          const int64_t input_height,
                          const int64_t input_width,
                          const int64_t output_height,
                          const int64_t output_width,
                          const float height_scale,
                          const float width_scale,
                          const std::vector<float>& roi,
                          const bool use_extrapolation,
                          const float extrapolation_value,
                          const T* const XdataBase,
                          T* const YdataBase,
                          AllocatorPtr& alloc,
                          const GetOriginalCoordinateFunc& get_original_coordinate,
                          concurrency::ThreadPool* tp) {
  BilinearParams p = SetupUpsampleBilinear(input_height, input_width, output_height, output_width,
                                           height_scale, width_scale, roi,
                                           alloc, get_original_coordinate, false);
  for (int64_t n = 0; n < batch_size; ++n) {
    const T* Xdata = XdataBase + n * (input_height * input_width) * num_channels;
    T* Ydata = YdataBase + n * (output_height * output_width) * num_channels;
    concurrency::ThreadPool::TryParallelFor(
        tp, output_height * output_width,
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          for (std::ptrdiff_t i = first; i < last; ++i) {
            const int64_t x = i % output_width;
            const int64_t y = i / output_width;
            for (int64_t c = 0; c < num_channels; ++c) {
              // when use_extrapolation is set and original index of x or y is out of the dim range
              // then use extrapolation_value as the output value.
              if (use_extrapolation &&
                  ((p.y_original[y] < 0 || p.y_original[y] > static_cast<float>(input_height - 1)) ||
                   (p.x_original[x] < 0 || p.x_original[x] > static_cast<float>(input_width - 1)))) {
                Ydata[(output_width * y + x) * num_channels + c] = static_cast<T>(extrapolation_value);
                continue;
              }

              T X11 = Xdata[(p.input_width_mul_y1[y] + p.in_x1[x]) * num_channels + c];
              T X21 = Xdata[(p.input_width_mul_y1[y] + p.in_x2[x]) * num_channels + c];
              T X12 = Xdata[(p.input_width_mul_y2[y] + p.in_x1[x]) * num_channels + c];
              T X22 = Xdata[(p.input_width_mul_y2[y] + p.in_x2[x]) * num_channels + c];

              Ydata[(output_width * y + x) * num_channels + c] = static_cast<T>(p.dx2[x] * p.dy2[y] * X11 +
                                                                                p.dx1[x] * p.dy2[y] * X21 +
                                                                                p.dx2[x] * p.dy1[y] * X12 +
                                                                                p.dx1[x] * p.dy1[y] * X22);
            }
          }
        });
  }
}

// TFLite are referred for the implementation.
inline void ComputeInterpolationValuesInteger(
    const int32_t value, const int32_t scale_10, const bool is_half_pixel,
    const int32_t input_size, int32_t* scaled_value, int32_t* lower_bound,
    int32_t* upper_bound) {
  if (is_half_pixel) {
    *scaled_value = value * scale_10 + scale_10 / 2 - (1 << 9);
  } else {
    *scaled_value = value * scale_10;
  }
  constexpr int32_t zero = 0;
  *lower_bound = std::max(*scaled_value / (1 << 10), zero);
  *upper_bound =
      std::min((*scaled_value + (1 << 10) - 1) / (1 << 10), input_size - 1);
}

// TFLite are referred for the implementation.
inline int Offset(const gsl::span<const int64_t>& dims_data, int i0, int i1, int i2, int i3) {
  assert(dims_data.size() == 4);
  assert(i0 >= 0 && i0 < dims_data[0]);
  assert(i1 >= 0 && i1 < dims_data[1]);
  assert(i2 >= 0 && i2 < dims_data[2]);
  assert(i3 >= 0 && i3 < dims_data[3]);
  return static_cast<int>(((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3);
}

// TFLite are referred for the implementation.
template <typename T>
inline void NhwcUpsampleBilinearInteger(
    bool is_half_pixel, bool is_align_corners,
    const int32_t batches, const int32_t input_height, const int32_t input_width, const int32_t depth,
    const gsl::span<const int64_t>& input_dims,
    const T* input_data,
    const int32_t output_height, const int32_t output_width,
    const gsl::span<const int64_t>& output_dims,
    T* output_data,
    int32_t height_scale_10, int32_t width_scale_10) {
  assert(!is_half_pixel || !is_align_corners);
  assert(input_dims.size() == 4);
  assert(output_dims.size() == 4);
  if (is_align_corners && output_height > 1) {
    height_scale_10 =
        ((1 << 10) * (input_height - 1) + (output_height - 1) / 2) /
        (output_height - 1);
  }
  if (is_align_corners && output_width > 1) {
    width_scale_10 = ((1 << 10) * (input_width - 1) + (output_width - 1) / 2) /
                     (output_width - 1);
  }

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32_t input_y, y0, y1;
      ComputeInterpolationValuesInteger(y, height_scale_10,
                                        is_half_pixel,
                                        input_height, &input_y, &y0, &y1);
      for (int x = 0; x < output_width; ++x) {
        int32_t input_x, x0, x1;
        ComputeInterpolationValuesInteger(x, width_scale_10,
                                          is_half_pixel,
                                          input_width, &input_x, &x0, &x1);
        for (int c = 0; c < depth; ++c) {
          const int64_t output_20_ll =
              static_cast<int64_t>(
                  input_data[Offset(input_dims, b, y0, x0, c)]) *
              ((1 << 10) - (input_y - (1 << 10) * y0)) *
              ((1 << 10) - (input_x - (1 << 10) * x0));
          const int64_t output_20_lu =
              static_cast<int64_t>(
                  input_data[Offset(input_dims, b, y1, x0, c)]) *
              (input_y - (1 << 10) * y0) *
              ((1 << 10) - (input_x - (1 << 10) * x0));
          const int64_t output_20_rl =
              static_cast<int64_t>(
                  input_data[Offset(input_dims, b, y0, x1, c)]) *
              ((1 << 10) - (input_y - (1 << 10) * y0)) *
              (input_x - (1 << 10) * x0);
          const int64_t output_20_ru =
              static_cast<int64_t>(
                  input_data[Offset(input_dims, b, y1, x1, c)]) *
              (input_y - (1 << 10) * y0) * (input_x - (1 << 10) * x0);
          const int64_t output_20 =
              output_20_ll + output_20_lu + output_20_rl + output_20_ru;
          const T interpolation =
              static_cast<T>(output_20 / (1 << 20));
          output_data[Offset(output_dims, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <core/common/inlined_containers_fwd.h>
#include "core/common/status.h"
#include <core/common/safeint.h>
#include <core/common/narrow.h>
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {

constexpr const char* UpsampleModeNN = "nearest";
constexpr const char* UpsampleModeLinear = "linear";
constexpr const char* UpsampleModeCubic = "cubic";

using GetNearestPixelFunc = int64_t (*)(float, bool);
using GetOriginalCoordinateFunc = float (*)(float, float, float, float, float, float);

enum UpsampleMode {
  NN = 0,      // nearest neighbor
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
  HALF_PIXEL_SYMMETRIC = 6,
};

enum ResizeNearestMode {
  SIMPLE = 0,  // For resize op 10
  ROUND_PREFER_FLOOR = 1,
  ROUND_PREFER_CEIL = 2,
  FLOOR = 3,
  CEIL = 4,
};

enum class AspectRatioPolicy {
  STRETCH,
  NOT_LARGER,
  NOT_SMALLER,
};

// Antialias types
template <typename T>
struct AccumulateType {
  using type = int32_t;
  using Dtype = T;
};

template <>
struct AccumulateType<int32_t> {
  using type = float;
};

template <>
struct AccumulateType<float> {
  using type = float;
};

template <>
struct AccumulateType<MLFloat16> {
  using type = float;
};

template <>
struct AccumulateType<double> {
  using type = double;
};

namespace antialias_constants {
constexpr float kCubicCoeffA = -0.75f;
constexpr float kSupportSize = 2.0f;
constexpr float kBiCubicSupportSize = 4.0f;
}  // namespace antialias_constants

namespace ConstValue {
constexpr int32_t mag_factor = 1 << (22 - 1);
// We use to multiply by 2, let's make a constant which is twice as big
constexpr int32_t mag_factor_x_2 = 1 << 22;
}  // namespace ConstValue

template <class T>
inline constexpr bool is_8bit_v = std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;

template <typename T>
void PrintAntiAliasBuffers(std::ostream& os, gsl::span<int64_t> bounds, gsl::span<int64_t> out_of_bounds,
                           gsl::span<T> weight_coefficients) {
  os << "#### Bounds: ";
  std::copy(bounds.begin(), bounds.end(), std::ostream_iterator<int64_t>(os, " "));
  os << std::endl;

  os << "#### Out of Bounds: ";
  std::copy(out_of_bounds.begin(), out_of_bounds.end(),
            std::ostream_iterator<int64_t>(os, " "));
  os << std::endl;

  os << "#### Scale Buffer: ";
  std::copy(weight_coefficients.begin(), weight_coefficients.end(),
            std::ostream_iterator<T>(os, " "));
  os << std::endl;
}

class UpsampleBase {
 public:
  // Make this available in other EP via provider bridge
  // it works iff output_shape is specified
  void AdjustOutputSizeAsPolicy(TensorShapeVector& output_dims, gsl::span<const int64_t> input_dims,
                                InlinedVector<float>& scales) const;

 protected:
  explicit UpsampleBase(const OpKernelInfo& info)
      : scales_cached_(false), roi_cached_(false), use_extrapolation_(false) {
    const auto& node = info.node();
    auto opset = node.SinceVersion();
    is_resize_ = (opset >= 10);

    std::string mode;
    ORT_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());
    mode_ = StringToUpsampleMode(mode);

    auto input_count = info.GetInputCount();
    if (input_count == 1) {  // opset < 10
      std::vector<float> scales;
      ORT_THROW_IF_ERROR(info.GetAttrs<float>("scales", scales));
      ORT_THROW_IF_ERROR(ScalesValidation(scales, mode_));
      scales_.assign(scales.cbegin(), scales.cend());
      scales_cached_ = true;
    }

    if (opset >= 18) {
      antialias_ = info.GetAttrOrDefault<int64_t>("antialias", 0) == 0 ? false : true;

      if (antialias_) {
        ORT_ENFORCE((UpsampleMode::LINEAR == mode_ || UpsampleMode::CUBIC == mode_),
                    "when anti-aliasing is set, Resize only supports mode `LINEAR` and `CUBIC`.");
      }

      // The attribute is absent in opset < 18, but the default value as if stretch.
      std::string keep_aspect_ratio_policy = info.GetAttrOrDefault<std::string>("keep_aspect_ratio_policy", "stretch");
      keep_aspect_ratio_policy_ = StringToKeepAspectRatioPolicy(keep_aspect_ratio_policy);

      // guard against unit tests that can add an attribute
      auto axes = info.GetAttrsOrDefault<int64_t>("axes");
      axes_.assign(axes.cbegin(), axes.cend());
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

    cubic_coeff_a_ = info.GetAttrOrDefault<float>("cubic_coeff_a", antialias_constants::kCubicCoeffA);
    exclude_outside_ = info.GetAttrOrDefault<int64_t>("exclude_outside", 0) == 0 ? false : true;

    if ((exclude_outside_ == 1 && mode_ != CUBIC) && (antialias_ == false || mode_ != LINEAR)) {
      ORT_THROW(
          "exclude_outside can be set to 1 when (1 mode is CUBIC. "
          "\n(2 mode is CUBIC or LINEAR when anti-aliasing is on"
          ". Current mode is set to " +
          mode + " and anti-aliasing is set to " + std::to_string(antialias_));
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
      auto x_shape = node.InputDefs()[0]->Shape();
      int64_t rank = x_shape ? x_shape->dim_size() : -1;
      if (get_scale && scale->Shape().Size() > 0 && ((opset < 18) || (rank > 0 && opset >= 18))) {
        ORT_THROW_IF_ERROR(ParseScalesData(scale, scales_, rank));
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
  AspectRatioPolicy keep_aspect_ratio_policy_{AspectRatioPolicy::STRETCH};
  GetNearestPixelFunc get_nearest_pixel_;
  float cubic_coeff_a_;
  bool exclude_outside_;
  bool antialias_{false};
  float extrapolation_value_;
  bool use_nearest2x_optimization_ = false;

  InlinedVector<float> scales_;
  InlinedVector<float> roi_;
  TensorShapeVector axes_;

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

  AspectRatioPolicy StringToKeepAspectRatioPolicy(const std::string& policy) {
    const static std::unordered_map<std::string_view, AspectRatioPolicy> policy_map{
        {"stretch", AspectRatioPolicy::STRETCH},
        {"not_larger", AspectRatioPolicy::NOT_LARGER},
        {"not_smaller", AspectRatioPolicy::NOT_SMALLER},
    };

    if (auto it = policy_map.find(policy); it != policy_map.end()) {
      return it->second;
    } else {
      ORT_THROW("keep_aspect_ratio of [" + policy + "] is not supported!");
    }
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
    if (coordinate_transform_mode_name == "half_pixel_symmetric") {
      return HALF_PIXEL_SYMMETRIC;
    }
    ORT_THROW("coordinate_transform_mode:[" + coordinate_transform_mode_name + "] is not supported!");
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
                          ? roi_start * (length_original - 1) +
                                (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                          : 0.5 * (roi_start + roi_end) * (length_original - 1);
          return static_cast<float>(orig);
        };
      case HALF_PIXEL_SYMMETRIC:
        return [](float x_resized, float x_scale, float length_resized, float length_original, float, float) {
          float output_width = x_scale * length_original;
          float adjustment = length_resized / output_width;
          float center = length_original / 2;
          float offset = center * (1 - adjustment);
          auto orig = offset + (x_resized + 0.5) / x_scale - 0.5;
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

  [[nodiscard]] Status ScalesValidation(gsl::span<const float> scales, const UpsampleMode mode) const {
    if (!is_resize_) {
      for (auto& scale : scales) {
        ORT_RETURN_IF_NOT(scale >= 1, "Scale value should be greater than or equal to 1.");
      }
    } else {
      for (auto& scale : scales) {
        ORT_RETURN_IF_NOT(scale > 0, "Scale value should be greater than 0.");
      }
    }

    if (UpsampleMode::LINEAR == mode) {
      ORT_RETURN_IF_NOT(scales.size() == 2 ||
                            (scales.size() == 4 && scales[0] == 1 && scales[1] == 1) ||
                            (scales.size() == 4 && scales[0] == 1 && scales[3] == 1) ||
                            scales.size() == 3 ||
                            (scales.size() == 5 && scales[0] == 1 && scales[1] == 1),
                        "'Linear' mode only supports:\n"
                        "  * 2-D inputs or\n"
                        "  * 3-D inputs ('Bilinear', 'Trilinear') or\n"
                        "  * 4-D inputs with the corresponding outermost 2 scale values being 1"
                        " or the corresponding outermost and innermost scale values being 1 or\n"
                        "  * 5-D inputs with the corresponding outermost 2 scale values being 1"
                        "in the ",
                        is_resize_ ? "Resize operator" : "Upsample operator");
    } else if (UpsampleMode::CUBIC == mode) {
      // we support cubic in NHWC format once anti-alias is enabled
      ORT_RETURN_IF_NOT(scales.size() == 2 || (scales.size() == 4 && scales[0] == 1 && scales[1] == 1) ||
                            (antialias_ && scales.size() == 4 && scales[0] == 1 && scales[3] == 1),
                        "'Cubic' mode only support 2-D inputs ('Bicubic') or 4-D inputs "
                        "with the corresponding outermost 2 scale values being 1 in the ",
                        is_resize_ ? "Resize operator" : "Upsample operator");
    }
    return Status::OK();
  }

  [[nodiscard]] Status
  ParseScalesData(const Tensor* scale, InlinedVector<float>& scales, int64_t rank) const {
    const auto* scale_data = scale->Data<float>();
    int64_t scales_size = scale->Shape().Size();
    ORT_RETURN_IF_NOT(scales_size > 0, "scales size should be greater than 0.");
    if (scales.empty()) {
      scales.resize(onnxruntime::narrow<size_t>(scales_size));
    }

    memcpy(scales.data(), scale_data, SafeInt<size_t>(scales_size) * sizeof(float));

    // since opset 18,
    // we allow scales only specified on axes of interest,
    // in which case the other axes is ignored and use default scale of 1
    // scales_size == axes_.size() should be guaranteed if axes is not empty
    if (rank > 0 && (scales_size != rank || axes_.size())) {
      InlinedVector<float> new_scales(size_t(rank), 1.0f);
      ORT_RETURN_IF_NOT(*std::max_element(axes_.begin(), axes_.end()) < rank && (int64_t(axes_.size()) == scales_size),
                        "all values in axes should be less than rank of the data");

      for (size_t i = 0; i < axes_.size(); i++) {
        new_scales[static_cast<size_t>(axes_[i])] = scales[i];
      }
      scales.swap(new_scales);
    }
    return ScalesValidation(scales, mode_);
  }

  void ParseRoiData(const Tensor* roi, InlinedVector<float>& roi_array) const {
    int64_t roi_size = roi->Shape().Size();
    if (roi_size > 0) {
      roi_array.resize(onnxruntime::narrow<size_t>(roi_size));
      memcpy(roi_array.data(), roi->Data<float>(), SafeInt<size_t>(roi_size) * sizeof(float));
    }
  }

  // output_shape is changeable in opset-18 or above.
  // It should be re-computed if axes is not empty.
  [[nodiscard]] Status ParseSizesData(const Tensor* sizes, TensorShapeVector& output_dims,
                                      gsl::span<const int64_t> input_dims) const {
    auto size_span = sizes->DataAsSpan<int64_t>();
    ORT_RETURN_IF_NOT(input_dims.size() >= size_span.size(),
                      "Resize: input tensor's rank does not match the output tensor's rank.");

    if (axes_.size()) {
      output_dims.assign(input_dims.begin(), input_dims.end());
      ORT_RETURN_IF_NOT(*std::max_element(axes_.begin(), axes_.end()) < int64_t(output_dims.size()),
                        "axes should be less than output_dims.size()");

      for (size_t i = 0; i < axes_.size(); i++) {
        output_dims[static_cast<size_t>(axes_[i])] = size_span[i];
      }
    } else {
      std::copy(size_span.begin(), size_span.end(), output_dims.begin());
    }
    return Status::OK();
  }

  // It's different in Opset 18 and before.
  // we will modify output_shape by sorts of policy even if it's specified
  [[nodiscard]] Status ParseScalesDataAndAdjustOutputSize(TensorShapeVector& output_dims,
                                                          gsl::span<const int64_t> input_dims,
                                                          InlinedVector<float>& scales) const {
    for (size_t i = 0, end = input_dims.size(); i < end; ++i) {
      // Handle corner case to avoid dividing by zero in the next step
      if (input_dims[i] == 0) {
        // Enforce that output_dim is 0, given that we cannot scale 0 by any factor to
        // result in any non-zero value
        ORT_RETURN_IF_NOT(output_dims[i] == 0,
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

    AdjustOutputSizeAsPolicy(output_dims, input_dims, scales);
    return ScalesValidation(scales, mode_);
  }

  void ComputeOutputShape(gsl::span<const float> scales,
                          gsl::span<const int64_t> input_dims,
                          TensorShapeVector& output_dims) const {
    for (std::size_t i = 0; i < input_dims.size(); i++) {
      output_dims[i] = static_cast<int64_t>(scales[i] * input_dims[i]);
    }
  }

  // Roi is redefined in Opset-18, we have a concept of axes.
  // So we need to update it accordingly.
  void ComputeROIWithAxes(InlinedVector<float>& roi_array, size_t rank) const {
    if (axes_.size()) {
      InlinedVector<float> roi_tmp(rank * 2, 0);
      for (size_t i = rank; i < rank * 2; ++i) {
        roi_tmp[i] = 1;
      }
      for (size_t i = 0; i < axes_.size(); i++) {
        auto v_in_axes = static_cast<size_t>(axes_[i]);
        roi_tmp[v_in_axes] = (roi_array[i]);
        roi_tmp[rank + v_in_axes] = (roi_array[axes_.size() + i]);
      }
      roi_array.swap(roi_tmp);
    }
  }

 public:
  static constexpr size_t kLookupTableSize = 1280;

  static const uint8_t* GetLookupTableShared() {
    // initialized once
    static const auto* lookup_table = []() {
      // if we have already initialized the lookup table, just return
      // ideally we could have a global lookup table, but that account for too much space.
      /* Handles values form -640 to 639. */
      static uint8_t table[kLookupTableSize] = {0};

      // taken from https://github.com/python-pillow/Pillow/blob/66add095a50d76c35c7f58643461f2edf78a3f05/src/libImaging/Resample.c#L94
      //  we need to handle negative values
      //  it's equivalent to :x = np.clip(x, 0, 255) where x \in [-640, 639]
      // we will accept a negative x for (&table[640])[x] means table +640 -x
      for (int i = 0; i < static_cast<int>(kLookupTableSize); ++i) {
        table[i] = static_cast<uint8_t>(std::min(std::max(i - 640, 0), 255));
      }
      return table;
    }();
    return lookup_table;
  }
};  // UpsampleBase

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

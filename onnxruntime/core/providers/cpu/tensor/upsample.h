// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cmath>

namespace onnxruntime {

constexpr const char* UpsampleModeNN = "nearest";
constexpr const char* UpsampleModeLinear = "linear";
constexpr const char* UpsampleModeCubic = "cubic";

// In case of cubic mode the grid used to calculate the interpolation value
// is a 4x4 matrix
const size_t CubicModeGridLength = 4;

using GetNearestPixelFunc = std::function<int64_t(float, bool)>;
using GetOriginalCoordinateFunc = std::function<float(float, float, float, float, float, float)>;

enum UpsampleMode {
  NN = 0,      // nearest neighbour
  LINEAR = 1,  // linear interpolation
  CUBIC = 2,   // cubic interpolation
};

class UpsampleBase {
 protected:
  UpsampleBase(OpKernelInfo info) : scales_cached_(false), roi_cached_(false), use_extrapolation_(false) {
    int start;
    int end;
    info.GetKernelDef().SinceVersion(&start, &end);
    is_resize = (start >= 10);

    std::string mode;
    ORT_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());
    mode_ = StringToUpsampleMode(mode);

    auto input_count = info.GetInputCount();
    if (input_count == 1) {
      ORT_ENFORCE(info.GetAttrs<float>("scales", scales_).IsOK());
      ScalesValidation(scales_, mode_);
      scales_cached_ = true;
    }

    extrapolation_value_ = info.GetAttrOrDefault<float>("extrapolation_value", 0.0f);

    // Coordinate transformation mode attr was introduced in version 11, before that asymmetric mode was the only available transformation mode
    std::string coordinate_transform_mode = start > 10
                                                ? info.GetAttrOrDefault<std::string>("coordinate_transformation_mode", "half_pixel")
                                                : "asymmetric";
    get_original_coordinate_ = GetOriginalCoordinateFromResizedCoordinate(coordinate_transform_mode);
    use_extrapolation_ = need_roi_input_ = coordinate_transform_mode == "tf_crop_and_resize" ? true : false;

    std::string nearest_mode = info.GetAttrOrDefault<std::string>("nearest_mode", "round_prefer_floor");
    get_nearest_pixel_ = GetNearestPixelFromOriginal(nearest_mode, start);

    cubic_coeff_a_ = info.GetAttrOrDefault<float>("cubic_coeff_a", -0.75f);
    exclude_outside_ = info.GetAttrOrDefault<int64_t>("exclude_outside", 0) == 0 ? false : true;

    if (exclude_outside_ == 1 && mode_ != CUBIC) {
      ORT_THROW("exclude_outside can be set to 1 only when mode is CUBIC. Current mode is set to " + mode);
    }

    // after version 11 update, this optimization is no longer applicable for all the available modes...
    // TODO : needs more testing to enable this for version 11
    use_nearest2x_optimization = start > 10 ? false : true;

    if (start > 10) {
      roi_input_idx_ = 1;
      scales_input_idx_ = 2;
      sizes_input_idx_ = 3;
    } else if (start <= 10 && input_count > 1) {
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

  GetOriginalCoordinateFunc get_original_coordinate_;
  GetNearestPixelFunc get_nearest_pixel_;
  float cubic_coeff_a_;
  bool exclude_outside_;
  float extrapolation_value_;
  UpsampleMode mode_;
  bool use_nearest2x_optimization = false;

  std::vector<float> scales_;
  std::vector<float> roi_;
  bool scales_cached_;
  bool roi_cached_;
  bool need_roi_input_;
  bool use_extrapolation_;
  bool is_resize = false;

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

  GetOriginalCoordinateFunc GetOriginalCoordinateFromResizedCoordinate(
      const std::string& coordinate_transform_mode) {
    if (coordinate_transform_mode == "asymmetric") {
      return [](float x_resized, float x_scale, float, float, float, float) {
        return x_resized / x_scale;
      };
    } else if (coordinate_transform_mode == "pytorch_half_pixel") {
      return [](float x_resized, float x_scale, float length_resized, float, float, float) {
        return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
      };
    } else if (coordinate_transform_mode == "tf_half_pixel_for_nn") {
      return [](float x_resized, float x_scale, float, float, float, float) {
        return (x_resized + 0.5f) / x_scale;
      };
    } else if (coordinate_transform_mode == "align_corners") {
      return [](float x_resized, float, float length_resized, float length_original, float, float) {
        return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
      };
    } else if (coordinate_transform_mode == "tf_crop_and_resize") {
      return [](float x_resized, float, float length_resized, float length_original, float roi_start, float roi_end) {
        auto orig = length_resized > 1
                        ? roi_start * (length_original - 1) + (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                        : 0.5 * (roi_start + roi_end) * (length_original - 1);
        return static_cast<float>(orig);
      };
    } else {  // "half_pixel"
      return [](float x_resized, float x_scale, float, float, float, float) {
        return ((x_resized + 0.5f) / x_scale) - 0.5f;
      };
    }
  }

  GetNearestPixelFunc GetNearestPixelFromOriginal(
      const std::string& nearest_mode, int opset_version) {
    // versions older than 11 did not have nearest_mode attr. Use the original logic in this case
    // to maintain backward compatibility
    if (opset_version < 11) {
      return [](float x_original, bool isDownSample) {
        if (isDownSample) {
          return static_cast<int64_t>(std::ceil(x_original));
        } else {
          return static_cast<int64_t>(x_original);
        }
      };
    }

    // if opset version >=11 choose the rounding mode based on nearest_mode attr
    if (nearest_mode == "round_prefer_ceil") {
      return [](float x_original, bool) {
        return static_cast<int64_t>(std::round(x_original));
      };
    } else if (nearest_mode == "floor") {
      return [](float x_original, bool) {
        return static_cast<int64_t>(std::floor(x_original));
      };
    } else if (nearest_mode == "ceil") {
      return [](float x_original, bool) {
        return static_cast<int64_t>(std::ceil(x_original));
      };
    } else {  // default is round_prefer_floor
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
    if (!is_resize) {
      for (auto& scale : scales) {
        ORT_ENFORCE(scale >= 1, "Scale value should be greater than or equal to 1.");
      }
    } else {
      for (auto& scale : scales) {
        ORT_ENFORCE(scale > 0, "Scale value should be greater than 0.");
      }
    }

    if (UpsampleMode::LINEAR == mode || UpsampleMode::CUBIC == mode) {
      ORT_ENFORCE(scales.size() == 2 || (scales.size() == 4 && scales[0] == 1 && scales[1] == 1),
                  "'Linear' mode and 'Cubic' mode only support 2-D inputs ('Bilinear', 'Bicubic') or 4-D inputs "
                  "with the corresponding outermost 2 scale values being 1 in the ",
                  is_resize ? "Resize operator" : "Upsample operator");
    }
  }

  void ParseScalesData(const Tensor* scale, std::vector<float>& scales) const {
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

  void ParseScalesDataFromOutputSize(const std::vector<int64_t>& output_dims,
                                     const std::vector<int64_t>& intput_dims,
                                     std::vector<float>& scales) const {
    for (size_t i = 0, end = intput_dims.size(); i < end; ++i) {
      scales[i] = static_cast<float>(output_dims[i]) / static_cast<float>(intput_dims[i]);
    }
    ScalesValidation(scales, mode_);
  }

  void ComputeOutputShape(const std::vector<float>& scales,
                          const std::vector<int64_t>& input_dims,
                          std::vector<int64_t>& output_dims) const {
    for (std::size_t i = 0; i < input_dims.size(); i++) {
      output_dims[i] = static_cast<int64_t>(scales[i] * input_dims[i]);
    }
  }
};

template <typename T>
class Upsample : public UpsampleBase, public OpKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  Status BaseCompute(OpKernelContext* context, const std::vector<float>& roi, const std::vector<float>& scales,
                     const std::vector<int64_t>& output_dims) const;
};

}  // namespace onnxruntime

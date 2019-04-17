// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

constexpr const char* UpsampleModeNN = "nearest";
constexpr const char* UpsampleModeLinear = "linear";

enum UpsampleMode {
  NN = 0,      // nearest neighbour
  LINEAR = 1,  // linear interpolation
};

class UpsampleBase {
 protected:
  UpsampleBase(OpKernelInfo info) : scales_cached_(false) {
    std::string mode;
    ORT_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());
    mode_ = StringToUpsampleMode(mode);

    auto input_count = info.GetInputCount();
    if (input_count == 1) {
      ORT_ENFORCE(info.GetAttrs<float>("scales", scales_).IsOK());
      ScalesValidation(scales_, mode_);
    }

    // opset 9
    if (input_count > 1) {
      const Tensor* scale;
      bool get_scale = info.TryGetConstantInput(1, &scale);

      if (get_scale) {
        ParseScalesData(scale, scales_);
        scales_cached_ = true;
      }
    }
  }

  UpsampleMode mode_;
  std::vector<float> scales_;
  bool scales_cached_;
  bool is_resize = false;

  UpsampleMode StringToUpsampleMode(const std::string& mode) {
    if (strcmp(mode.c_str(), UpsampleModeNN) == 0) {
      return UpsampleMode::NN;
    } else if (strcmp(mode.c_str(), UpsampleModeLinear) == 0) {
      return UpsampleMode::LINEAR;
    } else {
      ORT_THROW("mode attribute is " + mode + ". It can only be " +
                UpsampleModeNN + "(default) or " + UpsampleModeLinear + ".");
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

    if (UpsampleMode::LINEAR == mode) {
      ORT_ENFORCE(scales.size() == 4, "Upsample: linear mode upsample only support bilinear with 4 dimension.");
      ORT_ENFORCE(((scales[0] == 1) && (scales[1] == 1)),
                  "Upsample: linear mode upsample only support bilinear, the first 2 scales should be 1.");
    }
  }

  void ParseScalesData(const Tensor* scale, std::vector<float>& scales) const {
    const float* scale_data = scale->template Data<float>();
    int64_t scales_size = scale->Shape().Size();
    ORT_ENFORCE(scales_size > 0, "scales size should be greater than 0.");
    if (scales.size() == 0) {
      scales.resize(scales_size);
    }
    memcpy(scales.data(), scale_data, scales_size * sizeof(float));
    ScalesValidation(scales, mode_);
  }
};

template <typename T>
class Upsample : public UpsampleBase, public OpKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  Status BaseCompute(OpKernelContext* context, const std::vector<float>& scales) const;
};

}  // namespace onnxruntime

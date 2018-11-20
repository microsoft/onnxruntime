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
  UpsampleBase(OpKernelInfo info) {
    std::string mode;
    ONNXRUNTIME_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());

    mode_ = StringToUpsampleMode(mode);

    ONNXRUNTIME_ENFORCE(info.GetAttrs<float>("scales", scales_).IsOK());
    for (auto& scale : scales_) {
      ONNXRUNTIME_ENFORCE(scale >= 1, "Scale value should be greater than or equal to 1.");
    }

    if (UpsampleMode::LINEAR == mode_) {
      ONNXRUNTIME_ENFORCE(((scales_[0] == 1) && (scales_[1] == 1)),
          "Upsample: linear mode upsample only support bilinear, the first 2 scales should be 1.");
    }
  }

  UpsampleMode mode_;

  std::vector<float> scales_;

  UpsampleMode StringToUpsampleMode(const std::string& mode) {
    if (strcmp(mode.c_str(), UpsampleModeNN) == 0) {
      return UpsampleMode::NN;
    } else if (strcmp(mode.c_str(), UpsampleModeLinear) == 0) {
      return UpsampleMode::LINEAR;
    } else {
      ONNXRUNTIME_THROW("mode attribute is " + mode + ". It can only be " +
                        UpsampleModeNN + "(default) or " + UpsampleModeLinear + ".");
    }
  }
};

template <typename T>
class Upsample : public UpsampleBase, public OpKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime

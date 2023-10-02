// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/upsamplebase.h"

namespace onnxruntime::cuda {

template <typename T>
class ResizeGrad final : public UpsampleBase, public CudaKernel {
 public:
  ResizeGrad(const OpKernelInfo& info) : UpsampleBase(info), CudaKernel(info) {
    ORT_ENFORCE(!antialias_, "Antialiasing is not supported in ResizeGrad yet.");

    ORT_ENFORCE(axes_.empty(), "ReizeGrad does not support the `axes` attribute yet.");

    std::string coordinate_transform_mode =
        info.GetAttrOrDefault<std::string>("coordinate_transformation_mode", "half_pixel");
    coordinate_transform_mode_ = StringToCoordinateTransformationMode(coordinate_transform_mode);
    ORT_ENFORCE(coordinate_transform_mode_ == ResizeCoordinateTransformationMode::HALF_PIXEL ||
                    coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ALIGN_CORNERS,
                "ReizeGrad only supports the `HALF_PIXEL` and `ALIGN_CORNERS` coordinate_transform_mode ",
                coordinate_transform_mode, " is not supported yet.");

    ORT_ENFORCE(keep_aspect_ratio_policy_ == AspectRatioPolicy::STRETCH,
                "ReizeGrad only supports the `STRETCH` policy.");

    std::string mode;
    ORT_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());
    ORT_ENFORCE((UpsampleMode::LINEAR == mode_),
                "ReizeGrad only supports the `LINEAR` mode. ", mode, " mode is not supported yet.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace onnxruntime::cuda

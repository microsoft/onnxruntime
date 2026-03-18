// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cctype>
#include <string>

#include "core/common/common.h"
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

#ifdef SHARED_PROVIDER
Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr);
#else
inline Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr) {
  constexpr int64_t EXPECTED_NUM_ROI_DIMS = 2;
  constexpr int64_t EXPECTED_SECOND_ROI_DIM = 4;
  if (!X_ptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Null input X ptr");
  }
  if (!rois_ptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Null rois_ptr");
  }
  if (!batch_indices_ptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Null batch_indices_ptr");
  }

  const auto& rois_dims = rois_ptr->Shape();
  const auto& batch_indices_dims = batch_indices_ptr->Shape();

  if (batch_indices_dims.NumDimensions() != 1) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Number of dimensions for batch indices should be exactly 1");
  }

  if (rois_dims.NumDimensions() != EXPECTED_NUM_ROI_DIMS) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Number of dimensions for rois should be exactly " + std::to_string(EXPECTED_NUM_ROI_DIMS));
  }
  if (rois_dims[1] != EXPECTED_SECOND_ROI_DIM) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Second dimension for rois should be exactly " + std::to_string(EXPECTED_SECOND_ROI_DIM));
  }

  if (batch_indices_dims[0] != rois_dims[0]) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "First dimension (num_rois) of batch_indices and rois don't match");
  }

  if (batch_indices_ptr->Location().device.Type() == OrtDevice::CPU) {
    const int64_t batch_size = X_ptr->Shape()[0];
    const int64_t num_rois = batch_indices_dims[0];

    auto check_bounds = [batch_size, num_rois](const auto* batch_indices_data) -> Status {
      for (int64_t i = 0; i < num_rois; ++i) {
        if (batch_indices_data[i] < 0 || batch_indices_data[i] >= batch_size) {
          return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                        "batch_indices value " + std::to_string(batch_indices_data[i]) +
                            " at index " + std::to_string(i) +
                            " is out of range [0, " + std::to_string(batch_size) + ")");
        }
      }
      return Status::OK();
    };

    if (batch_indices_ptr->IsDataType<int64_t>()) {
      auto status = check_bounds(batch_indices_ptr->Data<int64_t>());
      if (!status.IsOK()) return status;
    } else if (batch_indices_ptr->IsDataType<int32_t>()) {
      auto status = check_bounds(batch_indices_ptr->Data<int32_t>());
      if (!status.IsOK()) return status;
    } else {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "batch_indices must be of type int64_t or int32_t");
    }
  }

  return Status::OK();
}
#endif

enum struct RoiAlignMode {
  avg = 0,
  max
};

class RoiAlignBase {
 public:
  template <typename TKernelInfo>
  explicit RoiAlignBase(const TKernelInfo& info) {
    std::string mode;
    if (info.template GetAttr<std::string>("mode", &mode).IsOK()) {
      std::transform(mode.begin(), mode.end(), mode.begin(), [](char i) { return static_cast<char>(::tolower(i)); });
      if (mode == "avg") {
        mode_ = RoiAlignMode::avg;
      } else if (mode == "max") {
        mode_ = RoiAlignMode::max;
      } else {
        ORT_THROW("Invalid mode of value ", mode, " specified. It should be either avg or max");
      }
    }

    int64_t output_height_tmp;
    if (info.template GetAttr<int64_t>("output_height", &output_height_tmp).IsOK()) {
      output_height_ = output_height_tmp;
    }

    int64_t output_width_tmp;
    if (info.template GetAttr<int64_t>("output_width", &output_width_tmp).IsOK()) {
      output_width_ = output_width_tmp;
    }

    int64_t sampling_ratio_tmp;
    if (info.template GetAttr<int64_t>("sampling_ratio", &sampling_ratio_tmp).IsOK()) {
      sampling_ratio_ = sampling_ratio_tmp;
      ORT_ENFORCE(sampling_ratio_ >= 0, "Sampling ratio should be >=0, but it was ", sampling_ratio_);
    }

    float spatial_scale_tmp;
    if (info.template GetAttr<float>("spatial_scale", &spatial_scale_tmp).IsOK()) {
      spatial_scale_ = spatial_scale_tmp;
    }

    std::string coordinate_transformation_mode;
    if (info.template GetAttr<std::string>("coordinate_transformation_mode", &coordinate_transformation_mode).IsOK()) {
      half_pixel_ = coordinate_transformation_mode == "half_pixel";
    }

    if (mode_ == RoiAlignMode::max && sampling_ratio_ != 1) {
      LOGS_DEFAULT(WARNING) << "The existing summation for max mode and sampling ratios besides 1 is incorrect "
                            << "and will be fixed in the next ORT 1.13 release. Thus the results of RoiAlign "
                            << "will be different.";
    }
  }

 protected:
  RoiAlignMode mode_{RoiAlignMode::avg};
  int64_t output_height_{1};
  int64_t output_width_{1};
  int64_t sampling_ratio_{0};
  float spatial_scale_{1.0f};
  bool half_pixel_{false};

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlignBase);
};

template <typename T>
class RoiAlign final : public OpKernel, public RoiAlignBase {
 public:
  explicit RoiAlign(const OpKernelInfo& info) : OpKernel(info), RoiAlignBase(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlign);
};
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/cpu/nn/pool.h"
#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {
namespace contrib {

class ReorderInput final : public OpKernel {
 public:
  ReorderInput(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("channels_last", &channels_last_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t channels_last_;
};

class ReorderOutput final : public OpKernel {
 public:
  ReorderOutput(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("channels", &channels_).IsOK());
    ORT_ENFORCE(channels_ > 0, "invalid channel count");
    ORT_ENFORCE(info.GetAttr<int64_t>("channels_last", &channels_last_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t channels_;
  int64_t channels_last_;
};

class NchwcConv final : public OpKernel {
 public:
  NchwcConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ConvAttributes conv_attrs_;

  MLAS_ACTIVATION activation_;
};

class NchwcPoolBase : public PoolBase {
 public:
  NchwcPoolBase(const OpKernelInfo& info) : PoolBase(info) {
    if (!pool_attrs_.global_pooling) {
      ORT_ENFORCE(pool_attrs_.kernel_shape.size() == 2, "kernel_shape num_dims is not compatible with X num_dims.");
    }
  }

  Status NchwcPool(OpKernelContext* context, MLAS_POOLING_KIND kind) const;
};

class NchwcMaxPool final : public OpKernel, public NchwcPoolBase {
 public:
  NchwcMaxPool(const OpKernelInfo& info) : OpKernel(info), NchwcPoolBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class NchwcAveragePool final : public OpKernel, public NchwcPoolBase {
 public:
  NchwcAveragePool(const OpKernelInfo& info) : OpKernel(info), NchwcPoolBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class NchwcUpsample final : public OpKernel {
 private:
  enum class TransformationMode {
    ASYMMETRIC,
    ALIGN_CORNERS,
    HALF_PIXEL,
  };

 public:
  NchwcUpsample(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttrs<int64_t>("scales", scales_).IsOK());
    ORT_ENFORCE(scales_.size() == 4);
    // Batch and channel dimensions cannot scale and spatial scaling must be positive.
    ORT_ENFORCE(scales_[0] == 1 && scales_[1] == 1 && scales_[2] >= 1 && scales_[3] >= 1);

    std::string transformation_mode;
    ORT_ENFORCE(info.GetAttr<std::string>("coordinate_transformation_mode", &transformation_mode).IsOK());
    if (transformation_mode == "asymmetric") {
      transformation_mode_ = TransformationMode::ASYMMETRIC;
    } else if (transformation_mode == "align_corners") {
      transformation_mode_ = TransformationMode::ALIGN_CORNERS;
    } else if (transformation_mode == "half_pixel") {
      transformation_mode_ = TransformationMode::HALF_PIXEL;
    } else {
      ORT_THROW("Unsupported transformation mode '" + transformation_mode + "' for NCHWc Upsample");
    }

    std::string mode;
    ORT_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());
    if (mode == "nearest") {
      nearest_mode_ = true;
      ORT_ENFORCE(transformation_mode_ == TransformationMode::ASYMMETRIC);
    } else if (mode == "linear") {
      nearest_mode_ = false;
    } else {
      ORT_THROW("Unsupported mode '" + mode + "' for NCHWc Upsample");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<float> ComputeInterpolation(int64_t input_length,
                                          int64_t output_length,
                                          int64_t scale) const;

  std::vector<int64_t> scales_;
  TransformationMode transformation_mode_;
  bool nearest_mode_;
};

}  // namespace contrib
}  // namespace onnxruntime

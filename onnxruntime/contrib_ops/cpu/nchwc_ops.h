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

class ReorderInput : public OpKernel {
 public:
  ReorderInput(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("channels_last", &channels_last_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t channels_last_;
};

class ReorderOutput : public OpKernel {
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

class NchwcConv : public OpKernel {
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

class NchwcMaxPool : public OpKernel, public NchwcPoolBase {
 public:
  NchwcMaxPool(const OpKernelInfo& info) : OpKernel(info), NchwcPoolBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class NchwcAveragePool : public OpKernel, public NchwcPoolBase {
 public:
  NchwcAveragePool(const OpKernelInfo& info) : OpKernel(info), NchwcPoolBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class NchwcUpsample : public OpKernel {
 public:
  NchwcUpsample(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttrs<int64_t>("scales", scales_).IsOK());
    ORT_ENFORCE(scales_.size() == 4);
    // Batch and channel dimensions cannot scale and spatial scaling must be positive.
    ORT_ENFORCE(scales_[0] == 1 && scales_[1] == 1 && scales_[2] >= 1 && scales_[3] >= 1);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> scales_;
};

}  // namespace contrib
}  // namespace onnxruntime

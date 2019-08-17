// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_base.h"
#include "core/providers/cpu/nn/pool.h"
#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class ReorderInput : public OpKernel {
 public:
  ReorderInput(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReorderOutput : public OpKernel {
 public:
  ReorderOutput(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("channels", &channels_).IsOK());
    ORT_ENFORCE(channels_ > 0, "invalid channel count");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t channels_;
};

class NchwcConv : public OpKernel, public ConvBase {
 public:
  NchwcConv(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  MLAS_ACTIVATION activation_;
};

class NchwcPoolBase : public PoolBase {
 public:
  NchwcPoolBase(const OpKernelInfo& info) : PoolBase(info) {
    if (!global_pooling_)
      ORT_ENFORCE(kernel_shape_.size() == 2, "kernel_shape num_dims is not compatible with X num_dims.");
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

}  // namespace contrib
}  // namespace onnxruntime

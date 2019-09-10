// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/conv_base.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

template <typename T>
class Conv : public OpKernel, public ConvBase {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <>
class Conv<float> : public OpKernel, public ConvBase {
 public:
  Conv<float>(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
    activation_.ActivationKind = MlasIdentityActivation;
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  MLAS_ACTIVATION activation_;
};

}  // namespace onnxruntime

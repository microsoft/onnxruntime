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

class ConvFloat : public OpKernel, public ConvBase {
 public:
  ConvFloat(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  MLAS_ACTIVATION activation_{MlasIdentityActivation};
};

}  // namespace onnxruntime

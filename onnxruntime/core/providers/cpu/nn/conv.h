// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

template <typename T>
class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ConvAttributes conv_attrs_;
};

template <>
class Conv<float> : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    activation_.ActivationKind = MlasIdentityActivation;
    SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  MLAS_ACTIVATION activation_;

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;

  ConvAttributes conv_attrs_;
};

}  // namespace onnxruntime

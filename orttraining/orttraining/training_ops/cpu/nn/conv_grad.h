// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class ConvGrad final : public OpKernel {
 public:
  explicit ConvGrad(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    mlas_backend_kernel_selector_config_.use_kleidiai =
        info.GetConfigOptions().GetConfigEntry(kOrtSessionOptionsMlasDisableKleidiai) != "1";
    mlas_backend_kernel_selector_config_.use_kleidiai_sme =
        info.GetConfigOptions().GetConfigEntry(kOrtSessionOptionsMlasDisableKleidiaiSME) != "1";
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  ConvAttributes conv_attrs_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvGrad);
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};

}  // namespace contrib
}  // namespace onnxruntime

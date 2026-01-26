// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/rnn/gru_io_utils.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime::contrib {

template <typename T>
class GRUGrad final : public OpKernel {
 public:
  GRUGrad(const OpKernelInfo& info) : OpKernel(info), attributes_(info) {
    mlas_backend_kernel_selector_config_.use_kleidiai =
        info.GetConfigOptions().GetConfigEntry(kOrtSessionOptionsMlasDisableKleidiai) != "1";
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  const gru::GRUAttributes attributes_;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};

}  // namespace onnxruntime::contrib

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/rnn/lstm_io_utils.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"

namespace onnxruntime::contrib {

template <typename T>
class LSTMGrad final : public OpKernel {
 public:
  LSTMGrad(const OpKernelInfo& info) : OpKernel(info), attributes_(info) {
    SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  const lstm::LSTMAttributes attributes_;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};

}  // namespace onnxruntime::contrib

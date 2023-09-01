// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/rnn/lstm_io_utils.h"

namespace onnxruntime::contrib {

template <typename T>
class LSTMTraining final : public OpKernel {
 public:
  LSTMTraining(const OpKernelInfo& info) : OpKernel(info), attributes_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  lstm::LSTMAttributes attributes_;
};

}  // namespace onnxruntime::contrib

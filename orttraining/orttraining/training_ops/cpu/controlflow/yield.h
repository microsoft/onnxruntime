// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class YieldOp final : public OpKernel {
 public:
  YieldOp(const OpKernelInfo& info) : OpKernel(info) {
    size_t num_inputs = static_cast<size_t>(info.GetInputCount());
    size_t num_outputs = static_cast<size_t>(info.GetOutputCount());

    std::vector<int64_t> non_differentiable_outputs = info.GetAttrsOrDefault<int64_t>("non_differentiable_outputs");
    ORT_ENFORCE(num_inputs == num_outputs + non_differentiable_outputs.size());
    non_differentiable_outputs_.resize(num_inputs, false);
    for (int64_t idx : non_differentiable_outputs) {
      ORT_ENFORCE(static_cast<size_t>(idx) < num_inputs);
      non_differentiable_outputs_[idx] = true;
    }

    std::vector<int64_t> full_shape_outputs;
    ORT_ENFORCE(info.GetAttrs<int64_t>("full_shape_outputs", full_shape_outputs).IsOK());
    full_shape_outputs_.resize(num_inputs, false);
    for (int64_t idx : full_shape_outputs) {
      ORT_ENFORCE(static_cast<size_t>(idx) < num_inputs);
      full_shape_outputs_[idx] = true;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<bool> non_differentiable_outputs_{};
  std::vector<bool> full_shape_outputs_{};
};

}  // namespace contrib
}  // namespace onnxruntime

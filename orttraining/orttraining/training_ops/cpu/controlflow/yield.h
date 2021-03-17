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
    std::vector<int64_t> non_differentiable_outputs = info.GetAttrsOrDefault<int64_t>("non_differentiable_outputs");
    non_differentiable_outputs_.insert(non_differentiable_outputs.begin(), non_differentiable_outputs.end());

    std::vector<int64_t> full_shape_outputs;
    ORT_ENFORCE(info.GetAttrs<int64_t>("full_shape_outputs", full_shape_outputs).IsOK());
    full_shape_outputs_.insert(full_shape_outputs.begin(), full_shape_outputs.end());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::unordered_set<int64_t> non_differentiable_outputs_{};
  std::unordered_set<int64_t> full_shape_outputs_{};
};

}  // namespace contrib
}  // namespace onnxruntime

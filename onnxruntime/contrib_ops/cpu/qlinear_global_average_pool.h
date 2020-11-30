// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class QLinearGlobalAveragePool final : public OpKernel {
 public:
  QLinearGlobalAveragePool(const OpKernelInfo& info) : OpKernel(info) {
    channels_last_ = (info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(0)) != 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool channels_last_;
};

}  // namespace contrib
}  // namespace onnxruntime

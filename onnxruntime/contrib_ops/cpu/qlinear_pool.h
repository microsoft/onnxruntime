// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace contrib {

class QLinearAveragePool final : public OpKernel, public PoolBase {
 public:
  QLinearAveragePool(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
    channels_last_ = (info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(0)) != 0);
  }

  ~QLinearAveragePool() override = default;

  Status Compute(OpKernelContext* context) const override;

 private:
  PoolProcessContext pool_context_;
  bool channels_last_;
};

}  // namespace contrib
}  // namespace onnxruntime

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
  QLinearAveragePool(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) { }

  ~QLinearAveragePool() override = default;

  Status Compute(OpKernelContext* context) const override;

private:
  PoolProcessContext pool_context_;

};

}  // namespace contrib
}  // namespace onnxruntime

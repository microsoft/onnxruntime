// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace cann {

template <typename T>
class MaxPool : public CannKernel, public PoolBase {
 public:
  explicit MaxPool(const OpKernelInfo& info) : CannKernel(info), PoolBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cann
}  // namespace onnxruntime

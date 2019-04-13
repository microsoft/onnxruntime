// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool.h"

namespace onnxruntime {
namespace contrib {

template <typename T, typename PoolType>
class NchwcPool : public Pool<T, PoolType> {
 public:
  NchwcPool(const OpKernelInfo& info) : Pool<T, PoolType>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

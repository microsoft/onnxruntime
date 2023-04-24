// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/pool_base.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <typename T, typename PoolType>
class Pool : public RocmKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : RocmKernel(info), PoolBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Pool<T, MaxPool<8>> final : public Pool<T, MaxPool<1>> {
 public:
  Pool(const OpKernelInfo& info) : Pool<T, MaxPool<1>>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename PoolType>
class GlobalPool final : public Pool<T, PoolType> {
 public:
  GlobalPool(const OpKernelInfo& info) : Pool<T, PoolType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime

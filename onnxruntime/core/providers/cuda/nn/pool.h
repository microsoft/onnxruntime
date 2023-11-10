// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename PoolType>
class Pool : public CudaKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : CudaKernel(info), PoolBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Pool<T, MaxPool<8>> final : public Pool<T, MaxPool<1>> {
 public:
  Pool(const OpKernelInfo& info) : Pool<T, MaxPool<1>>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime

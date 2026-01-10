// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename PoolType, bool NHWC>
class Pool : public CudaKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : CudaKernel(info), PoolBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, bool Layout>
class Pool<T, MaxPool<8>, Layout> final : public Pool<T, MaxPool<1>, Layout> {
 public:
  explicit Pool(const OpKernelInfo& info) : Pool<T, MaxPool<1>, Layout>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime

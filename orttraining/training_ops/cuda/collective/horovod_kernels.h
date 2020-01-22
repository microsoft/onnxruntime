// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

class HorovodAllReduce final : public CudaKernel {
 public:
  HorovodAllReduce(const OpKernelInfo& info) : CudaKernel(info) {
    unique_name = "AllReduceNode_" + info.node().Name();
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string unique_name;
};

class HorovodBarrier final : public CudaKernel {
 public:
  HorovodBarrier(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime

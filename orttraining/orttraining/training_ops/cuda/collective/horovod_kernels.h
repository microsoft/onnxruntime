// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "orttraining/core/graph/horovod_adapters.h"

namespace onnxruntime {
namespace cuda {

class HorovodAllReduce final : public CudaKernel {
 public:
  HorovodAllReduce(const OpKernelInfo& info) : CudaKernel(info) {
    unique_name = "AllReduceNode_" + info.node().Name();
    int64_t reduce_op;
    info.GetAttrOrDefault("reduce_op", &reduce_op, static_cast<int64_t>(hvd::ReduceOp::SUM));
    reduce_op_ = GetReduceOp(reduce_op);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string unique_name;
  hvd::ReduceOp reduce_op_;
};

class HorovodBarrier final : public CudaKernel {
 public:
  HorovodBarrier(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime

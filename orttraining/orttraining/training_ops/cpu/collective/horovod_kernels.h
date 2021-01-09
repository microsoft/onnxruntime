// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include "orttraining/core/graph/horovod_adapters.h"

namespace onnxruntime {
namespace contrib {

class HorovodAllReduce final : public OpKernel {
public:
 HorovodAllReduce(const OpKernelInfo& info) : OpKernel(info) {
   unique_name = "AllReduceNode_" + info.node().Name();
   int64_t reduce_op;
   info.GetAttrOrDefault("reduce_op", &reduce_op, static_cast<int64_t>(1));
   reduce_op_ = GetReduceOp(reduce_op);
}

 Status Compute(OpKernelContext* context) const override;
private:
  std::string unique_name;
  hvd::ReduceOp reduce_op_;
};

class HorovodBarrier final : public OpKernel {
 public:
  HorovodBarrier(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

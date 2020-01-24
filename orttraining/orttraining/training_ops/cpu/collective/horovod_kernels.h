// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class HorovodAllReduce final : public OpKernel {
public:
 HorovodAllReduce(const OpKernelInfo& info) : OpKernel(info) {
   unique_name = "AllReduceNode_" + info.node().Name();
}

 Status Compute(OpKernelContext* context) const override;
private:
  std::string unique_name;
};

class HorovodBarrier final : public OpKernel {
 public:
  HorovodBarrier(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

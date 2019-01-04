// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T, typename PoolType>
class Pool final : public onnxruntime::Pool<T, PoolType> {
 public:
  explicit Pool(const OpKernelInfo& info) : onnxruntime::Pool<T, PoolType>(info) {
    // Since there are multiple versions of Pooling kernels, we need to use
    // the opset version as part of the key for caching Pooling Primitives.
    int start, end;
    OpKernel::KernelDef().SinceVersion(&start, &end);
    opset_version_ = std::to_string(start);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string opset_version_;
};

}  // namespace mkl_dnn
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sharding_spec.h"
#include "sharding.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <algorithm>
#include <tuple>
#include <optional>
#include <string>
#include <nccl.h>
#include <sstream>

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

template <typename T>
class DistributedReduceBase : public DistributedKernel {
 public:
  explicit DistributedReduceBase(const OpKernelInfo& info, cudnnReduceTensorOp_t cudnn_reduce_op);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // ONNX attribute. If true, reduced axes are retained as dimensions with size one.
  // Otherwise, drop reduced axes.
  bool keepdims_;
  cudnnReduceTensorOp_t cudnn_reduce_op_;
};

template <typename T>
class DistributedReduceSum final : public DistributedReduceBase {
 public:
  explicit DistributedReduceSum(const OpKernelInfo& info);
};

template <typename T>
class DistributedReduceMean final : public DistributedReduceBase {
 public:
  explicit DistributedReduceMean(const OpKernelInfo& info);
};

template <typename T>
class DistributedReduceMax final : public DistributedReduceBase {
 public:
  explicit DistributedReduceMax(const OpKernelInfo& info);
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

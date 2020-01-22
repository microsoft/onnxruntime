// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <random>

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class TrainableDropout final : public OpKernel {
 public:
  TrainableDropout(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;

  const int64_t random_seed_;

  // random number generator state
  // inefficient but simple approach to thread-safety
  mutable std::mutex rng_mutex_;
  mutable std::default_random_engine rng_;
};

class TrainableDropoutGrad final : public OpKernel {
 public:
  TrainableDropoutGrad(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;
};
}  // namespace contrib
}  // namespace onnxruntime

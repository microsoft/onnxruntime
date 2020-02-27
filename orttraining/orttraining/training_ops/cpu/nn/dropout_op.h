// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <random>

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T1, typename T2>
class Dropout final : public OpKernel {
 public:
  Dropout(const OpKernelInfo& info) : OpKernel{info},
                                      random_seed_{info.GetAttrOrDefault<int64_t>(
                                          "seed",
                                          std::chrono::steady_clock::now().time_since_epoch().count())},
                                      rng_{static_cast<typename decltype(rng_)::result_type>(random_seed_)} {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  const int64_t random_seed_;

  // random number generator state
  // inefficient but simple approach to thread-safety
  mutable std::mutex rng_mutex_;
  mutable std::default_random_engine rng_;
};

template <typename T1, typename T2>
class DropoutGrad final : public OpKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace contrib
}  // namespace onnxruntime

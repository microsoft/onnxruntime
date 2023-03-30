// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/cann/cann_kernel.h"
#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace cann {

template <typename T1, typename T2>
class Dropout final : public CannKernel {
 public:
  Dropout(const OpKernelInfo& info) : CannKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<RandomGenerator>(seed);
    }
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  mutable std::unique_ptr<RandomGenerator> generator_;
};

}  // namespace cann
}  // namespace onnxruntime

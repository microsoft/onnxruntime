// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace contrib {

template <typename T1, typename T2, bool trainable_dropout>
class Dropout final: public OpKernel {
 public:
  Dropout(const OpKernelInfo& info) : OpKernel{info} {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = onnxruntime::make_unique<RandomGenerator>(seed);
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<RandomGenerator> generator_;
};

template <typename T1, typename T2>
class DropoutGrad final : public OpKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : OpKernel{info} {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

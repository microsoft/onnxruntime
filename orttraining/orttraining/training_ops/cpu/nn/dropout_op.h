// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace contrib {

template <typename T1, typename T2>
class Dropout : public OpKernel {
 public:
  Dropout(const OpKernelInfo& info) : OpKernel{info} {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = onnxruntime::make_unique<RandomGenerator>(seed);
    }

    trainable_dropout_ = false;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<RandomGenerator> generator_;

 protected: 
  bool trainable_dropout_;
};

template <typename T1, typename T2>
class DropoutGrad : public OpKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : OpKernel{info} {
    trainable_dropout_ = false;
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  bool trainable_dropout_;
};

template <typename T1, typename T2>
class TrainableDropout final : public Dropout<T1, T2> {
 public:
  TrainableDropout(const OpKernelInfo& info) : Dropout<T1, T2>{info} {
    Dropout<T1, T2>::trainable_dropout_ = true;
  }
};

template <typename T1, typename T2>
class TrainableDropoutGrad final : public DropoutGrad<T1, T2> {
 public:
  TrainableDropoutGrad(const OpKernelInfo& info) : DropoutGrad<T1, T2>{info} {
    DropoutGrad<T1, T2>::trainable_dropout_ = true;
  }

};

}  // namespace contrib
}  // namespace onnxruntime

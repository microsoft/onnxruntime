// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "orttraining/training_ops/cuda/nn/dropout_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2>
class Dropout : public CudaKernel {
 public:
  Dropout(const OpKernelInfo& info) : CudaKernel(info), default_ratio_(0.5) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = onnxruntime::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
      trainable_dropout_ = false;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  const float default_ratio_;

 protected:
  bool trainable_dropout_;
};

template <typename T1, typename T2>
class DropoutGrad : public CudaKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : CudaKernel(info), default_ratio_(0.5) {
    trainable_dropout_ = false;
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  const float default_ratio_;

 protected:
  bool trainable_dropout_;
};

template <typename T1, typename T2>
class TrainableDropout final : public Dropout<T1, T2> {
 public:
  TrainableDropout(const OpKernelInfo& info) : Dropout<T1, T2>{info} {
    trainable_dropout_ = true;
  }
};

template <typename T1, typename T2>
class TrainableDropoutGrad final : public DropoutGrad<T1, T2> {
 public:
  TrainableDropoutGrad(const OpKernelInfo& info) : DropoutGrad<T1, T2>{info} {
    trainable_dropout_ = true;
  }
};

}  // namespace cuda
}  // namespace onnxruntime

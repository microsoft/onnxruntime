// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/batch_norm.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class BatchNorm final : public onnxruntime::BatchNorm<T> {
 public:
   explicit BatchNorm(const OpKernelInfo& info) : onnxruntime::BatchNorm<T>(info) {}
  Status Compute(OpKernelContext* context) const override;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime

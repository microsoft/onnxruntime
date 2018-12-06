// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/activation/activations.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class Relu : public onnxruntime::Relu<T> {
 public:
  Relu(const OpKernelInfo& info) : onnxruntime::Relu<T>(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace mkl_dnn
}  // namespace onnxruntime
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T, bool use_approximation>
class BiasGelu : public OpKernel {
 public:
  BiasGelu(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;

 protected:
  void AddBiasGelu(const T* input, const T* bias, T* temp, T* output, int64_t count) const;
};

}  // namespace contrib
}  // namespace onnxruntime

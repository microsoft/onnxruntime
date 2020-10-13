// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T, bool simplified>
class LayerNormGrad final : public OpKernel {
 public:
  LayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* op_kernel_context) const override;

 private:
  int64_t axis_;
};

template <typename T>
class InvertibleLayerNormGrad final : public OpKernel {
 public:
  InvertibleLayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* op_kernel_context) const override;

 private:
  int64_t axis_;
};

}  // namespace contrib
}  // namespace onnxruntime

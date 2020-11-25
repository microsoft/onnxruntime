// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T, typename GeluComputationMode>
class GeluGrad final : public OpKernel {
 public:
  GeluGrad(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T, typename GeluComputationMode>
class BiasGeluGrad_dX final : public OpKernel {
 public:
  BiasGeluGrad_dX(const OpKernelInfo& info) : OpKernel{info} {}
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

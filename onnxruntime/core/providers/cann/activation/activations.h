// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"

namespace onnxruntime {
namespace cann {

class Activations : public CannKernel {
 protected:
  explicit Activations(const OpKernelInfo& info) : CannKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  template <typename T>
  Status Prepare(OpKernelContext* ctx, CannPreparation& prepare) const;
};

template <typename T>
class Relu final : public Activations {
 public:
  Relu(const OpKernelInfo& info) : Activations(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace cann
}  // namespace onnxruntime

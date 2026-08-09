// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cann {

class BinaryElementwise : public CannKernel {
 protected:
  explicit BinaryElementwise(const OpKernelInfo& info) : CannKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  template <typename T>
  Status Prepare(OpKernelContext* ctx, CannPreparation& prepare) const;
};

template <typename T>
class Add final : public BinaryElementwise {
 public:
  Add(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Sub final : public BinaryElementwise {
 public:
  Sub(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Mul final : public BinaryElementwise {
 public:
  Mul(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class Div final : public BinaryElementwise {
 public:
  Div(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace cann
}  // namespace onnxruntime

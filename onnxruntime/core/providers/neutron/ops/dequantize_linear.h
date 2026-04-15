// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/neutron/neutron_kernel.h"

namespace onnxruntime {
namespace neutron {

template <typename T>
class DequantizeLinear final : public NeutronKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DequantizeLinear);

  explicit DequantizeLinear(const OpKernelInfo& info) : NeutronKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace neutron
}  // namespace onnxruntime

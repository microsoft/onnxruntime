// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"

namespace onnxruntime {
namespace cann {

template <typename T>
class Flatten final : public CannKernel {
 public:
  Flatten(const OpKernelInfo& info) : CannKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace cann
}  // namespace onnxruntime

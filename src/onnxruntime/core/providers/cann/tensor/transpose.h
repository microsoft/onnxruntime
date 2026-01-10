// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/cann/cann_kernel.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace cann {

template <typename T>
class Transpose final : public CannKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : CannKernel(info), TransposeBase(info) {
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace cann
}  // namespace onnxruntime

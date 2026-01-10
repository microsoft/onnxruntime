// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace webgpu {

class Upsample : public UpsampleBase, public WebGpuKernel {
 public:
  explicit Upsample(const OpKernelInfo& info) : UpsampleBase(info), WebGpuKernel(info) {};

  Status ComputeInternal(ComputeContext& context) const override;
  Status BaseCompute(ComputeContext& context, gsl::span<const float> roi, gsl::span<const float> scales,
                     gsl::span<const int64_t> output_dims) const;
};

}  // namespace webgpu
}  // namespace onnxruntime

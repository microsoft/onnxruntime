// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

class MatMulBlockQuantizedFp8Weight final : public onnxruntime::webgpu::WebGpuKernel {
 public:
  explicit MatMulBlockQuantizedFp8Weight(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  uint32_t block_size_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime

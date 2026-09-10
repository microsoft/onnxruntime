// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <limits>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/webgpu/moe/moe_base.h"
#include "contrib_ops/webgpu/moe/moe.h"
#include "core/providers/webgpu/math/matmul.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class QMoE final : public MoE {
 public:
  QMoE(const OpKernelInfo& info) : MoE(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
    ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
                "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
    block_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("block_size", 0));
    // ``zero_point_offset`` (fractional zero-point center) is a CUDA-only feature. Reject it here so
    // a model authored for CUDA is not silently evaluated with the default integer center on WebGPU,
    // which would violate the documented dequant formula (code - zero_point_offset) * scale.
    const float zero_point_offset =
        info.GetAttrOrDefault<float>("zero_point_offset", std::numeric_limits<float>::quiet_NaN());
    ORT_ENFORCE(std::isnan(zero_point_offset),
                "WebGPU QMoE does not support the 'zero_point_offset' attribute; it is only "
                "implemented by the CUDA execution provider.");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t expert_weight_bits_;
  int64_t block_size_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime

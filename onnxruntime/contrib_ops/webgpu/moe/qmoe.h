// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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
    fc1_expert_weight_bits_ = info.GetAttrOrDefault<int64_t>("fc1_expert_weight_bits", expert_weight_bits_);
    fc2_expert_weight_bits_ = info.GetAttrOrDefault<int64_t>("fc2_expert_weight_bits", expert_weight_bits_);
    fc3_expert_weight_bits_ = info.GetAttrOrDefault<int64_t>("fc3_expert_weight_bits", expert_weight_bits_);
    ORT_ENFORCE((fc1_expert_weight_bits_ == 2 || fc1_expert_weight_bits_ == 4 || fc1_expert_weight_bits_ == 8) &&
                    (fc2_expert_weight_bits_ == 2 || fc2_expert_weight_bits_ == 4 || fc2_expert_weight_bits_ == 8) &&
                    (fc3_expert_weight_bits_ == 2 || fc3_expert_weight_bits_ == 4 || fc3_expert_weight_bits_ == 8),
                "FC-specific expert weight bits must be 2, 4, or 8.");
    ORT_ENFORCE(swiglu_fusion_ == 0 || fc3_expert_weight_bits_ == fc1_expert_weight_bits_,
                "Fused SwiGLU requires FC1 and FC3 expert weight bits to match.");
    block_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("block_size", 0));
    quant_type_ = info.GetAttrOrDefault<std::string>("quant_type", "int");
    is_block_fp8_ = quant_type_ == "fp8" && block_size_ == 128;
    ORT_ENFORCE(quant_type_ == "int" || is_block_fp8_,
                "WebGPU QMoE supports quant_type='int' and block-scaled quant_type='fp8' with block_size=128; "
                "fp4, nvfp4, global-scale fp8, and wfp4afp8 formats are not supported.");
    ORT_ENFORCE(!is_block_fp8_ ||
                    (expert_weight_bits_ == 8 && fc1_expert_weight_bits_ == 8 &&
                     fc2_expert_weight_bits_ == 8 && fc3_expert_weight_bits_ == 8),
                "WebGPU block-scaled FP8 QMoE requires 8-bit expert weights for every projection.");
    const auto weights_prepacked = info.GetAttrOrDefault<int64_t>("weights_prepacked", -1);
    ORT_ENFORCE(weights_prepacked != 1,
                "WebGPU QMoE does not support provider-specific prepacked expert weights. "
                "Use weights_prepacked=0 or omit the attribute.");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t expert_weight_bits_;
  int64_t fc1_expert_weight_bits_;
  int64_t fc2_expert_weight_bits_;
  int64_t fc3_expert_weight_bits_;
  int64_t block_size_;
  std::string quant_type_;
  bool is_block_fp8_ = false;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <array>
#include <mutex>
#include "core/providers/webgpu/webgpu_context.h"


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
    ORT_ENFORCE(expert_weight_bits_ == 2 || expert_weight_bits_ == 3 || expert_weight_bits_ == 4 || expert_weight_bits_ == 8,
                "expert_weight_bits must be 2, 3, 4, or 8, but got ", expert_weight_bits_);
    fc1_expert_weight_bits_ = info.GetAttrOrDefault<int64_t>("fc1_expert_weight_bits", expert_weight_bits_);
    fc2_expert_weight_bits_ = info.GetAttrOrDefault<int64_t>("fc2_expert_weight_bits", expert_weight_bits_);
    fc3_expert_weight_bits_ = info.GetAttrOrDefault<int64_t>("fc3_expert_weight_bits", expert_weight_bits_);
    ORT_ENFORCE((fc1_expert_weight_bits_ == 2 || fc1_expert_weight_bits_ == 3 || fc1_expert_weight_bits_ == 4 || fc1_expert_weight_bits_ == 8) &&
                    (fc2_expert_weight_bits_ == 2 || fc2_expert_weight_bits_ == 3 || fc2_expert_weight_bits_ == 4 || fc2_expert_weight_bits_ == 8) &&
                    (fc3_expert_weight_bits_ == 2 || fc3_expert_weight_bits_ == 3 || fc3_expert_weight_bits_ == 4 || fc3_expert_weight_bits_ == 8),
                "FC-specific expert weight bits must be 2, 3, 4, or 8.");
    ORT_ENFORCE(swiglu_fusion_ == 0 || fc3_expert_weight_bits_ == fc1_expert_weight_bits_,
                "Fused SwiGLU requires FC1 and FC3 expert weight bits to match.");
    block_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("block_size", 0));
    quant_type_ = info.GetAttrOrDefault<std::string>("quant_type", "int");
    is_block_fp8_ = quant_type_ == "fp8" && block_size_ == 128;
    is_nvfp4_ = quant_type_ == "nvfp4";
    if (is_nvfp4_ && block_size_ == -1) {
      block_size_ = 16;
    }
    ORT_ENFORCE(quant_type_ == "int" || is_block_fp8_ || is_nvfp4_,
                "WebGPU QMoE supports integer, block-128 FP8, and block-16 NVFP4 weights.");
    ORT_ENFORCE(!is_nvfp4_ ||
                    (block_size_ == 16 && expert_weight_bits_ == 4 && fc1_expert_weight_bits_ == 4 &&
                     fc2_expert_weight_bits_ == 4 && fc3_expert_weight_bits_ == 4),
                "WebGPU NVFP4 QMoE requires block_size=16 and 4-bit weights for every projection.");
    ORT_ENFORCE(!is_block_fp8_ ||
                    (expert_weight_bits_ == 8 && fc1_expert_weight_bits_ == 8 &&
                     fc2_expert_weight_bits_ == 8 && fc3_expert_weight_bits_ == 8),
                "WebGPU block-scaled FP8 QMoE requires 8-bit expert weights for every projection.");
    const auto weights_prepacked = info.GetAttrOrDefault<int64_t>("weights_prepacked", -1);
    ORT_ENFORCE(!is_nvfp4_ || (weights_prepacked >= -1 && weights_prepacked <= 1),
                "NVFP4 weights_prepacked must be -1, 0, or 1.");
    nvfp4_k_packed_ = is_nvfp4_ && weights_prepacked == 1;
    ORT_ENFORCE(is_nvfp4_ || weights_prepacked != 1,
                "WebGPU QMoE does not support provider-specific prepacked expert weights. "
                "Use weights_prepacked=0 or omit the attribute.");
  }

  ~QMoE() override;
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  bool CanFuseFc1Activation(ComputeContext& context, bool enabled) const;
  bool CanFuseFc2FinalMix(ComputeContext& context, bool enabled) const;
  struct ReplayEntry {
    std::vector<uint64_t> key;
    uint64_t last_used = 0;
    std::vector<onnxruntime::webgpu::CapturedCommandInfo> commands;
  };
  Status ComputeWithWorkspace(ComputeContext& context,
                             std::array<std::optional<Tensor>, 5>* scratch) const;
  mutable std::mutex replay_mutex_;
  mutable std::array<std::optional<Tensor>, 5> replay_scratch_;
  mutable std::vector<uint64_t> replay_workspace_key_;
  mutable std::vector<ReplayEntry> replay_entries_;
  mutable uint64_t replay_requests_ = 0;
  mutable uint64_t replay_hits_ = 0;
  mutable uint64_t replay_capacity_fallbacks_ = 0;
  mutable uint64_t replay_captures_ = 0;
  mutable uint64_t replay_evictions_ = 0;
  mutable uint64_t replay_shape_invalidations_ = 0;
  mutable uint64_t replay_clock_ = 0;

  int64_t expert_weight_bits_;
  int64_t fc1_expert_weight_bits_;
  int64_t fc2_expert_weight_bits_;
  int64_t fc3_expert_weight_bits_;
  int64_t block_size_;
  std::string quant_type_;
  bool is_block_fp8_ = false;
  bool is_nvfp4_ = false;
  bool nvfp4_k_packed_ = false;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime

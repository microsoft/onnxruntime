// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/gated_delta_net_plan.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gated_delta_net {

const char* EngineName(Engine e) {
  switch (e) {
    case Engine::kChunked:
      return "chunked";
    case Engine::kRecurrent:
      return "recurrent";
    case Engine::kCudnn:
      return "cudnn";
    default:
      return "auto";
  }
}

Engine EngineFromName(const std::string& name) {
  if (name == "chunked") return Engine::kChunked;
  if (name == "recurrent") return Engine::kRecurrent;
  if (name == "cudnn") return Engine::kCudnn;
  return Engine::kAuto;
}

bool Descriptor::operator==(const Descriptor& o) const noexcept {
  return total_tokens == o.total_tokens && batch == o.batch && num_heads_q == o.num_heads_q &&
         num_heads_k == o.num_heads_k && num_heads_v == o.num_heads_v &&
         head_size_qk == o.head_size_qk && head_size_v == o.head_size_v &&
         chunk_size == o.chunk_size && state_checkpoints == o.state_checkpoints &&
         update_rule == o.update_rule && gate_activation == o.gate_activation &&
         beta_activation == o.beta_activation && io_type == o.io_type &&
         qk_l2_norm == o.qk_l2_norm && decay_per_key_dim == o.decay_per_key_dim &&
         has_decay == o.has_decay && has_beta == o.has_beta &&
         has_initial_state == o.has_initial_state && ragged == o.ragged &&
         sm_major == o.sm_major && sm_minor == o.sm_minor;
}

size_t DescriptorHash::operator()(const Descriptor& d) const noexcept {
  size_t h = 1469598103934665603ULL;
  auto mix = [&h](uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  mix(static_cast<uint64_t>(d.total_tokens));
  mix(static_cast<uint64_t>(d.batch));
  mix(static_cast<uint64_t>(d.num_heads_q));
  mix(static_cast<uint64_t>(d.num_heads_k));
  mix(static_cast<uint64_t>(d.num_heads_v));
  mix(static_cast<uint64_t>(d.head_size_qk));
  mix(static_cast<uint64_t>(d.head_size_v));
  mix(static_cast<uint64_t>(d.chunk_size));
  mix(static_cast<uint64_t>(d.state_checkpoints));
  mix(static_cast<uint64_t>(d.update_rule));
  mix(static_cast<uint64_t>(d.gate_activation));
  mix(static_cast<uint64_t>(d.beta_activation));
  mix(static_cast<uint64_t>(d.io_type));
  mix(static_cast<uint64_t>(d.sm_major) << 8 | static_cast<uint64_t>(d.sm_minor));
  mix((d.qk_l2_norm ? 1ULL : 0ULL) | (d.decay_per_key_dim ? 2ULL : 0ULL) |
      (d.has_decay ? 4ULL : 0ULL) | (d.has_beta ? 8ULL : 0ULL) |
      (d.has_initial_state ? 16ULL : 0ULL) | (d.ragged ? 32ULL : 0ULL));
  return h;
}

namespace {

// Shared-memory footprint of the chunked engine, mirroring CarveChunked in the .cu.
size_t ChunkedSmemBytes(int bt, int dk, int dvb) {
  const int ld_kh = dk + 8;
  const int ld_vh = dvb + 8;
  const int ld_mh = bt + 8;
  const int ld_vf = std::max(dvb, bt) + 4;
  const size_t f = sizeof(float) * static_cast<size_t>(dk * ld_vf + bt * ld_vf + 2 * bt);
  const size_t h = sizeof(uint16_t) * static_cast<size_t>(2 * bt * ld_kh + 2 * bt * ld_vh +
                                                          4 * bt * ld_mh + dk * ld_vh);
  return f + h;
}

}  // namespace

Plan SelectPlan(const Descriptor& desc, int sm_count, size_t smem_per_block_optin) {
  Plan plan;
  plan.chunk_size = desc.chunk_size > 0 ? desc.chunk_size : 64;

  // Token checkpoints are a per-token series; only the sequential engine can produce them.
  const bool needs_checkpoints = desc.state_checkpoints > 0;

  // Below the crossover the chunked engine still pays for a full chunk, so a handful of
  // tokens costs the same as 64. Measured crossover on H200 at the Qwen3.8 geometry is
  // ~30 tokens (chunked 46.5 us at T=1 against 17.9 us for a sequential recurrence).
  const int64_t kChunkedMinTokens = 32;
  const bool long_enough = desc.total_tokens >= kChunkedMinTokens * std::max(desc.batch, 1);

  const bool shape_ok = desc.head_size_qk == 128 && desc.head_size_v == 128 &&
                        plan.chunk_size == 64 && desc.num_heads_q == desc.num_heads_k &&
                        desc.num_heads_v % desc.num_heads_q == 0;

  // The chunked engine folds the decay into [BT x BT] gram matrices, which assumes one
  // scalar decay per (token, v-head). Per-key-dim decay (KDA) stays sequential.
  const bool rule_ok = !desc.decay_per_key_dim;

  // Its GEMM operands are fp16 mma fragments, so float input would be silently narrowed;
  // float stays on the scalar engine. bfloat16 needs bf16 fragments, not yet written.
  const bool type_ok = desc.io_type == IoType::kFloat16;

  if (!needs_checkpoints && long_enough && shape_ok && rule_ok && type_ok && desc.sm_major >= 8) {
    plan.v_block = 64;
    plan.threads = 512;
    plan.smem_bytes = ChunkedSmemBytes(plan.chunk_size, desc.head_size_qk, plan.v_block);
    if (plan.smem_bytes <= smem_per_block_optin) {
      plan.engine = Engine::kChunked;
      plan.workspace_bytes = 0;  // the state never leaves shared memory
      plan.supported = true;
      return plan;
    }
    plan.reject_reason = "chunked engine exceeds the device shared-memory opt-in limit";
  }

  plan.engine = Engine::kRecurrent;
  plan.cols_per_block = 32;
  plan.smem_bytes = 0;
  plan.workspace_bytes = 0;
  plan.supported = true;
  if (plan.reject_reason == nullptr) {
    if (needs_checkpoints) {
      plan.reject_reason = "token checkpoints requested";
    } else if (!long_enough) {
      plan.reject_reason = "below the chunked-engine token threshold";
    } else if (!shape_ok) {
      plan.reject_reason = "unsupported head geometry or chunk size";
    } else if (!rule_ok) {
      plan.reject_reason = "per-key-dimension decay";
    } else if (!type_ok) {
      plan.reject_reason = "chunked engine requires float16 input";
    }
  }
  (void)sm_count;
  return plan;
}

PlanCache& PlanCache::Instance() {
  static PlanCache instance;
  return instance;
}

Plan PlanCache::GetOrCreate(const Descriptor& desc, int sm_count, size_t smem_per_block_optin) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = cache_.find(desc);
  if (it != cache_.end()) {
    return it->second;
  }
  Plan plan = SelectPlan(desc, sm_count, smem_per_block_optin);
  cache_.emplace(desc, plan);
  return plan;
}

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

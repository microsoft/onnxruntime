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

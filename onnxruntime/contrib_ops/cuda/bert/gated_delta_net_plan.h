// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plan selection for GatedDeltaNet, modelled on the cuDNN frontend's execution flow:
// a problem descriptor is hashed into a cache, a heuristic picks an engine, the engine
// reports its workspace requirement, and execution binds a variant pack of pointers.
//
// cuDNN itself cannot serve as a backend here: its GDN/KDA graph nodes exist only in the
// Python frontend (`python/cudnn/linear_attention/`, nothing in `include/`) and its FROST
// engines refuse anything below SM100. `Engine::kCudnn` is therefore reserved, not wired.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gated_delta_net {

enum class UpdateRule : int { kLinear = 0, kGated = 1, kDelta = 2, kGatedDelta = 3 };
enum class GateActivation : int { kNone = 0, kQwen = 1 };
enum class BetaActivation : int { kNone = 0, kSigmoid = 1 };

// Element type of the q/k/v/output tensors. The state is always float.
enum class IoType : int { kFloat = 0, kFloat16 = 1, kBFloat16 = 2 };

enum class Engine : int {
  kAuto = 0,
  kChunked = 1,    // tensor-core chunked scan; prefill
  kRecurrent = 2,  // sequential per-token recurrence; decode / MTP verify, emits checkpoints
  kCudnn = 3,      // reserved: SM100-only and C++-inaccessible today
};

const char* EngineName(Engine e);
Engine EngineFromName(const std::string& name);

// Everything that can change the generated code or the launch geometry. Two calls with
// equal descriptors may share a plan.
struct Descriptor {
  int64_t total_tokens = 0;
  int batch = 0;
  int num_heads_q = 0;
  int num_heads_k = 0;
  int num_heads_v = 0;
  int head_size_qk = 0;
  int head_size_v = 0;
  int chunk_size = 64;
  int state_checkpoints = 0;
  UpdateRule update_rule = UpdateRule::kGatedDelta;
  GateActivation gate_activation = GateActivation::kNone;
  BetaActivation beta_activation = BetaActivation::kNone;
  IoType io_type = IoType::kFloat16;
  bool qk_l2_norm = false;
  bool decay_per_key_dim = false;
  bool has_decay = false;
  bool has_beta = false;
  bool has_initial_state = false;
  bool ragged = false;  // cu_seqlens supplied
  int sm_major = 0;
  int sm_minor = 0;

  bool operator==(const Descriptor& o) const noexcept;
};

struct DescriptorHash {
  size_t operator()(const Descriptor& d) const noexcept;
};

struct Plan {
  Engine engine = Engine::kRecurrent;
  int chunk_size = 64;
  int v_block = 64;      // dv columns owned by one CTA (chunked engine)
  int threads = 512;     // CTA size
  int cols_per_block = 32;  // dv columns per CTA (recurrent engine)
  size_t smem_bytes = 0;
  size_t workspace_bytes = 0;
  bool supported = false;
  const char* reject_reason = nullptr;
};

// The analogue of cudnn's heuristic mode A. Pure arithmetic on the descriptor and the
// device limits, so it is defined here: tests can then exercise the architecture-dependent
// choices -- notably the consumer-Blackwell shared-memory budget -- without linking the
// provider module or owning that hardware.
inline size_t ChunkedSmemBytes(int bt, int dk, int dvb) {
  const int ld_kh = dk + 8;
  const int ld_vh = dvb + 8;
  const int ld_mh = bt + 8;
  const int ld_vf = (dvb > bt ? dvb : bt) + 4;
  const size_t f = sizeof(float) * static_cast<size_t>(dk * ld_vf + bt * ld_vf + 2 * bt);
  const size_t h = sizeof(uint16_t) * static_cast<size_t>(2 * bt * ld_kh + 2 * bt * ld_vh +
                                                          4 * bt * ld_mh + dk * ld_vh);
  return f + h;
}

inline Plan SelectPlan(const Descriptor& desc, int sm_count, size_t smem_per_block_optin) {
  Plan plan;
  plan.chunk_size = desc.chunk_size > 0 ? desc.chunk_size : 64;

  // Token checkpoints are a per-token series; only the sequential engine can produce them.
  const bool needs_checkpoints = desc.state_checkpoints > 0;

  // Below the crossover the chunked engine still pays for a full chunk, so a handful of
  // tokens costs the same as 64. Measured crossover on H200 at the Qwen3.8 geometry is
  // ~30 tokens (chunked 46.5 us at T=1 against 17.9 us for a sequential recurrence).
  const int64_t kChunkedMinTokens = 32;
  const int64_t batch = desc.batch > 1 ? desc.batch : 1;
  const bool long_enough = desc.total_tokens >= kChunkedMinTokens * batch;

  const bool shape_ok = desc.head_size_qk == 128 && desc.head_size_v == 128 &&
                        desc.num_heads_q == desc.num_heads_k && desc.num_heads_q > 0 &&
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
    // BT=64 is the fastest configuration measured on SM90 but needs 157 KB of shared
    // memory. Consumer Blackwell (SM120) allows only 99 KB per block, where BT=32 fits in
    // 96 KB for about a 10% cost. Take the widest chunk the device can actually hold; an
    // explicit chunk_size of 32 pins the narrow one so that path is reachable on SM90.
    const bool pin_narrow = desc.chunk_size == 32;
    const int candidates[2] = {64, 32};
    for (int bt : candidates) {
      if (pin_narrow && bt != 32) continue;
      const size_t bytes = ChunkedSmemBytes(bt, desc.head_size_qk, plan.v_block);
      if (bytes <= smem_per_block_optin) {
        plan.chunk_size = bt;
        plan.smem_bytes = bytes;
        plan.engine = Engine::kChunked;
        plan.workspace_bytes = 0;  // the state never leaves shared memory
        plan.supported = true;
        return plan;
      }
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
      plan.reject_reason = "unsupported head geometry";
    } else if (!rule_ok) {
      plan.reject_reason = "per-key-dimension decay";
    } else if (!type_ok) {
      plan.reject_reason = "chunked engine requires float16 input";
    }
  }
  (void)sm_count;
  return plan;
}

// Process-wide memoisation of SelectPlan, mirroring the frontend's plan cache.
class PlanCache {
 public:
  static PlanCache& Instance();
  Plan GetOrCreate(const Descriptor& desc, int sm_count, size_t smem_per_block_optin);

 private:
  std::mutex mutex_;
  std::unordered_map<Descriptor, Plan, DescriptorHash> cache_;
};

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

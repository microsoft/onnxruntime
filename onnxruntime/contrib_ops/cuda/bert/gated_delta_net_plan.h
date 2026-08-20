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

// The analogue of cudnn's heuristic mode A. Pure function of the descriptor and the device,
// so it is safe to memoise.
Plan SelectPlan(const Descriptor& desc, int sm_count, size_t smem_per_block_optin);

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

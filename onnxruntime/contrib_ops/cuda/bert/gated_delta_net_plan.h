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

#include <cmath>
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

enum class UpdateRule : int { kLinear = 0,
                              kGated = 1,
                              kDelta = 2,
                              kGatedDelta = 3 };
enum class GateActivation : int { kNone = 0,
                                  kQwen = 1 };
enum class BetaActivation : int { kNone = 0,
                                  kSigmoid = 1 };

// Element type of the q/k/v/output tensors. The state is always float.
enum class IoType : int { kFloat = 0,
                          kFloat16 = 1,
                          kBFloat16 = 2 };

enum class Engine : int {
  kAuto = 0,
  kChunked = 1,       // fused tensor-core chunked scan; prefill
  kRecurrent = 2,     // sequential per-token recurrence; decode / MTP verify, emits checkpoints
  kCudnn = 3,         // reserved: SM100-only and C++-inaccessible today
  kChunkedSplit = 4,  // chunked scan split into a token-parallel prepare and a state-only scan
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
  // Benchmarking override. kAuto lets the heuristic choose.
  Engine preferred_engine = Engine::kAuto;
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
  int v_block = 64;         // dv columns owned by one CTA (chunked engine)
  int threads = 512;        // CTA size
  int cols_per_block = 32;  // dv columns per CTA (recurrent engine)
  // Split engine only: chunks whose prepare output is live in the workspace at once. The
  // sequence is walked in passes of this many chunks so the workspace stays bounded.
  int chunks_per_pass = 0;
  // Recurrent engine only: one warp per v-column with lanes spanning K, instead of one CTA
  // per v-head with the state in shared memory.
  bool warp_specialized = false;
  size_t smem_bytes = 0;          // chunked / split-scan kernel
  size_t smem_bytes_prepare = 0;  // split-prepare kernel
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

// Split engine. K1 holds q, k, the full-width v, and the [BT x BT] matrices of the inverse;
// it never sees the state. The Neumann iterate shares M's tile, so three, not four.
inline size_t SplitPrepareSmemBytes(int bt, int dk, int dv) {
  const int ld_kh = dk + 8;
  const int ld_vh = dv + 8;
  const int ld_mh = bt + 8;
  const int ld_f1 = (dv > bt ? dv : bt) + 4;
  const size_t f = sizeof(float) * static_cast<size_t>(bt * ld_f1 + 2 * bt);
  const size_t h =
      sizeof(uint16_t) * static_cast<size_t>(2 * bt * ld_kh + bt * ld_vh + 3 * bt * ld_mh);
  return f + h;
}

// K2 holds the fp16 state, W/Qg/Kd, its own v-block of U, P, and one fp32 scratch that must
// be wide enough for the [DK x DVB] state update.
inline size_t SplitScanSmemBytes(int bt, int dk, int dvb) {
  const int ld_kh = dk + 8;
  const int ld_bh = dvb + 8;
  const int ld_mh = bt + 8;
  const int ld_f2 = (dvb > bt ? dvb : bt) + 4;
  const size_t f = sizeof(float) * static_cast<size_t>(dk) * ld_f2;
  const size_t h = sizeof(uint16_t) * static_cast<size_t>(dk * ld_bh + 3 * bt * ld_kh +
                                                          bt * ld_bh + bt * ld_mh);
  return f + h;
}

// Workspace one (head, chunk) needs to hand K1's output to K2: W, Qg, Kd [BT x DK],
// Uv [BT x DV] and P [BT x BT] in fp16, plus the chunk's scalar decay.
inline size_t SplitTileBytes(int bt, int dk, int dv) {
  return sizeof(uint16_t) * static_cast<size_t>(bt) * (3 * dk + dv + bt) + sizeof(float);
}

inline Plan SelectPlan(const Descriptor& desc, int sm_count, size_t smem_per_block_optin) {
  Plan plan;
  plan.chunk_size = desc.chunk_size > 0 ? desc.chunk_size : 64;

  const int64_t batch = desc.batch > 1 ? desc.batch : 1;

  // Token checkpoints are a per-token series, and only the sequential engine can produce one.
  // A request longer than the window cannot be rolled back into it anyway, so it takes the
  // normal plan and only the committed last slot is written.
  const bool needs_checkpoints =
      desc.state_checkpoints > 0 &&
      desc.total_tokens <= static_cast<int64_t>(desc.state_checkpoints) * batch;

  // Below the crossover the chunked engine still pays for a full chunk, so a handful of
  // tokens costs the same as 64. Measured crossover on H200 at the Qwen3.8 geometry is
  // ~30 tokens (chunked 46.5 us at T=1 against 17.9 us for a sequential recurrence).
  const int64_t kChunkedMinTokens = 32;
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

    // Split engine: a token-parallel prepare launch followed by a scan that only carries
    // the state. It exists to fill a machine the fused engine leaves idle, so it is chosen
    // on the fused engine's own wave-quantisation efficiency rather than on a shape guess.
    //
    // The fused grid is one CTA per (sequence, v-head, 64 v-columns), so it costs
    // ceil(waves) waves to do `waves` waves of work. Measured on H200 (132 SMs) at the
    // Qwen3.8 geometry, split/fused runtime tracks that efficiency and nothing else:
    //
    //   batch  CTAs  waves  efficiency   T=256  T=1024  T=4096
    //     1      96   0.73     73%        0.90    0.80    0.83
    //     2     192   1.45     73%        0.80    0.88    0.88
    //     3     288   2.18     73%        0.85    0.91    0.90
    //     4     384   2.91     97%        0.94    1.12    1.13
    //     8     768   5.82     97%        1.22    1.31    1.32
    //
    // The fused per-token cost jumps 0.392 -> 0.298 us across that same boundary, so once
    // the fused engine stops wasting a wave the split engine's extra launch and its
    // round-trip through the workspace are pure overhead. Below two chunks there is no
    // sequence to pipeline either and the extra launch dominates (T=64 measures 1.18x).
    const int bt = 64;
    const int64_t longest_seq = (desc.total_tokens + batch - 1) / batch;
    const int64_t fused_ctas = batch * desc.num_heads_v * (desc.head_size_v / plan.v_block);
    const double waves = sm_count > 0 ? static_cast<double>(fused_ctas) / sm_count : 0.0;
    const double wave_efficiency = waves > 0.0 ? waves / std::ceil(waves) : 0.0;
    const bool fused_underfills = sm_count > 0 && wave_efficiency < 0.85;
    const bool long_enough_to_pipeline = longest_seq >= 2 * bt;

    const bool want_split = desc.preferred_engine == Engine::kChunkedSplit ||
                            (desc.preferred_engine != Engine::kChunked && fused_underfills &&
                             long_enough_to_pipeline);
    if (want_split) {
      const size_t prep = SplitPrepareSmemBytes(bt, desc.head_size_qk, desc.head_size_v);
      // Narrow v-blocks only pay off once the state-independent work has been hoisted out
      // of the scan, which is what this engine does: all four of the scan's GEMMs then
      // scale with the block width. Take the narrow block when two CTAs of it fit on an
      // SM, since the wide one leaves the grid under a single wave at batch 1.
      const int narrow = 32;
      const size_t scan_narrow = SplitScanSmemBytes(bt, desc.head_size_qk, narrow);
      const int dvb = (2 * scan_narrow <= smem_per_block_optin) ? narrow : 64;
      const size_t scan = SplitScanSmemBytes(bt, desc.head_size_qk, dvb);
      if (prep <= smem_per_block_optin && scan <= smem_per_block_optin) {
        plan.engine = Engine::kChunkedSplit;
        plan.chunk_size = bt;
        plan.v_block = dvb;
        plan.smem_bytes = scan;
        plan.smem_bytes_prepare = prep;
        // Cap the live prepare output so a long sequence walks in passes instead of asking
        // for a workspace proportional to its length.
        const size_t tile = SplitTileBytes(bt, desc.head_size_qk, desc.head_size_v);
        const size_t per_chunk = tile * static_cast<size_t>(batch) * desc.num_heads_v;
        const int64_t longest = longest_seq;
        const int total_chunks = static_cast<int>((longest + bt - 1) / bt);
        const size_t kWorkspaceCap = 64u << 20;
        int per_pass = static_cast<int>(kWorkspaceCap / (per_chunk > 0 ? per_chunk : 1));
        per_pass = per_pass < 1 ? 1 : per_pass;
        per_pass = per_pass > total_chunks ? total_chunks : per_pass;
        plan.chunks_per_pass = per_pass;
        plan.workspace_bytes = per_chunk * static_cast<size_t>(per_pass);
        if (per_pass < total_chunks) {
          // A multi-pass walk carries the state between launches through the workspace,
          // because the caller is not obliged to have asked for final_state.
          plan.workspace_bytes += sizeof(float) * static_cast<size_t>(batch) *
                                  desc.num_heads_v * desc.head_size_v * desc.head_size_qk;
        }
        plan.supported = true;
        return plan;
      }
      plan.v_block = 64;  // fall through to the fused engine
    }

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
  // The warp kernel keeps its slice of the state in registers, so it only needs a head size
  // that divides evenly into 32-lane strips.
  plan.warp_specialized =
      desc.head_size_qk == 64 || desc.head_size_qk == 128 || desc.head_size_qk == 256;
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

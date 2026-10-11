// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plan selection for GatedDeltaNet, modeled on the cuDNN frontend's execution flow:
// a heuristic picks an engine, the engine reports its workspace requirement, and execution
// binds a variant pack of pointers.
//
// cuDNN itself cannot serve as a backend here: its GDN/KDA graph nodes exist only in the
// Python frontend (`python/cudnn/linear_attention/`, nothing in `include/`) and its FROST
// engines refuse anything below SM100. `Engine::kCudnn` is therefore reserved, not wired.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

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
  kRecurrent = 2,     // sequential per-token recurrence; decode / speculative verify
  kCudnn = 3,         // reserved: SM100-only and C++-inaccessible today
  kChunkedSplit = 4,  // chunked scan split into a token-parallel prepare and a state-only scan
};

inline constexpr int kChunkedMinTokens = 32;

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
  int state_update_capacity = 0;
  bool state_update_active = false;
  UpdateRule update_rule = UpdateRule::kGatedDelta;
  GateActivation gate_activation = GateActivation::kNone;
  BetaActivation beta_activation = BetaActivation::kNone;
  IoType io_type = IoType::kFloat16;
  bool qk_l2_norm = false;
  bool decay_per_key_dim = false;
  bool ragged = false;  // cu_seqlens supplied
  // Benchmarking override. kAuto lets the heuristic choose.
  Engine preferred_engine = Engine::kAuto;
  int sm_major = 0;
};

struct Plan {
  Engine engine = Engine::kRecurrent;
  int chunk_size = 64;
  int v_block = 64;  // dv columns owned by one CTA (chunked engine)
  // Split engine only: chunks whose prepare output is live in the workspace at once. The
  // sequence is walked in passes of this many chunks so the workspace stays bounded.
  int chunks_per_pass = 0;
  // Recurrent engine only: one warp per v-column with lanes spanning K, instead of one CTA
  // per v-head with the state in shared memory.
  bool warp_specialized = false;
  // Compact state updates are requested per row by a device capture_count. The fused pass
  // skips rows with a positive count and a recurrent tail emits their transition factors.
  bool state_update_tail_pass = false;
  // Ragged prefill can share a launch with decode rows. Keep rows below the chunked crossover on
  // the recurrent engine so their arithmetic does not depend on other scheduled requests.
  bool short_row_tail_pass = false;
  size_t smem_bytes = 0;          // chunked / split-scan kernel
  size_t smem_bytes_prepare = 0;  // split-prepare kernel
  size_t workspace_bytes = 0;
  bool supported = false;
  const char* reject_reason = nullptr;
};

// The analog of cudnn's heuristic mode A. Pure arithmetic on the descriptor and the
// device limits, so it is defined here: tests can then exercise the architecture-dependent
// choices -- notably the consumer-Blackwell shared-memory budget -- without linking the
// provider module or owning that hardware.
inline size_t RecurrentSmemBytes(int dk, int dv) {
  const int reduction_scratch = dv > 2 ? dv : 2;
  return sizeof(float) *
         (static_cast<size_t>(dk) * dv + 2 * dk + reduction_scratch + dk);
}

// The warp kernel keeps its slice of the state in registers, so it only needs a head size
// that divides evenly into 32-lane strips.
inline bool RecurrentIsWarpSpecialized(int dk) {
  return dk == 64 || dk == 128 || dk == 256;
}

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

inline Plan SelectPlan(const Descriptor& desc, size_t smem_per_block_optin) {
  Plan plan;
  plan.chunk_size = desc.chunk_size > 0 ? desc.chunk_size : 64;

  const int64_t batch = desc.batch > 1 ? desc.batch : 1;

  const bool wants_state_update_tail_pass =
      desc.state_update_capacity > 0 && desc.state_update_active;
  const bool wants_short_row_tail_pass = desc.ragged;

  // Below the crossover the chunked engine still pays for a full chunk, so a handful of
  // tokens costs the same as 64. Measured crossover on H200 at the Qwen3.8 geometry is
  // ~30 tokens (chunked 46.5 us at T=1 against 17.9 us for a sequential recurrence).
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

  if (long_enough && shape_ok && rule_ok && type_ok && desc.sm_major >= 8) {
    plan.v_block = 64;

    // The split engine remains an explicit benchmarking override. The fused chunked engine
    // is the production prefill path selected by the automatic heuristic.
    const int bt = 64;
    const int64_t longest_seq = (desc.total_tokens + batch - 1) / batch;
    const bool want_split = !wants_state_update_tail_pass &&
                            !wants_short_row_tail_pass &&
                            desc.preferred_engine == Engine::kChunkedSplit;
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
      plan.reject_reason = "split engine exceeds the device shared-memory opt-in limit";
      plan.v_block = 64;  // fall through to the fused engine
    }

    // BT=64 is the fastest configuration measured on SM90 but needs 157 KB of shared
    // memory. Consumer Blackwell (SM120) allows only 99 KB per block, where BT=32 fits in
    // 96 KB for about a 10% cost. Take the widest chunk the device can actually hold; an
    // explicit chunk_size of 32 pins the narrow one so that path is reachable on SM90.
    const bool pin_narrow = desc.chunk_size == 32;
    const int candidates[2] = {64, 32};
    for (int candidate_bt : candidates) {
      if (pin_narrow && candidate_bt != 32) continue;
      const size_t bytes = ChunkedSmemBytes(candidate_bt, desc.head_size_qk, plan.v_block);
      if (bytes <= smem_per_block_optin) {
        plan.chunk_size = candidate_bt;
        plan.smem_bytes = bytes;
        plan.engine = Engine::kChunked;
        plan.workspace_bytes = 0;  // the state never leaves shared memory
        plan.state_update_tail_pass = wants_state_update_tail_pass;
        plan.short_row_tail_pass = wants_short_row_tail_pass;
        plan.supported = true;
        return plan;
      }
    }
    plan.reject_reason = "chunked engine exceeds the device shared-memory opt-in limit";
  }

  plan.engine = Engine::kRecurrent;
  plan.smem_bytes = 0;
  plan.workspace_bytes = 0;
  plan.supported = true;
  plan.warp_specialized = RecurrentIsWarpSpecialized(desc.head_size_qk);
  if (plan.reject_reason == nullptr) {
    if (!long_enough) {
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

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

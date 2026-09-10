// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

namespace onnxruntime {
namespace contrib {
namespace sparse_attention_indexer {

// Values accepted by the policy_mode attribute of com.microsoft.SparseAttentionIndexer.
constexpr const char* kPolicyModeQsa = "qsa";
constexpr const char* kPolicyModeCsa = "csa";

enum class Policy {
  kQsa,
  kCsa,
};

inline bool TryParsePolicy(const std::string& policy_mode, Policy& policy) {
  if (policy_mode == kPolicyModeQsa) {
    policy = Policy::kQsa;
    return true;
  }
  if (policy_mode == kPolicyModeCsa) {
    policy = Policy::kCsa;
    return true;
  }
  return false;
}

// Input slots. Slots 5-6 belong to policy_mode="qsa" and slots 7-13 to policy_mode="csa";
// a slot that does not belong to the active policy must be omitted from the node.
enum InputIndex : int {
  kQuery = 0,
  kKey = 1,
  kKeyNormWeight = 2,
  kCosCache = 3,
  kSinCache = 4,
  kMask = 5,
  kPastKey = 6,
  kGate = 7,
  kPositionBias = 8,
  kHeadWeights = 9,
  kPositionIds = 10,
  kPastCompressedKey = 11,
  kPastKvBuffer = 12,
  kPastGateBuffer = 13,
  kInputCount = 14,
};

// Output slots. Slot 1 belongs to policy_mode="qsa" and slots 2-4 to policy_mode="csa".
enum OutputIndex : int {
  kSelectedIndices = 0,
  kPresentKey = 1,
  kPresentCompressedKey = 2,
  kPresentKvBuffer = 3,
  kPresentGateBuffer = 4,
  kOutputCount = 5,
};

// A "qsa" node declares selected_indices + present_key; a "csa" node declares every slot so that
// the three csa state outputs keep their fixed indices (slot 1 is left as a missing optional).
constexpr int kQsaOutputCount = 2;
constexpr int kCsaOutputCount = 5;

// Number of selected entries emitted per query. The capacity only depends on attributes, so it is
// a compile-time constant of the graph rather than a function of the data.
inline int64_t SelectedCapacity(Policy policy, int64_t token_budget, int64_t index_topk,
                                int64_t compress_ratio) {
  return policy == Policy::kQsa ? token_budget + compress_ratio - 1 : index_topk;
}

// How the "csa" token buffer is split and how many new compressed entries this call emits.
//
// past_kv_buffer / past_gate_buffer carry `overlap_length` tokens of the previous complete window
// (the Ca operand of the next window) followed by `leftover_length` tokens of an incomplete window.
// Since `leftover_length` is always < compress_ratio, a buffer length >= compress_ratio uniquely
// means "the previous complete window is present", so no extra state tensor is needed.
struct CsaWindowPlan {
  int64_t overlap_length = 0;         // compress_ratio, or 0 before the first complete window
  int64_t leftover_length = 0;        // tokens of the current incomplete window
  int64_t new_window_count = 0;       // complete windows closed by this call
  int64_t present_buffer_length = 0;  // length of present_kv_buffer / present_gate_buffer
  int64_t present_buffer_start = 0;   // offset of that buffer inside [past buffer | new tokens]
};

inline bool TryComputeCsaWindowPlan(int64_t past_buffer_length, int64_t sequence_length,
                                    int64_t compress_ratio, CsaWindowPlan& plan) {
  if (compress_ratio <= 0 || sequence_length < 0 || past_buffer_length < 0 ||
      past_buffer_length >= 2 * compress_ratio) {
    return false;
  }

  plan.overlap_length = past_buffer_length >= compress_ratio ? compress_ratio : 0;
  plan.leftover_length = past_buffer_length - plan.overlap_length;

  const int64_t pending = plan.leftover_length + sequence_length;
  plan.new_window_count = pending / compress_ratio;
  if (plan.new_window_count > 0) {
    plan.present_buffer_length = compress_ratio + pending % compress_ratio;
    plan.present_buffer_start = plan.overlap_length + (plan.new_window_count - 1) * compress_ratio;
  } else {
    plan.present_buffer_length = past_buffer_length + sequence_length;
    plan.present_buffer_start = 0;
  }
  return true;
}

}  // namespace sparse_attention_indexer
}  // namespace contrib
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

// nvcc recognizes __host__/__device__ as built-in qualifiers in any translation unit it compiles
// (no CUDA header include required), but a plain host compiler does not know these tokens. This
// header is shared by CPU-only graph/schema code and by CUDA device code, so the annotation is
// only emitted when nvcc is compiling the translation unit that includes this header.
#if defined(__CUDACC__)
#define SAI_HOST_DEVICE __host__ __device__
#else
#define SAI_HOST_DEVICE
#endif

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

// Input slots. mask belongs to policy_mode="qsa"; gate, position_bias, head_weights, position_ids,
// and past_proj_buffer belong to policy_mode="csa". The past_key and past_sequence_length state is
// shared by both policies.
enum InputIndex : int {
  kQuery = 0,
  kKey = 1,
  kQueryNormWeight = 2,
  kKeyNormWeight = 3,
  kCosCache = 4,
  kSinCache = 5,
  kMask = 6,
  kPastKey = 7,
  kGate = 8,
  kPositionBias = 9,
  kHeadWeights = 10,
  kPositionIds = 11,
  kPastSequenceLength = 12,
  kPastProjBuffer = 13,
  kInputCount = 14,
};

// present_key is shared by both policies; present_proj_buffer is only used by policy_mode="csa".
enum OutputIndex : int {
  kSelectedIndices = 0,
  kPresentKey = 1,
  kPresentProjBuffer = 2,
  kOutputCount = 3,
};

// A "qsa" node declares selected_indices + present_key. A "csa" node additionally declares
// present_proj_buffer.
constexpr int kQsaOutputCount = 2;
constexpr int kCsaOutputCount = 3;

// Number of selected entries emitted per query. The capacity only depends on attributes, so it is
// a compile-time constant of the graph rather than a function of the data.
SAI_HOST_DEVICE inline int64_t SelectedCapacity(Policy policy, int64_t token_budget, int64_t index_topk,
                                                int64_t compress_ratio) {
  return policy == Policy::kQsa ? token_budget + compress_ratio - 1 : index_topk;
}

// How the "csa" token buffer is split and how many new compressed entries this call emits.
//
// Each plane of past_proj_buffer carries `overlap_length` tokens of the previous complete window
// (the Ca operand of the next window) followed by `leftover_length` tokens of an incomplete window.
// Since `leftover_length` is always < compress_ratio, a buffer length >= compress_ratio uniquely
// means "the previous complete window is present", so no extra state tensor is needed.
struct CsaWindowPlan {
  int64_t overlap_length = 0;         // compress_ratio, or 0 before the first complete window
  int64_t leftover_length = 0;        // tokens of the current incomplete window
  int64_t new_window_count = 0;       // complete windows closed by this call
  int64_t present_buffer_length = 0;  // sequence length of present_proj_buffer
  int64_t present_buffer_start = 0;   // offset of that buffer inside [past buffer | new tokens]
};

SAI_HOST_DEVICE inline bool TryComputeCsaWindowPlan(int64_t past_buffer_length, int64_t sequence_length,
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

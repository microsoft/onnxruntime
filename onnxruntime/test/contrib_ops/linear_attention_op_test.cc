// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

using namespace onnxruntime::test;


namespace onnxruntime {
namespace test {

namespace {

// Reference implementation of the linear attention recurrence.
// Processes all tokens sequentially and returns output + final_state.
void LinearAttentionReference(
    const std::string& update_rule,
    int batch_size, int num_heads, int seq_length, int head_dim_k, int head_dim_v,
    float scale,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>* initial_state,
    const std::vector<float>* decay,
    const std::vector<float>* beta,
    std::vector<float>& output,
    std::vector<float>& final_state) {

    int bht = batch_size * num_heads * seq_length;
    bool decay_broadcast_dk = (decay != nullptr && static_cast<int>(decay->size()) == bht);

      // State: (B, H, dk, dv)
  final_state.resize(batch_size * num_heads * head_dim_k * head_dim_v, 0.0f);
  output.resize(batch_size * num_heads * seq_length * head_dim_v, 0.0f);

  // Initialize state from initial_state if provided
  if (initial_state != nullptr) {
    final_state = *initial_state;
  }

  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_heads; h++) {
      // State for this (b, h): dk x dv
      auto state_offset = [&](int k, int v) {
        return ((b * num_heads + h) * head_dim_k + k) * head_dim_v + v;
      };

      for (int t = 0; t < seq_length; t++) {
        auto qkv_offset = [&](int dim) {
          return ((b * num_heads + h) * seq_length + t) * dim;
        };

        // Load q, k for this token
        std::vector<float> q_vec(head_dim_k), k_vec(head_dim_k), v_vec(head_dim_v);
        for (int i = 0; i < head_dim_k; i++) {
          q_vec[i] = query[qkv_offset(head_dim_k) + i];
          k_vec[i] = key[qkv_offset(head_dim_k) + i];
        }
        for (int i = 0; i < head_dim_v; i++) {
          v_vec[i] = value[qkv_offset(head_dim_v) + i];
        }

        // Step 1: Apply decay (gated, gated_delta)
        if (update_rule == "gated" || update_rule == "gated_delta") {
          for (int k = 0; k < head_dim_k; k++) {
            float exp_g;
            if (decay_broadcast_dk) {
              int decay_idx = (b * num_heads + h) * seq_length + t;
              exp_g = std::exp((*decay)[decay_idx]);
            } else {
              int decay_idx = ((b * num_heads + h) * seq_length + t) * head_dim_k + k;
              exp_g = std::exp((*decay)[decay_idx]);
            }
            for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
              final_state[state_offset(k, v_idx)] *= exp_g;
            }
          }
        }

        // Step 2: Compute state update
        if (update_rule == "delta" || update_rule == "gated_delta") {
          // retrieved = S^T @ k (for each v dimension)
          std::vector<float> retrieved(head_dim_v, 0.0f);
          for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
            for (int k = 0; k < head_dim_k; k++) {
              retrieved[v_idx] += final_state[state_offset(k, v_idx)] * k_vec[k];
            }
          }

          // delta = beta * (v - retrieved)
          int beta_idx = (b * num_heads + h) * seq_length + t;
          float beta_val = (*beta)[beta_idx];
          std::vector<float> delta(head_dim_v);
          for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
            delta[v_idx] = beta_val * (v_vec[v_idx] - retrieved[v_idx]);
          }

          // S += k ⊗ delta
          for (int k = 0; k < head_dim_k; k++) {
            for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
              final_state[state_offset(k, v_idx)] += k_vec[k] * delta[v_idx];
            }
          }
        } else {
          // linear, gated: S += k ⊗ v
          for (int k = 0; k < head_dim_k; k++) {
            for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
              final_state[state_offset(k, v_idx)] += k_vec[k] * v_vec[v_idx];
            }
          }
        }

        // Step 3: Compute output = scale * S^T @ q
        for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
          float sum = 0.0f;
          for (int k = 0; k < head_dim_k; k++) {
            sum += final_state[state_offset(k, v_idx)] * q_vec[k];
          }
          int out_idx = ((b * num_heads + h) * seq_length + t) * head_dim_v + v_idx;
          output[out_idx] = scale * sum;
        }
      }
    }
  }
}

// Convert data from 4D (B,H,T,D) layout to 3D packed (B,T,H*D) layout
std::vector<float> PackBHTD_to_BTHD(const std::vector<float>& data_4d,
                                     int B, int H, int T, int D) {
  std::vector<float> packed(B * T * H * D);
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
          int src_idx = ((b * H + h) * T + t) * D + d;
          int dst_idx = (b * T + t) * (H * D) + h * D + d;
          packed[dst_idx] = data_4d[src_idx];
        }
      }
    }
  }
  return packed;
}

// Convert decay/beta from (B,H,T) layout to (B,T,H) layout
std::vector<float> TransposeBHT_to_BTH(const std::vector<float>& data,
                                        int B, int H, int T) {
  std::vector<float> transposed(B * T * H);
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int t = 0; t < T; t++) {
        int src_idx = (b * H + h) * T + t;
        int dst_idx = (b * T + t) * H + h;
        transposed[dst_idx] = data[src_idx];
      }
    }
  }
  return transposed;
}

void RunLinearAttentionTest(
    const std::string& update_rule,
    int batch_size, int num_heads, int seq_length, int head_dim_k, int head_dim_v,
    float scale,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>* initial_state,
    const std::vector<float>* decay,
    const std::vector<float>* beta_data) {
  // Compute reference output (reference works in 4D layout)
  std::vector<float> expected_output_4d, expected_state;
  LinearAttentionReference(update_rule, batch_size, num_heads, seq_length,
                           head_dim_k, head_dim_v, scale,
                           query, key, value, initial_state, decay, beta_data,
                           expected_output_4d, expected_state);

  bool enable_webgpu = (nullptr != DefaultWebGpuExecutionProvider().get());
  if (!enable_webgpu) {
    return;
  }

  int bht = batch_size * num_heads * seq_length;
  bool decay_broadcast_dk = (decay != nullptr && static_cast<int>(decay->size()) == bht);

  // Convert from 4D (B,H,T,D) to 3D packed (B,T,H*D) for OpTester
  auto query_3d = PackBHTD_to_BTHD(query, batch_size, num_heads, seq_length, head_dim_k);
  auto key_3d = PackBHTD_to_BTHD(key, batch_size, num_heads, seq_length, head_dim_k);
  auto value_3d = PackBHTD_to_BTHD(value, batch_size, num_heads, seq_length, head_dim_v);
  auto output_3d = PackBHTD_to_BTHD(expected_output_4d, batch_size, num_heads, seq_length, head_dim_v);

  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", update_rule);
  tester.AddAttribute<float>("scale", scale);
  tester.AddAttribute<int64_t>("q_num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(num_heads));

  // Add required inputs — 3D packed (B, T, H*D)
  std::vector<int64_t> qk_dims = {batch_size, seq_length, num_heads * head_dim_k};
  std::vector<int64_t> v_dims = {batch_size, seq_length, num_heads * head_dim_v};
  tester.AddInput<float>("query", qk_dims, query_3d);
  tester.AddInput<float>("key", qk_dims, key_3d);
  tester.AddInput<float>("value", v_dims, value_3d);

  // Optional: past_state (4D, same format as before)
  if (initial_state != nullptr) {
    std::vector<int64_t> state_dims = {batch_size, num_heads, head_dim_k, head_dim_v};
    tester.AddInput<float>("past_state", state_dims, *initial_state);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  // Optional: decay — convert from (B,H,T[,dk]) to (B,T,H[*dk])
  if (decay != nullptr) {
    if (decay_broadcast_dk) {
      // (B,H,T) → (B,T,H)
      auto decay_3d = TransposeBHT_to_BTH(*decay, batch_size, num_heads, seq_length);
      std::vector<int64_t> decay_dims = {batch_size, seq_length, num_heads};
      tester.AddInput<float>("decay", decay_dims, decay_3d);
    } else {
      // (B,H,T,dk) → (B,T,H*dk)
      auto decay_3d = PackBHTD_to_BTHD(*decay, batch_size, num_heads, seq_length, head_dim_k);
      std::vector<int64_t> decay_dims = {batch_size, seq_length, num_heads * head_dim_k};
      tester.AddInput<float>("decay", decay_dims, decay_3d);
    }
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  // Optional: beta — convert from (B*H*T) flat to (B,T,H)
  if (beta_data != nullptr) {
    auto beta_3d = TransposeBHT_to_BTH(*beta_data, batch_size, num_heads, seq_length);
    std::vector<int64_t> beta_dims = {batch_size, seq_length, num_heads};
    tester.AddInput<float>("beta", beta_dims, beta_3d);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  // Add outputs — output is 3D packed, state is 4D
  std::vector<int64_t> out_dims = {batch_size, seq_length, num_heads * head_dim_v};
  std::vector<int64_t> state_dims = {batch_size, num_heads, head_dim_k, head_dim_v};
  tester.AddOutput<float>("output", out_dims, output_3d, false, 0.005f, 0.005f);
  tester.AddOutput<float>("present_state", state_dims, expected_state, false, 0.005f, 0.005f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

// ===========================================================================
// Test: Linear update rule (simplest - no decay, no beta)
// ===========================================================================
TEST(ContribOpLinearAttentionTest, LinearRule_SingleToken) {
  const int B = 1, H = 1, T = 1, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {1.0f, 0.0f, 0.5f, -0.5f};
  std::vector<float> key = {0.5f, 0.5f, 0.0f, 1.0f};
  std::vector<float> value = {1.0f, 2.0f, 3.0f, 4.0f};

  RunLinearAttentionTest("linear", B, H, T, dk, dv, scale,
                         query, key, value,
                         nullptr, nullptr, nullptr);
}

TEST(ContribOpLinearAttentionTest, LinearRule_MultiToken) {
  const int B = 1, H = 1, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {
      1.0f, 0.0f, 0.5f, -0.5f,
      0.5f, 1.0f, -0.5f, 0.0f,
      0.0f, -1.0f, 1.0f, 0.5f};
  std::vector<float> key = {
      0.5f, 0.5f, 0.0f, 1.0f,
      1.0f, 0.0f, 1.0f, 0.5f,
      -0.5f, 1.0f, 0.5f, 0.0f};
  std::vector<float> value = {
      1.0f, 2.0f, 3.0f, 4.0f,
      2.0f, 1.0f, 0.0f, 3.0f,
      3.0f, 0.0f, 1.0f, 2.0f};

  RunLinearAttentionTest("linear", B, H, T, dk, dv, scale,
                         query, key, value,
                         nullptr, nullptr, nullptr);
}

TEST(ContribOpLinearAttentionTest, LinearRule_WithInitialState) {
  const int B = 1, H = 1, T = 2, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {
      1.0f, 0.0f, 0.5f, -0.5f,
      0.5f, 1.0f, -0.5f, 0.0f};
  std::vector<float> key = {
      0.5f, 0.5f, 0.0f, 1.0f,
      1.0f, 0.0f, 1.0f, 0.5f};
  std::vector<float> value = {
      1.0f, 2.0f, 3.0f, 4.0f,
      2.0f, 1.0f, 0.0f, 3.0f};

  // Non-zero initial state
  std::vector<float> initial_state(dk * dv, 0.1f);

  RunLinearAttentionTest("linear", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, nullptr, nullptr);
}

// ===========================================================================
// Test: Gated update rule (decay, no beta)
// ===========================================================================
TEST(ContribOpLinearAttentionTest, GatedRule_SingleToken) {
  const int B = 1, H = 1, T = 1, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {1.0f, 0.0f, 0.5f, -0.5f};
  std::vector<float> key = {0.5f, 0.5f, 0.0f, 1.0f};
  std::vector<float> value = {1.0f, 2.0f, 3.0f, 4.0f};

  // Decay in log-space (small negative values for slight decay)
  std::vector<float> decay = {-0.1f, -0.2f, -0.05f, -0.15f};

  // Initial state (needed to see decay effect)
  std::vector<float> initial_state(dk * dv, 1.0f);

  RunLinearAttentionTest("gated", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, nullptr);
}

TEST(ContribOpLinearAttentionTest, GatedRule_MultiToken) {
  const int B = 1, H = 1, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {
      1.0f, 0.0f, 0.5f, -0.5f,
      0.5f, 1.0f, -0.5f, 0.0f,
      0.0f, -1.0f, 1.0f, 0.5f};
  std::vector<float> key = {
      0.5f, 0.5f, 0.0f, 1.0f,
      1.0f, 0.0f, 1.0f, 0.5f,
      -0.5f, 1.0f, 0.5f, 0.0f};
  std::vector<float> value = {
      1.0f, 2.0f, 3.0f, 4.0f,
      2.0f, 1.0f, 0.0f, 3.0f,
      3.0f, 0.0f, 1.0f, 2.0f};
  std::vector<float> decay = {
      -0.1f, -0.2f, -0.05f, -0.15f,
      -0.2f, -0.1f, -0.3f, -0.05f,
      -0.05f, -0.15f, -0.1f, -0.2f};

  std::vector<float> initial_state(dk * dv, 0.5f);

  RunLinearAttentionTest("gated", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, nullptr);
}

// ===========================================================================
// Test: Delta update rule (no decay, uses beta)
// ===========================================================================
TEST(ContribOpLinearAttentionTest, DeltaRule_SingleToken) {
  const int B = 1, H = 1, T = 1, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {1.0f, 0.0f, 0.5f, -0.5f};
  std::vector<float> key = {0.5f, 0.5f, 0.0f, 1.0f};
  std::vector<float> value = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> beta = {0.8f};  // shape: (1,1,1,1)

  std::vector<float> initial_state(dk * dv, 0.5f);

  RunLinearAttentionTest("delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, nullptr, &beta);
}

TEST(ContribOpLinearAttentionTest, DeltaRule_MultiToken) {
  const int B = 1, H = 1, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {
      1.0f, 0.0f, 0.5f, -0.5f,
      0.5f, 1.0f, -0.5f, 0.0f,
      0.0f, -1.0f, 1.0f, 0.5f};
  std::vector<float> key = {
      0.5f, 0.5f, 0.0f, 1.0f,
      1.0f, 0.0f, 1.0f, 0.5f,
      -0.5f, 1.0f, 0.5f, 0.0f};
  std::vector<float> value = {
      1.0f, 2.0f, 3.0f, 4.0f,
      2.0f, 1.0f, 0.0f, 3.0f,
      3.0f, 0.0f, 1.0f, 2.0f};
  std::vector<float> beta = {0.8f, 0.6f, 0.9f};  // shape: (1,1,3,1)

  RunLinearAttentionTest("delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         nullptr, nullptr, &beta);
}

// ===========================================================================
// Test: GatedDelta update rule (full - decay + beta)
// ===========================================================================
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_SingleToken) {
  const int B = 1, H = 1, T = 1, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {1.0f, 0.0f, 0.5f, -0.5f};
  std::vector<float> key = {0.5f, 0.5f, 0.0f, 1.0f};
  std::vector<float> value = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> decay = {-0.1f, -0.2f, -0.05f, -0.15f};
  std::vector<float> beta = {0.8f};

  std::vector<float> initial_state(dk * dv, 1.0f);

  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, &beta);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_MultiToken) {
  const int B = 1, H = 1, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {
      1.0f, 0.0f, 0.5f, -0.5f,
      0.5f, 1.0f, -0.5f, 0.0f,
      0.0f, -1.0f, 1.0f, 0.5f};
  std::vector<float> key = {
      0.5f, 0.5f, 0.0f, 1.0f,
      1.0f, 0.0f, 1.0f, 0.5f,
      -0.5f, 1.0f, 0.5f, 0.0f};
  std::vector<float> value = {
      1.0f, 2.0f, 3.0f, 4.0f,
      2.0f, 1.0f, 0.0f, 3.0f,
      3.0f, 0.0f, 1.0f, 2.0f};
  std::vector<float> decay = {
      -0.1f, -0.2f, -0.05f, -0.15f,
      -0.2f, -0.1f, -0.3f, -0.05f,
      -0.05f, -0.15f, -0.1f, -0.2f};
  std::vector<float> beta = {0.8f, 0.6f, 0.9f};

  std::vector<float> initial_state(dk * dv, 0.5f);

  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, &beta);
}

// ===========================================================================
// Test: Gated rule with B,H,T decay (broadcast across dk)
// ===========================================================================
TEST(ContribOpLinearAttentionTest, GatedRule_BroadcastDecay) {
  const int B = 1, H = 1, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {
      1.0f, 0.0f, 0.5f, -0.5f,
      0.5f, 1.0f, -0.5f, 0.0f,
      0.0f, -1.0f, 1.0f, 0.5f};
  std::vector<float> key = {
      0.5f, 0.5f, 0.0f, 1.0f,
      1.0f, 0.0f, 1.0f, 0.5f,
      -0.5f, 1.0f, 0.5f, 0.0f};
  std::vector<float> value = {
      1.0f, 2.0f, 3.0f, 4.0f,
      2.0f, 1.0f, 0.0f, 3.0f,
      3.0f, 0.0f, 1.0f, 2.0f};
  // Decay shape: (B, H, T) = (1, 1, 3) — one scalar per token
  std::vector<float> decay = {-0.1f, -0.2f, -0.05f};

  std::vector<float> initial_state(dk * dv, 0.5f);

  RunLinearAttentionTest("gated", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, nullptr);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_BroadcastDecay) {
  const int B = 1, H = 1, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query = {
      1.0f, 0.0f, 0.5f, -0.5f,
      0.5f, 1.0f, -0.5f, 0.0f,
      0.0f, -1.0f, 1.0f, 0.5f};
  std::vector<float> key = {
      0.5f, 0.5f, 0.0f, 1.0f,
      1.0f, 0.0f, 1.0f, 0.5f,
      -0.5f, 1.0f, 0.5f, 0.0f};
  std::vector<float> value = {
      1.0f, 2.0f, 3.0f, 4.0f,
      2.0f, 1.0f, 0.0f, 3.0f,
      3.0f, 0.0f, 1.0f, 2.0f};
  // Decay shape: (B, H, T) = (1, 1, 3) — one scalar per token
  std::vector<float> decay = {-0.1f, -0.2f, -0.05f};
  std::vector<float> beta = {0.8f, 0.6f, 0.9f};

  std::vector<float> initial_state(dk * dv, 0.5f);

  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, &beta);
}

// ===========================================================================
// Test: Multi-batch, multi-head
// ===========================================================================
TEST(ContribOpLinearAttentionTest, LinearRule_MultiBatchMultiHead) {
  const int B = 2, H = 2, T = 2, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  // Total: B*H*T*dk = 2*2*2*4 = 32 values for q/k, B*H*T*dv = 32 for v
  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);

  // Fill with deterministic pattern
  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = std::sin(static_cast<float>(i) * 0.3f);
    key[i] = std::cos(static_cast<float>(i) * 0.5f);
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = std::sin(static_cast<float>(i) * 0.7f + 1.0f);
  }

  RunLinearAttentionTest("linear", B, H, T, dk, dv, scale,
                         query, key, value,
                         nullptr, nullptr, nullptr);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_MultiBatchMultiHead) {
  const int B = 2, H = 2, T = 2, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  std::vector<float> decay(B * H * T * dk);
  std::vector<float> beta(B * H * T);

  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = std::sin(static_cast<float>(i) * 0.3f);
    key[i] = std::cos(static_cast<float>(i) * 0.5f);
    decay[i] = -0.1f - 0.1f * std::sin(static_cast<float>(i) * 0.2f);
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = std::sin(static_cast<float>(i) * 0.7f + 1.0f);
  }
  for (int i = 0; i < B * H * T; i++) {
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i));
  }

  std::vector<float> initial_state(B * H * dk * dv, 0.1f);

  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, &beta);
}

// ===========================================================================
// Test: Default scale (should use 1/sqrt(dk))
// ===========================================================================
TEST(ContribOpLinearAttentionTest, LinearRule_DefaultScale) {
  const int B = 1, H = 1, T = 1, dk = 4, dv = 4;

  std::vector<float> query = {1.0f, 0.0f, 0.5f, -0.5f};
  std::vector<float> key = {0.5f, 0.5f, 0.0f, 1.0f};
  std::vector<float> value = {1.0f, 2.0f, 3.0f, 4.0f};

  // Compute with explicit scale for reference
  float actual_scale = 1.0f / std::sqrt(static_cast<float>(dk));
  std::vector<float> expected_output, expected_state;
  LinearAttentionReference("linear", B, H, T, dk, dv, actual_scale,
                           query, key, value, nullptr, nullptr, nullptr,
                           expected_output, expected_state);

  bool enable_webgpu = (nullptr != DefaultWebGpuExecutionProvider().get());
  if (!enable_webgpu) {
    return;
  }

  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", std::string("linear"));
  tester.AddAttribute<int64_t>("q_num_heads", static_cast<int64_t>(H));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(H));
  // Don't set scale — use default (0.0 triggers 1/sqrt(dk))

  // Convert to 3D packed for B=1, H=1 (flat data is identical)
  std::vector<int64_t> qk_dims = {B, T, H * dk};
  std::vector<int64_t> v_dims = {B, T, H * dv};
  tester.AddInput<float>("query", qk_dims, query);
  tester.AddInput<float>("key", qk_dims, key);
  tester.AddInput<float>("value", v_dims, value);
  tester.AddOptionalInputEdge<float>();  // past_state
  tester.AddOptionalInputEdge<float>();  // decay
  tester.AddOptionalInputEdge<float>();  // beta

  std::vector<int64_t> out_dims = {B, T, H * dv};
  std::vector<int64_t> state_dims = {B, H, dk, dv};
  tester.AddOutput<float>("output", out_dims, expected_output, false, 0.005f, 0.005f);
  tester.AddOutput<float>("present_state", state_dims, expected_state, false, 0.005f, 0.005f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// ===========================================================================
// Test: Longer sequence
// ===========================================================================
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_LongerSequence) {
  const int B = 1, H = 2, T = 16, dk = 8, dv = 8;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  std::vector<float> decay(B * H * T * dk);
  std::vector<float> beta(B * H * T);

  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.1f * std::sin(static_cast<float>(i) * 0.13f);
    key[i] = 0.1f * std::cos(static_cast<float>(i) * 0.17f);
    decay[i] = -0.05f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.07f));
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.1f * std::sin(static_cast<float>(i) * 0.23f + 0.5f);
  }
  for (int i = 0; i < B * H * T; i++) {
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }

  std::vector<float> initial_state(B * H * dk * dv, 0.01f);

  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, &beta);
}

// Test with Qwen3.5-like dimensions: dk=128, dv=128, broadcast decay
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_Qwen35Like) {
  const int B = 1, H = 2, T = 8, dk = 128, dv = 128;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  // Broadcast decay: (B, H, T) — one scalar per head per token, like Qwen3.5
  std::vector<float> decay(B * H * T);
  std::vector<float> beta(B * H * T);

  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.05f * std::sin(static_cast<float>(i) * 0.013f);
    key[i] = 0.05f * std::cos(static_cast<float>(i) * 0.017f);
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.05f * std::sin(static_cast<float>(i) * 0.023f + 0.5f);
  }
  for (int i = 0; i < B * H * T; i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.3f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }

  std::vector<float> initial_state(B * H * dk * dv, 0.01f);

  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, &beta);
}

// Test with non-power-of-2 dk to trigger workgroup padding bug
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_NonPowerOf2DK) {
  const int B = 1, H = 1, T = 3, dk = 3, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  std::vector<float> decay(B * H * T);
  std::vector<float> beta(B * H * T);

  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.5f * std::sin(static_cast<float>(i) * 0.3f);
    key[i] = 0.5f * std::cos(static_cast<float>(i) * 0.5f);
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.5f * std::sin(static_cast<float>(i) * 0.7f + 1.0f);
  }
  for (int i = 0; i < B * H * T; i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.3f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }

  std::vector<float> initial_state(B * H * dk * dv, 0.5f);

  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, &beta);
}

}  // namespace test
}  // namespace onnxruntime

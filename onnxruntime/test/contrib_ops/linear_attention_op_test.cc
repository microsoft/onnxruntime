// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include <cmath>
#include <vector>

namespace onnxruntime {
namespace test {

namespace {

// Helper to run a LinearAttention test
void RunLinearAttentionTest(
    const std::vector<float>& query_data,          // (B, H, T, d_k)
    const std::vector<float>& key_data,            // (B, H, T, d_k)
    const std::vector<float>& value_data,          // (B, H, T, d_v)
    const std::vector<float>& past_state_data,     // (B, H, d_k, d_v) or empty
    const std::vector<float>& decay_data,          // (B, H, T, decay_dim) or empty
    const std::vector<float>& beta_data,           // (B, H, T, 1) or empty
    const std::vector<float>& expected_output,     // (B, H, T, d_v)
    const std::vector<float>& expected_state,      // (B, H, d_k, d_v)
    int batch_size,
    int num_heads,
    int seq_len,
    int key_dim,
    int value_dim,
    const std::string& update_rule = "gated_delta",
    float scale = 0.0f,
    int decay_key_dim = 0,
    float tolerance = 1e-4f) {
  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", update_rule);
  if (scale != 0.0f) {
    tester.AddAttribute<float>("scale", scale);
  }

  tester.AddInput<float>("query", {batch_size, num_heads, seq_len, key_dim}, query_data);
  tester.AddInput<float>("key", {batch_size, num_heads, seq_len, key_dim}, key_data);
  tester.AddInput<float>("value", {batch_size, num_heads, seq_len, value_dim}, value_data);

  if (!past_state_data.empty()) {
    tester.AddInput<float>("past_state", {batch_size, num_heads, key_dim, value_dim}, past_state_data);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  if (!decay_data.empty()) {
    int actual_decay_dim = (decay_key_dim > 0) ? decay_key_dim : key_dim;
    tester.AddInput<float>("decay", {batch_size, num_heads, seq_len, actual_decay_dim}, decay_data);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  if (!beta_data.empty()) {
    tester.AddInput<float>("beta", {batch_size, num_heads, seq_len, 1}, beta_data);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  tester.AddOutput<float>("output", {batch_size, num_heads, seq_len, value_dim}, expected_output, false, tolerance);
  tester.AddOutput<float>("present_state", {batch_size, num_heads, key_dim, value_dim}, expected_state, false, tolerance);

  // Run on CPU
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Reference implementation for computing expected outputs
void ComputeLinearAttentionReference(
    const std::vector<float>& q_data,
    const std::vector<float>& k_data,
    const std::vector<float>& v_data,
    const std::vector<float>& past_state_data,
    const std::vector<float>& decay_data,
    const std::vector<float>& beta_data,
    std::vector<float>& output,
    std::vector<float>& state,
    int B, int H, int T, int dk, int dv,
    const std::string& rule,
    float scale,
    int decay_key_dim) {
  bool is_gated = (rule == "gated" || rule == "gated_delta");
  bool is_delta = (rule == "delta" || rule == "gated_delta");

  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(dk));
  }

  state.resize(static_cast<size_t>(B * H * dk * dv));
  output.resize(static_cast<size_t>(B * H * T * dv));

  // Initialize state
  if (!past_state_data.empty()) {
    state = past_state_data;
  } else {
    std::fill(state.begin(), state.end(), 0.0f);
  }

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      float* S = state.data() + (b * H + h) * dk * dv;

      for (int t = 0; t < T; ++t) {
        const float* q_t = q_data.data() + ((b * H + h) * T + t) * dk;
        const float* k_t = k_data.data() + ((b * H + h) * T + t) * dk;
        const float* v_t = v_data.data() + ((b * H + h) * T + t) * dv;
        float* o_t = output.data() + ((b * H + h) * T + t) * dv;

        // Decay
        if (is_gated) {
          for (int i = 0; i < dk; ++i) {
            int decay_idx = ((b * H + h) * T + t) * decay_key_dim +
                            ((decay_key_dim == 1) ? 0 : i);
            float g = std::exp(decay_data[static_cast<size_t>(decay_idx)]);
            for (int j = 0; j < dv; ++j) {
              S[i * dv + j] *= g;
            }
          }
        }

        if (is_delta) {
          // retrieved = S^T k
          std::vector<float> retrieved(static_cast<size_t>(dv), 0.0f);
          for (int i = 0; i < dk; ++i) {
            for (int j = 0; j < dv; ++j) {
              retrieved[static_cast<size_t>(j)] += S[i * dv + j] * k_t[i];
            }
          }

          float beta_val = beta_data[static_cast<size_t>((b * H + h) * T + t)];
          for (int i = 0; i < dk; ++i) {
            for (int j = 0; j < dv; ++j) {
              float delta = beta_val * (v_t[j] - retrieved[static_cast<size_t>(j)]);
              S[i * dv + j] += k_t[i] * delta;
            }
          }
        } else {
          for (int i = 0; i < dk; ++i) {
            for (int j = 0; j < dv; ++j) {
              S[i * dv + j] += k_t[i] * v_t[j];
            }
          }
        }

        // Output
        for (int j = 0; j < dv; ++j) {
          float sum = 0.0f;
          for (int i = 0; i < dk; ++i) {
            sum += q_t[i] * S[i * dv + j];
          }
          o_t[j] = sum * scale;
        }
      }
    }
  }
}

}  // namespace

// Test 1: Linear mode (vanilla linear attention), single token
TEST(LinearAttentionTest, LinearMode_SingleToken) {
  int B = 1, H = 1, T = 1, dk = 2, dv = 2;
  float scale = 1.0f;

  std::vector<float> q = {1.0f, 0.5f};
  std::vector<float> k = {0.5f, 1.0f};
  std::vector<float> v = {1.0f, 2.0f};

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, {}, {}, {}, expected_output, expected_state,
                                  B, H, T, dk, dv, "linear", scale, 0);

  RunLinearAttentionTest(q, k, v, {}, {}, {}, expected_output, expected_state,
                         B, H, T, dk, dv, "linear", scale);
}

// Test 2: Linear mode with past state
TEST(LinearAttentionTest, LinearMode_WithPastState) {
  int B = 1, H = 1, T = 1, dk = 2, dv = 2;
  float scale = 1.0f;

  std::vector<float> q = {1.0f, 0.0f};
  std::vector<float> k = {1.0f, 0.0f};
  std::vector<float> v = {1.0f, 1.0f};
  std::vector<float> past_state = {0.5f, 0.5f, 0.5f, 0.5f};  // (1,1,2,2)

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, past_state, {}, {}, expected_output, expected_state,
                                  B, H, T, dk, dv, "linear", scale, 0);

  RunLinearAttentionTest(q, k, v, past_state, {}, {}, expected_output, expected_state,
                         B, H, T, dk, dv, "linear", scale);
}

// Test 3: Gated mode with scalar decay
TEST(LinearAttentionTest, GatedMode_ScalarDecay) {
  int B = 1, H = 1, T = 1, dk = 2, dv = 2;
  float scale = 1.0f;

  std::vector<float> q = {1.0f, 1.0f};
  std::vector<float> k = {0.5f, 0.5f};
  std::vector<float> v = {1.0f, 0.0f};
  std::vector<float> past_state = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> decay = {-0.5f};  // scalar decay (B, H, T, 1)

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, past_state, decay, {}, expected_output, expected_state,
                                  B, H, T, dk, dv, "gated", scale, 1);

  RunLinearAttentionTest(q, k, v, past_state, decay, {}, expected_output, expected_state,
                         B, H, T, dk, dv, "gated", scale, 1);
}

// Test 4: Delta mode
TEST(LinearAttentionTest, DeltaMode_SingleToken) {
  int B = 1, H = 1, T = 1, dk = 2, dv = 2;
  float scale = 1.0f;

  std::vector<float> q = {1.0f, 0.0f};
  std::vector<float> k = {1.0f, 0.0f};
  std::vector<float> v = {2.0f, 3.0f};
  std::vector<float> past_state = {1.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> beta = {0.5f};

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, past_state, {}, beta, expected_output, expected_state,
                                  B, H, T, dk, dv, "delta", scale, 0);

  RunLinearAttentionTest(q, k, v, past_state, {}, beta, expected_output, expected_state,
                         B, H, T, dk, dv, "delta", scale);
}

// Test 5: Gated Delta mode (the most complex variant)
TEST(LinearAttentionTest, GatedDeltaMode_SingleToken) {
  int B = 1, H = 1, T = 1, dk = 2, dv = 2;
  float scale = 1.0f;

  std::vector<float> q = {1.0f, 0.5f};
  std::vector<float> k = {0.5f, 1.0f};
  std::vector<float> v = {2.0f, 1.0f};
  std::vector<float> past_state = {1.0f, 0.5f, 0.5f, 1.0f};
  std::vector<float> decay = {-0.1f, -0.2f};  // per-key-dim decay
  std::vector<float> beta = {0.8f};

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, past_state, decay, beta, expected_output, expected_state,
                                  B, H, T, dk, dv, "gated_delta", scale, 2);

  RunLinearAttentionTest(q, k, v, past_state, decay, beta, expected_output, expected_state,
                         B, H, T, dk, dv, "gated_delta", scale, 2);
}

// Test 6: Multiple tokens (sequence)
TEST(LinearAttentionTest, LinearMode_MultipleTokens) {
  int B = 1, H = 1, T = 3, dk = 2, dv = 2;
  float scale = 1.0f;

  // q, k, v for 3 tokens: each (1, 1, 3, 2)
  std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f};
  std::vector<float> v = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, {}, {}, {}, expected_output, expected_state,
                                  B, H, T, dk, dv, "linear", scale, 0);

  RunLinearAttentionTest(q, k, v, {}, {}, {}, expected_output, expected_state,
                         B, H, T, dk, dv, "linear", scale);
}

// Test 7: Multiple heads
TEST(LinearAttentionTest, LinearMode_MultipleHeads) {
  int B = 1, H = 2, T = 1, dk = 2, dv = 2;
  float scale = 1.0f;

  // (1, 2, 1, 2) = 4 elements per tensor
  std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, {}, {}, {}, expected_output, expected_state,
                                  B, H, T, dk, dv, "linear", scale, 0);

  RunLinearAttentionTest(q, k, v, {}, {}, {}, expected_output, expected_state,
                         B, H, T, dk, dv, "linear", scale);
}

// Test 8: Batch > 1
TEST(LinearAttentionTest, LinearMode_MultipleBatches) {
  int B = 2, H = 1, T = 1, dk = 2, dv = 2;
  float scale = 1.0f;

  // (2, 1, 1, 2) = 4 elements per tensor
  std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> k = {0.5f, 0.5f, 1.0f, 0.0f};
  std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, {}, {}, {}, expected_output, expected_state,
                                  B, H, T, dk, dv, "linear", scale, 0);

  RunLinearAttentionTest(q, k, v, {}, {}, {}, expected_output, expected_state,
                         B, H, T, dk, dv, "linear", scale);
}

// Test 9: Default scale (1/sqrt(d_k))
TEST(LinearAttentionTest, LinearMode_DefaultScale) {
  int B = 1, H = 1, T = 1, dk = 4, dv = 2;
  // scale = 0.0 means use 1/sqrt(4) = 0.5

  std::vector<float> q = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> k = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> v = {2.0f, 4.0f};

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, {}, {}, {}, expected_output, expected_state,
                                  B, H, T, dk, dv, "linear", 0.0f, 0);

  RunLinearAttentionTest(q, k, v, {}, {}, {}, expected_output, expected_state,
                         B, H, T, dk, dv, "linear", 0.0f);
}

// Test 10: Gated Delta mode with multiple tokens and past state
TEST(LinearAttentionTest, GatedDeltaMode_MultiTokenWithState) {
  int B = 1, H = 1, T = 2, dk = 2, dv = 2;
  float scale = 1.0f;

  std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> k = {0.5f, 0.5f, 1.0f, 0.0f};
  std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> past_state = {0.5f, 0.5f, 0.5f, 0.5f};
  std::vector<float> decay = {-0.1f, -0.2f, -0.3f, -0.1f};  // (1,1,2,2)
  std::vector<float> beta = {0.5f, 0.8f};                     // (1,1,2,1)

  std::vector<float> expected_output, expected_state;
  ComputeLinearAttentionReference(q, k, v, past_state, decay, beta, expected_output, expected_state,
                                  B, H, T, dk, dv, "gated_delta", scale, 2);

  RunLinearAttentionTest(q, k, v, past_state, decay, beta, expected_output, expected_state,
                         B, H, T, dk, dv, "gated_delta", scale, 2);
}

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

using namespace onnxruntime::test;

namespace linear_attention {

enum class UpdateRule {
  kLinear,       // S = S + k^T v
  kGated,        // S = exp(g) * S + k^T v
  kDelta,        // S = S + k^T ((v - S^T k) * beta)
  kGatedDelta,   // S = exp(g) * S + k^T ((v - exp(g) * S^T k) * beta)
};

/// 4D tensor accessor with (d0, d1, d2, d3) layout (row-major).
template <typename T>
struct Tensor4D {
  T* data;
  int d0, d1, d2, d3;

  Tensor4D(T* data, int d0, int d1, int d2, int d3)
      : data(data), d0(d0), d1(d1), d2(d2), d3(d3) {}

  T& operator()(int i0, int i1, int i2, int i3) {
    return data[((i0 * d1 + i1) * d2 + i2) * d3 + i3];
  }
  const T& operator()(int i0, int i1, int i2, int i3) const {
    return data[((i0 * d1 + i1) * d2 + i2) * d3 + i3];
  }

  int size() const { return d0 * d1 * d2 * d3; }
};

/// 3D tensor accessor with (d0, d1, d2) layout (row-major).
template <typename T>
struct Tensor3D {
  T* data;
  int d0, d1, d2;

  Tensor3D(T* data, int d0, int d1, int d2)
      : data(data), d0(d0), d1(d1), d2(d2) {}

  T& operator()(int i0, int i1, int i2) {
    return data[(i0 * d1 + i1) * d2 + i2];
  }
  const T& operator()(int i0, int i1, int i2) const {
    return data[(i0 * d1 + i1) * d2 + i2];
  }

  int size() const { return d0 * d1 * d2; }
};

/// Compute one recurrence step for a single (batch, head) pair.
///
/// state:     (d_k, d_v)  — carried across time steps (modified in place)
/// q_t:       (d_k,)      — query for this step
/// k_t:       (d_k,)      — key for this step
/// v_t:       (d_v,)      — value for this step
/// decay_t:   scalar      — log-space decay gate
/// beta_t:    scalar      — update rate
/// output_t:  (d_v,)      — output for this step (written)
template <typename T>
inline void recurrence_step(
    T* state,          // (d_k, d_v), modified in place
    const T* q_t,      // (d_k,)
    const T* k_t,      // (d_k,)
    const T* v_t,      // (d_v,)
    T decay_t,         // scalar (log-space)
    T beta_t,          // scalar
    T* output_t,       // (d_v,), written
    int d_k,
    int d_v,
    UpdateRule rule) {
  const bool uses_decay = (rule == UpdateRule::kGated || rule == UpdateRule::kGatedDelta);
  const bool uses_beta = (rule == UpdateRule::kDelta || rule == UpdateRule::kGatedDelta);

  // 1. State decay: state *= exp(decay)
  T g_exp = T(1);
  if (uses_decay) {
    g_exp = std::exp(decay_t);
    for (int i = 0; i < d_k * d_v; ++i) {
      state[i] *= g_exp;
    }
  }

  // 2. Retrieval: retrieval[j] = sum_i k[i] * state[i, j]  (k @ state)
  //    This computes k_row (1, d_k) @ state (d_k, d_v) -> (1, d_v)
  std::vector<T> retrieval(d_v, T(0));
  for (int i = 0; i < d_k; ++i) {
    for (int j = 0; j < d_v; ++j) {
      retrieval[j] += k_t[i] * state[i * d_v + j];
    }
  }

  // 3. Compute delta
  std::vector<T> delta(d_v);
  if (uses_beta) {
    // delta = (v - retrieval) * beta
    for (int j = 0; j < d_v; ++j) {
      delta[j] = (v_t[j] - retrieval[j]) * beta_t;
    }
  } else {
    // delta = v
    std::copy(v_t, v_t + d_v, delta.data());
  }

  // 4. State update: state += k^T @ delta  (outer product)
  //    state[i, j] += k[i] * delta[j]
  for (int i = 0; i < d_k; ++i) {
    for (int j = 0; j < d_v; ++j) {
      state[i * d_v + j] += k_t[i] * delta[j];
    }
  }

  // 5. Output: output[j] = sum_i q[i] * new_state[i, j]  (q @ new_state)
  std::fill(output_t, output_t + d_v, T(0));
  for (int i = 0; i < d_k; ++i) {
    for (int j = 0; j < d_v; ++j) {
      output_t[j] += q_t[i] * state[i * d_v + j];
    }
  }
}

/// Expand Q/K heads for GQA: repeat each head `ratio` times.
///
/// src:  (B, H_kv, T, d)
/// dst:  (B, H, T, d)       where H = H_kv * ratio
template <typename T>
inline void expand_kv_heads(
    const T* src, T* dst,
    int B, int H_kv, int T_len, int d, int ratio) {
  if (ratio == 1) {
    std::memcpy(dst, src, B * H_kv * T_len * d * sizeof(T));
    return;
  }
  for (int b = 0; b < B; ++b) {
    for (int h_kv = 0; h_kv < H_kv; ++h_kv) {
      const T* src_head = src + ((b * H_kv + h_kv) * T_len) * d;
      for (int r = 0; r < ratio; ++r) {
        int h = h_kv * ratio + r;
        T* dst_head = dst + ((b * (H_kv * ratio) + h) * T_len) * d;
        std::memcpy(dst_head, src_head, T_len * d * sizeof(T));
      }
    }
  }
}

/// Run the full LinearAttention operator.
///
/// query:         (B, H_kv, T, d_k)  — pre-scaled by 1/sqrt(d_k)
/// key:           (B, H_kv, T, d_k)  — L2-normalized
/// value:         (B, H, T, d_v)
/// past_state:    (B, H, d_k, d_v)   — recurrent state from previous chunk
/// decay:         (B, H, T)           — log-space decay gate
/// beta:          (B, H, T)           — sigmoid update rate
/// output:        (B, H, T, d_v)     — attention output     [written]
/// present_state: (B, H, d_k, d_v)   — updated state        [written]
///
/// H must be divisible by H_kv (GQA ratio = H / H_kv).
template <typename T>
void linear_attention_forward(
    const T* query,         // (B, H_kv, T, d_k)
    const T* key,           // (B, H_kv, T, d_k)
    const T* value,         // (B, H, T, d_v)
    const T* past_state,    // (B, H, d_k, d_v)
    const T* decay,         // (B, H, T)
    const T* beta,          // (B, H, T)
    T* output,              // (B, H, T, d_v)
    T* present_state,       // (B, H, d_k, d_v)
    int B,
    int H_kv,
    int H,
    int T_len,
    int d_k,
    int d_v,
    float scale,
    UpdateRule rule = UpdateRule::kGatedDelta) {
  assert(H % H_kv == 0 && "H must be divisible by H_kv for GQA");
  const int ratio = H / H_kv;

  // --- GQA: expand Q/K heads to match V head count ---
  std::vector<T> q_expanded(B * H * T_len * d_k);
  std::vector<T> k_expanded(B * H * T_len * d_k);
  expand_kv_heads(query, q_expanded.data(), B, H_kv, T_len, d_k, ratio);
  expand_kv_heads(key, k_expanded.data(), B, H_kv, T_len, d_k, ratio);

  // Accessors for expanded Q/K
  Tensor4D<const T> Q(q_expanded.data(), B, H, T_len, d_k);
  Tensor4D<const T> K(k_expanded.data(), B, H, T_len, d_k);
  Tensor4D<const T> V(value, B, H, T_len, d_v);
  Tensor3D<const T> D(decay, B, H, T_len);
  Tensor4D<T> O(output, B, H, T_len, d_v);

  // Copy past_state into present_state (we'll modify it in place)
  const int state_size = B * H * d_k * d_v;
  std::memcpy(present_state, past_state, state_size * sizeof(T));
  Tensor4D<T> S(present_state, B, H, d_k, d_v);

  // --- Sequential recurrence over time ---
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      T* state_bh = &S(b, h, 0, 0);  // (d_k, d_v) slice

      for (int t = 0; t < T_len; ++t) {
        recurrence_step(
            state_bh,
            &Q(b, h, t, 0),
            &K(b, h, t, 0),
            &V(b, h, t, 0),
            D(b, h, t),
            beta ? beta[((b * H + h) * T_len) + t] : T(0),
            &O(b, h, t, 0),
            d_k, d_v,
            rule);
      }
    }
  }
    // linear_attention_forward doesn't apply scale; apply it here.
  int output_size = B * H * T_len * d_v;
  for (int i = 0; i < output_size; ++i) {
    output[i] *= scale;
  }

}

/// Parse a string into an UpdateRule enum.
inline UpdateRule parse_update_rule(const std::string& s) {
  if (s == "linear") return UpdateRule::kLinear;
  if (s == "gated") return UpdateRule::kGated;
  if (s == "delta") return UpdateRule::kDelta;
  if (s == "gated_delta") return UpdateRule::kGatedDelta;
  assert(false && "Unknown update_rule");
  return UpdateRule::kGatedDelta;
}

}  // namespace linear_attention

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

    if (false && decay_broadcast_dk) {
      output.resize(batch_size * num_heads * seq_length * head_dim_v, 0.0f);
      final_state.resize(batch_size * num_heads * head_dim_k * head_dim_v, 0.0f);
      if (initial_state != nullptr) {
        final_state = *initial_state;
      }
      linear_attention::linear_attention_forward<float>(
        query.data(),             // (B, H_kv, T, d_k)
        key.data(),               // (B, H_kv, T, d_k)
        value.data(),             // (B, H, T, d_v)
        final_state.data(),       // (B, H, d_k, d_v)
        decay->data(),            // (B, H, T)
        beta ? beta->data() : nullptr,  // (B, H, T)
        output.data(),            // (B, H, T, d_v)
        final_state.data(),       // (B, H, d_k, d_v)
        batch_size,
        num_heads,
        num_heads,
        seq_length,
        head_dim_k,
        head_dim_v,
        scale,
        linear_attention::parse_update_rule(update_rule));
      return;
  }

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
  // Compute reference output
  std::vector<float> expected_output, expected_state;
  LinearAttentionReference(update_rule, batch_size, num_heads, seq_length,
                           head_dim_k, head_dim_v, scale,
                           query, key, value, initial_state, decay, beta_data,
                           expected_output, expected_state);

  bool enable_webgpu = (nullptr != DefaultWebGpuExecutionProvider().get());
  if (!enable_webgpu) {
    return;
  }

  int bht = batch_size * num_heads * seq_length;
  bool decay_broadcast_dk = (decay != nullptr && static_cast<int>(decay->size()) == bht);
  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", update_rule);
  tester.AddAttribute<float>("scale", scale);

  // Add required inputs
  std::vector<int64_t> qk_dims = {batch_size, num_heads, seq_length, head_dim_k};
  std::vector<int64_t> v_dims = {batch_size, num_heads, seq_length, head_dim_v};
  tester.AddInput<float>("query", qk_dims, query);
  tester.AddInput<float>("key", qk_dims, key);
  tester.AddInput<float>("value", v_dims, value);

  // Optional: initial_state
  if (initial_state != nullptr) {
    std::vector<int64_t> state_dims = {batch_size, num_heads, head_dim_k, head_dim_v};
    tester.AddInput<float>("initial_state", state_dims, *initial_state);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  // Optional: decay
  if (decay != nullptr) {
    if (decay_broadcast_dk) {
      std::vector<int64_t> decay_dims = {batch_size, num_heads, seq_length};
      tester.AddInput<float>("decay", decay_dims, *decay);
    } else {
      std::vector<int64_t> decay_dims = {batch_size, num_heads, seq_length, head_dim_k};
      tester.AddInput<float>("decay", decay_dims, *decay);
    }
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  // Optional: beta
  if (beta_data != nullptr) {
    std::vector<int64_t> beta_dims = {batch_size, num_heads, seq_length, 1};
    tester.AddInput<float>("beta", beta_dims, *beta_data);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  // Add outputs
  std::vector<int64_t> out_dims = {batch_size, num_heads, seq_length, head_dim_v};
  std::vector<int64_t> state_dims = {batch_size, num_heads, head_dim_k, head_dim_v};
  tester.AddOutput<float>("output", out_dims, expected_output, false, 0.005f, 0.005f);
  tester.AddOutput<float>("final_state", state_dims, expected_state, false, 0.005f, 0.005f);

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
  // Don't set scale — use default (0.0 triggers 1/sqrt(dk))

  std::vector<int64_t> qk_dims = {B, H, T, dk};
  std::vector<int64_t> v_dims = {B, H, T, dv};
  tester.AddInput<float>("query", qk_dims, query);
  tester.AddInput<float>("key", qk_dims, key);
  tester.AddInput<float>("value", v_dims, value);
  tester.AddOptionalInputEdge<float>();  // initial_state
  tester.AddOptionalInputEdge<float>();  // decay
  tester.AddOptionalInputEdge<float>();  // beta

  std::vector<int64_t> out_dims = {B, H, T, dv};
  std::vector<int64_t> state_dims = {B, H, dk, dv};
  tester.AddOutput<float>("output", out_dims, expected_output, false, 0.005f, 0.005f);
  tester.AddOutput<float>("final_state", state_dims, expected_state, false, 0.005f, 0.005f);

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

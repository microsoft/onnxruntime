// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

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

// GQA-aware reference implementation.
// Q has q_num_heads heads, K has n_k_heads heads, V/state have kv_num_heads heads.
// Standard GQA: q_num_heads >= kv_num_heads, heads_per_group = q_num_heads / kv_num_heads.
// K-to-KV sharing: kv_per_k_head = kv_num_heads / n_k_heads.
void LinearAttentionGQAReference(
    const std::string& update_rule,
    int batch_size, int q_num_heads, int kv_num_heads, int n_k_heads,
    int seq_length, int head_dim_k, int head_dim_v,
    float scale,
    const std::vector<float>& query,          // (B, q_num_heads, T, dk)
    const std::vector<float>& key,            // (B, n_k_heads, T, dk)
    const std::vector<float>& value,          // (B, kv_num_heads, T, dv)
    const std::vector<float>* initial_state,  // (B, kv_num_heads, dk, dv)
    const std::vector<float>* decay,          // (B, kv_num_heads, T[, dk])
    const std::vector<float>* beta,           // (B, kv_num_heads, T)
    std::vector<float>& output,               // (B, kv_num_heads, T, dv)
    std::vector<float>& final_state) {        // (B, kv_num_heads, dk, dv)
  int bht_kv = batch_size * kv_num_heads * seq_length;
  bool decay_broadcast_dk = (decay != nullptr && static_cast<int>(decay->size()) == bht_kv);
  int kv_per_k_head = kv_num_heads / n_k_heads;
  bool inverse_gqa = q_num_heads < kv_num_heads;
  int heads_per_group = inverse_gqa ? 0 : q_num_heads / kv_num_heads;
  // Output head count = max(q,kv): standard GQA emits one output per Q head (q_num_heads),
  // inverse GQA one per KV head. Matches the op's output_hidden = max(q,kv)*d_v.
  int out_heads = std::max(q_num_heads, kv_num_heads);

  final_state.resize(batch_size * kv_num_heads * head_dim_k * head_dim_v, 0.0f);
  output.resize(batch_size * out_heads * seq_length * head_dim_v, 0.0f);

  if (initial_state != nullptr) {
    final_state = *initial_state;
  }

  for (int b = 0; b < batch_size; b++) {
    for (int kv_h = 0; kv_h < kv_num_heads; kv_h++) {
      int k_head = kv_h / kv_per_k_head;

      auto state_offset = [&](int k, int v) {
        return ((b * kv_num_heads + kv_h) * head_dim_k + k) * head_dim_v + v;
      };

      for (int t = 0; t < seq_length; t++) {
        // Load k from the K-head that this KV-head maps to
        std::vector<float> k_vec(head_dim_k), v_vec(head_dim_v);
        int k_base = ((b * n_k_heads + k_head) * seq_length + t) * head_dim_k;
        for (int i = 0; i < head_dim_k; i++) k_vec[i] = key[k_base + i];
        int v_base = ((b * kv_num_heads + kv_h) * seq_length + t) * head_dim_v;
        for (int i = 0; i < head_dim_v; i++) v_vec[i] = value[v_base + i];

        // Step 1: Apply decay
        if (update_rule == "gated" || update_rule == "gated_delta") {
          for (int k = 0; k < head_dim_k; k++) {
            float exp_g;
            if (decay_broadcast_dk) {
              exp_g = std::exp((*decay)[(b * kv_num_heads + kv_h) * seq_length + t]);
            } else {
              exp_g = std::exp((*decay)[((b * kv_num_heads + kv_h) * seq_length + t) * head_dim_k + k]);
            }
            for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
              final_state[state_offset(k, v_idx)] *= exp_g;
            }
          }
        }

        // Step 2: Update state
        if (update_rule == "delta" || update_rule == "gated_delta") {
          std::vector<float> retrieved(head_dim_v, 0.0f);
          for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
            for (int k = 0; k < head_dim_k; k++) {
              retrieved[v_idx] += final_state[state_offset(k, v_idx)] * k_vec[k];
            }
          }
          int beta_idx = (b * kv_num_heads + kv_h) * seq_length + t;
          float beta_val = (*beta)[beta_idx];
          std::vector<float> delta(head_dim_v);
          for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
            delta[v_idx] = beta_val * (v_vec[v_idx] - retrieved[v_idx]);
          }
          for (int k = 0; k < head_dim_k; k++) {
            for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
              final_state[state_offset(k, v_idx)] += k_vec[k] * delta[v_idx];
            }
          }
        } else {
          for (int k = 0; k < head_dim_k; k++) {
            for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
              final_state[state_offset(k, v_idx)] += k_vec[k] * v_vec[v_idx];
            }
          }
        }

        // Step 3: Compute output
        if (!inverse_gqa) {
          // Standard GQA/MHA: one output per Q head
          for (int g = 0; g < heads_per_group; g++) {
            int q_h = kv_h * heads_per_group + g;
            int q_base = ((b * q_num_heads + q_h) * seq_length + t) * head_dim_k;
            for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
              float sum = 0.0f;
              for (int k = 0; k < head_dim_k; k++) {
                sum += final_state[state_offset(k, v_idx)] * query[q_base + k];
              }
              // Standard GQA: output head == q head, indexed with out_heads (= q_num_heads) stride.
              int out_idx = ((b * out_heads + q_h) * seq_length + t) * head_dim_v + v_idx;
              output[out_idx] = scale * sum;
            }
          }
        } else {
          // Inverse GQA: output indexed by kv_head, Q broadcast
          int q_h = kv_h * q_num_heads / kv_num_heads;
          int q_base = ((b * q_num_heads + q_h) * seq_length + t) * head_dim_k;
          for (int v_idx = 0; v_idx < head_dim_v; v_idx++) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim_k; k++) {
              sum += final_state[state_offset(k, v_idx)] * query[q_base + k];
            }
            int out_idx = ((b * out_heads + kv_h) * seq_length + t) * head_dim_v + v_idx;
            output[out_idx] = scale * sum;
          }
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

// Returns a WebGPU EP if it is available and has the LinearAttention kernel registered,
// or nullptr otherwise.
std::unique_ptr<IExecutionProvider> TryGetEpWithLinearAttention() {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    ep = DefaultCpuExecutionProvider();
  }

  auto kernel_registry = ep->GetKernelRegistry();
  if (kernel_registry) {
    const KernelCreateInfo* info = nullptr;
    KernelRegistry::TypeConstraintMap type_constraints;
    auto status = kernel_registry->TryFindKernel(
        ep->Type(), "LinearAttention", kMSDomain, 1,
        type_constraints, DefaultLoggingManager().DefaultLogger(), &info);
    if (!status.IsOK()) return nullptr;
  }
  return ep;
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
  auto ep = TryGetEpWithLinearAttention();
  if (!ep) {
    GTEST_SKIP() << "LinearAttention kernel not registered";
    return;
  }

  // Compute reference output (reference works in 4D layout)
  std::vector<float> expected_output_4d, expected_state;
  LinearAttentionReference(update_rule, batch_size, num_heads, seq_length,
                           head_dim_k, head_dim_v, scale,
                           query, key, value, initial_state, decay, beta_data,
                           expected_output_4d, expected_state);

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
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// GQA-aware test harness.
// Q: (B, q_num_heads, T, dk), K: (B, n_k_heads, T, dk), V: (B, kv_num_heads, T, dv)
void RunLinearAttentionGQATest(
    const std::string& update_rule,
    int batch_size, int q_num_heads, int kv_num_heads, int n_k_heads,
    int seq_length, int head_dim_k, int head_dim_v,
    float scale,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>* initial_state,
    const std::vector<float>* decay,
    const std::vector<float>* beta_data) {
  auto ep = TryGetEpWithLinearAttention();
  if (!ep) {
    GTEST_SKIP() << "LinearAttention kernel not registered";
    return;
  }

  std::vector<float> expected_output_4d, expected_state;
  LinearAttentionGQAReference(update_rule, batch_size, q_num_heads, kv_num_heads, n_k_heads,
                              seq_length, head_dim_k, head_dim_v, scale,
                              query, key, value, initial_state, decay, beta_data,
                              expected_output_4d, expected_state);

  int bht_kv = batch_size * kv_num_heads * seq_length;
  bool decay_broadcast_dk = (decay != nullptr && static_cast<int>(decay->size()) == bht_kv);

  // Output head count = max(q,kv), matching the op's output_hidden = max(q,kv)*d_v.
  int out_heads = std::max(q_num_heads, kv_num_heads);

  // Pack to 3D — each tensor uses its own head count
  auto query_3d = PackBHTD_to_BTHD(query, batch_size, q_num_heads, seq_length, head_dim_k);
  auto key_3d = PackBHTD_to_BTHD(key, batch_size, n_k_heads, seq_length, head_dim_k);
  auto value_3d = PackBHTD_to_BTHD(value, batch_size, kv_num_heads, seq_length, head_dim_v);
  auto output_3d = PackBHTD_to_BTHD(expected_output_4d, batch_size, out_heads, seq_length, head_dim_v);

  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", update_rule);
  tester.AddAttribute<float>("scale", scale);
  tester.AddAttribute<int64_t>("q_num_heads", static_cast<int64_t>(q_num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<float>("query", {batch_size, seq_length, q_num_heads * head_dim_k}, query_3d);
  tester.AddInput<float>("key", {batch_size, seq_length, n_k_heads * head_dim_k}, key_3d);
  tester.AddInput<float>("value", {batch_size, seq_length, kv_num_heads * head_dim_v}, value_3d);

  if (initial_state != nullptr) {
    tester.AddInput<float>("past_state", {batch_size, kv_num_heads, head_dim_k, head_dim_v}, *initial_state);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  if (decay != nullptr) {
    if (decay_broadcast_dk) {
      auto decay_3d = TransposeBHT_to_BTH(*decay, batch_size, kv_num_heads, seq_length);
      tester.AddInput<float>("decay", {batch_size, seq_length, kv_num_heads}, decay_3d);
    } else {
      auto decay_3d = PackBHTD_to_BTHD(*decay, batch_size, kv_num_heads, seq_length, head_dim_k);
      tester.AddInput<float>("decay", {batch_size, seq_length, kv_num_heads * head_dim_k}, decay_3d);
    }
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  if (beta_data != nullptr) {
    auto beta_3d = TransposeBHT_to_BTH(*beta_data, batch_size, kv_num_heads, seq_length);
    tester.AddInput<float>("beta", {batch_size, seq_length, kv_num_heads}, beta_3d);
  } else {
    tester.AddOptionalInputEdge<float>();
  }

  tester.AddOutput<float>("output", {batch_size, seq_length, out_heads * head_dim_v},
                          output_3d, false, 0.005f, 0.005f);
  tester.AddOutput<float>("present_state", {batch_size, kv_num_heads, head_dim_k, head_dim_v},
                          expected_state, false, 0.005f, 0.005f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#ifdef USE_CUDA
void RunCudaLinearAttentionShapeFailure(
    const std::vector<int64_t>& query_dims,
    const std::vector<int64_t>& key_dims,
    const std::vector<int64_t>& value_dims,
    const std::vector<int64_t>& output_dims,
    const std::vector<int64_t>& state_dims,
    const std::string& expected_error,
    const std::vector<int64_t>& decay_dims = {}) {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
    return;
  }

  auto element_count = [](const std::vector<int64_t>& dims) {
    size_t count = 1;
    for (int64_t dim : dims) {
      count *= static_cast<size_t>(dim);
    }
    return count;
  };

  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", "linear");
  tester.AddAttribute<int64_t>("q_num_heads", 2);
  tester.AddAttribute<int64_t>("kv_num_heads", 2);
  tester.AddInput<float>("query", query_dims, std::vector<float>(element_count(query_dims), 0.1f));
  tester.AddInput<float>("key", key_dims, std::vector<float>(element_count(key_dims), 0.1f));
  tester.AddInput<float>("value", value_dims, std::vector<float>(element_count(value_dims), 0.1f));
  tester.AddOptionalInputEdge<float>();
  if (decay_dims.empty()) {
    tester.AddOptionalInputEdge<float>();
  } else {
    tester.AddInput<float>("decay", decay_dims, std::vector<float>(element_count(decay_dims), 0.1f));
  }
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("output", output_dims, std::vector<float>(element_count(output_dims), 0.0f));
  tester.AddOutput<float>("present_state", state_dims, std::vector<float>(element_count(state_dims), 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectFailure, expected_error, {}, nullptr, &execution_providers);
}
#endif

}  // namespace
// ===========================================================================
#ifdef USE_CUDA
TEST(ContribOpLinearAttentionTest, CudaRejectsNonDivisibleQueryHidden) {
  RunCudaLinearAttentionShapeFailure({1, 1, 3}, {1, 1, 4}, {1, 1, 4},
                                     {1, 1, 4}, {1, 2, 1, 2}, "query last dim");
}

TEST(ContribOpLinearAttentionTest, CudaRejectsNonDivisibleValueHidden) {
  RunCudaLinearAttentionShapeFailure({1, 1, 4}, {1, 1, 4}, {1, 1, 3},
                                     {1, 1, 2}, {1, 2, 2, 1}, "value last dim");
}

TEST(ContribOpLinearAttentionTest, CudaRejectsInvalidDecayRank) {
  RunCudaLinearAttentionShapeFailure({1, 1, 4}, {1, 1, 4}, {1, 1, 4},
                                     {1, 1, 4}, {1, 2, 2, 2}, "decay must be rank 3", {1, 2});
}
#endif

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
  auto ep = TryGetEpWithLinearAttention();
  if (!ep) {
    GTEST_SKIP() << "LinearAttention kernel not registered on WebGPU EP (or EP not available)";
    return;
  }

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
  execution_providers.push_back(std::move(ep));
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

// ===========================================================================
// Tests: Larger dimensions exercising multi-tile vec4 path (tile_v > 1)
// ===========================================================================
TEST(ContribOpLinearAttentionTest, LinearRule_LargerDims) {
  const int B = 1, H = 2, T = 4, dk = 16, dv = 64;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);

  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.1f * std::sin(static_cast<float>(i) * 0.13f);
    key[i] = 0.1f * std::cos(static_cast<float>(i) * 0.17f);
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.1f * std::sin(static_cast<float>(i) * 0.23f + 0.5f);
  }

  RunLinearAttentionTest("linear", B, H, T, dk, dv, scale,
                         query, key, value,
                         nullptr, nullptr, nullptr);
}

TEST(ContribOpLinearAttentionTest, GatedRule_LargerDims) {
  const int B = 1, H = 2, T = 4, dk = 32, dv = 64;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  std::vector<float> decay(B * H * T * dk);

  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.1f * std::sin(static_cast<float>(i) * 0.13f);
    key[i] = 0.1f * std::cos(static_cast<float>(i) * 0.17f);
    decay[i] = -0.05f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.07f));
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.1f * std::sin(static_cast<float>(i) * 0.23f + 0.5f);
  }

  std::vector<float> initial_state(B * H * dk * dv, 0.01f);

  RunLinearAttentionTest("gated", B, H, T, dk, dv, scale,
                         query, key, value,
                         &initial_state, &decay, nullptr);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_LargerDims) {
  const int B = 2, H = 2, T = 4, dk = 32, dv = 64;
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

// ===========================================================================
// Tests: GQA (Grouped Query Attention) — q_num_heads != kv_num_heads
// ===========================================================================
// Tests: GQA — K has fewer heads than KV (n_k < kv_num_heads)
// Schema requires q_num_heads == kv_num_heads; K head count is derived from
// the key tensor shape. Multiple KV heads share one K head via kv_per_k_head.
// ===========================================================================

// Small K-GQA: q=kv=4, n_k=2 → each K head serves 2 KV heads
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_KGQA_Small) {
  const int B = 1, q_H = 4, kv_H = 4, n_k = 2, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * q_H * T * dk);
  std::vector<float> key(B * n_k * T * dk);
  std::vector<float> value(B * kv_H * T * dv);
  std::vector<float> decay(B * kv_H * T);  // broadcast
  std::vector<float> beta(B * kv_H * T);

  for (int i = 0; i < B * q_H * T * dk; i++) {
    query[i] = 0.5f * std::sin(static_cast<float>(i) * 0.13f);
  }
  for (int i = 0; i < B * n_k * T * dk; i++) {
    key[i] = 0.5f * std::cos(static_cast<float>(i) * 0.17f);
  }
  for (int i = 0; i < B * kv_H * T * dv; i++) {
    value[i] = 0.5f * std::sin(static_cast<float>(i) * 0.23f + 0.5f);
  }
  for (int i = 0; i < B * kv_H * T; i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.3f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }

  std::vector<float> initial_state(B * kv_H * dk * dv, 0.1f);

  RunLinearAttentionGQATest("gated_delta", B, q_H, kv_H, n_k, T, dk, dv, scale,
                            query, key, value,
                            &initial_state, &decay, &beta);
}

// Linear rule with K-GQA: q=kv=4, n_k=2
TEST(ContribOpLinearAttentionTest, LinearRule_KGQA) {
  const int B = 1, q_H = 4, kv_H = 4, n_k = 2, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * q_H * T * dk);
  std::vector<float> key(B * n_k * T * dk);
  std::vector<float> value(B * kv_H * T * dv);

  for (int i = 0; i < B * q_H * T * dk; i++) {
    query[i] = 0.5f * std::sin(static_cast<float>(i) * 0.13f);
  }
  for (int i = 0; i < B * n_k * T * dk; i++) {
    key[i] = 0.5f * std::cos(static_cast<float>(i) * 0.17f);
  }
  for (int i = 0; i < B * kv_H * T * dv; i++) {
    value[i] = 0.5f * std::sin(static_cast<float>(i) * 0.23f + 0.5f);
  }

  RunLinearAttentionGQATest("linear", B, q_H, kv_H, n_k, T, dk, dv, scale,
                            query, key, value,
                            nullptr, nullptr, nullptr);
}

// Qwen3.5 9B-like: q=kv=32, n_k=16 (K has half the heads),
// dk=128, dv=128, broadcast decay
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_Qwen35_KGQA) {
  const int B = 1, q_H = 32, kv_H = 32, n_k = 16, T = 4, dk = 128, dv = 128;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * q_H * T * dk);
  std::vector<float> key(B * n_k * T * dk);
  std::vector<float> value(B * kv_H * T * dv);
  std::vector<float> decay(B * kv_H * T);  // broadcast
  std::vector<float> beta(B * kv_H * T);

  for (int i = 0; i < B * q_H * T * dk; i++) {
    query[i] = 0.05f * std::sin(static_cast<float>(i) * 0.013f);
  }
  for (int i = 0; i < B * n_k * T * dk; i++) {
    key[i] = 0.05f * std::cos(static_cast<float>(i) * 0.017f);
  }
  for (int i = 0; i < B * kv_H * T * dv; i++) {
    value[i] = 0.05f * std::sin(static_cast<float>(i) * 0.023f + 0.5f);
  }
  for (int i = 0; i < B * kv_H * T; i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.3f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }

  std::vector<float> initial_state(B * kv_H * dk * dv, 0.01f);

  RunLinearAttentionGQATest("gated_delta", B, q_H, kv_H, n_k, T, dk, dv, scale,
                            query, key, value,
                            &initial_state, &decay, &beta);
}

// ===========================================================================
// Tests: Inverse GQA — q_num_heads < kv_num_heads
// Each KV head has its own output slot; Q is broadcast across KV groups.
// ===========================================================================

// Small inverse GQA: q=2, kv=4 → each Q head shared by 2 KV heads
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_InverseGQA_Small) {
  const int B = 1, q_H = 2, kv_H = 4, n_k = 4, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * q_H * T * dk);
  std::vector<float> key(B * n_k * T * dk);
  std::vector<float> value(B * kv_H * T * dv);
  std::vector<float> decay(B * kv_H * T);  // broadcast
  std::vector<float> beta(B * kv_H * T);

  for (int i = 0; i < B * q_H * T * dk; i++) {
    query[i] = 0.5f * std::sin(static_cast<float>(i) * 0.13f);
  }
  for (int i = 0; i < B * n_k * T * dk; i++) {
    key[i] = 0.5f * std::cos(static_cast<float>(i) * 0.17f);
  }
  for (int i = 0; i < B * kv_H * T * dv; i++) {
    value[i] = 0.5f * std::sin(static_cast<float>(i) * 0.23f + 0.5f);
  }
  for (int i = 0; i < B * kv_H * T; i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.3f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }

  std::vector<float> initial_state(B * kv_H * dk * dv, 0.1f);

  RunLinearAttentionGQATest("gated_delta", B, q_H, kv_H, n_k, T, dk, dv, scale,
                            query, key, value,
                            &initial_state, &decay, &beta);
}

// Linear rule with inverse GQA: q=2, kv=4
TEST(ContribOpLinearAttentionTest, LinearRule_InverseGQA) {
  const int B = 1, q_H = 2, kv_H = 4, n_k = 4, T = 3, dk = 4, dv = 4;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * q_H * T * dk);
  std::vector<float> key(B * n_k * T * dk);
  std::vector<float> value(B * kv_H * T * dv);

  for (int i = 0; i < B * q_H * T * dk; i++) {
    query[i] = 0.5f * std::sin(static_cast<float>(i) * 0.13f);
  }
  for (int i = 0; i < B * n_k * T * dk; i++) {
    key[i] = 0.5f * std::cos(static_cast<float>(i) * 0.17f);
  }
  for (int i = 0; i < B * kv_H * T * dv; i++) {
    value[i] = 0.5f * std::sin(static_cast<float>(i) * 0.23f + 0.5f);
  }

  RunLinearAttentionGQATest("linear", B, q_H, kv_H, n_k, T, dk, dv, scale,
                            query, key, value,
                            nullptr, nullptr, nullptr);
}

// Larger inverse GQA with K-head sharing: q=2, kv=8, n_k=4, dk=16, dv=64
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_InverseGQA_LargerDims) {
  const int B = 1, q_H = 2, kv_H = 8, n_k = 4, T = 4, dk = 16, dv = 64;
  float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * q_H * T * dk);
  std::vector<float> key(B * n_k * T * dk);
  std::vector<float> value(B * kv_H * T * dv);
  std::vector<float> decay(B * kv_H * T);  // broadcast
  std::vector<float> beta(B * kv_H * T);

  for (int i = 0; i < B * q_H * T * dk; i++) {
    query[i] = 0.1f * std::sin(static_cast<float>(i) * 0.013f);
  }
  for (int i = 0; i < B * n_k * T * dk; i++) {
    key[i] = 0.1f * std::cos(static_cast<float>(i) * 0.017f);
  }
  for (int i = 0; i < B * kv_H * T * dv; i++) {
    value[i] = 0.1f * std::sin(static_cast<float>(i) * 0.023f + 0.5f);
  }
  for (int i = 0; i < B * kv_H * T; i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.3f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }

  std::vector<float> initial_state(B * kv_H * dk * dv, 0.01f);

  RunLinearAttentionGQATest("gated_delta", B, q_H, kv_H, n_k, T, dk, dv, scale,
                            query, key, value,
                            &initial_state, &decay, &beta);
}

TEST(ContribOpLinearAttentionTest, LinearRule_MultiChunk) {
  const int B = 1, H = 2, T = 200, dk = 32, dv = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.1f * std::sin(static_cast<float>(i) * 0.013f);
    key[i] = 0.1f * std::cos(static_cast<float>(i) * 0.017f);
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.1f * std::sin(static_cast<float>(i) * 0.023f + 0.5f);
  }

  RunLinearAttentionTest("linear", B, H, T, dk, dv, scale,
                         query, key, value, nullptr, nullptr, nullptr);
}

TEST(ContribOpLinearAttentionTest, GatedRule_MultiChunk_PerDimDecay) {
  const int B = 1, H = 2, T = 150, dk = 32, dv = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  std::vector<float> decay(B * H * T * dk);
  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.1f * std::sin(static_cast<float>(i) * 0.013f);
    key[i] = 0.1f * std::cos(static_cast<float>(i) * 0.017f);
    decay[i] = -0.05f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.07f));
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.1f * std::sin(static_cast<float>(i) * 0.023f + 0.5f);
  }

  std::vector<float> initial_state(B * H * dk * dv, 0.01f);
  RunLinearAttentionTest("gated", B, H, T, dk, dv, scale,
                         query, key, value, &initial_state, &decay, nullptr);
}

TEST(ContribOpLinearAttentionTest, DeltaRule_MultiChunk) {
  const int B = 2, H = 2, T = 130, dk = 32, dv = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  std::vector<float> beta(B * H * T);
  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.1f * std::sin(static_cast<float>(i) * 0.013f);
    key[i] = 0.1f * std::cos(static_cast<float>(i) * 0.017f);
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.1f * std::sin(static_cast<float>(i) * 0.023f + 0.5f);
  }
  for (int i = 0; i < B * H * T; i++) {
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.031f);
  }

  RunLinearAttentionTest("delta", B, H, T, dk, dv, scale,
                         query, key, value, nullptr, nullptr, &beta);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_MultiChunk_Qwen35Like) {
  const int B = 1, H = 2, T = 192, dk = 128, dv = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  std::vector<float> decay(B * H * T);
  std::vector<float> beta(B * H * T);
  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.05f * std::sin(static_cast<float>(i) * 0.0013f);
    key[i] = 0.05f * std::cos(static_cast<float>(i) * 0.0017f);
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.05f * std::sin(static_cast<float>(i) * 0.0023f + 0.5f);
  }
  for (int i = 0; i < B * H * T; i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.03f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.031f);
  }

  std::vector<float> initial_state(B * H * dk * dv, 0.01f);
  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value, &initial_state, &decay, &beta);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_MultiChunk_PerDimDecay) {
  const int B = 1, H = 3, T = 175, dk = 64, dv = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * H * T * dk);
  std::vector<float> key(B * H * T * dk);
  std::vector<float> value(B * H * T * dv);
  std::vector<float> decay(B * H * T * dk);
  std::vector<float> beta(B * H * T);
  for (int i = 0; i < B * H * T * dk; i++) {
    query[i] = 0.08f * std::sin(static_cast<float>(i) * 0.011f);
    key[i] = 0.08f * std::cos(static_cast<float>(i) * 0.019f);
    decay[i] = -0.04f - 0.04f * std::abs(std::sin(static_cast<float>(i) * 0.05f));
  }
  for (int i = 0; i < B * H * T * dv; i++) {
    value[i] = 0.08f * std::sin(static_cast<float>(i) * 0.017f + 0.3f);
  }
  for (int i = 0; i < B * H * T; i++) {
    beta[i] = 0.4f + 0.4f * std::sin(static_cast<float>(i) * 0.023f);
  }

  std::vector<float> initial_state(B * H * dk * dv, 0.005f);
  RunLinearAttentionTest("gated_delta", B, H, T, dk, dv, scale,
                         query, key, value, &initial_state, &decay, &beta);
}

static void RunStandardGQA(const std::string& rule, int q_H, int kv_H, int dk, int dv) {
  const int B = 2, n_k = kv_H, T = 10;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
  const bool has_decay = rule == "gated" || rule == "gated_delta";
  const bool has_beta = rule == "delta" || rule == "gated_delta";

  std::vector<float> query(B * q_H * T * dk);
  std::vector<float> key(B * n_k * T * dk);
  std::vector<float> value(B * kv_H * T * dv);
  std::vector<float> decay(B * kv_H * T);
  std::vector<float> beta(B * kv_H * T);
  for (int i = 0; i < B * q_H * T * dk; i++) {
    query[i] = 0.1f * std::sin(static_cast<float>(i) * 0.013f);
  }
  for (int i = 0; i < B * n_k * T * dk; i++) {
    key[i] = 0.1f * std::cos(static_cast<float>(i) * 0.017f);
  }
  for (int i = 0; i < B * kv_H * T * dv; i++) {
    value[i] = 0.1f * std::sin(static_cast<float>(i) * 0.023f + 0.5f);
  }
  for (int i = 0; i < B * kv_H * T; i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.3f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }
  std::vector<float> initial_state(B * kv_H * dk * dv, 0.01f);

  RunLinearAttentionGQATest(rule, B, q_H, kv_H, n_k, T, dk, dv, scale,
                            query, key, value, &initial_state,
                            has_decay ? &decay : nullptr,
                            has_beta ? &beta : nullptr);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StandardGQA_N2) {
  RunStandardGQA("gated_delta", 4, 2, 32, 64);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StandardGQA_N4) {
  RunStandardGQA("gated_delta", 8, 2, 32, 64);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StandardGQA_N8) {
  RunStandardGQA("gated_delta", 16, 2, 32, 64);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StandardGQA_N16) {
  RunStandardGQA("gated_delta", 16, 1, 32, 64);
}

TEST(ContribOpLinearAttentionTest, LinearRule_StandardGQA_N4) {
  RunStandardGQA("linear", 8, 2, 32, 64);
}

TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StandardGQA_N4_Dim128) {
  RunStandardGQA("gated_delta", 8, 2, 128, 128);
}

// The state tensors grow linearly with state_window, so the schema caps it at 8.
TEST(ContribOpLinearAttentionTest, StateWindowAboveMaxIsRejected) {
  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", "linear");
  tester.AddAttribute<int64_t>("q_num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<int64_t>("state_window", 9);
  tester.AddInput<float>("query", {1, 1, 1}, {1.0f});
  tester.AddInput<float>("key", {1, 1, 1}, {1.0f});
  tester.AddInput<float>("value", {1, 1, 1}, {1.0f});
  tester.AddOutput<float>("output", {1, 1, 1}, {1.0f});
  tester.AddOutput<float>("present_state", {9, 1, 1, 1, 1}, std::vector<float>(9, 0.0f));
  tester.Run(OpTester::ExpectResult::kExpectFailure, "state_window must be in [0, 8]");
}

// state_window: past_state / present_state hold the last W per-token states, right-aligned, so
// slot j is the state after token (T - W + j) and slot W-1 is the state after the last token (the
// tensor the unwindowed op produces). past_state is read from slot W-1. Slots below max(0, W - T)
// hold no token from this call and come back zeroed.
#ifdef USE_CUDA
TEST(ContribOpLinearAttentionTest, StateWindowRejectsEmptySequence) {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
    return;
  }

  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", "linear");
  tester.AddAttribute<int64_t>("q_num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<int64_t>("state_window", 1);
  tester.AddInput<float>("query", {1, 0, 1}, {});
  tester.AddInput<float>("key", {1, 0, 1}, {});
  tester.AddInput<float>("value", {1, 0, 1}, {});
  tester.AddOutput<float>("output", {1, 0, 1}, {});
  tester.AddOutput<float>("present_state", {1, 1, 1, 1, 1}, {0.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectFailure, "sequence length must be positive", {}, nullptr,
             &execution_providers);
}

// The state_window attribute is only implemented by the CUDA kernel. The four CUDA kernel families
// (generic recurrent, fixed-shape recurrent, warp-per-column decode, column-per-thread decode) each
// compute the window slot offsets themselves, so every family gets its own shape below.
static void RunLinearAttentionStateWindowTest(int B, int q_H, int kv_H, int n_k, int T, int dk, int dv, int W,
                                              bool with_past_state = true) {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
    return;
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  std::vector<float> query(B * q_H * T * dk);
  std::vector<float> key(B * n_k * T * dk);
  std::vector<float> value(B * kv_H * T * dv);
  std::vector<float> decay(B * kv_H * T);
  std::vector<float> beta(B * kv_H * T);
  for (size_t i = 0; i < query.size(); i++) query[i] = 0.5f * std::sin(static_cast<float>(i) * 0.13f);
  for (size_t i = 0; i < key.size(); i++) key[i] = 0.5f * std::cos(static_cast<float>(i) * 0.17f);
  for (size_t i = 0; i < value.size(); i++) value[i] = 0.5f * std::sin(static_cast<float>(i) * 0.23f + 0.5f);
  for (size_t i = 0; i < decay.size(); i++) {
    decay[i] = -0.1f - 0.05f * std::abs(std::sin(static_cast<float>(i) * 0.3f));
    beta[i] = 0.5f + 0.3f * std::sin(static_cast<float>(i) * 0.31f);
  }
  std::vector<float> initial_state(static_cast<size_t>(B) * kv_H * dk * dv, 0.0f);
  if (with_past_state) {
    for (size_t i = 0; i < initial_state.size(); i++) {
      initial_state[i] = 0.1f * std::cos(static_cast<float>(i) * 0.07f);
    }
  }
  const std::vector<float>* past = with_past_state ? &initial_state : nullptr;

  // Slice a (B, H, T, D) tensor down to its first `prefix` tokens.
  auto slice_prefix = [](const std::vector<float>& src, int seq, int dim, int prefix) {
    std::vector<float> dst(src.size() / seq * prefix);
    const int bh = static_cast<int>(src.size()) / (seq * dim);
    for (int i = 0; i < bh; i++) {
      for (int t = 0; t < prefix; t++) {
        for (int d = 0; d < dim; d++) {
          dst[(i * prefix + t) * dim + d] = src[(i * seq + t) * dim + d];
        }
      }
    }
    return dst;
  };

  std::vector<float> expected_output_4d, expected_state;
  LinearAttentionGQAReference("gated_delta", B, q_H, kv_H, n_k, T, dk, dv, scale,
                              query, key, value, past, &decay, &beta,
                              expected_output_4d, expected_state);

  // Slot j = the recurrent state after running the first (T - W + j + 1) tokens; slots for
  // non-positive prefixes are never computed by the kernel and stay zero. The window axis leads
  // the batch axis, so a slot is exactly one contiguous (B, H_kv, d_k, d_v) reference block.
  const size_t slot_elems = static_cast<size_t>(kv_H) * dk * dv;
  const size_t batch_slot_elems = static_cast<size_t>(B) * slot_elems;
  std::vector<float> expected_state_window(static_cast<size_t>(W) * batch_slot_elems, 0.0f);
  for (int j = 0; j < W; j++) {
    const int prefix = T - W + j + 1;
    if (prefix <= 0) continue;
    std::vector<float> prefix_state;
    if (prefix == T) {
      prefix_state = expected_state;
    } else {
      const std::vector<float> decay_prefix = slice_prefix(decay, T, 1, prefix);
      const std::vector<float> beta_prefix = slice_prefix(beta, T, 1, prefix);
      std::vector<float> prefix_output;
      LinearAttentionGQAReference("gated_delta", B, q_H, kv_H, n_k, prefix, dk, dv, scale,
                                  slice_prefix(query, T, dk, prefix),
                                  slice_prefix(key, T, dk, prefix),
                                  slice_prefix(value, T, dv, prefix),
                                  past, &decay_prefix, &beta_prefix,
                                  prefix_output, prefix_state);
    }
    std::copy_n(prefix_state.begin(), batch_slot_elems,
                expected_state_window.begin() + static_cast<size_t>(j) * batch_slot_elems);
  }

  OpTester tester("LinearAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("update_rule", "gated_delta");
  tester.AddAttribute<float>("scale", scale);
  tester.AddAttribute<int64_t>("q_num_heads", static_cast<int64_t>(q_H));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_H));
  tester.AddAttribute<int64_t>("state_window", static_cast<int64_t>(W));

  tester.AddInput<float>("query", {B, T, q_H * dk}, PackBHTD_to_BTHD(query, B, q_H, T, dk));
  tester.AddInput<float>("key", {B, T, n_k * dk}, PackBHTD_to_BTHD(key, B, n_k, T, dk));
  tester.AddInput<float>("value", {B, T, kv_H * dv}, PackBHTD_to_BTHD(value, B, kv_H, T, dv));
  if (with_past_state) {
    // past_state is windowed too, and only slot W-1 is read. Poison the earlier slots to prove it.
    std::vector<float> past_state_window(static_cast<size_t>(W) * batch_slot_elems, -1e4f);
    std::copy_n(initial_state.begin(), batch_slot_elems,
                past_state_window.begin() + static_cast<size_t>(W - 1) * batch_slot_elems);
    tester.AddInput<float>("past_state", {W, B, kv_H, dk, dv}, past_state_window);
  } else {
    tester.AddOptionalInputEdge<float>();
  }
  tester.AddInput<float>("decay", {B, T, kv_H}, TransposeBHT_to_BTH(decay, B, kv_H, T));
  tester.AddInput<float>("beta", {B, T, kv_H}, TransposeBHT_to_BTH(beta, B, kv_H, T));

  tester.AddOutput<float>("output", {B, T, kv_H * dv},
                          PackBHTD_to_BTHD(expected_output_4d, B, kv_H, T, dv),
                          false, 0.005f, 0.005f);
  tester.AddOutput<float>("present_state", {W, B, kv_H, dk, dv}, expected_state_window,
                          false, 0.005f, 0.005f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// d_k = 4 is not a decode fast-path shape, so this lands on the generic recurrent kernel.
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StateWindow) {
  RunLinearAttentionStateWindowTest(/*B=*/1, /*q_H=*/2, /*kv_H=*/2, /*n_k=*/2, /*T=*/5,
                                    /*dk=*/4, /*dv=*/4, /*W=*/3);
}

// B > 1 exercises the batch stride of the window axis on the generic recurrent kernel.
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StateWindow_MultiBatch) {
  RunLinearAttentionStateWindowTest(/*B=*/2, /*q_H=*/2, /*kv_H=*/2, /*n_k=*/2, /*T=*/5,
                                    /*dk=*/4, /*dv=*/4, /*W=*/3);
}

// T <= 16 with d_k = 128 and d_v % 32 == 0 selects the column-per-thread decode kernel.
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StateWindow_DecodeColKernel) {
  RunLinearAttentionStateWindowTest(/*B=*/2, /*q_H=*/2, /*kv_H=*/2, /*n_k=*/1, /*T=*/4,
                                    /*dk=*/128, /*dv=*/128, /*W=*/3);
}

// d_k = 256 falls back to the warp-per-column decode kernel.
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StateWindow_DecodeWarpKernel) {
  RunLinearAttentionStateWindowTest(/*B=*/1, /*q_H=*/2, /*kv_H=*/2, /*n_k=*/1, /*T=*/4,
                                    /*dk=*/256, /*dv=*/64, /*W=*/3);
}

// T > 16 with a (d_k, d_v) fast-path pair selects the compile-time specialized recurrent kernel
// when the device has enough shared memory, or the column kernel fallback otherwise.
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StateWindow_FixedShapeKernel) {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
    return;
  }

  int device = 0;
  int multiprocessor_count = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  ASSERT_EQ(cudaSuccess,
            cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  constexpr int kv_num_heads = 2;
  const int batch_size = (multiprocessor_count + kv_num_heads - 1) / kv_num_heads;

  RunLinearAttentionStateWindowTest(/*B=*/batch_size, /*q_H=*/2, /*kv_H=*/kv_num_heads, /*n_k=*/1, /*T=*/17,
                                    /*dk=*/128, /*dv=*/128, /*W=*/4);
}

// W > T is the shape genai actually runs during MTP decode: the leading W - T slots belong to
// positions before this call, so the kernel skips them and they come back zeroed.
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StateWindow_WiderThanSequence) {
  RunLinearAttentionStateWindowTest(/*B=*/2, /*q_H=*/2, /*kv_H=*/2, /*n_k=*/1, /*T=*/2,
                                    /*dk=*/128, /*dv=*/128, /*W=*/5, /*with_past_state=*/false);
}

// Same shape with a past_state: present_state is a fresh allocation, so the skipped leading slots
// must be zeroed rather than left as uninitialized device memory.
TEST(ContribOpLinearAttentionTest, GatedDeltaRule_StateWindow_WiderThanSequenceWithPastState) {
  RunLinearAttentionStateWindowTest(/*B=*/2, /*q_H=*/2, /*kv_H=*/2, /*n_k=*/1, /*T=*/2,
                                    /*dk=*/128, /*dv=*/128, /*W=*/5, /*with_past_state=*/true);
}
#endif  // USE_CUDA
}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
enum class TensorType {
  kFloat,
  kFloat16
};

// Reference implementation for linear attention recurrent update
void LinearAttentionRecurrentReference(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& past_state,
    const std::vector<float>* decay,
    const std::vector<float>* beta,
    std::vector<float>& output,
    std::vector<float>& present_state,
    int batch_size,
    int num_heads,
    int head_dim_k,
    int head_dim_v,
    const std::string& update_rule,
    float scale) {
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));
  }

  // Copy past_state to present_state first
  present_state = past_state;

  output.resize(batch_size * num_heads * head_dim_v);

  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      int bh = b * num_heads + h;
      int state_base = bh * head_dim_k * head_dim_v;
      int qkv_base = bh * head_dim_k;
      int v_base = bh * head_dim_v;

      // Apply decay if gated or gated_delta
      if (update_rule == "gated" || update_rule == "gated_delta") {
        for (int k = 0; k < head_dim_k; ++k) {
          float g = (*decay)[qkv_base + k];
          float exp_g = std::exp(g);
          for (int v = 0; v < head_dim_v; ++v) {
            present_state[state_base + k * head_dim_v + v] *= exp_g;
          }
        }
      }

      // Compute update
      if (update_rule == "linear" || update_rule == "gated") {
        // S += k ⊗ v
        for (int k = 0; k < head_dim_k; ++k) {
          float k_val = key[qkv_base + k];
          for (int v = 0; v < head_dim_v; ++v) {
            float v_val = value[v_base + v];
            present_state[state_base + k * head_dim_v + v] += k_val * v_val;
          }
        }
      } else if (update_rule == "delta" || update_rule == "gated_delta") {
        // Compute retrieved = S^T @ k
        std::vector<float> retrieved(head_dim_v, 0.0f);
        for (int v = 0; v < head_dim_v; ++v) {
          for (int k = 0; k < head_dim_k; ++k) {
            float k_val = key[qkv_base + k];
            // For gated_delta, retrieval uses decayed state (already applied above)
            // For delta, uses original past_state
            float s_val = (update_rule == "gated_delta")
                              ? present_state[state_base + k * head_dim_v + v]
                              : past_state[state_base + k * head_dim_v + v];
            retrieved[v] += s_val * k_val;
          }
        }

        // Compute delta and update
        float beta_val = (*beta)[bh];
        for (int k = 0; k < head_dim_k; ++k) {
          float k_val = key[qkv_base + k];
          for (int v = 0; v < head_dim_v; ++v) {
            float v_val = value[v_base + v];
            float delta = beta_val * (v_val - retrieved[v]);
            present_state[state_base + k * head_dim_v + v] += k_val * delta;
          }
        }
      }

      // Compute output = scale * q^T @ S
      for (int v = 0; v < head_dim_v; ++v) {
        float out_val = 0.0f;
        for (int k = 0; k < head_dim_k; ++k) {
          float q_val = query[qkv_base + k];
          out_val += q_val * present_state[state_base + k * head_dim_v + v];
        }
        output[v_base + v] = out_val * scale;
      }
    }
  }
}

// Reference implementation for linear attention chunk parallel (full sequence)
// This is the sequential version that processes one step at a time.
void LinearAttentionChunkParallelReference(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>* initial_state,
    const std::vector<float>* decay,
    const std::vector<float>* beta,
    std::vector<float>& output,
    std::vector<float>& final_state,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim_k,
    int head_dim_v,
    const std::string& update_rule,
    float scale) {
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));
  }

  output.resize(batch_size * num_heads * seq_length * head_dim_v);
  final_state.resize(batch_size * num_heads * head_dim_k * head_dim_v);

  int state_size = head_dim_k * head_dim_v;

  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      int bh = b * num_heads + h;

      // Initialize state
      std::vector<float> state(state_size, 0.0f);
      if (initial_state != nullptr) {
        int init_base = bh * state_size;
        for (int i = 0; i < state_size; ++i) {
          state[i] = (*initial_state)[init_base + i];
        }
      }

      // Process each timestep sequentially
      for (int t = 0; t < seq_length; ++t) {
        int qk_base = (bh * seq_length + t) * head_dim_k;
        int v_base = (bh * seq_length + t) * head_dim_v;

        // 1. Apply decay if gated or gated_delta
        if (update_rule == "gated" || update_rule == "gated_delta") {
          for (int ki = 0; ki < head_dim_k; ++ki) {
            float g = (*decay)[qk_base + ki];
            float exp_g = std::exp(g);
            for (int vi = 0; vi < head_dim_v; ++vi) {
              state[ki * head_dim_v + vi] *= exp_g;
            }
          }
        }

        // 2. Update state
        if (update_rule == "linear" || update_rule == "gated") {
          // S += k ⊗ v
          for (int ki = 0; ki < head_dim_k; ++ki) {
            float k_val = key[qk_base + ki];
            for (int vi = 0; vi < head_dim_v; ++vi) {
              float v_val = value[v_base + vi];
              state[ki * head_dim_v + vi] += k_val * v_val;
            }
          }
        } else if (update_rule == "delta" || update_rule == "gated_delta") {
          // Compute retrieved = S^T @ k
          std::vector<float> retrieved(head_dim_v, 0.0f);
          for (int vi = 0; vi < head_dim_v; ++vi) {
            for (int ki = 0; ki < head_dim_k; ++ki) {
              retrieved[vi] += state[ki * head_dim_v + vi] * key[qk_base + ki];
            }
          }

          float beta_val = (*beta)[bh * seq_length + t];
          for (int ki = 0; ki < head_dim_k; ++ki) {
            float k_val = key[qk_base + ki];
            for (int vi = 0; vi < head_dim_v; ++vi) {
              float v_val = value[v_base + vi];
              float delta_val = beta_val * (v_val - retrieved[vi]);
              state[ki * head_dim_v + vi] += k_val * delta_val;
            }
          }
        }

        // 3. Compute output = scale * q^T @ S
        int out_base = (bh * seq_length + t) * head_dim_v;
        for (int vi = 0; vi < head_dim_v; ++vi) {
          float out_val = 0.0f;
          for (int ki = 0; ki < head_dim_k; ++ki) {
            out_val += query[qk_base + ki] * state[ki * head_dim_v + vi];
          }
          output[out_base + vi] = out_val * scale;
        }
      }

      // Copy final state
      int final_base = bh * state_size;
      for (int i = 0; i < state_size; ++i) {
        final_state[final_base + i] = state[i];
      }
    }
  }
}

}  // anonymous namespace

static void RunLinearAttentionRecurrentTest(
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    const std::vector<float>& past_state_data,
    const std::vector<float>* decay_data,
    const std::vector<float>* beta_data,
    const std::vector<float>& expected_output,
    const std::vector<float>& expected_state,
    int batch_size,
    int num_heads,
    int head_dim_k,
    int head_dim_v,
    const std::string& update_rule,
    float scale,
    TensorType tensor_type) {
  std::vector<int64_t> query_shape = {batch_size, num_heads, 1, head_dim_k};
  std::vector<int64_t> key_shape = {batch_size, num_heads, 1, head_dim_k};
  std::vector<int64_t> value_shape = {batch_size, num_heads, 1, head_dim_v};
  std::vector<int64_t> state_shape = {batch_size, num_heads, head_dim_k, head_dim_v};
  std::vector<int64_t> decay_shape = {batch_size, num_heads, 1, head_dim_k};
  std::vector<int64_t> beta_shape = {batch_size, num_heads, 1, 1};
  std::vector<int64_t> output_shape = {batch_size, num_heads, 1, head_dim_v};

  std::string op_type = "LinearAttentionRecurrent";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;

  bool enable_webgpu = nullptr != DefaultWebGpuExecutionProvider().get();

  if (enable_webgpu) {
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
  }

  if (execution_providers.empty()) {
    // Skip if no providers available
    return;
  }

  for (auto& ep : execution_providers) {
    OpTester test(op_type.c_str(), 1, onnxruntime::kMSDomain);
    test.AddAttribute<std::string>("update_rule", update_rule);
    test.AddAttribute<float>("scale", scale);

    if (tensor_type == TensorType::kFloat) {
      test.AddInput<float>("query", query_shape, query_data);
      test.AddInput<float>("key", key_shape, key_data);
      test.AddInput<float>("value", value_shape, value_data);
      test.AddInput<float>("past_state", state_shape, past_state_data);

      if (decay_data != nullptr) {
        test.AddInput<float>("decay", decay_shape, *decay_data);
      } else {
        test.AddOptionalInputEdge<float>();
      }

      if (beta_data != nullptr) {
        test.AddInput<float>("beta", beta_shape, *beta_data);
      } else {
        test.AddOptionalInputEdge<float>();
      }

      test.AddOutput<float>("output", output_shape, expected_output);
      test.AddOutput<float>("present_state", state_shape, expected_state);
    } else {
      test.AddInput<MLFloat16>("query", query_shape, ToFloat16(query_data));
      test.AddInput<MLFloat16>("key", key_shape, ToFloat16(key_data));
      test.AddInput<MLFloat16>("value", value_shape, ToFloat16(value_data));
      test.AddInput<MLFloat16>("past_state", state_shape, ToFloat16(past_state_data));

      if (decay_data != nullptr) {
        test.AddInput<MLFloat16>("decay", decay_shape, ToFloat16(*decay_data));
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }

      if (beta_data != nullptr) {
        test.AddInput<MLFloat16>("beta", beta_shape, ToFloat16(*beta_data));
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }

      test.AddOutput<MLFloat16>("output", output_shape, ToFloat16(expected_output));
      test.AddOutput<MLFloat16>("present_state", state_shape, ToFloat16(expected_state));
    }

    test.SetOutputAbsErr("output", 0.01f);
    test.SetOutputAbsErr("present_state", 0.01f);

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
  }
}

static void RunLinearAttentionRecurrentTests(
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    const std::vector<float>& past_state_data,
    const std::vector<float>* decay_data,
    const std::vector<float>* beta_data,
    int batch_size,
    int num_heads,
    int head_dim_k,
    int head_dim_v,
    const std::string& update_rule,
    float scale = 0.0f) {
  // Compute expected output using reference implementation
  std::vector<float> expected_output;
  std::vector<float> expected_state;
  LinearAttentionRecurrentReference(
      query_data, key_data, value_data, past_state_data,
      decay_data, beta_data,
      expected_output, expected_state,
      batch_size, num_heads, head_dim_k, head_dim_v,
      update_rule, scale);

  // FP32 test
  RunLinearAttentionRecurrentTest(
      query_data, key_data, value_data, past_state_data,
      decay_data, beta_data,
      expected_output, expected_state,
      batch_size, num_heads, head_dim_k, head_dim_v,
      update_rule, scale, TensorType::kFloat);

  // FP16 test
  RunLinearAttentionRecurrentTest(
      query_data, key_data, value_data, past_state_data,
      decay_data, beta_data,
      expected_output, expected_state,
      batch_size, num_heads, head_dim_k, head_dim_v,
      update_rule, scale, TensorType::kFloat16);
}

// =============================================================================
// LinearAttentionRecurrent Tests
// =============================================================================

TEST(ContribOpLinearAttentionTest, LinearAttentionRecurrent_Linear_Basic) {
  int batch_size = 1;
  int num_heads = 2;
  int head_dim_k = 4;
  int head_dim_v = 4;

  // Query: (1, 2, 1, 4)
  std::vector<float> query_data = {
      0.5f, 0.3f, -0.2f, 0.1f,  // head 0
      -0.4f, 0.6f, 0.2f, -0.3f  // head 1
  };

  // Key: (1, 2, 1, 4)
  std::vector<float> key_data = {
      0.1f, 0.2f, 0.3f, 0.4f,
      0.2f, -0.1f, 0.3f, 0.1f};

  // Value: (1, 2, 1, 4)
  std::vector<float> value_data = {
      0.4f, 0.3f, 0.2f, 0.1f,
      -0.2f, 0.4f, 0.1f, 0.3f};

  // Past state: (1, 2, 4, 4) - initialized to small values
  std::vector<float> past_state_data(batch_size * num_heads * head_dim_k * head_dim_v, 0.1f);

  RunLinearAttentionRecurrentTests(
      query_data, key_data, value_data, past_state_data,
      nullptr, nullptr,
      batch_size, num_heads, head_dim_k, head_dim_v,
      "linear");
}

TEST(ContribOpLinearAttentionTest, LinearAttentionRecurrent_Gated_Basic) {
  int batch_size = 1;
  int num_heads = 2;
  int head_dim_k = 4;
  int head_dim_v = 4;

  std::vector<float> query_data = {
      0.5f, 0.3f, -0.2f, 0.1f,
      -0.4f, 0.6f, 0.2f, -0.3f};

  std::vector<float> key_data = {
      0.1f, 0.2f, 0.3f, 0.4f,
      0.2f, -0.1f, 0.3f, 0.1f};

  std::vector<float> value_data = {
      0.4f, 0.3f, 0.2f, 0.1f,
      -0.2f, 0.4f, 0.1f, 0.3f};

  std::vector<float> past_state_data(batch_size * num_heads * head_dim_k * head_dim_v, 0.1f);

  // Decay: (1, 2, 1, 4) - negative values for decay
  std::vector<float> decay_data = {
      -0.1f, -0.1f, -0.1f, -0.1f,
      -0.2f, -0.2f, -0.2f, -0.2f};

  RunLinearAttentionRecurrentTests(
      query_data, key_data, value_data, past_state_data,
      &decay_data, nullptr,
      batch_size, num_heads, head_dim_k, head_dim_v,
      "gated");
}

TEST(ContribOpLinearAttentionTest, LinearAttentionRecurrent_Delta_Basic) {
  int batch_size = 1;
  int num_heads = 2;
  int head_dim_k = 4;
  int head_dim_v = 4;

  std::vector<float> query_data = {
      0.5f, 0.3f, -0.2f, 0.1f,
      -0.4f, 0.6f, 0.2f, -0.3f};

  // L2-normalized keys for delta rule
  std::vector<float> key_data = {
      0.1826f, 0.3651f, 0.5477f, 0.7303f,  // normalized
      0.5345f, -0.2673f, 0.8018f, 0.2673f  // normalized
  };

  std::vector<float> value_data = {
      0.4f, 0.3f, 0.2f, 0.1f,
      -0.2f, 0.4f, 0.1f, 0.3f};

  std::vector<float> past_state_data(batch_size * num_heads * head_dim_k * head_dim_v, 0.1f);

  // Beta: (1, 2, 1, 1)
  std::vector<float> beta_data = {0.5f, 0.7f};

  RunLinearAttentionRecurrentTests(
      query_data, key_data, value_data, past_state_data,
      nullptr, &beta_data,
      batch_size, num_heads, head_dim_k, head_dim_v,
      "delta");
}

TEST(ContribOpLinearAttentionTest, LinearAttentionRecurrent_GatedDelta_Basic) {
  int batch_size = 1;
  int num_heads = 2;
  int head_dim_k = 4;
  int head_dim_v = 4;

  std::vector<float> query_data = {
      0.5f, 0.3f, -0.2f, 0.1f,
      -0.4f, 0.6f, 0.2f, -0.3f};

  // L2-normalized keys
  std::vector<float> key_data = {
      0.1826f, 0.3651f, 0.5477f, 0.7303f,
      0.5345f, -0.2673f, 0.8018f, 0.2673f};

  std::vector<float> value_data = {
      0.4f, 0.3f, 0.2f, 0.1f,
      -0.2f, 0.4f, 0.1f, 0.3f};

  std::vector<float> past_state_data(batch_size * num_heads * head_dim_k * head_dim_v, 0.1f);

  // Decay: (1, 2, 1, 4)
  std::vector<float> decay_data = {
      -0.1f, -0.1f, -0.1f, -0.1f,
      -0.2f, -0.2f, -0.2f, -0.2f};

  // Beta: (1, 2, 1, 1)
  std::vector<float> beta_data = {0.5f, 0.7f};

  RunLinearAttentionRecurrentTests(
      query_data, key_data, value_data, past_state_data,
      &decay_data, &beta_data,
      batch_size, num_heads, head_dim_k, head_dim_v,
      "gated_delta");
}

TEST(ContribOpLinearAttentionTest, LinearAttentionRecurrent_LargerBatch) {
  int batch_size = 2;
  int num_heads = 4;
  int head_dim_k = 8;
  int head_dim_v = 8;

  int qkv_size = batch_size * num_heads * head_dim_k;
  int value_size = batch_size * num_heads * head_dim_v;
  int state_size = batch_size * num_heads * head_dim_k * head_dim_v;

  // Generate random-ish data
  std::vector<float> query_data(qkv_size);
  std::vector<float> key_data(qkv_size);
  std::vector<float> value_data(value_size);
  std::vector<float> past_state_data(state_size);
  std::vector<float> decay_data(qkv_size);
  std::vector<float> beta_data(batch_size * num_heads);

  for (int i = 0; i < qkv_size; ++i) {
    query_data[i] = 0.1f * (i % 10 - 5);
    key_data[i] = 0.1f * ((i + 3) % 10 - 5);
    decay_data[i] = -0.1f * ((i % 3) + 1);
  }
  for (int i = 0; i < value_size; ++i) {
    value_data[i] = 0.1f * ((i + 7) % 10 - 5);
  }
  for (int i = 0; i < state_size; ++i) {
    past_state_data[i] = 0.05f * (i % 10 - 5);
  }
  for (int i = 0; i < batch_size * num_heads; ++i) {
    beta_data[i] = 0.3f + 0.1f * (i % 5);
  }

  RunLinearAttentionRecurrentTests(
      query_data, key_data, value_data, past_state_data,
      &decay_data, &beta_data,
      batch_size, num_heads, head_dim_k, head_dim_v,
      "gated_delta");
}

// =============================================================================
// LinearAttentionChunkParallel Tests
// =============================================================================

static void RunLinearAttentionChunkParallelTest(
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    const std::vector<float>* initial_state_data,
    const std::vector<float>* decay_data,
    const std::vector<float>* beta_data,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim_k,
    int head_dim_v,
    const std::string& update_rule,
    int64_t chunk_size,
    float scale,
    TensorType tensor_type) {
  std::vector<int64_t> query_shape = {batch_size, num_heads, seq_length, head_dim_k};
  std::vector<int64_t> key_shape = {batch_size, num_heads, seq_length, head_dim_k};
  std::vector<int64_t> value_shape = {batch_size, num_heads, seq_length, head_dim_v};
  std::vector<int64_t> state_shape = {batch_size, num_heads, head_dim_k, head_dim_v};
  std::vector<int64_t> decay_shape = {batch_size, num_heads, seq_length, head_dim_k};
  std::vector<int64_t> beta_shape = {batch_size, num_heads, seq_length, 1};
  std::vector<int64_t> output_shape = {batch_size, num_heads, seq_length, head_dim_v};

  // Compute reference expected output
  std::vector<float> expected_output;
  std::vector<float> expected_state;
  LinearAttentionChunkParallelReference(
      query_data, key_data, value_data,
      initial_state_data, decay_data, beta_data,
      expected_output, expected_state,
      batch_size, num_heads, seq_length, head_dim_k, head_dim_v,
      update_rule, scale);

  std::string op_type = "LinearAttentionChunkParallel";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;

  bool enable_webgpu = nullptr != DefaultWebGpuExecutionProvider().get();

  if (enable_webgpu) {
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
  }

  if (execution_providers.empty()) {
    return;
  }

  for (auto& ep : execution_providers) {
    OpTester test(op_type.c_str(), 1, onnxruntime::kMSDomain);
    test.AddAttribute<std::string>("update_rule", update_rule);
    test.AddAttribute<int64_t>("chunk_size", chunk_size);
    test.AddAttribute<float>("scale", scale);

    if (tensor_type == TensorType::kFloat) {
      test.AddInput<float>("query", query_shape, query_data);
      test.AddInput<float>("key", key_shape, key_data);
      test.AddInput<float>("value", value_shape, value_data);

      if (initial_state_data != nullptr) {
        test.AddInput<float>("initial_state", state_shape, *initial_state_data);
      } else {
        test.AddOptionalInputEdge<float>();
      }

      if (decay_data != nullptr) {
        test.AddInput<float>("decay", decay_shape, *decay_data);
      } else {
        test.AddOptionalInputEdge<float>();
      }

      if (beta_data != nullptr) {
        test.AddInput<float>("beta", beta_shape, *beta_data);
      } else {
        test.AddOptionalInputEdge<float>();
      }

      test.AddOutput<float>("output", output_shape, expected_output);
      test.AddOutput<float>("final_state", state_shape, expected_state);
    } else {
      test.AddInput<MLFloat16>("query", query_shape, ToFloat16(query_data));
      test.AddInput<MLFloat16>("key", key_shape, ToFloat16(key_data));
      test.AddInput<MLFloat16>("value", value_shape, ToFloat16(value_data));

      if (initial_state_data != nullptr) {
        test.AddInput<MLFloat16>("initial_state", state_shape, ToFloat16(*initial_state_data));
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }

      if (decay_data != nullptr) {
        test.AddInput<MLFloat16>("decay", decay_shape, ToFloat16(*decay_data));
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }

      if (beta_data != nullptr) {
        test.AddInput<MLFloat16>("beta", beta_shape, ToFloat16(*beta_data));
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }

      test.AddOutput<MLFloat16>("output", output_shape, ToFloat16(expected_output));
      test.AddOutput<MLFloat16>("final_state", state_shape, ToFloat16(expected_state));
    }

    test.SetOutputAbsErr("output", 0.01f);
    test.SetOutputAbsErr("final_state", 0.01f);

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
  }
}

TEST(ContribOpLinearAttentionTest, LinearAttentionChunkParallel_Linear_Basic) {
  int batch_size = 1;
  int num_heads = 2;
  int seq_length = 8;
  int head_dim_k = 4;
  int head_dim_v = 4;

  int qkv_size = batch_size * num_heads * seq_length * head_dim_k;
  int value_size = batch_size * num_heads * seq_length * head_dim_v;

  std::vector<float> query_data(qkv_size);
  std::vector<float> key_data(qkv_size);
  std::vector<float> value_data(value_size);

  for (int i = 0; i < qkv_size; ++i) {
    query_data[i] = 0.1f * (i % 10 - 5);
    key_data[i] = 0.1f * ((i + 3) % 10 - 5);
  }
  for (int i = 0; i < value_size; ++i) {
    value_data[i] = 0.1f * ((i + 7) % 10 - 5);
  }

  RunLinearAttentionChunkParallelTest(
      query_data, key_data, value_data,
      nullptr, nullptr, nullptr,
      batch_size, num_heads, seq_length, head_dim_k, head_dim_v,
      "linear", 4, 0.0f, TensorType::kFloat);
}

TEST(ContribOpLinearAttentionTest, LinearAttentionChunkParallel_Gated_Basic) {
  int batch_size = 1;
  int num_heads = 2;
  int seq_length = 8;
  int head_dim_k = 4;
  int head_dim_v = 4;

  int qkv_size = batch_size * num_heads * seq_length * head_dim_k;
  int value_size = batch_size * num_heads * seq_length * head_dim_v;
  int decay_size = batch_size * num_heads * seq_length * head_dim_k;

  std::vector<float> query_data(qkv_size);
  std::vector<float> key_data(qkv_size);
  std::vector<float> value_data(value_size);
  std::vector<float> decay_data(decay_size);

  for (int i = 0; i < qkv_size; ++i) {
    query_data[i] = 0.1f * (i % 10 - 5);
    key_data[i] = 0.1f * ((i + 3) % 10 - 5);
  }
  for (int i = 0; i < value_size; ++i) {
    value_data[i] = 0.1f * ((i + 7) % 10 - 5);
  }
  for (int i = 0; i < decay_size; ++i) {
    decay_data[i] = -0.1f * ((i % 3) + 1);
  }

  RunLinearAttentionChunkParallelTest(
      query_data, key_data, value_data,
      nullptr, &decay_data, nullptr,
      batch_size, num_heads, seq_length, head_dim_k, head_dim_v,
      "gated", 4, 0.0f, TensorType::kFloat);
}

TEST(ContribOpLinearAttentionTest, LinearAttentionChunkParallel_GatedDelta_WithInitialState) {
  int batch_size = 1;
  int num_heads = 2;
  int seq_length = 16;
  int head_dim_k = 4;
  int head_dim_v = 4;

  int qkv_size = batch_size * num_heads * seq_length * head_dim_k;
  int value_size = batch_size * num_heads * seq_length * head_dim_v;
  int state_size = batch_size * num_heads * head_dim_k * head_dim_v;
  int decay_size = batch_size * num_heads * seq_length * head_dim_k;
  int beta_size = batch_size * num_heads * seq_length;

  std::vector<float> query_data(qkv_size);
  std::vector<float> key_data(qkv_size);
  std::vector<float> value_data(value_size);
  std::vector<float> initial_state_data(state_size);
  std::vector<float> decay_data(decay_size);
  std::vector<float> beta_data(beta_size);

  for (int i = 0; i < qkv_size; ++i) {
    query_data[i] = 0.1f * (i % 10 - 5);
    key_data[i] = 0.1f * ((i + 3) % 10 - 5);
  }
  for (int i = 0; i < value_size; ++i) {
    value_data[i] = 0.1f * ((i + 7) % 10 - 5);
  }
  for (int i = 0; i < state_size; ++i) {
    initial_state_data[i] = 0.05f;
  }
  for (int i = 0; i < decay_size; ++i) {
    decay_data[i] = -0.1f * ((i % 3) + 1);
  }
  for (int i = 0; i < beta_size; ++i) {
    beta_data[i] = 0.5f;
  }

  RunLinearAttentionChunkParallelTest(
      query_data, key_data, value_data,
      &initial_state_data, &decay_data, &beta_data,
      batch_size, num_heads, seq_length, head_dim_k, head_dim_v,
      "gated_delta", 8, 0.0f, TensorType::kFloat);
}

TEST(ContribOpLinearAttentionTest, LinearAttentionChunkParallel_LargerSequence) {
  int batch_size = 2;
  int num_heads = 4;
  int seq_length = 64;
  int head_dim_k = 8;
  int head_dim_v = 8;

  int qkv_size = batch_size * num_heads * seq_length * head_dim_k;
  int value_size = batch_size * num_heads * seq_length * head_dim_v;
  int decay_size = batch_size * num_heads * seq_length * head_dim_k;
  int beta_size = batch_size * num_heads * seq_length;

  std::vector<float> query_data(qkv_size);
  std::vector<float> key_data(qkv_size);
  std::vector<float> value_data(value_size);
  std::vector<float> decay_data(decay_size);
  std::vector<float> beta_data(beta_size);

  for (int i = 0; i < qkv_size; ++i) {
    query_data[i] = 0.05f * (i % 20 - 10);
    key_data[i] = 0.05f * ((i + 7) % 20 - 10);
  }
  for (int i = 0; i < value_size; ++i) {
    value_data[i] = 0.05f * ((i + 13) % 20 - 10);
  }
  for (int i = 0; i < decay_size; ++i) {
    decay_data[i] = -0.05f * ((i % 5) + 1);
  }
  for (int i = 0; i < beta_size; ++i) {
    beta_data[i] = 0.3f + 0.1f * (i % 5);
  }

  RunLinearAttentionChunkParallelTest(
      query_data, key_data, value_data,
      nullptr, &decay_data, &beta_data,
      batch_size, num_heads, seq_length, head_dim_k, head_dim_v,
      "gated_delta", 16, 0.0f, TensorType::kFloat);

  // Also test FP16
  RunLinearAttentionChunkParallelTest(
      query_data, key_data, value_data,
      nullptr, &decay_data, &beta_data,
      batch_size, num_heads, seq_length, head_dim_k, head_dim_v,
      "gated_delta", 16, 0.0f, TensorType::kFloat16);
}

}  // namespace test
}  // namespace onnxruntime

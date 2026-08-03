// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// End-to-end correctness tests for the WebGPU PagedAttention kernel.
// Each test runs the full gather → unpack → ApplyFlashAttention → repack
// pipeline and compares against a CPU scaled-dot-product-attention reference.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "default_providers.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// Index into a (num_blocks, block_size, kv_num_heads, head_size) tensor.
int CacheIndex(int block_id, int slot_in_block, int kv_head, int dim,
               int block_size, int kv_num_heads, int head_size) {
  return ((block_id * block_size + slot_in_block) * kv_num_heads + kv_head) * head_size + dim;
}

}  // namespace

namespace {

struct EndToEndCase {
  int batch_size = 1;
  int token_count = 1;
  int num_heads = 1;
  int kv_num_heads = 1;
  int head_size = 8;
  int block_size = 256;
  int num_blocks = 2;
  int max_num_blocks_per_seq = 1;
  float scale = 0.0f;  // 0 means kernel default = 1/sqrt(head_size)
  std::vector<int32_t> cumulative_seqlens_q;
  std::vector<int32_t> past_seqlens;
  std::vector<int32_t> block_table;
};

// Softmax with causal masking: masked positions get -inf → 0 after exp.
// Uses fp32 throughout to establish a reference the fp16 kernel is compared
// against with a loose tolerance.
void CausalSoftmax(std::vector<float>& scores, int q_pos) {
  const int len = static_cast<int>(scores.size());
  float max_val = -std::numeric_limits<float>::infinity();
  for (int i = 0; i <= q_pos && i < len; ++i) {
    max_val = std::max(max_val, scores[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < len; ++i) {
    if (i > q_pos) {
      scores[i] = 0.0f;
    } else {
      scores[i] = std::exp(scores[i] - max_val);
      sum += scores[i];
    }
  }
  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < len; ++i) {
    scores[i] *= inv_sum;
  }
}

void RunEndToEndCase(const EndToEndCase& c) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  ASSERT_EQ(c.cumulative_seqlens_q.size(), static_cast<size_t>(c.batch_size + 1));
  ASSERT_EQ(c.past_seqlens.size(), static_cast<size_t>(c.batch_size));
  ASSERT_EQ(c.block_table.size(), static_cast<size_t>(c.batch_size * c.max_num_blocks_per_seq));
  ASSERT_EQ(c.cumulative_seqlens_q.back(), c.token_count);
  ASSERT_EQ(c.num_heads % c.kv_num_heads, 0);

  const int hidden_size = c.num_heads * c.head_size;
  const int kv_hidden_size = c.kv_num_heads * c.head_size;
  const int cache_elems = c.num_blocks * c.block_size * c.kv_num_heads * c.head_size;
  const int gqa_factor = c.num_heads / c.kv_num_heads;

  // ----- Inputs: tight values so fp16 SDPA stays within tolerance.
  std::vector<float> query_f(c.token_count * hidden_size);
  std::vector<float> key_f(c.token_count * kv_hidden_size);
  std::vector<float> value_f(c.token_count * kv_hidden_size);
  for (size_t i = 0; i < query_f.size(); ++i) {
    // Small range so QK products stay ~O(head_size * 0.01) → well within fp16.
    query_f[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 7) - 3);
  }
  for (size_t i = 0; i < key_f.size(); ++i) {
    key_f[i] = 0.03f * static_cast<float>((static_cast<int>(i) % 5) - 2);
    value_f[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 11) - 5);
  }
  std::vector<float> key_cache_f(cache_elems);
  std::vector<float> value_cache_f(cache_elems);
  for (int i = 0; i < cache_elems; ++i) {
    key_cache_f[i] = 0.01f * static_cast<float>((i % 9) - 4);
    value_cache_f[i] = 0.02f * static_cast<float>((i % 13) - 6);
  }

  const float scale = c.scale != 0.0f ? c.scale : 1.0f / std::sqrt(static_cast<float>(c.head_size));

  // ----- CPU reference.
  // 1) Apply scatter to a copy of the initial cache. This gives us the K/V
  //    that the WebGPU pipeline will see for each batch.
  std::vector<float> expected_key_cache_f = key_cache_f;
  std::vector<float> expected_value_cache_f = value_cache_f;
  for (int t = 0; t < c.token_count; ++t) {
    int seq_idx = 0;
    for (int b = 0; b < c.batch_size; ++b) {
      if (t < c.cumulative_seqlens_q[b + 1]) {
        seq_idx = b;
        break;
      }
    }
    const int local_tok = t - c.cumulative_seqlens_q[seq_idx];
    const int abs_slot = c.past_seqlens[seq_idx] + local_tok;
    const int block_off = abs_slot / c.block_size;
    const int slot_in_bk = abs_slot % c.block_size;
    const int block_id = c.block_table[seq_idx * c.max_num_blocks_per_seq + block_off];
    for (int h = 0; h < c.kv_num_heads; ++h) {
      for (int d = 0; d < c.head_size; ++d) {
        const int src = t * kv_hidden_size + h * c.head_size + d;
        const int dst = CacheIndex(block_id, slot_in_bk, h, d, c.block_size, c.kv_num_heads, c.head_size);
        expected_key_cache_f[dst] = key_f[src];
        expected_value_cache_f[dst] = value_f[src];
      }
    }
  }

  // 2) For each token, run causal SDPA against the (past + new) K/V window.
  std::vector<float> expected_output_f(c.token_count * hidden_size, 0.0f);
  for (int b = 0; b < c.batch_size; ++b) {
    const int cum_lo = c.cumulative_seqlens_q[b];
    const int cum_hi = c.cumulative_seqlens_q[b + 1];
    const int q_len = cum_hi - cum_lo;
    const int past = c.past_seqlens[b];
    const int total_kv_len = past + q_len;

    // Build K/V for this batch: shape (kv_num_heads, total_kv_len, head_size).
    // Each slot reads from the (post-scatter) paged cache.
    std::vector<float> k_window(c.kv_num_heads * total_kv_len * c.head_size);
    std::vector<float> v_window(c.kv_num_heads * total_kv_len * c.head_size);
    for (int s = 0; s < total_kv_len; ++s) {
      const int abs_slot = s;
      const int block_off = abs_slot / c.block_size;
      const int slot_in_bk = abs_slot % c.block_size;
      const int block_id = c.block_table[b * c.max_num_blocks_per_seq + block_off];
      for (int h = 0; h < c.kv_num_heads; ++h) {
        for (int d = 0; d < c.head_size; ++d) {
          const int src = CacheIndex(block_id, slot_in_bk, h, d, c.block_size, c.kv_num_heads, c.head_size);
          const int dst = (h * total_kv_len + s) * c.head_size + d;
          k_window[dst] = expected_key_cache_f[src];
          v_window[dst] = expected_value_cache_f[src];
        }
      }
    }

    // For each new query token, do causal SDPA.
    for (int local_tok = 0; local_tok < q_len; ++local_tok) {
      const int t = cum_lo + local_tok;
      const int q_pos = past + local_tok;  // Absolute position in the KV window.
      for (int n_q = 0; n_q < c.num_heads; ++n_q) {
        const int h_kv = n_q / gqa_factor;

        // scores[s] = scale * dot(q, k[s])
        std::vector<float> scores(total_kv_len, 0.0f);
        for (int s = 0; s < total_kv_len; ++s) {
          float dot = 0.0f;
          for (int d = 0; d < c.head_size; ++d) {
            const float q = query_f[t * hidden_size + n_q * c.head_size + d];
            const float k = k_window[(h_kv * total_kv_len + s) * c.head_size + d];
            dot += q * k;
          }
          scores[s] = dot * scale;
        }
        CausalSoftmax(scores, q_pos);

        // out[t, n_q * head_size + d] = sum_s scores[s] * v[s, d]
        for (int d = 0; d < c.head_size; ++d) {
          float acc = 0.0f;
          for (int s = 0; s <= q_pos && s < total_kv_len; ++s) {
            const float v = v_window[(h_kv * total_kv_len + s) * c.head_size + d];
            acc += scores[s] * v;
          }
          expected_output_f[t * hidden_size + n_q * c.head_size + d] = acc;
        }
      }
    }
  }

  OpTester test("PagedAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", c.num_heads);
  test.AddAttribute<int64_t>("kv_num_heads", c.kv_num_heads);
  test.AddAttribute<float>("scale", c.scale);
  test.AddAttribute<int64_t>("do_rotary", 0);

  test.AddInput<MLFloat16>("query", {c.token_count, hidden_size}, FloatsToMLFloat16s(query_f));
  test.AddInput<MLFloat16>("key", {c.token_count, kv_hidden_size}, FloatsToMLFloat16s(key_f));
  test.AddInput<MLFloat16>("value", {c.token_count, kv_hidden_size}, FloatsToMLFloat16s(value_f));
  test.AddInput<MLFloat16>("key_cache",
                           {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                           FloatsToMLFloat16s(key_cache_f));
  test.AddInput<MLFloat16>("value_cache",
                           {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                           FloatsToMLFloat16s(value_cache_f));
  test.AddInput<int32_t>("cumulative_sequence_length", {c.batch_size + 1}, c.cumulative_seqlens_q);
  test.AddInput<int32_t>("past_seqlens", {c.batch_size}, c.past_seqlens);
  test.AddInput<int32_t>("block_table", {c.batch_size, c.max_num_blocks_per_seq}, c.block_table);
  test.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  test.AddOptionalInputEdge<MLFloat16>();  // sin_cache

  test.AddOutput<MLFloat16>("output", {c.token_count, hidden_size}, FloatsToMLFloat16s(expected_output_f));
  test.AddOutput<MLFloat16>("key_cache_out",
                            {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                            FloatsToMLFloat16s(expected_key_cache_f));
  test.AddOutput<MLFloat16>("value_cache_out",
                            {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                            FloatsToMLFloat16s(expected_value_cache_f));

  // fp16 softmax + dot-product accumulation over a short KV window.
  // Reference is fp32; kernel is fp16. 2e-2 is a comfortable envelope for
  // these value ranges and short sequences.
  test.SetOutputAbsErr("output", 2e-2f);
  test.SetOutputAbsErr("key_cache_out", 1e-3f);
  test.SetOutputAbsErr("value_cache_out", 1e-3f);

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

// Decode tier: single batch, single new Q token, non-zero past. The FA
// tier selector uses `sequence_length_ < 32` → split-reduce path.
TEST(WebGpuPagedAttention, EndToEnd_Decode_SingleBatch_WithPast) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 1;
  c.cumulative_seqlens_q = {0, 1};
  c.past_seqlens = {4};  // total_kv_len = 5
  c.block_table = {0};
  RunEndToEndCase(c);
}

// Prefill tier: single batch, multiple new Q tokens, zero past. Exercises
// causal masking across new tokens.
TEST(WebGpuPagedAttention, EndToEnd_Prefill_SingleBatch_NoPast) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 4;
  c.cumulative_seqlens_q = {0, 4};
  c.past_seqlens = {0};
  c.block_table = {0};
  RunEndToEndCase(c);
}

// Multi-batch decode with differing past lengths — exercises variable-length
// packing across batches.
TEST(WebGpuPagedAttention, EndToEnd_Decode_MultiBatch_VariablePast) {
  EndToEndCase c{};
  c.batch_size = 2;
  c.token_count = 2;  // one new Q per batch
  c.num_blocks = 3;
  c.cumulative_seqlens_q = {0, 1, 2};
  c.past_seqlens = {3, 6};  // total_kv_len = 4 and 7 respectively
  c.block_table = {0, 2};   // seq 0 → block 0, seq 1 → block 2
  RunEndToEndCase(c);
}

// GQA (num_heads > kv_num_heads): broadcasts each KV head across gqa_factor
// query heads. Verifies the head-index mapping matches FA's convention.
TEST(WebGpuPagedAttention, EndToEnd_Decode_GQA) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 1;
  c.num_heads = 2;
  c.kv_num_heads = 1;  // GQA broadcast factor = 2
  c.cumulative_seqlens_q = {0, 1};
  c.past_seqlens = {3};
  c.block_table = {0};
  RunEndToEndCase(c);
}

// Mixed variable-length prefill (seq 0: 3 tokens, seq 1: 2 tokens) with
// non-zero past on both — the most realistic paged-attention scenario.
TEST(WebGpuPagedAttention, EndToEnd_MixedPrefillDecode_MultiBatch_VariablePast) {
  EndToEndCase c{};
  c.batch_size = 2;
  c.token_count = 5;
  c.num_heads = 2;
  c.kv_num_heads = 1;  // GQA
  c.num_blocks = 3;
  c.cumulative_seqlens_q = {0, 3, 5};
  c.past_seqlens = {2, 4};
  c.block_table = {0, 2};
  RunEndToEndCase(c);
}

}  // namespace test
}  // namespace onnxruntime

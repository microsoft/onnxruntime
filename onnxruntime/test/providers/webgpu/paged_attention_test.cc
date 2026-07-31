// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Per-program correctness tests for the WebGPU PagedAttention kernel.
//
// Phase 1b.1 covers the scatter K/V program (non-packed, no rotary).
// Phase 1b.2 adds a fused rotary + scatter path: rotated Q is routed into
// `output` (a temporary layering — the attention kernel dispatch lands in
// Phase 1b.3 and will overwrite `output` with real attention results) and
// rotated K + untouched V flow through the same scatter program from 1b.1.
// See docs/design/webgpu_paged_attention.md §5 for the phased plan.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "default_providers.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// Small helper: index into a (num_blocks, block_size, kv_num_heads, head_size)
// tensor stored in row-major order.
int CacheIndex(int block_id, int slot_in_block, int kv_head, int dim,
               int block_size, int kv_num_heads, int head_size) {
  return ((block_id * block_size + slot_in_block) * kv_num_heads + kv_head) * head_size + dim;
}

struct ScatterCase {
  int batch_size = 1;
  int token_count = 1;
  int num_heads = 1;
  int kv_num_heads = 1;
  int head_size = 8;     // helper enforces head_size % 8 == 0
  int block_size = 256;  // helper enforces block_size % 256 == 0
  int num_blocks = 2;
  int max_num_blocks_per_seq = 1;
  std::vector<int32_t> cumulative_seqlens_q;  // size batch_size + 1
  std::vector<int32_t> past_seqlens;          // size batch_size
  std::vector<int32_t> block_table;           // size batch_size * max_num_blocks_per_seq
};

// Build inputs, run the scatter path through PagedAttention, and validate that
// each new (token, kv_head, dim) element lands in the correct cache slot while
// all untouched slots retain their initial values.
void RunScatterCase(const ScatterCase& c) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const int hidden_size = c.num_heads * c.head_size;
  const int kv_hidden_size = c.kv_num_heads * c.head_size;
  const int cache_elems = c.num_blocks * c.block_size * c.kv_num_heads * c.head_size;

  ASSERT_EQ(c.cumulative_seqlens_q.size(), static_cast<size_t>(c.batch_size + 1));
  ASSERT_EQ(c.past_seqlens.size(), static_cast<size_t>(c.batch_size));
  ASSERT_EQ(c.block_table.size(), static_cast<size_t>(c.batch_size * c.max_num_blocks_per_seq));
  ASSERT_EQ(c.cumulative_seqlens_q.back(), c.token_count);

  // ----- Query: zeros (unused by scatter, but the op requires the tensor).
  std::vector<float> query_f(c.token_count * hidden_size, 0.0f);

  // ----- Key / value: distinct, deterministic patterns so mis-scatters
  // surface as concrete diffs. Keep values in a tight fp16-friendly range.
  std::vector<float> key_f(c.token_count * kv_hidden_size);
  std::vector<float> value_f(c.token_count * kv_hidden_size);
  for (size_t i = 0; i < key_f.size(); ++i) {
    key_f[i] = 0.01f * static_cast<float>(i + 1);
    value_f[i] = -0.01f * static_cast<float>(i + 1);
  }

  // ----- Initial K/V cache: distinct pattern per slot so untouched slots are
  // provably preserved.
  std::vector<float> key_cache_f(cache_elems);
  std::vector<float> value_cache_f(cache_elems);
  for (int i = 0; i < cache_elems; ++i) {
    key_cache_f[i] = 0.001f * static_cast<float>(i + 1);
    value_cache_f[i] = -0.001f * static_cast<float>(i + 1);
  }

  // ----- CPU reference: apply the same scatter to a copy of the initial
  // cache. Address model must match paged_attention_scatter_kv.wgsl.template.
  std::vector<float> expected_key_cache_f = key_cache_f;
  std::vector<float> expected_value_cache_f = value_cache_f;
  for (int t = 0; t < c.token_count; ++t) {
    // Find which sequence token t belongs to.
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

  // Output tensor is zero-filled by the phase-1b.1 kernel (attention path not
  // yet dispatched). Expected value is exact zero.
  const int output_elems = c.token_count * hidden_size;
  std::vector<float> expected_output_f(output_elems, 0.0f);

  OpTester test("PagedAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", c.num_heads);
  test.AddAttribute<int64_t>("kv_num_heads", c.kv_num_heads);
  test.AddAttribute<float>("scale", 0.0f);
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
  test.AddInput<int32_t>("cumulative_sequence_length", {c.batch_size + 1},
                         c.cumulative_seqlens_q);
  test.AddInput<int32_t>("past_seqlens", {c.batch_size}, c.past_seqlens);
  test.AddInput<int32_t>("block_table", {c.batch_size, c.max_num_blocks_per_seq},
                         c.block_table);
  test.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  test.AddOptionalInputEdge<MLFloat16>();  // sin_cache

  test.AddOutput<MLFloat16>("output", {c.token_count, hidden_size},
                            FloatsToMLFloat16s(expected_output_f));
  test.AddOutput<MLFloat16>("key_cache_out",
                            {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                            FloatsToMLFloat16s(expected_key_cache_f));
  test.AddOutput<MLFloat16>("value_cache_out",
                            {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                            FloatsToMLFloat16s(expected_value_cache_f));

  // fp16 quantisation tolerance. Both the scattered values and the untouched
  // initial values live in a tight range, so a small absolute error is fine.
  test.SetOutputAbsErr("output", 1e-3f);
  test.SetOutputAbsErr("key_cache_out", 1e-3f);
  test.SetOutputAbsErr("value_cache_out", 1e-3f);

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

// -----------------------------------------------------------------------------
// Scatter-only (no rotary, non-packed) — Phase 1b.1
// -----------------------------------------------------------------------------

// Simplest case: one sequence, one new token, no past. Verifies the scatter
// hits slot [block_id=0, slot=0, head=0, :] and leaves every other slot alone.
TEST(WebGpuPagedAttention, ScatterOnly_SingleToken_SingleBatch_NoPast) {
  ScatterCase c{};
  c.batch_size = 1;
  c.token_count = 1;
  c.cumulative_seqlens_q = {0, 1};
  c.past_seqlens = {0};
  c.block_table = {0};
  RunScatterCase(c);
}

// Multiple new tokens for a single sequence with a non-zero past. Exercises
// the `abs_slot = past_len + local_tok` arithmetic across several slots within
// one block.
TEST(WebGpuPagedAttention, ScatterOnly_MultiToken_SingleBatch_WithPast) {
  ScatterCase c{};
  c.batch_size = 1;
  c.token_count = 3;
  c.cumulative_seqlens_q = {0, 3};
  c.past_seqlens = {5};  // new tokens land in slots 5, 6, 7 of block 0
  c.block_table = {0};
  RunScatterCase(c);
}

// Two sequences with different past lengths and different physical blocks —
// exercises the batch→block_table indirection alongside the multi-KV-head
// layout.
TEST(WebGpuPagedAttention, ScatterOnly_MultiBatch_MultiHead_WithPast) {
  ScatterCase c{};
  c.batch_size = 2;
  c.token_count = 5;   // seq 0: 3 tokens, seq 1: 2 tokens
  c.num_heads = 2;     // hidden_size = 2 * 8 = 16
  c.kv_num_heads = 1;  // GQA broadcast factor = 2
  c.num_blocks = 3;    // seqs pull from blocks 0 and 2
  c.cumulative_seqlens_q = {0, 3, 5};
  c.past_seqlens = {4, 0};  // seq 0 slots 4-6 of block 0; seq 1 slots 0-1 of block 2
  c.block_table = {0, 2};   // seq 0 → block 0, seq 1 → block 2
  RunScatterCase(c);
}

// -----------------------------------------------------------------------------
// Rotary + scatter (non-packed) — Phase 1b.2
//
// Same address model as the scatter-only tests, plus a rotary embedding stage
// that mirrors paged_attention_impl.cu::RotaryEmbeddingTNH. The op writes the
// rotated Q into `output` (a temporary layering until the attention kernel
// lands in 1b.3) and scatters the rotated K + untouched V into the paged
// cache. Values in the head are pass-through when `h >= rotary_dim`.
// -----------------------------------------------------------------------------

namespace {

struct RotaryCase {
  int batch_size = 1;
  int token_count = 1;
  int num_heads = 1;
  int kv_num_heads = 1;
  int head_size = 16;   // rotary path helper enforces head_size % 16 == 0
  int rotary_dim = 16;  // must divide 2 evenly and be <= head_size
  int block_size = 256;
  int num_blocks = 2;
  int max_num_blocks_per_seq = 1;
  bool interleaved = false;
  std::vector<int32_t> cumulative_seqlens_q;
  std::vector<int32_t> past_seqlens;
  std::vector<int32_t> block_table;
};

// Apply the rotary transform to one packed 2-D token tensor
// (token_count, n_heads*head_size). Address model must match
// paged_attention_rotary.wgsl.template exactly.
std::vector<float> ApplyRotary(const std::vector<float>& in,
                               int token_count, int n_heads, int head_size,
                               int rotary_dim, bool interleaved,
                               const std::vector<int32_t>& cum_seqlens,
                               const std::vector<int32_t>& past_seqlens,
                               int batch_size,
                               const std::vector<float>& cos_cache,
                               const std::vector<float>& sin_cache,
                               int cos_cols) {
  const int half_rot = rotary_dim / 2;
  std::vector<float> out(in.size(), 0.0f);
  for (int t = 0; t < token_count; ++t) {
    int seq_idx = 0;
    for (int b = 0; b < batch_size; ++b) {
      if (t < cum_seqlens[b + 1]) {
        seq_idx = b;
        break;
      }
    }
    const int local_tok = t - cum_seqlens[seq_idx];
    const int position_id = past_seqlens[seq_idx] + local_tok;
    const int hidden = n_heads * head_size;
    for (int n = 0; n < n_heads; ++n) {
      const int head_off = n * head_size;
      for (int h = 0; h < head_size; ++h) {
        const int dst = t * hidden + head_off + h;
        if (h >= rotary_dim) {
          out[dst] = in[dst];
          continue;
        }
        int cache_idx = 0;
        int j = 0;
        bool sign_pos = false;
        if (interleaved) {
          cache_idx = (h / 2) % half_rot;
          const bool even = (h % 2 == 0);
          j = even ? (h + 1) : (h - 1);
          sign_pos = !even;
        } else {
          cache_idx = h % half_rot;
          j = (h + half_rot) % rotary_dim;
          sign_pos = (h >= half_rot);
        }
        const float cos_v = cos_cache[position_id * cos_cols + cache_idx];
        const float sin_v = sin_cache[position_id * cos_cols + cache_idx];
        const float x_h = in[t * hidden + head_off + h];
        const float x_j = in[t * hidden + head_off + j];
        const float sin_term = sign_pos ? (x_j * sin_v) : (-x_j * sin_v);
        out[dst] = x_h * cos_v + sin_term;
      }
    }
  }
  return out;
}

void RunRotaryCase(const RotaryCase& c) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  ASSERT_EQ(c.rotary_dim % 2, 0);
  ASSERT_LE(c.rotary_dim, c.head_size);
  ASSERT_EQ(c.cumulative_seqlens_q.size(), static_cast<size_t>(c.batch_size + 1));
  ASSERT_EQ(c.past_seqlens.size(), static_cast<size_t>(c.batch_size));
  ASSERT_EQ(c.block_table.size(), static_cast<size_t>(c.batch_size * c.max_num_blocks_per_seq));
  ASSERT_EQ(c.cumulative_seqlens_q.back(), c.token_count);

  const int hidden_size = c.num_heads * c.head_size;
  const int kv_hidden_size = c.kv_num_heads * c.head_size;
  const int cache_elems = c.num_blocks * c.block_size * c.kv_num_heads * c.head_size;

  // Enough positions to cover the largest (past + local_tok) across all seqs.
  int max_position = 0;
  for (int b = 0; b < c.batch_size; ++b) {
    const int seq_len = c.cumulative_seqlens_q[b + 1] - c.cumulative_seqlens_q[b];
    max_position = std::max(max_position, c.past_seqlens[b] + seq_len);
  }
  const int cos_rows = std::max(max_position, 1);
  const int cos_cols = c.rotary_dim / 2;

  // Deterministic Q / K / V patterns.
  std::vector<float> query_f(c.token_count * hidden_size);
  std::vector<float> key_f(c.token_count * kv_hidden_size);
  std::vector<float> value_f(c.token_count * kv_hidden_size);
  for (size_t i = 0; i < query_f.size(); ++i) {
    query_f[i] = 0.005f * static_cast<float>(i + 1);
  }
  for (size_t i = 0; i < key_f.size(); ++i) {
    key_f[i] = 0.01f * static_cast<float>(i + 1);
    value_f[i] = -0.01f * static_cast<float>(i + 1);
  }

  // Cos / sin caches: bounded values so fp16 accumulation stays tight.
  std::vector<float> cos_cache_f(cos_rows * cos_cols);
  std::vector<float> sin_cache_f(cos_rows * cos_cols);
  for (int r = 0; r < cos_rows; ++r) {
    for (int col = 0; col < cos_cols; ++col) {
      const float theta = 0.1f * static_cast<float>(r) + 0.05f * static_cast<float>(col);
      cos_cache_f[r * cos_cols + col] = std::cos(theta);
      sin_cache_f[r * cos_cols + col] = std::sin(theta);
    }
  }

  // Cache pattern reused from the scatter tests.
  std::vector<float> key_cache_f(cache_elems);
  std::vector<float> value_cache_f(cache_elems);
  for (int i = 0; i < cache_elems; ++i) {
    key_cache_f[i] = 0.001f * static_cast<float>(i + 1);
    value_cache_f[i] = -0.001f * static_cast<float>(i + 1);
  }

  // ----- CPU reference.
  // 1) Rotate Q → expected_output.
  const std::vector<float> expected_output_f =
      ApplyRotary(query_f, c.token_count, c.num_heads, c.head_size,
                  c.rotary_dim, c.interleaved, c.cumulative_seqlens_q,
                  c.past_seqlens, c.batch_size, cos_cache_f, sin_cache_f, cos_cols);
  // 2) Rotate K → then scatter into cache.
  const std::vector<float> rotated_key_f =
      ApplyRotary(key_f, c.token_count, c.kv_num_heads, c.head_size,
                  c.rotary_dim, c.interleaved, c.cumulative_seqlens_q,
                  c.past_seqlens, c.batch_size, cos_cache_f, sin_cache_f, cos_cols);

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
        expected_key_cache_f[dst] = rotated_key_f[src];
        expected_value_cache_f[dst] = value_f[src];
      }
    }
  }

  OpTester test("PagedAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", c.num_heads);
  test.AddAttribute<int64_t>("kv_num_heads", c.kv_num_heads);
  test.AddAttribute<float>("scale", 0.0f);
  test.AddAttribute<int64_t>("do_rotary", 1);
  test.AddAttribute<int64_t>("rotary_interleaved", c.interleaved ? 1 : 0);

  test.AddInput<MLFloat16>("query", {c.token_count, hidden_size}, FloatsToMLFloat16s(query_f));
  test.AddInput<MLFloat16>("key", {c.token_count, kv_hidden_size}, FloatsToMLFloat16s(key_f));
  test.AddInput<MLFloat16>("value", {c.token_count, kv_hidden_size}, FloatsToMLFloat16s(value_f));
  test.AddInput<MLFloat16>("key_cache",
                           {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                           FloatsToMLFloat16s(key_cache_f));
  test.AddInput<MLFloat16>("value_cache",
                           {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                           FloatsToMLFloat16s(value_cache_f));
  test.AddInput<int32_t>("cumulative_sequence_length", {c.batch_size + 1},
                         c.cumulative_seqlens_q);
  test.AddInput<int32_t>("past_seqlens", {c.batch_size}, c.past_seqlens);
  test.AddInput<int32_t>("block_table", {c.batch_size, c.max_num_blocks_per_seq},
                         c.block_table);
  test.AddInput<MLFloat16>("cos_cache", {cos_rows, cos_cols}, FloatsToMLFloat16s(cos_cache_f));
  test.AddInput<MLFloat16>("sin_cache", {cos_rows, cos_cols}, FloatsToMLFloat16s(sin_cache_f));

  test.AddOutput<MLFloat16>("output", {c.token_count, hidden_size},
                            FloatsToMLFloat16s(expected_output_f));
  test.AddOutput<MLFloat16>("key_cache_out",
                            {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                            FloatsToMLFloat16s(expected_key_cache_f));
  test.AddOutput<MLFloat16>("value_cache_out",
                            {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                            FloatsToMLFloat16s(expected_value_cache_f));

  // fp16 rotation involves a multiply-add of two fp16 values plus a cos/sin
  // stored in fp16 → 3e-3 is a comfortable envelope for the value ranges above.
  test.SetOutputAbsErr("output", 3e-3f);
  test.SetOutputAbsErr("key_cache_out", 3e-3f);
  test.SetOutputAbsErr("value_cache_out", 3e-3f);

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

// Full-head rotary, non-interleaved (split) layout, single new token.
TEST(WebGpuPagedAttention, Rotary_NonInterleaved_SingleToken) {
  RotaryCase c{};
  c.batch_size = 1;
  c.token_count = 1;
  c.head_size = 16;
  c.rotary_dim = 16;
  c.interleaved = false;
  c.cumulative_seqlens_q = {0, 1};
  c.past_seqlens = {0};
  c.block_table = {0};
  RunRotaryCase(c);
}

// Full-head rotary, interleaved layout, multiple new tokens with non-zero
// past — exercises the `position_id = past + s` shift across positions.
TEST(WebGpuPagedAttention, Rotary_Interleaved_MultiToken) {
  RotaryCase c{};
  c.batch_size = 1;
  c.token_count = 3;
  c.head_size = 16;
  c.rotary_dim = 16;
  c.interleaved = true;
  c.cumulative_seqlens_q = {0, 3};
  c.past_seqlens = {2};
  c.block_table = {0};
  RunRotaryCase(c);
}

// rotary_dim < head_size, multi-head, multi-batch — exercises tail
// pass-through and both batch-branching and GQA broadcast simultaneously.
// (Helper requires rotary_dim/2 to be a multiple of 8, so head_size=32 and
// rotary_dim=16 is the smallest tail-exercising configuration.)
TEST(WebGpuPagedAttention, Rotary_PartialDim_MultiBatch_MultiHead) {
  RotaryCase c{};
  c.batch_size = 2;
  c.token_count = 5;  // seq 0: 3 tokens, seq 1: 2 tokens
  c.num_heads = 2;
  c.kv_num_heads = 1;  // GQA broadcast factor = 2
  c.head_size = 32;
  c.rotary_dim = 16;  // tail dims 16..31 must be copied through
  c.interleaved = false;
  c.num_blocks = 3;
  c.cumulative_seqlens_q = {0, 3, 5};
  c.past_seqlens = {4, 0};
  c.block_table = {0, 2};
  RunRotaryCase(c);
}

}  // namespace test
}  // namespace onnxruntime

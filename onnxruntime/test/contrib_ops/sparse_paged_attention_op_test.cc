// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "default_providers.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// A deterministic single-token / single-head configuration. The query is all
// zeros, so every logit is exactly 0 and the softmax reduces to a plain average
// over the attended value vectors. That makes the expected outputs exact in
// FP16 and independent of the reduction order the shaders happen to use.
constexpr int kHeadSize = 8;
constexpr int kBlockSize = 16;
constexpr int kCacheElems = kBlockSize * kHeadSize;

void AddCommonInputs(OpTester& tester, float current_value) {
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(current_value)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kBlockSize * kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kBlockSize * kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {0});
  tester.AddInput<int32_t>("selected_indices", {1, 2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
}

void RunCuda(OpTester& tester) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

void RunWebGpu(OpTester& tester,
               OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
               const std::string& expected_failure_string = "") {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(webgpu_ep, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  tester.Run(expect_result, expected_failure_string, {}, nullptr, &execution_providers);
}

// Same layout as AddCommonInputs, but with caller-controlled cache contents and
// slot_mapping so the KV-cache write path can be observed directly.
void AddCacheProbeInputs(OpTester& tester,
                         float current_value,
                         float cached_value_at_slot0,
                         int32_t slot_mapping_value) {
  std::vector<MLFloat16> value_cache_data(kCacheElems, MLFloat16(0.0f));
  for (int i = 0; i < kHeadSize; ++i) {
    value_cache_data[i] = MLFloat16(cached_value_at_slot0);
  }

  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(current_value)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache_data);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {slot_mapping_value});
  tester.AddInput<int32_t>("selected_indices", {1, 2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
}

// A row of a (num_blocks, block_size, 1 kv head, head_size) cache, addressed by
// the flat slot index (physical_block * block_size + slot_in_block).
void SetCacheRow(std::vector<MLFloat16>& cache, int slot, int head_size,
                 const std::vector<float>& row) {
  for (int i = 0; i < head_size; ++i) {
    cache[slot * head_size + i] =
        MLFloat16(i < static_cast<int>(row.size()) ? row[i] : 0.0f);
  }
}

// Every value vector in the numeric tests below is constant across head_size, so
// a single scalar describes it and the expected output is the same in each
// dimension.
void SetConstantCacheRow(std::vector<MLFloat16>& cache, int slot, int head_size, float value) {
  for (int i = 0; i < head_size; ++i) {
    cache[slot * head_size + i] = MLFloat16(value);
  }
}

// One softmax over the given logits, weighting the matching scalar values. This
// is the definition the op must implement across *all* of its candidate
// sources, and it is what the staged WebGPU shaders produce after merging their
// partial softmax states.
float JointSoftmax(const std::vector<float>& logits, const std::vector<float>& values) {
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    const double weight = std::exp(static_cast<double>(logits[i] - max_logit));
    numerator += weight * values[i];
    denominator += weight;
  }
  return static_cast<float>(numerator / denominator);
}

}  // namespace

TEST(SparsePagedAttention, Cuda_SelectedMainWritesAndReadsPagedCache) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 2.0f);
  tester.AddOutput<MLFloat16>("output", {1, 8},
                              std::vector<MLFloat16>(8, MLFloat16(2.0f)));
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_LocalAndSharedAuxiliaryUseJointSoftmax) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 1.0f);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddAttribute<int64_t>("auxiliary_kv_shared", 1);
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 1, 1, 8},
                             std::vector<MLFloat16>(8, MLFloat16(3.0f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, 8},
                              std::vector<MLFloat16>(8, MLFloat16(2.0f)));
  RunCuda(tester);
}

// ---------------------------------------------------------------------------
// WebGPU
// ---------------------------------------------------------------------------

// selected_only + selected_kv_source='main': the current token is written into
// the paged cache through slot_mapping and then attended through the selection,
// so the output must be the value that was just stored.
TEST(SparsePagedAttention, WebGpu_SelectedMainWritesAndReadsPagedCache) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 2.0f);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(2.0f)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// local_plus_selected + selected_kv_source='auxiliary' with a shared auxiliary
// K/V tensor. The local window contributes the main-cache value (1.0) and the
// selection contributes the auxiliary value (3.0). Because both partial states
// are merged into a single softmax, the result is their mean (2.0). Two
// independent softmaxes averaged afterwards would give the same number only by
// coincidence here, so the CUDA reference test uses the same configuration and
// both kernels must agree.
TEST(SparsePagedAttention, WebGpu_LocalAndSharedAuxiliaryUseJointSoftmax) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 1.0f);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddAttribute<int64_t>("auxiliary_kv_shared", 1);
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 1, 1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(3.0f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(2.0f)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// A non-negative slot_mapping entry stores the current K/V and the cache
// outputs must observe that store.
TEST(SparsePagedAttention, WebGpu_SlotMappingWritesCacheOutputs) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> expected_value_cache(kCacheElems, MLFloat16(0.0f));
  for (int i = 0; i < kHeadSize; ++i) {
    expected_value_cache[i] = MLFloat16(7.0f);
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCacheProbeInputs(tester, /*current_value*/ 7.0f, /*cached_value_at_slot0*/ 5.0f,
                      /*slot_mapping_value*/ 0);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(7.0f)));
  tester.AddOutput<MLFloat16>("key_cache_out", {1, kBlockSize, 1, kHeadSize},
                              std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("value_cache_out", {1, kBlockSize, 1, kHeadSize},
                              expected_value_cache);
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// slot_mapping == -1 suppresses the store. The previously cached value must
// survive in value_cache_out and must be what the selection reads back, proving
// the suppression happens before the write rather than after it.
TEST(SparsePagedAttention, WebGpu_NegativeSlotMappingSuppressesCacheWrite) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> expected_value_cache(kCacheElems, MLFloat16(0.0f));
  for (int i = 0; i < kHeadSize; ++i) {
    expected_value_cache[i] = MLFloat16(5.0f);
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCacheProbeInputs(tester, /*current_value*/ 7.0f, /*cached_value_at_slot0*/ 5.0f,
                      /*slot_mapping_value*/ -1);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(5.0f)));
  tester.AddOutput<MLFloat16>("key_cache_out", {1, kBlockSize, 1, kHeadSize},
                              std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("value_cache_out", {1, kBlockSize, 1, kHeadSize},
                              expected_value_cache);
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// Validation: selected_indices must be 2-D (token_count, max_selected_entries).
// A rank-1 tensor is type-correct, so only the kernel can reject it, and it must
// do so explicitly instead of reinterpreting the buffer.
TEST(SparsePagedAttention, WebGpu_RejectsMalformedSelectedIndices) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(1.0f)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {0});
  tester.AddInput<int32_t>("selected_indices", {2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  RunWebGpu(tester, OpTester::ExpectResult::kExpectFailure,
            "selected_indices must have shape");
}

// Validation: auxiliary inputs are meaningless when the selection addresses the
// main cache, and are rejected rather than silently ignored.
TEST(SparsePagedAttention, WebGpu_RejectsAuxiliaryInputsWithMainSelection) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 1.0f);
  tester.AddAttribute<std::string>("selected_kv_source", "main");
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 1, 1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(3.0f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  RunWebGpu(tester, OpTester::ExpectResult::kExpectFailure,
            "Auxiliary inputs must be absent");
}

// The distinguishing property of this op is that the main-cache candidates and
// the auxiliary candidates share *one* softmax. This test makes that observable:
// the logits are non-zero and non-uniform, the two sources contribute a
// different number of candidates (3 local vs. 2 auxiliary), and the source with
// the larger logits carries the smaller values. Averaging two independent
// softmaxes would land far away from the joint result, and the test asserts that
// the two references really do differ so it cannot silently stop discriminating.
TEST(SparsePagedAttention, WebGpu_JointSoftmaxDiffersFromIndependentAveraging) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  // scale=1 and a one-hot query turn every logit into the key's first element,
  // so the expected distribution is exact rather than a floating-point artifact
  // of 1/sqrt(head_size).
  std::vector<MLFloat16> query(kHeadSize, MLFloat16(0.0f));
  query[0] = MLFloat16(1.0f);

  std::vector<MLFloat16> key_cache(kCacheElems, MLFloat16(0.0f));
  SetCacheRow(key_cache, 0, kHeadSize, {2.0f});
  SetCacheRow(key_cache, 1, kHeadSize, {1.0f});
  SetCacheRow(key_cache, 2, kHeadSize, {0.0f});

  std::vector<MLFloat16> value_cache(kCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(value_cache, 0, kHeadSize, 1.0f);
  SetConstantCacheRow(value_cache, 1, kHeadSize, 2.0f);
  SetConstantCacheRow(value_cache, 2, kHeadSize, 3.0f);

  constexpr int kAuxCapacity = 4;
  std::vector<MLFloat16> auxiliary_key(kAuxCapacity * kHeadSize, MLFloat16(0.0f));
  SetCacheRow(auxiliary_key, 0, kHeadSize, {-1.0f});
  SetCacheRow(auxiliary_key, 1, kHeadSize, {-2.0f});
  std::vector<MLFloat16> auxiliary_value(kAuxCapacity * kHeadSize, MLFloat16(0.0f));
  SetConstantCacheRow(auxiliary_value, 0, kHeadSize, 4.0f);
  SetConstantCacheRow(auxiliary_value, 1, kHeadSize, 5.0f);

  const std::vector<float> main_logits{2.0f, 1.0f, 0.0f};
  const std::vector<float> main_values{1.0f, 2.0f, 3.0f};
  const std::vector<float> aux_logits{-1.0f, -2.0f};
  const std::vector<float> aux_values{4.0f, 5.0f};
  std::vector<float> all_logits(main_logits);
  all_logits.insert(all_logits.end(), aux_logits.begin(), aux_logits.end());
  std::vector<float> all_values(main_values);
  all_values.insert(all_values.end(), aux_values.begin(), aux_values.end());

  const float expected = JointSoftmax(all_logits, all_values);
  const float independent_average =
      0.5f * (JointSoftmax(main_logits, main_values) + JointSoftmax(aux_logits, aux_values));
  ASSERT_GT(std::abs(expected - independent_average), 0.5f)
      << "the configuration no longer separates a joint softmax from independent averaging";

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddAttribute<int64_t>("local_window_size", 3);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize}, key_cache);
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  // The token sits at position 2, so the causal local window of 3 covers the
  // pre-populated cache slots 0..2.
  tester.AddInput<int32_t>("past_seqlens", {1}, {2});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  // -1 keeps the current K/V out of the cache so the attended rows are exactly
  // the ones written above.
  tester.AddInput<int32_t>("slot_mapping", {1}, {-1});
  tester.AddInput<int32_t>("selected_indices", {1, 3}, {0, 1, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {2});
  tester.AddInput<MLFloat16>("auxiliary_key", {1, kAuxCapacity, 1, kHeadSize}, auxiliary_key);
  tester.AddInput<MLFloat16>("auxiliary_value", {1, kAuxCapacity, 1, kHeadSize}, auxiliary_value);
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {2});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(expected)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// Two requests packed into one call, two query heads sharing one KV head, and a
// per-token selection. The two heads read the same cache rows through different
// query vectors, so a broken GQA head mapping or a broken request lookup shows
// up as a wrong value rather than as a shape error.
TEST(SparsePagedAttention, WebGpu_GroupedQueryMultiRequestSelection) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  constexpr int kGqaHeadSize = 4;
  constexpr int kNumHeads = 2;
  constexpr int kNumBlocks = 2;
  constexpr int kGqaCacheElems = kNumBlocks * kBlockSize * kGqaHeadSize;

  // head 0 reads key element 0, head 1 reads key element 1.
  const std::vector<MLFloat16> query{
      MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),   // token 0, head 0
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f),   // token 0, head 1
      MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),   // token 1, head 0
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f),   // token 1, head 1
      MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),   // token 2, head 0
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f)};  // token 2, head 1

  // The current K/V of all three tokens is stored through slot_mapping and then
  // read back through the selection, so the scatter is part of what is verified.
  const std::vector<MLFloat16> key{
      MLFloat16(1.0f), MLFloat16(0.5f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.5f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f)};
  const std::vector<MLFloat16> value{
      MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f),
      MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(2.0f),
      MLFloat16(3.0f), MLFloat16(3.0f), MLFloat16(3.0f), MLFloat16(3.0f)};

  // Request 1 owns block 1; its position 0 is pre-populated, its position 3 is
  // the token stored below at flat slot 1 * kBlockSize + 3.
  std::vector<MLFloat16> key_cache(kGqaCacheElems, MLFloat16(0.0f));
  SetCacheRow(key_cache, kBlockSize, kGqaHeadSize, {2.0f, 1.0f});
  std::vector<MLFloat16> value_cache(kGqaCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(value_cache, kBlockSize, kGqaHeadSize, 4.0f);

  const float token0_head0 = JointSoftmax({1.0f}, {1.0f});
  const float token0_head1 = JointSoftmax({0.5f}, {1.0f});
  const float token1_head0 = JointSoftmax({1.0f, 0.0f}, {1.0f, 2.0f});
  const float token1_head1 = JointSoftmax({0.5f, 1.0f}, {1.0f, 2.0f});
  const float token2_head0 = JointSoftmax({2.0f, 0.5f}, {4.0f, 3.0f});
  const float token2_head1 = JointSoftmax({1.0f, 0.0f}, {4.0f, 3.0f});
  ASSERT_NE(token1_head0, token1_head1);
  ASSERT_NE(token2_head0, token2_head1);

  std::vector<MLFloat16> expected;
  for (float head_value : {token0_head0, token0_head1, token1_head0, token1_head1, token2_head0,
                           token2_head1}) {
    expected.insert(expected.end(), kGqaHeadSize, MLFloat16(head_value));
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", kNumHeads);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddInput<MLFloat16>("query", {3, kNumHeads * kGqaHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {3, kGqaHeadSize}, key);
  tester.AddInput<MLFloat16>("value", {3, kGqaHeadSize}, value);
  tester.AddInput<MLFloat16>("key_cache", {kNumBlocks, kBlockSize, 1, kGqaHeadSize}, key_cache);
  tester.AddInput<MLFloat16>("value_cache", {kNumBlocks, kBlockSize, 1, kGqaHeadSize}, value_cache);
  // Request 0 owns tokens 0..1 at positions 0..1; request 1 owns token 2 at
  // position 3 (three cached positions precede it).
  tester.AddInput<int32_t>("cumulative_sequence_length", {3}, {0, 2, 3});
  tester.AddInput<int32_t>("past_seqlens", {2}, {0, 3});
  tester.AddInput<int32_t>("block_table", {2, 1}, {0, 1});
  tester.AddInput<int32_t>("slot_mapping", {3}, {0, 1, kBlockSize + 3});
  tester.AddInput<int32_t>("selected_indices", {3, 2}, {0, -1, 0, 1, 0, 3});
  tester.AddInput<int32_t>("selected_counts", {3}, {1, 2, 2});
  tester.AddOutput<MLFloat16>("output", {3, kNumHeads * kGqaHeadSize}, expected);
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// past_seqlens is caller-owned device data that the op never reads on the host.
// An extreme value must not overflow the i32 metadata arithmetic or turn into an
// unbounded local-window walk: the sanitized lengths are clamped to the
// positions the block table can address, here the 16 slots of the single block.
// With a zero query every logit is 0, so the result is the plain mean of those
// 16 cached value rows. Without the clamp this configuration asks the shader for
// ~2^31 candidates.
TEST(SparsePagedAttention, WebGpu_ExtremePastSeqlenClampsToCacheCapacity) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> value_cache(kCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(value_cache, 0, kHeadSize, 8.0f);
  const float expected = 8.0f / static_cast<float>(kBlockSize);

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  // An unbounded local window, so the candidate count is decided purely by the
  // sanitized lengths.
  tester.AddAttribute<int64_t>("local_window_size", -1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {std::numeric_limits<int32_t>::max()});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {-1});
  // Position 0 is already inside the local window, so the de-duplication rule
  // must keep it from being weighted twice.
  tester.AddInput<int32_t>("selected_indices", {1, 2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(expected)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// The same guarantee for cumulative_sequence_length: a length far beyond the
// packed token count must not overflow main_length. Sanitization clamps the
// cumulative range to the token count, so the single token keeps position 0 and
// attends the one selected cache row.
TEST(SparsePagedAttention, WebGpu_ExtremeCumulativeSequenceLengthIsSanitized) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> value_cache(kCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(value_cache, 0, kHeadSize, 6.0f);

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2},
                           {0, std::numeric_limits<int32_t>::max()});
  tester.AddInput<int32_t>("past_seqlens", {1}, {std::numeric_limits<int32_t>::max()});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {-1});
  tester.AddInput<int32_t>("selected_indices", {1, 2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(6.0f)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

}  // namespace test
}  // namespace onnxruntime

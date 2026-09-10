// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

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

}  // namespace test
}  // namespace onnxruntime

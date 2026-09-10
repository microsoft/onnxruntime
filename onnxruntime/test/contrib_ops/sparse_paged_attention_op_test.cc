// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "default_providers.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

void AddCommonInputs(OpTester& tester, float current_value) {
  constexpr int kHeadSize = 8;
  constexpr int kBlockSize = 16;
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

}  // namespace test
}  // namespace onnxruntime

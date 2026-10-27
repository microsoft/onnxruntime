#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_merge_common.h"
#include "core/graph/constants.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

void RunMerge(int64_t cached_rows, int64_t base_capacity, int64_t output_capacity,
              const std::vector<int32_t>& base, const std::vector<int32_t>& counts,
              const std::vector<int32_t>& rows, const std::vector<int32_t>& starts,
              const std::vector<int32_t>& ends, const std::vector<int32_t>& expected,
              const std::vector<int32_t>& expected_counts, const std::vector<int32_t>& status) {
  bool tested = false;
  for (int provider_kind = 0; provider_kind < 2; ++provider_kind) {
    std::unique_ptr<IExecutionProvider> provider;
    if (provider_kind == 0) provider = DefaultCudaExecutionProvider();
#ifdef USE_WEBGPU
    if (provider_kind == 1) provider = DefaultWebGpuExecutionProvider();
#endif
    if (!provider) continue;
    tested = true;
    OpTester test("PackedSparseAttentionIndexerMerge", 1, kMSDomain);
    test.AddAttribute("policy_mode", std::string("append_range"));
    test.AddAttribute("max_output_entries", output_capacity);
    const int64_t queries = static_cast<int64_t>(rows.size());
    test.AddInput<int32_t>("base_indices", {cached_rows, base_capacity}, base);
    test.AddInput<int32_t>("base_counts", {cached_rows}, counts);
    test.AddInput<int32_t>("base_row_indices", {queries}, rows);
    test.AddInput<int32_t>("range_starts", {queries}, starts);
    test.AddInput<int32_t>("range_ends", {queries}, ends);
    test.AddOutput<int32_t>("selected_indices", {queries, output_capacity}, expected);
    test.AddOutput<int32_t>("selected_counts", {queries}, expected_counts);
    test.AddOutput<int32_t>("status", {queries}, status);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
  if (!tested) GTEST_SKIP() << "Requires CUDA or WebGPU";
}

}  // namespace

TEST(PackedSparseAttentionIndexerMerge, StableOverlapAndSharedRows) {
  RunMerge(2, 6, 7, {9, 1, 9, 3, 1, -1, 6, 6, 2, 7, -1, -1}, {5, 4},
           {1, 0, 0}, {5, 1, 10}, {8, 5, 12},
           {6, 2, 7, 5, -1, -1, -1, 9, 1, 3, 2, 4, -1, -1, 9, 1, 3, 10, 11, -1, -1},
           {4, 5, 5}, {0, 0, 0});
}

TEST(PackedSparseAttentionIndexerMerge, HashCapacityUsesInsertionBound) {
  using contrib::indexer_merge::HashCapacity;
  EXPECT_EQ(HashCapacity(int64_t{1} << 30, 3), 8);
  EXPECT_EQ(HashCapacity(8, 8), 16);
  EXPECT_EQ(HashCapacity(4, INT32_MAX), 16);
  EXPECT_EQ(HashCapacity(int64_t{1} << 30, INT32_MAX), int64_t{1} << 32);
  EXPECT_EQ(HashCapacity(8, 0), 1);
}

TEST(PackedSparseAttentionIndexerMerge, PowerOfTwoCapacityAndDuplicatePrefixes) {
  RunMerge(1, 4, 4, {0, 1, 2, 3}, {4}, {0}, {0}, {4}, {0, 1, 2, 3}, {4}, {0});
  RunMerge(1, 8, 2, {1, 1, 1, 1, 2, 2, 2, 2}, {8}, {0}, {1}, {3}, {1, 2}, {2}, {0});
  RunMerge(1, 8, 2, {1, 1, 2, 2, 3, 3, 4, 4}, {8}, {0}, {0}, {0}, {-1, -1}, {0}, {2});
  RunMerge(1, 513, 2, std::vector<int32_t>(513, 7), {513}, {0}, {7}, {9}, {7, 8}, {2}, {0});
}

TEST(PackedSparseAttentionIndexerMerge, AppendIndicesDuplicateChunks) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) GTEST_SKIP() << "Requires CUDA";
  OpTester test("PackedSparseAttentionIndexerMerge", 1, kMSDomain);
  test.AddAttribute("policy_mode", std::string("append_indices"));
  test.AddAttribute("max_output_entries", int64_t{2});
  test.AddInput<int32_t>("base", {1, 257}, std::vector<int32_t>(257, 7));
  test.AddInput<int32_t>("counts", {1}, {257});
  test.AddInput<int32_t>("rows", {1}, {0});
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<int32_t>("additional", {1, 256}, std::vector<int32_t>(256, 8));
  test.AddInput<int32_t>("additional_counts", {1}, {256});
  test.AddOutput<int32_t>("selected", {1, 2}, {7, 8});
  test.AddOutput<int32_t>("selected_counts", {1}, {2});
  test.AddOutput<int32_t>("status", {1}, {0});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(PackedSparseAttentionIndexerMerge, InvalidMetadataAndOverflow) {
  RunMerge(2, 3, 3, {1, 2, 3, -1, 2, 3}, {3, 3},
           {-1, 2, 0, 0, 0, 1, 0}, {0, 0, -1, 5, 3, 0, 0},
           {0, 0, 1, 4, 5, 0, std::numeric_limits<int32_t>::max()},
           std::vector<int32_t>(21, -1), {0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 2, 1, 2});
}

TEST(PackedSparseAttentionIndexerMerge, InvalidCounts) {
  RunMerge(2, 2, 3, {1, 2, 3, 4}, {-1, 3}, {0, 1}, {0, 0}, {0, 0},
           std::vector<int32_t>(6, -1), {0, 0}, {1, 1});
}

TEST(PackedSparseAttentionIndexerMerge, EmptyBaseAndQueries) {
  RunMerge(1, 0, 4, {}, {0}, {0}, {2}, {5}, {2, 3, 4, -1}, {3}, {0});
  RunMerge(1, 2, 4, {1, 2}, {2}, {}, {}, {}, {}, {}, {});
  RunMerge(0, 2, 4, {}, {}, {0}, {0}, {0}, {-1, -1, -1, -1}, {0}, {1});
}

TEST(PackedSparseAttentionIndexerMerge, PackedName) {
  RunMerge(1, 2, 4, {1, 2}, {2}, {0}, {2}, {4}, {1, 2, 3, -1}, {3}, {0});
}

TEST(PackedSparseAttentionIndexerMerge, AppendIndicesStableUnionAndErrors) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) GTEST_SKIP() << "Requires CUDA";
  OpTester test("PackedSparseAttentionIndexerMerge", 1, kMSDomain);
  test.AddAttribute("policy_mode", std::string("append_indices"));
  test.AddAttribute("max_output_entries", int64_t{4});
  test.AddInput<int32_t>("base", {2, 3}, {3, 1, 3, 5, 6, 7});
  test.AddInput<int32_t>("counts", {2}, {3, 3});
  test.AddInput<int32_t>("rows", {4}, {0, 1, 0, -1});
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<int32_t>("additional", {4, 3}, {1, 8, 8, 8, 9, 10, -1, 0, 0, 0, 0, 0});
  test.AddInput<int32_t>("additional_counts", {4}, {3, 3, 1, 0});
  test.AddOutput<int32_t>("selected", {4, 4}, {3, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
  test.AddOutput<int32_t>("selected_counts", {4}, {3, 0, 0, 0});
  test.AddOutput<int32_t>("status", {4}, {0, 2, 1, 1});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(PackedSparseAttentionIndexerMerge, QwenCapacity) {
  std::vector<int32_t> base(2051);
  for (int32_t column = 0; column < 2051; ++column) base[column] = column;
  auto expected = base;
  for (int32_t column = 2051; column < 2058; ++column) expected.push_back(column);
  RunMerge(1, 2051, 2058, base, {2051}, {0}, {2051}, {2058}, expected, {2058}, {0});
}

TEST(PackedSparseAttentionIndexerMerge, LargeOutputCapacityWithSmallBase) {
  std::vector<int32_t> expected(4097, -1);
  expected[0] = 7;
  expected[1] = 5;
  expected[2] = 6;
  RunMerge(1, 5, 4097, {7, 7, 5, 5, -1}, {4}, {0}, {5}, {8}, expected, {3}, {0});
}

TEST(PackedSparseAttentionIndexerMerge, StableCompactionAcrossWorkgroupChunks) {
  std::vector<int32_t> base(513);
  for (int32_t column = 0; column < 513; ++column) base[column] = (column * 37) % 257;
  std::vector<int32_t> expected(base.begin(), base.begin() + 257);
  for (int32_t value = 257; value < 265; ++value) expected.push_back(value);
  expected.resize(270, -1);
  expected.insert(expected.end(), base.begin(), base.begin() + 257);
  expected.resize(540, -1);
  expected.resize(810, -1);
  RunMerge(1, 513, 270, base, {513}, {0, 0, 0}, {250, 0, 0}, {265, 0, 300},
           expected, {265, 257, 0}, {0, 0, 2});

  base.back() = -1;
  RunMerge(1, 513, 2, base, {513}, {0}, {0}, {0}, {-1, -1}, {0}, {1});
}

#ifndef ORT_NO_EXCEPTIONS
TEST(PackedSparseAttentionIndexerMerge, RejectUnsupportedPolicy) {
  OpTester test("PackedSparseAttentionIndexerMerge", 1, kMSDomain);
  test.AddAttribute("policy_mode", std::string("invalid_policy"));
  test.AddAttribute("max_output_entries", int64_t{1});
  test.AddInput<int32_t>("base", {1, 1}, {0});
  test.AddInput<int32_t>("counts", {1}, {1});
  test.AddInput<int32_t>("rows", {1}, {0});
  test.AddInput<int32_t>("starts", {1}, {0});
  test.AddInput<int32_t>("ends", {1}, {0});
  test.AddOutput<int32_t>("selected", {1, 1}, {0});
  test.AddOutput<int32_t>("selected_counts", {1}, {1});
  test.AddOutput<int32_t>("status", {1}, {0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid PackedSparseAttentionIndexerMerge policy or capacity");
}
#endif

}  // namespace onnxruntime::test
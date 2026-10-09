#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
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

TEST(PackedSparseAttentionIndexerMerge, LargeCapacityFallback) {
  std::vector<int32_t> expected(4097, -1);
  expected[0] = 7;
  expected[1] = 5;
  expected[2] = 6;
  RunMerge(1, 5, 4097, {7, 7, 5, 5, -1}, {4}, {0}, {5}, {8}, expected, {3}, {0});
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
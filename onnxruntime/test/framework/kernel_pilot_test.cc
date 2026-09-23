// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include <numeric>

#include "core/framework/kernel_pilot.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

void ExpectSelectedExperts(const KernelPilot::MoeExpertSelection& usage, const InlinedVector<int>& expected) {
  gsl::span<const int> selected;
  ASSERT_STATUS_OK(usage.GetSelectedExperts(selected));
  EXPECT_EQ((InlinedVector<int>{selected.begin(), selected.end()}), expected);
}

TEST(MoeExpertSelectionTest, RejectsUninitializedCollection) {
  KernelPilot::MoeExpertSelection usage;
  EXPECT_FALSE(usage.IsInitialized());
  EXPECT_FALSE(usage.Collect({}).IsOK());
  const int sentinel[] = {7};
  gsl::span<const int> selected = sentinel;
  EXPECT_FALSE(usage.GetSelectedExperts(selected).IsOK());
  EXPECT_EQ(selected.data(), sentinel);
  EXPECT_FALSE(usage.BeginInvocation(0).IsOK());
  EXPECT_FALSE(usage.IsInitialized());
}

TEST(MoeExpertSelectionTest, UnionsSelectedExpertsAcrossBatches) {
  KernelPilot::MoeExpertSelection usage;
  ASSERT_STATUS_OK(usage.BeginInvocation(4));
  EXPECT_TRUE(usage.IsInitialized());
  ExpectSelectedExperts(usage, {});
  const int first[] = {0, 2, 0, 2};
  const int second[] = {2, 3, 3};
  ASSERT_STATUS_OK(usage.Collect(first));
  ASSERT_STATUS_OK(usage.Collect(second));
  ASSERT_STATUS_OK(usage.Collect({}));
  ExpectSelectedExperts(usage, {0, 2, 3});
}

TEST(MoeExpertSelectionTest, ResetsSelectionAndReusesStorage) {
  KernelPilot::MoeExpertSelection usage;
  ASSERT_STATUS_OK(usage.BeginInvocation(64));
  InlinedVector<int> all(64);
  std::iota(all.begin(), all.end(), 0);
  ASSERT_STATUS_OK(usage.Collect(all));
  gsl::span<const int> selected;
  ASSERT_STATUS_OK(usage.GetSelectedExperts(selected));
  const int* storage = selected.data();
  ExpectSelectedExperts(usage, all);
  for (size_t expert_count : {size_t{64}, size_t{32}, size_t{64}}) {
    ASSERT_STATUS_OK(usage.BeginInvocation(expert_count));
    ExpectSelectedExperts(usage, {});
    const int repeated[] = {1, 1};
    ASSERT_STATUS_OK(usage.Collect(repeated));
    ASSERT_STATUS_OK(usage.GetSelectedExperts(selected));
    EXPECT_EQ(selected.data(), storage);
    ExpectSelectedExperts(usage, {1});
  }
}

TEST(MoeExpertSelectionTest, InvalidBatchDoesNotPartiallyChangeSelection) {
  KernelPilot::MoeExpertSelection usage;
  ASSERT_STATUS_OK(usage.BeginInvocation(4));
  const int first[] = {2};
  ASSERT_STATUS_OK(usage.Collect(first));
  for (const int invalid : {-1, 4}) {
    const int batch[] = {0, invalid};
    EXPECT_FALSE(usage.Collect(batch).IsOK());
    ExpectSelectedExperts(usage, {2});
  }
  EXPECT_FALSE(usage.BeginInvocation(0).IsOK());
  ExpectSelectedExperts(usage, {2});
  const int valid[] = {0, 1};
  ASSERT_STATUS_OK(usage.Collect(valid));
  ExpectSelectedExperts(usage, {2, 0, 1});
}

TEST(MoeExpertSelectionTest, KeepsInstancesIndependent) {
  KernelPilot::MoeExpertSelection first, second;
  ASSERT_STATUS_OK(first.BeginInvocation(2));
  ASSERT_STATUS_OK(second.BeginInvocation(4));
  const int first_ids[] = {0, 0};
  const int second_ids[] = {3, 3};
  ASSERT_STATUS_OK(first.Collect(first_ids));
  ASSERT_STATUS_OK(second.Collect(second_ids));
  ExpectSelectedExperts(first, {0});
  ExpectSelectedExperts(second, {3});
  ASSERT_STATUS_OK(first.BeginInvocation(4));
  ASSERT_STATUS_OK(first.Collect(second_ids));
  ExpectSelectedExperts(first, {3});
  ExpectSelectedExperts(second, {3});
}

}  // namespace
}  // namespace onnxruntime::test
#endif

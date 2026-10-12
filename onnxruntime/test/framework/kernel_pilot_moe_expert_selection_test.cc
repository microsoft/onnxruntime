// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include <limits>
#include <numeric>

#include "core/framework/kernel_pilot_moe_expert_selection.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

void ExpectSelectedExperts(const KernelPilotMoeExpertSelection& usage, const InlinedVector<int>& expected) {
  gsl::span<const int> selected;
  ASSERT_STATUS_OK(usage.GetSelectedExperts(selected));
  EXPECT_EQ((InlinedVector<int>{selected.begin(), selected.end()}), expected);
}

void ExpectSelectedProbabilities(const KernelPilotMoeExpertSelection& usage,
                                 const InlinedVector<float>& expected) {
  gsl::span<const float> probabilities;
  ASSERT_STATUS_OK(usage.GetSelectedExpertProbabilities(probabilities));
  EXPECT_EQ((InlinedVector<float>{probabilities.begin(), probabilities.end()}), expected);
}

TEST(KernelPilotMoeExpertSelectionTest, RejectsUninitializedCollection) {
  KernelPilotMoeExpertSelection usage;
  EXPECT_FALSE(usage.IsInitialized());
  EXPECT_FALSE(usage.Collect({}).IsOK());
  const int sentinel[] = {7};
  gsl::span<const int> selected = sentinel;
  EXPECT_FALSE(usage.GetSelectedExperts(selected).IsOK());
  EXPECT_EQ(selected.data(), sentinel);
  gsl::span<const float> probabilities;
  EXPECT_FALSE(usage.GetSelectedExpertProbabilities(probabilities).IsOK());
  EXPECT_FALSE(usage.BeginInvocation(0).IsOK());
  EXPECT_FALSE(usage.IsInitialized());
}

TEST(KernelPilotMoeExpertSelectionTest, UnionsSelectedExpertsAcrossBatches) {
  KernelPilotMoeExpertSelection usage;
  ASSERT_STATUS_OK(usage.BeginInvocation(4));
  EXPECT_TRUE(usage.IsInitialized());
  ExpectSelectedExperts(usage, {});
  const int first[] = {0, 2, 0, 2};
  const int second[] = {2, 3, 3};
  ASSERT_STATUS_OK(usage.Collect(first));
  ASSERT_STATUS_OK(usage.Collect(second));
  ASSERT_STATUS_OK(usage.Collect({}));
  ExpectSelectedExperts(usage, {0, 2, 3});
  ExpectSelectedProbabilities(usage, {});
}

TEST(KernelPilotMoeExpertSelectionTest, RetainsMaximumProbabilityForRepeatedExperts) {
  KernelPilotMoeExpertSelection usage;
  ASSERT_STATUS_OK(usage.BeginInvocation(4));
  const int first_ids[] = {0, 2, 0};
  const float first_probabilities[] = {0.6f, 0.2f, 0.8f};
  ASSERT_STATUS_OK(usage.CollectWeighted(first_ids, first_probabilities));
  const int second_ids[] = {2, 3};
  const float second_probabilities[] = {0.1f, 0.4f};
  ASSERT_STATUS_OK(usage.CollectWeighted(second_ids, second_probabilities));
  ExpectSelectedExperts(usage, {0, 2, 3});
  ExpectSelectedProbabilities(usage, {0.8f, 0.2f, 0.4f});
}

TEST(KernelPilotMoeExpertSelectionTest, ValidatesWeightedCollection) {
  KernelPilotMoeExpertSelection usage;
  ASSERT_STATUS_OK(usage.BeginInvocation(4));
  const int ids[] = {0, 2};
  const float one_probability[] = {0.5f};
  EXPECT_FALSE(usage.CollectWeighted(ids, one_probability).IsOK());
  ExpectSelectedExperts(usage, {});

  for (const float invalid_probability : {
           -0.1f,
           1.1f,
           std::numeric_limits<float>::infinity(),
           std::numeric_limits<float>::quiet_NaN()}) {
    const float probabilities[] = {0.5f, invalid_probability};
    EXPECT_FALSE(usage.CollectWeighted(ids, probabilities).IsOK());
    ExpectSelectedExperts(usage, {});
  }

  const int invalid_ids[] = {0, 4};
  const float valid_probabilities[] = {0.5f, 0.5f};
  EXPECT_FALSE(usage.CollectWeighted(invalid_ids, valid_probabilities).IsOK());
  ExpectSelectedExperts(usage, {});

  ASSERT_STATUS_OK(usage.CollectWeighted(ids, valid_probabilities));
  EXPECT_FALSE(usage.Collect(ids).IsOK());

  ASSERT_STATUS_OK(usage.BeginInvocation(4));
  ASSERT_STATUS_OK(usage.Collect(ids));
  EXPECT_FALSE(usage.CollectWeighted(ids, valid_probabilities).IsOK());
}

TEST(KernelPilotMoeExpertSelectionTest, ResetsSelectionAndReusesStorage) {
  KernelPilotMoeExpertSelection usage;
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

TEST(KernelPilotMoeExpertSelectionTest, InvalidBatchDoesNotPartiallyChangeSelection) {
  KernelPilotMoeExpertSelection usage;
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

TEST(KernelPilotMoeExpertSelectionTest, KeepsInstancesIndependent) {
  KernelPilotMoeExpertSelection first, second;
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Ledger-only tests: no GPU, no ORT runtime and no complete model are needed. They use device ordinals that no real
// device has and group names that are unique per test, and compare registry sizes against a baseline so they do not
// depend on what other tests registered. Both the in-tree CUDA target and the plugin-internal target compile it.
#include "core/providers/cuda/cuda_device_memory_budget.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr int kDevice = 9001;
constexpr int kOtherDevice = 9002;
constexpr size_t kUnlimited = std::numeric_limits<size_t>::max();

}  // namespace

// A group whose sessions are all gone has to be forgotten. Otherwise a later group with the same name inherits the old
// limit: one created with 64 bytes and recreated with 128 kept 64 and rejected a 96-byte reservation.
TEST(CudaBudgetGroupRegistryTest, RetiredGroupStartsFromTheNewLimit) {
  const std::string name = "retired_group_starts_from_the_new_limit";
  const size_t baseline = GetCudaBudgetGroupCount();

  {
    auto group = GetCudaBudgetGroupByName(kDevice, name, 64);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->GetGroupStats().limit, 64u);
    EXPECT_EQ(GetCudaBudgetGroupCount(), baseline + 1);
  }
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline);

  {
    auto group = GetCudaBudgetGroupByName(kDevice, name, 128);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->GetGroupStats().limit, 128u);

    SessionBudgetAccount account(GetCudaDeviceMemoryBudgetByOrdinal(kDevice), kUnlimited, group);
    account.Reserve(96).Commit();
    EXPECT_EQ(group->GetGroupStats().committed, 96u);
    EXPECT_THROW(account.Reserve(64), std::runtime_error);

    account.BeginFree(96);
    account.FinishFree(96, true);
    EXPECT_EQ(group->GetGroupStats().committed, 0u);
  }
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline);
}

TEST(CudaBudgetGroupRegistryTest, LiveGroupIsSharedAndKeepsTheSmallerLimit) {
  const std::string name = "live_group_is_shared";
  const size_t baseline = GetCudaBudgetGroupCount();

  auto first = GetCudaBudgetGroupByName(kDevice, name, 64);
  auto second = GetCudaBudgetGroupByName(kDevice, name, 128);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first.get(), second.get());
  EXPECT_EQ(second->GetGroupStats().limit, 64u);
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline + 1);

  // Every handle holds a lease, so the group stays registered until the last one is gone.
  first.reset();
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline + 1);
  EXPECT_EQ(second->GetGroupStats().limit, 64u);

  second.reset();
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline);
}

TEST(CudaBudgetGroupRegistryTest, CopiesOfAHandleShareOneLease) {
  const std::string name = "copies_of_a_handle_share_one_lease";
  const size_t baseline = GetCudaBudgetGroupCount();

  auto handle = GetCudaBudgetGroupByName(kDevice, name, 64);
  auto copy = handle;
  handle.reset();
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline + 1);

  copy.reset();
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline);
}

TEST(CudaBudgetGroupRegistryTest, AccountHoldsTheGroupAlive) {
  const std::string name = "account_holds_the_group_alive";
  const size_t baseline = GetCudaBudgetGroupCount();

  {
    SessionBudgetAccount account(GetCudaDeviceMemoryBudgetByOrdinal(kDevice), kUnlimited,
                                 GetCudaBudgetGroupByName(kDevice, name, 64));
    EXPECT_EQ(GetCudaBudgetGroupCount(), baseline + 1);

    account.Reserve(64).Commit();
    EXPECT_THROW(account.Reserve(1), std::runtime_error);

    account.BeginFree(64);
    account.FinishFree(64, true);
  }
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline);
}

// Charges that outlive every handle (for example an allocation freed after its session went away) still count against
// the group, so the group has to stay registered with its limit until they are released.
TEST(CudaBudgetGroupRegistryTest, GroupWithOutstandingChargesStaysRegistered) {
  const std::string name = "group_with_outstanding_charges";
  const size_t baseline = GetCudaBudgetGroupCount();
  auto ledger = GetCudaDeviceMemoryBudgetByOrdinal(kDevice);

  {
    SessionBudgetAccount account(ledger, kUnlimited, GetCudaBudgetGroupByName(kDevice, name, 64));
    account.Reserve(32).Commit();
  }
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline + 1);

  auto group = GetCudaBudgetGroupByName(kDevice, name, 128);
  EXPECT_EQ(group->GetGroupStats().limit, 64u);
  EXPECT_EQ(group->GetGroupStats().committed, 32u);
  {
    SessionBudgetAccount account(ledger, kUnlimited, group);
    EXPECT_THROW(account.Reserve(48), std::runtime_error);
    account.Reserve(32).Commit();
    account.BeginFree(32);
    account.FinishFree(32, true);
  }

  // Release the charge left by the first account, which no account owns any more.
  group->GroupBudget().Release(32);
  ledger->Release(32);
  EXPECT_EQ(group->GetGroupStats().committed, 0u);

  group.reset();
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline);
}

TEST(CudaBudgetGroupRegistryTest, GroupWithQuarantinedChargesStaysRegistered) {
  const std::string name = "group_with_quarantined_charges";
  const size_t baseline = GetCudaBudgetGroupCount();
  auto ledger = GetCudaDeviceMemoryBudgetByOrdinal(kDevice);

  {
    SessionBudgetAccount account(ledger, kUnlimited, GetCudaBudgetGroupByName(kDevice, name, 64));
    account.Reserve(16).Commit();
    account.BeginFree(16);
    // The driver refused the free, so the bytes are still held.
    account.FinishFree(16, false);
  }
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline + 1);

  auto group = GetCudaBudgetGroupByName(kDevice, name, 128);
  EXPECT_EQ(group->GetGroupStats().limit, 64u);
  EXPECT_EQ(group->GetGroupStats().quarantined, 16u);

  group->GroupBudget().DisownState(0, 0, 0, 16);
  ledger->DisownState(0, 0, 0, 16);
  group.reset();
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline);
}

TEST(CudaBudgetGroupRegistryTest, GroupsAreKeyedByDeviceAndName) {
  const std::string name = "groups_are_keyed_by_device_and_name";
  const size_t baseline = GetCudaBudgetGroupCount();

  auto on_device = GetCudaBudgetGroupByName(kDevice, name, 64);
  auto on_other_device = GetCudaBudgetGroupByName(kOtherDevice, name, 128);
  auto other_name = GetCudaBudgetGroupByName(kDevice, name + "_other", 256);

  EXPECT_NE(on_device.get(), on_other_device.get());
  EXPECT_NE(on_device.get(), other_name.get());
  EXPECT_EQ(on_device->GetGroupStats().limit, 64u);
  EXPECT_EQ(on_other_device->GetGroupStats().limit, 128u);
  EXPECT_EQ(other_name->GetGroupStats().limit, 256u);
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline + 3);

  on_device.reset();
  on_other_device.reset();
  other_name.reset();
  EXPECT_EQ(GetCudaBudgetGroupCount(), baseline);
}

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Pure decision logic: no GPU, no ORT runtime and no complete model are needed. Both the in-tree CUDA target and the
// plugin-internal target compile it.
#include "core/providers/cuda/plugin/cuda_ep_account_binding.h"

#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

using cuda_plugin::AccountBindingAction;
using cuda_plugin::AccountBindingDecision;
using cuda_plugin::DecideAccountBinding;
using cuda_plugin::PendingAccountCandidate;

constexpr uintptr_t kOwnerA = 0x1000;
constexpr uintptr_t kOwnerB = 0x2000;
constexpr uintptr_t kOwnerC = 0x3000;

constexpr bool kFinite = false;
constexpr bool kUnrestricted = true;

// An id of a thread other than the calling one. The caller is alive, so the two ids cannot be equal.
std::thread::id AnotherThreadId() {
  std::thread::id id;
  std::thread([&id] { id = std::this_thread::get_id(); }).join();
  return id;
}

void ExpectDecision(const AccountBindingDecision& decision, AccountBindingAction action, uintptr_t owner = 0) {
  EXPECT_EQ(decision.action, action);
  EXPECT_EQ(decision.owner, owner);
}

}  // namespace

TEST(CudaEpAccountBindingTest, HostIdentityBindsExactlyThatPendingAccount) {
  const auto main_thread = std::this_thread::get_id();
  const std::vector<PendingAccountCandidate> pending = {
      {kOwnerA, kFinite, main_thread},
      {kOwnerB, kUnrestricted, main_thread},
  };

  ExpectDecision(DecideAccountBinding(kOwnerA, false, AnotherThreadId(), pending),
                 AccountBindingAction::kBindPending, kOwnerA);
  ExpectDecision(DecideAccountBinding(kOwnerB, false, main_thread, pending),
                 AccountBindingAction::kBindPending, kOwnerB);
  // The identity decides, even when the request also carries a limit.
  ExpectDecision(DecideAccountBinding(kOwnerA, true, main_thread, pending),
                 AccountBindingAction::kBindPending, kOwnerA);
}

TEST(CudaEpAccountBindingTest, HostIdentityWithoutPendingAccountCreatesAccountForRequest) {
  const auto main_thread = std::this_thread::get_id();
  const std::vector<PendingAccountCandidate> pending = {{kOwnerA, kFinite, main_thread}};

  ExpectDecision(DecideAccountBinding(kOwnerC, false, main_thread, pending), AccountBindingAction::kCreateFresh);
  ExpectDecision(DecideAccountBinding(kOwnerC, true, main_thread, pending), AccountBindingAction::kCreateExplicit);
  ExpectDecision(DecideAccountBinding(kOwnerC, false, main_thread, {}), AccountBindingAction::kCreateFresh);
}

TEST(CudaEpAccountBindingTest, ExplicitLimitWithoutIdentityCreatesExplicitAccount) {
  const auto main_thread = std::this_thread::get_id();
  const std::vector<PendingAccountCandidate> pending = {{kOwnerA, kFinite, main_thread}};

  ExpectDecision(DecideAccountBinding(std::nullopt, true, main_thread, pending),
                 AccountBindingAction::kCreateExplicit);
  ExpectDecision(DecideAccountBinding(std::nullopt, true, main_thread, {}), AccountBindingAction::kCreateExplicit);
}

TEST(CudaEpAccountBindingTest, LegacyRequestWithoutPendingAccountCreatesFreshAccount) {
  ExpectDecision(DecideAccountBinding(std::nullopt, false, std::this_thread::get_id(), {}),
                 AccountBindingAction::kCreateFresh);
}

// A host that does not send the OrtEp identity, one session with a finite budget, and ordinary caller arena options
// (no explicit limit on the request). The allocator used to get an unlimited account, so a 4096-byte session could
// allocate 8192 bytes while staying charged zero.
TEST(CudaEpAccountBindingTest, LegacyRequestBindsSingleFinitePendingAccount) {
  const auto creator_thread = std::this_thread::get_id();
  const std::vector<PendingAccountCandidate> pending = {{kOwnerA, kFinite, creator_thread}};

  ExpectDecision(DecideAccountBinding(std::nullopt, false, creator_thread, pending),
                 AccountBindingAction::kBindPending, kOwnerA);
  // A single waiting account belongs to the requester whichever thread asks.
  ExpectDecision(DecideAccountBinding(std::nullopt, false, AnotherThreadId(), pending),
                 AccountBindingAction::kBindPending, kOwnerA);
}

// Two overlapping sessions without limits both queue an account. Neither carries a limit that could be lost, so the
// request must not fail.
TEST(CudaEpAccountBindingTest, LegacyOverlappingUnrestrictedSessionsKeepWorking) {
  const auto main_thread = std::this_thread::get_id();
  const std::vector<PendingAccountCandidate> same_creator = {
      {kOwnerA, kUnrestricted, main_thread},
      {kOwnerB, kUnrestricted, main_thread},
  };
  ExpectDecision(DecideAccountBinding(std::nullopt, false, main_thread, same_creator),
                 AccountBindingAction::kCreateFresh);
  ExpectDecision(DecideAccountBinding(std::nullopt, false, AnotherThreadId(), same_creator),
                 AccountBindingAction::kCreateFresh);
}

TEST(CudaEpAccountBindingTest, LegacyRequestBindsTheAccountItsOwnThreadCreated) {
  const auto main_thread = std::this_thread::get_id();
  const auto other_thread = AnotherThreadId();
  ASSERT_NE(main_thread, other_thread);
  const std::vector<PendingAccountCandidate> pending = {
      {kOwnerA, kFinite, main_thread},
      {kOwnerB, kUnrestricted, other_thread},
  };

  ExpectDecision(DecideAccountBinding(std::nullopt, false, main_thread, pending),
                 AccountBindingAction::kBindPending, kOwnerA);
  ExpectDecision(DecideAccountBinding(std::nullopt, false, other_thread, pending),
                 AccountBindingAction::kBindPending, kOwnerB);
}

// With several waiting sessions where one carries a limit, binding by guess could run the limited session without its
// limit, so the request has to be rejected.
TEST(CudaEpAccountBindingTest, LegacyRequestThatCannotBeAttributedToALimitedSessionIsRejected) {
  const auto main_thread = std::this_thread::get_id();

  // The requester created none of the waiting accounts.
  const std::vector<PendingAccountCandidate> created_elsewhere = {
      {kOwnerA, kFinite, main_thread},
      {kOwnerB, kUnrestricted, main_thread},
  };
  ExpectDecision(DecideAccountBinding(std::nullopt, false, AnotherThreadId(), created_elsewhere),
                 AccountBindingAction::kRejectAmbiguous);

  // The requester created more than one of them, so its thread does not identify one either.
  ExpectDecision(DecideAccountBinding(std::nullopt, false, main_thread, created_elsewhere),
                 AccountBindingAction::kRejectAmbiguous);
}

}  // namespace test
}  // namespace onnxruntime

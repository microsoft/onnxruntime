// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

/// A session budget account that CreateEp queued and that no allocator request has claimed yet.
struct PendingAccountCandidate {
  /// Identity of the OrtEp that owns the account.
  uintptr_t owner{0};
  /// True when the account has no finite session limit and no budget group, so charging it by mistake cannot drop
  /// or add a limit.
  bool unrestricted{true};
  /// Thread that created the OrtEp.
  std::thread::id creator_thread{};
};

enum class AccountBindingAction {
  /// Charge the allocator to the pending account of AccountBindingDecision::owner.
  kBindPending,
  /// The request carries gpu_device_mem_limit: create an account with exactly that limit.
  kCreateExplicit,
  /// No pending account can be attributed to the request: create a new account.
  kCreateFresh,
  /// Several sessions are waiting, at least one carries a limit or budget group, and the requester is unknown.
  /// Guessing could silently drop that limit, so the request is rejected.
  kRejectAmbiguous,
};

struct AccountBindingDecision {
  AccountBindingAction action{AccountBindingAction::kCreateFresh};
  uintptr_t owner{0};
};

/// Decides which session budget account an allocator request is charged to.
///
/// Hosts that identify the requesting OrtEp pass `owning_ep` and the account is matched exactly. Older hosts send no
/// identity, so ownership is inferred, never from limits or queue order:
///   1. The host creates an OrtEp and requests its allocators on the same thread, so the one account created by the
///      requesting thread is the requester's.
///   2. A single waiting account can only belong to the requester.
///   3. Several waiting accounts without a limit or budget group are interchangeable, so the request gets a new
///      account and ordinary sessions keep working.
///   4. Otherwise the request is rejected rather than risk running a limited session without its limit.
inline AccountBindingDecision DecideAccountBinding(std::optional<uintptr_t> owning_ep,
                                                   bool has_explicit_limit,
                                                   std::thread::id requesting_thread,
                                                   const std::vector<PendingAccountCandidate>& pending) {
  if (owning_ep.has_value()) {
    const bool found = std::any_of(pending.begin(), pending.end(), [&](const PendingAccountCandidate& candidate) {
      return candidate.owner == *owning_ep;
    });
    if (found) {
      return {AccountBindingAction::kBindPending, *owning_ep};
    }
    return {has_explicit_limit ? AccountBindingAction::kCreateExplicit : AccountBindingAction::kCreateFresh, 0};
  }

  if (has_explicit_limit) {
    return {AccountBindingAction::kCreateExplicit, 0};
  }

  if (pending.empty()) {
    return {AccountBindingAction::kCreateFresh, 0};
  }

  const auto same_thread_count = std::count_if(pending.begin(), pending.end(),
                                               [&](const PendingAccountCandidate& candidate) {
                                                 return candidate.creator_thread == requesting_thread;
                                               });
  if (same_thread_count == 1) {
    const auto same_thread = std::find_if(pending.begin(), pending.end(),
                                          [&](const PendingAccountCandidate& candidate) {
                                            return candidate.creator_thread == requesting_thread;
                                          });
    return {AccountBindingAction::kBindPending, same_thread->owner};
  }

  if (pending.size() == 1) {
    return {AccountBindingAction::kBindPending, pending.front().owner};
  }

  const bool all_unrestricted = std::all_of(pending.begin(), pending.end(),
                                            [](const PendingAccountCandidate& candidate) {
                                              return candidate.unrestricted;
                                            });
  return {all_unrestricted ? AccountBindingAction::kCreateFresh : AccountBindingAction::kRejectAmbiguous, 0};
}

}  // namespace cuda_plugin
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

// Provider-independent contract for collecting the local expert IDs selected by one MoE kernel
// invocation. Implementations may accept IDs directly from host routing or adapt provider-specific
// transfers, but expose the completed selection through the same session-owned interface.
//
// KernelPilotMoeExpertState owns one KernelPilot, and therefore one selection, for each registered
// MoE kernel. One invocation proceeds as follows:
// 1. The kernel obtains its KernelPilot from OpKernelContext and calls BeginInvocation().
// 2. Routing selects local expert IDs. CPU kernels call Collect() directly; CUDA kernels transfer
//    IDs through KernelPilotMoeExpertSelectionCuda, which delegates collection to this interface.
// 3. Tiled routing may call Collect() repeatedly; the selection retains each expert only once.
// 4. If Compute() succeeds, the executor calls KernelPilot::RecordUsage().
// 5. RecordUsage() reads GetSelectedExperts(), decays that kernel's persistent counters, and adds
//    the configured contribution for every selected expert. Failed kernel invocations are not
//    committed.
//
// The selection is therefore transient invocation output. It does not own persistent counters,
// placement policy, or the decision to commit an invocation.
class IKernelPilotMoeExpertSelection {
 public:
  virtual ~IKernelPilotMoeExpertSelection() = default;

  // Starts a new invocation for a node with expert_count local experts and discards the previous
  // invocation's selection. Returns an error when expert_count is zero.
  virtual Status BeginInvocation(size_t expert_count) = 0;

  // Returns whether BeginInvocation() has established a positive expert count.
  virtual bool IsInitialized() const noexcept = 0;

  // Returns the number of local experts configured for the current invocation, or zero before
  // successful initialization.
  virtual size_t ExpertCount() const noexcept = 0;

  // Adds local expert IDs to the current invocation's selection. Implementations deduplicate IDs
  // across calls, including calls made for separate routing tiles, and reject out-of-range IDs.
  virtual Status Collect(gsl::span<const int> expert_ids) = 0;

  // Sets expert_ids to a view of the unique local expert IDs selected during this invocation.
  // The view remains owned by the implementation and is valid until its next non-const operation.
  virtual Status GetSelectedExperts(gsl::span<const int>& expert_ids) const = 0;
};

// Collects the union of local expert IDs selected across one MoE kernel invocation,
// including tiled routing.
class KernelPilotMoeExpertSelection final : public IKernelPilotMoeExpertSelection {
 public:
  // Creates an uninitialized selection. BeginInvocation() must be called before collecting
  // or reading selected experts.
  KernelPilotMoeExpertSelection();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelPilotMoeExpertSelection);

  // Retains the selection's allocated storage across invocations.
  Status BeginInvocation(size_t expert_count) override;

  bool IsInitialized() const noexcept override;

  size_t ExpertCount() const noexcept override;

  // Validation is atomic: an out-of-range ID leaves the current selection unchanged. IDs are
  // retained in first-seen order.
  // expert_count is small (the number of experts configured for this MoE node), so a linear
  // scan of the already-selected list is cheaper than maintaining a separate expert_count-sized
  // membership mask.
  Status Collect(gsl::span<const int> expert_ids) override;

  Status GetSelectedExperts(gsl::span<const int>& expert_ids) const override;

 private:
  friend class KernelPilot;
  bool HasPendingInvocation() const noexcept;
  void FinishInvocation() noexcept;

  size_t expert_count_{0};
  InlinedVector<int> selected_experts_;
  bool invocation_pending_{false};
};

}  // namespace onnxruntime

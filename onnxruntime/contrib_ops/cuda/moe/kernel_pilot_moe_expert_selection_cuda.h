// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/kernel_pilot_moe_expert_selection.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

class KernelPilotMoeExpertSelectionCuda final : public IKernelPilotMoeExpertSelection {
 public:
  explicit KernelPilotMoeExpertSelectionCuda(AllocatorPtr pinned_allocator);
  ~KernelPilotMoeExpertSelectionCuda() override;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelPilotMoeExpertSelectionCuda);

  // Attaches a session-owned selection for subsequent selection calls and starts an invocation.
  Status BeginInvocation(IKernelPilotMoeExpertSelection& usage, size_t expert_count);
  // Starts another invocation using the selection attached by the two-argument overload.
  Status BeginInvocation(size_t expert_count) override;
  bool IsInitialized() const noexcept override;
  size_t ExpertCount() const noexcept override;
  Status Collect(gsl::span<const int> expert_ids) override;
  Status GetSelectedExperts(gsl::span<const int>& expert_ids) const override;

  Status Capture(const int* expert_ids, size_t count, cudaStream_t stream);

  // The fused-routing runner invokes this on the calling CPU thread, before launching expert GEMMs.
  static void CaptureRouting(void* snapshot, const int* expert_ids, size_t count, cudaStream_t stream);

  Status Consume();

 private:
  Status WaitForCopy();

  AllocatorPtr pinned_allocator_;
  IKernelPilotMoeExpertSelection* usage_{nullptr};
  IAllocatorUniquePtr<int> host_ids_;
  size_t capacity_{0};
  size_t captured_count_{0};
  cudaEvent_t copy_ready_{nullptr};
  cudaStream_t copy_stream_{nullptr};
  bool copy_pending_{false};
  bool copy_recorded_{false};
};

}  // namespace onnxruntime::contrib::cuda

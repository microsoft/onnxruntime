// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {

// Provider-side adapter used by the session-global MoE placement policy. Implementations own
// device-specific transfer resources and keep the published mapping usable while a swap is pending.
class IKernelPilotMoeExpertCache {
 public:
  virtual ~IKernelPilotMoeExpertCache() = default;

  virtual int DeviceId() const noexcept = 0;
  virtual bool HasPendingSwap() const noexcept = 0;
  virtual Status ReclaimCompletedSwap() = 0;
  virtual Status StartSwap(int cuda_expert_id, int cpu_expert_id) = 0;
};

}  // namespace onnxruntime

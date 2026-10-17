// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <gsl/gsl>

#include "core/common/common.h"

namespace onnxruntime {

struct KernelPilotMoeExpertUsageSnapshot {
  gsl::span<const double> counters;
  // One entry per expert: maximum selected probability, or -1 for an unselected expert.
  gsl::span<const float> selected_probabilities;
  gsl::span<const int64_t> first_selected_routes;
};

// Provider-owned asynchronous invocation results, consumed before end-of-run placement decisions.
class IKernelPilotMoeExpertUsage {
 public:
  virtual ~IKernelPilotMoeExpertUsage() = default;
  virtual bool HasPendingUsage() const noexcept = 0;
  virtual Status CompleteUsage(KernelPilotMoeExpertUsageSnapshot& snapshot) = 0;
};

}  // namespace onnxruntime

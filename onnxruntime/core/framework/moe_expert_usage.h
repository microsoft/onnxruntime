// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"

namespace onnxruntime {

// Internal C++ interface shared by the runtime and built-in execution providers.
// A session owns one instance per registered kernel; callers must not destroy it.
class MoeExpertUsage {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MoeExpertUsage);

  // IDs are local to this kernel. Repeated IDs contribute only once per invocation.
  virtual Status RecordUsage(gsl::span<const int> used_expert_ids) = 0;
  // During a Run, only this kernel may read its counters.
  virtual Status GetCounters(InlinedVector<double>& counters) const = 0;

 protected:
  MoeExpertUsage() = default;
  virtual ~MoeExpertUsage() = default;
};

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <map>
#include <optional>
#include <string>

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/framework/ortdevice.h"
#include "core/framework/resource_accountant.h"
#include "core/framework/workspace_requirement.h"

namespace onnxruntime {

class SessionState;

struct SessionMemoryAllocation {
  std::string name;
  OrtDevice location;
  std::optional<size_t> size_bytes;
  // Inclusive positions in one ordered execution stream, not host completion times.
  size_t first_step = 0;
  size_t last_step = 0;
};

struct SessionMemoryDeviceReport {
  size_t initializers_before_kernel_creation_bytes = 0;
  size_t initializers_after_prepack_bytes = 0;
  // L1 components remain separate: their allocation identities/sharing are not declared.
  size_t l1_persistent_prepack_estimate_bytes = 0;
  size_t l1_initialization_scratch_estimate_bytes = 0;
  size_t modeled_run_live_peak_bytes = 0;
  std::optional<size_t> initialization_backing_peak_bytes;
  std::optional<size_t> run_backing_peak_bytes;
};

struct SessionMemoryReport {
  std::map<OrtDevice, SessionMemoryDeviceReport> devices;
  InlinedVector<SessionMemoryAllocation> allocations;
  InlinedVector<std::string> uncovered_terms;
  bool ordered_lifetimes = false;
};

struct SessionMemoryAllocatorSnapshot {
  std::string name;
  OrtDevice location;
  // Allocator-wide observations, potentially shared by sessions. Never additive to live bytes.
  std::optional<size_t> reported_allocated_bytes;
  std::optional<size_t> reported_in_use_bytes;
};

using MemoryReportWorkspaceMap = InlinedHashMap<size_t, InlinedVector<WorkspaceRequirement>>;

Status CaptureSessionMemoryInitializers(const SessionState& session_state, SessionMemoryReport& report);

Status CompleteSessionMemoryReport(
    const SessionState& session_state,
    const MemoryReportWorkspaceMap& workspace,
    const NodeWorkspaceReservationMap* reservations,
    SessionMemoryReport& report);

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

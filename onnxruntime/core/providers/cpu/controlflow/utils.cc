// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/controlflow/utils.h"

#include <vector>
#include "core/common/common.h"
#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace controlflow {
namespace detail {

static const OrtAllocatorInfo& FindAllocatorInfoForValue(const OrtValueNameIdxMap& map,
                                                         const SequentialExecutionPlan& plan,
                                                         const std::string& name) {
  int idx = -1;
  auto status = map.GetIdx(name, idx);
  ORT_THROW_IF_ERROR(status);

  const auto& location = plan.GetLocation(idx);
  return location;
}

const OrtAllocatorInfo& FindAllocatorInfoForValue(const SessionState& session_state,
                                                  const std::string& name) {
  const auto* exec_plan_ptr = session_state.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr);

  return FindAllocatorInfoForValue(session_state.GetOrtValueNameIdxMap(), *exec_plan_ptr, name);
}

common::Status FindDevicesForValues(const SessionState& session_state,
                                    const std::vector<std::string>& names,
                                    std::vector<OrtDevice>& devices,
                                    size_t start_at) {
  devices.resize(names.size());

  const auto& map = session_state.GetOrtValueNameIdxMap();
  const auto* exec_plan_ptr = session_state.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr);

  for (size_t i = start_at, end = names.size(); i < end; ++i) {
    const auto& location = FindAllocatorInfoForValue(map, *exec_plan_ptr, names[i]);
    devices[i] = location.device;
  }

  return Status::OK();
}

}  // namespace detail
}  // namespace controlflow
}  // namespace onnxruntime

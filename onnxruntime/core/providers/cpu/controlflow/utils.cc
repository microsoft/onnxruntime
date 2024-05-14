// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/controlflow/utils.h"

#include <vector>
#include "core/common/common.h"
#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace controlflow {
namespace detail {

common::Status FindDevicesForValues(const SessionState& session_state,
                                    const std::vector<std::string>& names,
                                    std::vector<OrtDevice>& devices,
                                    size_t start_at) {
  devices.resize(names.size());

  for (size_t i = start_at, end = names.size(); i < end; ++i) {
    const auto& location = utils::FindDeviceForValue(session_state, names[i]);
    devices[i] = location;
  }

  return Status::OK();
}

}  // namespace detail
}  // namespace controlflow
}  // namespace onnxruntime

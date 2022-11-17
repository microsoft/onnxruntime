// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/platform/windows/env.h"
#include <vector>

namespace onnxruntime {
namespace test {

bool WindowsEnvTester::SetCpuInfo(const CpuInfo& cpu_info) {
  if (cpu_info.empty()) {
    return false;
  }
  std::vector<Core> new_cores;
  std::vector<Group> new_groups;
  // for every group
  for (const auto& cpu_cores : cpu_info) {
    Group new_group;
    if (cpu_cores.empty()) {
      return false;
    }
    // for every core in the group
    for (const auto& cpu_core : cpu_cores) {
      if (cpu_core.empty()) {
        return false;
      }
      Core new_core;
      new_core.group_id = new_groups.size();
      for (const auto& logic_processor_id : cpu_core) {
        new_group.num_processors++;
        auto logical_processor_bitmask = BitOne << logic_processor_id;
        new_group.processor_bitmask |= logical_processor_bitmask;
        new_core.processor_bitmask |= logical_processor_bitmask;
      }
      new_cores.push_back(std::move(new_core));
    }
    new_groups.push_back(std::move(new_group));
  }
  cores_ = std::move(new_cores);
  groups_ = std::move(new_groups);
  return true;
}

}
}  // namespace onnxruntime
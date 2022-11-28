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
  cores_.clear();
  global_processor_info_map_.clear();
  int global_processor_id = 0;
  for (int group_id = 0; group_id < cpu_info.size(); ++group_id) {
    int local_processor_id = 0;
    for (int core_id = 0; core_id < cpu_info[group_id].size(); ++core_id) {
      onnxruntime::LogicalProcessors logical_processors;
      for (int i = 0; i < cpu_info[group_id][core_id].size(); ++i) {
        logical_processors.push_back(global_processor_id);
        global_processor_info_map_[global_processor_id] = {group_id, local_processor_id};
        local_processor_id++;
        global_processor_id++;
      }
      cores_.push_back(std::move(logical_processors));
    }
  }
  return true;
}

}  // namespace test
}  // namespace onnxruntime
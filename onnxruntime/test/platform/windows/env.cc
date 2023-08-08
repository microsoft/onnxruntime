// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/platform/windows/env.h"
#include <vector>

namespace onnxruntime {
namespace test {

/*
Mock cpu layout for windows
"global_processor_id" means "index" of a logical processor in the system,
"local_processor_id" refers to "index" of a logical processor in its belonging group,
the "local" here means group-local.
Unlike the definition of affinity API, the id here starts from 0, not 1.
E.g.
assume cpu_info is like:
{
{{0,1},{2,3}} // group 1 of two cores
{{0,1},{2,3}} // group 2 of two cores
}
then we will have eight logical processors with global id be like:
0,1,2,3,4,5,6,7
and local id be like:
0,1,2,3,0,1,2,3
*/
bool WindowsEnvTester::SetCpuInfo(const CpuInfo& cpu_info) {
  if (cpu_info.empty()) {
    return false;
  }
  cores_.clear();
  global_processor_info_map_.clear();
  int global_processor_id = 0;
  for (int group_id = 0; group_id < static_cast<int>(cpu_info.size()); ++group_id) {
    int local_processor_id = 0;
    for (int core_id = 0; core_id < static_cast<int>(cpu_info[group_id].size()); ++core_id) {
      onnxruntime::LogicalProcessors logical_processors;
      for (int i = 0; i < static_cast<int>(cpu_info[group_id][core_id].size()); ++i) {
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
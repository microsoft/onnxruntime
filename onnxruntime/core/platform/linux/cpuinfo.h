// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/status.h"

#include <vector>

namespace onnxruntime {

struct CpuInfoFileProcessorInfo {
  size_t processor;
  std::string vendor;

  // There are plenty of other fields. We can add more if needed.
};

Status ParseCpuInfoFile(const std::string& cpu_info_file, std::vector<CpuInfoFileProcessorInfo>& cpu_infos);

inline Status ParseCpuInfoFile(std::vector<CpuInfoFileProcessorInfo>& cpu_info) {
  return ParseCpuInfoFile("/proc/cpuinfo", cpu_info);
}

}  // namespace onnxruntime

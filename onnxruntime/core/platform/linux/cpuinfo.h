// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/status.h"

#include <vector>

namespace onnxruntime {

struct CpuInfoFileProcessorInfo {
  size_t processor;
  std::string vendor_id;

  // There are plenty of other fields. We can add more if needed.
};

using CpuInfo = std::vector<CpuInfoFileProcessorInfo>;

Status ParseCpuInfoFile(const std::string& cpu_info_file, CpuInfo& cpu_info);

inline Status ParseCpuInfoFile(CpuInfo& cpu_info) {
  return ParseCpuInfoFile("/proc/cpuinfo", cpu_info);
}

}  // namespace onnxruntime

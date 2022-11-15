// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/env.h"

namespace onnxruntime {
namespace test {

using CpuLogicProcessorId = int32_t; // an id of a logical processor starting from 0
using CpuCore = std::vector<CpuLogicProcessorId>; // a core of multiple logical processors
using CpuCores = std::vector<CpuCore>; // a group of cores
using CpuInfo = std::vector<CpuCores>; // groups

class WindowsEnvTester : public WindowsEnv {
 public:
  bool SetCpuInfo(const CpuInfo& cpu_info);
};

} // namespace test
}  // namespace onnxruntime
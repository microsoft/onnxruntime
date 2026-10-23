// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/env.h"

namespace onnxruntime {
namespace test {

using CpuLogicProcessorId = int;                   // an id of a logical processor starting from 0
using CpuCore = std::vector<CpuLogicProcessorId>;  // a core of multiple logical processors
using CpuGroup = std::vector<CpuCore>;             // core group
using CpuInfo = std::vector<CpuGroup>;             // groups
// EfficiencyClass of each core, in the same order the cores appear in CpuInfo. Windows
// reports a higher value for a higher performance core.
using CpuEfficiencyClasses = std::vector<BYTE>;

class WindowsEnvTester : public WindowsEnv {
 public:
  WindowsEnvTester() = default;
  ~WindowsEnvTester() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WindowsEnvTester);
  bool SetCpuInfo(const CpuInfo& cpu_info);
  // Sets the cores as above and, when the classes are not all equal, records the cores of
  // the highest one as the performance cores. The size must match the total number of cores.
  bool SetCpuInfo(const CpuInfo& cpu_info, const CpuEfficiencyClasses& efficiency_classes);
};

}  // namespace test
}  // namespace onnxruntime
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/env.h"

namespace onnxruntime {
namespace test {

using CpuLogicProcessorId = int;                   // an id of a logical processor starting from 0
using CpuCore = std::vector<CpuLogicProcessorId>;  // a core of multiple logical processors
using CpuGroup = std::vector<CpuCore>;             // core group
using CpuInfo = std::vector<CpuGroup>;             // groups

class WindowsEnvTester : public WindowsEnv {
 public:
  WindowsEnvTester() = default;
  ~WindowsEnvTester() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WindowsEnvTester);
  bool SetCpuInfo(const CpuInfo& cpu_info);
};

}  // namespace test
}  // namespace onnxruntime
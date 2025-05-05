// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>

namespace onnxruntime {
struct HardwareCoreEnumerator {
  HardwareCoreEnumerator() = delete;
  static uint32_t DefaultIntraOpNumThreads();
};
typedef struct {
  bool is_intel;
  bool is_intel_specified_platform;
} CheckIntelResult;
CheckIntelResult CheckIntel();
}  // namespace onnxruntime
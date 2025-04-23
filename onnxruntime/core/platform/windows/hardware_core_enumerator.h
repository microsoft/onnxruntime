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
  bool isIntel;
  bool isIntelSpecifiedPlatform;
} IntelChecks;
IntelChecks checkIntel();
}  // namespace onnxruntime
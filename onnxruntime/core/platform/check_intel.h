// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
typedef struct {
  bool is_intel;
  bool is_intel_specified_platform;
} CheckIntelResult;

CheckIntelResult CheckIntel();
}  // namespace onnxruntime

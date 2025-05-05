// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
struct CheckIntelResult {
  bool is_intel;
  bool is_intel_specified_platform;
};

CheckIntelResult CheckIntel();
}  // namespace onnxruntime

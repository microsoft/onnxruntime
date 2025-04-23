#pragma once

namespace onnxruntime {
struct IntelChecks {
  bool isIntel;
  bool isIntelSpecifiedPlatform;
};

IntelChecks checkIntel();
}  // namespace onnxruntime

#pragma once

namespace onnxruntime {
struct IntelChecks {
  bool is_Intel;
  bool isIntel_SpecifiedPlatform;
};

IntelChecks CheckIntel();
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "EpContextDataReadPolicy.h"

#include <cmath>
#include <limits>

namespace onnxruntimejsi {

namespace {

constexpr const char* kMaxDataSizeOption =
    "session option \"epContextDataRead.maxDataSize\"";

}  // namespace

MaxDataSizeParseResult parseMaxDataSize(double raw) {
  MaxDataSizeParseResult result;

  if (std::isnan(raw) || std::isinf(raw)) {
    result.error = std::string(kMaxDataSizeOption) + " must be a finite number";
    return result;
  }
  if (raw != std::trunc(raw)) {
    result.error = std::string(kMaxDataSizeOption) + " must be an integer";
    return result;
  }
  if (raw <= 0.0) {
    result.error =
        std::string(kMaxDataSizeOption) + " must be greater than zero";
    return result;
  }
  if (raw > kMaxSafeInteger) {
    result.error = std::string(kMaxDataSizeOption) +
                   " must not exceed Number.MAX_SAFE_INTEGER (9007199254740991)";
    return result;
  }
  // ORT requires a value strictly below SIZE_MAX. On 32-bit targets size_t tops out well below
  // the safe-integer range, so a valid JavaScript number may still be unrepresentable.
  if (raw >= static_cast<double>(std::numeric_limits<size_t>::max())) {
    result.error = std::string(kMaxDataSizeOption) +
                   " is too large to represent as size_t on this platform";
    return result;
  }

  result.ok = true;
  result.value = static_cast<size_t>(raw);
  return result;
}

DataSizeCheckResult checkDataSize(size_t dataSize, size_t maxDataSize,
                                  const std::string& name) {
  DataSizeCheckResult result;

  if (dataSize > maxDataSize) {
    result.error = "EPContext data \"" + name + "\" is " +
                   std::to_string(dataSize) +
                   " bytes, which exceeds the configured "
                   "epContextDataRead.maxDataSize of " +
                   std::to_string(maxDataSize) + " bytes";
    return result;
  }

  result.ok = true;
  return result;
}

}  // namespace onnxruntimejsi

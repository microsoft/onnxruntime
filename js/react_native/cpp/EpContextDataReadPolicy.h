// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <string>

namespace onnxruntimejsi {

/**
 * @brief Largest integer a JavaScript number represents exactly (Number.MAX_SAFE_INTEGER).
 */
inline constexpr double kMaxSafeInteger = 9007199254740991.0;

struct MaxDataSizeParseResult {
  bool ok = false;
  size_t value = 0;
  std::string error;
};

/**
 * @brief Validate the `epContextDataRead.maxDataSize` session option.
 *
 * The value must be a finite, positive, safe integer that is representable as size_t on the
 * current platform. size_t is narrower than the safe-integer range on 32-bit targets, so the
 * representability check is not redundant there.
 */
MaxDataSizeParseResult parseMaxDataSize(double raw);

struct DataSizeCheckResult {
  bool ok = false;
  std::string error;
};

/**
 * @brief Enforce the configured payload limit. Callers must run this before asking the ORT
 * allocator for memory.
 */
DataSizeCheckResult checkDataSize(size_t dataSize, size_t maxDataSize,
                                  const std::string& name);

}  // namespace onnxruntimejsi

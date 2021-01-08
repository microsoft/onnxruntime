// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {

enum class ArenaExtendStrategy : int32_t {
  kNextPowerOfTwo = 0,
  kSameAsRequested,
};

}  // namespace onnxruntime

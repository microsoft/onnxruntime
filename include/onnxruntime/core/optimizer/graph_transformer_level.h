// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {

enum class TransformerLevel : uint32_t {
  Default = 0,
  Level1,
  Level2,
  // Convenience enum to always get the max available value. 
  // This way when we add more levels code which iterates over this enum does not need to change.
  MaxTransformerLevel
};

}  // namespace onnxruntime

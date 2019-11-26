// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {

enum class TransformerLevel : int {
  Default = 0,
  Level1,
  Level2,
  Level3,
  Level4,
  // The max level should always be same as the last level.
  MaxLevel = Level4
};

}  // namespace onnxruntime

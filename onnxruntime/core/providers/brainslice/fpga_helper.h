// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"

namespace onnxruntime {
namespace fpga {
class FPGAUtil {
 public:
  // Flip integer.
  static uint32_t FlipUint32(const uint32_t in);
};
}  // namespace fpga
}  // namespace onnxruntime

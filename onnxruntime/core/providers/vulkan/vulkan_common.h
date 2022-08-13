// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vulkan/vulkan.h>

#include "core/common/common.h"

// Call into the Vulkan library (Something like CUDA_CALL)
#define VK_CALL(func)                                \
  {                                                  \
    auto status = (func);                            \
    if (status != VK_SUCCESS) {                      \
      ORT_THROW("Vulkan command invocation failed"); \
    }                                                \
  }

#define VK_CALL_RETURNS_VOID(func) \
  {                                \
    (func);                        \
  }
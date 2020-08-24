// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdexcept>
#include <string>
#include "onnxruntime/core/common/common.h"

#define MTI_ASSERT(condition)                                           \
  if (!(condition)) {                                                   \
    std::string error_msg = "Not satsified: " #condition                \
                            ": line " +                                 \
                            std::to_string(__LINE__) +                  \
                            " in file " + std::string(__FILE__) + "\n"; \
    ORT_THROW(error_msg);                                               \
  }

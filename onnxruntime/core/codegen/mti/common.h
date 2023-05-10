// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdexcept>
#include <string>

#define MTI_ASSERT(condition)                                           \
  if (!(condition)) {                                                   \
    std::string error_msg = "Not satsified: " #condition                \
                            ": line " +                                 \
                            std::to_string(__LINE__) +                  \
                            " in file " + std::string(__FILE__) + "\n"; \
    throw std::runtime_error(error_msg);                                \
  }

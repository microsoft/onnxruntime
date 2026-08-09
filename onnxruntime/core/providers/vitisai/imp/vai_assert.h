// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cassert>
#include <iostream>

#define vai_assert(exp, err_msg)                                     \
  do {                                                               \
    if (!(exp)) {                                                    \
      std::cerr << "check failure: " #exp << (err_msg) << std::endl; \
      std::abort();                                                  \
    }                                                                \
  } while (0)

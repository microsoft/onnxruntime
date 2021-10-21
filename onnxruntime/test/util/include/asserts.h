// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "gtest/gtest.h"

// helpers to run a function and check the status, outputting any error if it fails.
// note: wrapped in do{} while(false) so the _tmp_status variable has limited scope
#define ASSERT_STATUS_OK(function)                  \
  do {                                              \
    Status _tmp_status = (function);                \
    ASSERT_TRUE(_tmp_status.IsOK()) << _tmp_status; \
  } while (false)

#define EXPECT_STATUS_OK(function)                  \
  do {                                              \
    Status _tmp_status = (function);                \
    EXPECT_TRUE(_tmp_status.IsOK()) << _tmp_status; \
  } while (false)

#define ASSERT_STATUS_NOT_OK(function) \
  do {                                 \
    Status _tmp_status = (function);   \
    ASSERT_FALSE(_tmp_status.IsOK());  \
  } while (false)

#define EXPECT_STATUS_NOT_OK(function) \
  do {                                 \
    Status _tmp_status = (function);   \
    EXPECT_FALSE(_tmp_status.IsOK());  \
  } while (false)

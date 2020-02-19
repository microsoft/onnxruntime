// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gtest/gtest.h"

#include "core/common/status.h"

// helpers to run a function and check the status, outputting any error if it fails.
// note: wrapped in do{} while(0) so the status variable has limited scope
#define ASSERT_STATUS_OK(expr)                           \
  do {                                                   \
    const common::Status status = (expr);                \
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage(); \
  } while (0)

#define EXPECT_STATUS_OK(expr)                           \
  do {                                                   \
    const common::Status status = (expr);                \
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage(); \
  } while (0)

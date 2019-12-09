// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gtest/gtest.h"

#include "core/common/status.h"

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

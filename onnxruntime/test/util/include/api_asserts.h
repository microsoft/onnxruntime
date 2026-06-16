// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/status.h"
#include "core/session/onnxruntime_cxx_api.h"

// asserts for the public API
#define ASSERT_ORTSTATUS_OK(function)                                 \
  do {                                                                \
    Ort::Status _tmp_status{(function)};                              \
    ASSERT_TRUE(_tmp_status.IsOK()) << _tmp_status.GetErrorMessage(); \
  } while (false)

#define EXPECT_ORTSTATUS_OK(api, function)                            \
  do {                                                                \
    Ort::Status _tmp_status{(api->function)};                         \
    EXPECT_TRUE(_tmp_status.IsOK()) << _tmp_status.GetErrorMessage(); \
  } while (false)

#define ASSERT_ORTSTATUS_NOT_OK(api, function) \
  do {                                         \
    Ort::Status _tmp_status{(api->function)};  \
    ASSERT_TRUE(_tmp_status.IsOK());           \
  } while (false)

#define EXPECT_ORTSTATUS_NOT_OK(api, function) \
  do {                                         \
    Ort::Status _tmp_status{(api->function)};  \
    EXPECT_FALSE(_tmp_status.IsOK());          \
  } while (false)

#define ASSERT_CXX_ORTSTATUS_OK(function) \
  ASSERT_ORTSTATUS_OK(function)

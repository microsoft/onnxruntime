// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

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

#define ASSERT_STATUS_NOT_OK_CHECK_MSG(function, msg)                   \
  do {                                                                  \
    Status _tmp_status = (function);                                    \
    ASSERT_FALSE(_tmp_status.IsOK());                                   \
    ASSERT_THAT(_tmp_status.ErrorMessage(), ::testing::HasSubstr(msg)); \
  } while (false)

#define EXPECT_STATUS_NOT_OK_CHECK_MSG(function, msg)                   \
  do {                                                                  \
    Status _tmp_status = (function);                                    \
    EXPECT_FALSE(_tmp_status.IsOK());                                   \
    EXPECT_THAT(_tmp_status.ErrorMessage(), ::testing::HasSubstr(msg)); \
  } while (false)

// Same helpers for public API OrtStatus. Get the 'api' instance using:
//   const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
#define ASSERT_ORTSTATUS_OK(api, function)                                \
  do {                                                                    \
    OrtStatusPtr _tmp_status = (api->function);                           \
    ASSERT_EQ(_tmp_status, nullptr) << api->GetErrorMessage(_tmp_status); \
    if (_tmp_status) api->ReleaseStatus(_tmp_status);                     \
  } while (false)

#define EXPECT_ORTSTATUS_OK(api, function)                                \
  do {                                                                    \
    OrtStatusPtr _tmp_status = (api->function);                           \
    EXPECT_EQ(_tmp_status, nullptr) << api->GetErrorMessage(_tmp_status); \
    if (_tmp_status) api->ReleaseStatus(_tmp_status);                     \
  } while (false)

#define ASSERT_ORTSTATUS_NOT_OK(api, function)        \
  do {                                                \
    OrtStatusPtr _tmp_status = (api->function);       \
    ASSERT_NE(_tmp_status, nullptr);                  \
    if (_tmp_status) api->ReleaseStatus(_tmp_status); \
  } while (false)

#define EXPECT_ORTSTATUS_NOT_OK(api, function)        \
  do {                                                \
    OrtStatusPtr _tmp_status = (api->function);       \
    EXPECT_NE(_tmp_status, nullptr);                  \
    if (_tmp_status) api->ReleaseStatus(_tmp_status); \
  } while (false)

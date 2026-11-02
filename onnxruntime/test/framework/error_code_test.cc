// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <string>

#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(ErrorCodeTest, CreateStatusPreservesNullAndEmptyMessages) {
  for (const char* message : {static_cast<const char*>(nullptr), ""}) {
    OrtStatus* status = OrtApis::CreateStatus(ORT_FAIL, message);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(OrtApis::GetErrorCode(status), ORT_FAIL);
    EXPECT_STREQ(OrtApis::GetErrorMessage(status), "");
    OrtApis::ReleaseStatus(status);
  }
}

TEST(ErrorCodeTest, CreateStatusPreservesMessageLimit) {
  const std::string message(kMaxStrLen + 1, 'x');
  OrtStatus* status = OrtApis::CreateStatus(ORT_FAIL, message.c_str());
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(OrtApis::GetErrorCode(status), ORT_FAIL);
  EXPECT_EQ(std::strlen(OrtApis::GetErrorMessage(status)), kMaxStrLen);
  OrtApis::ReleaseStatus(status);
}

TEST(ErrorCodeTest, UnknownExceptionIncludesFunctionName) {
  OrtStatus* status = CreateUnknownExceptionStatus("TestFunction");
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(OrtApis::GetErrorCode(status), ORT_FAIL);
  EXPECT_STREQ(OrtApis::GetErrorMessage(status), "Unknown exception in TestFunction");
  OrtApis::ReleaseStatus(status);
}

TEST(ErrorCodeTest, UnknownExceptionTruncatesLongFunctionName) {
  const std::string function_name(400, 'x');
  OrtStatus* status = CreateUnknownExceptionStatus(function_name.c_str());
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(OrtApis::GetErrorCode(status), ORT_FAIL);
  EXPECT_EQ(std::strlen(OrtApis::GetErrorMessage(status)), 255U);
  OrtApis::ReleaseStatus(status);
}

#ifndef ORT_NO_EXCEPTIONS
static OrtStatus* ThrowUnknownException() noexcept {
  API_IMPL_BEGIN
  throw 42;
  API_IMPL_END
}

TEST(ErrorCodeTest, CatchAllPreservesFunctionContext) {
  OrtStatus* status = ThrowUnknownException();
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(OrtApis::GetErrorCode(status), ORT_FAIL);
  EXPECT_STREQ(OrtApis::GetErrorMessage(status), "Unknown exception in ThrowUnknownException");
  OrtApis::ReleaseStatus(status);
}
#endif

}  // namespace test
}  // namespace onnxruntime

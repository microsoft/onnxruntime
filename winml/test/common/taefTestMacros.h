// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "WexTestClass.h"

using namespace WEX::Logging;
using namespace WEX::Common;
using namespace WEX::TestExecution;

#define WINML_EXPECT_NO_THROW(statement) VERIFY_NO_THROW(statement)

#define WINML_TEST_CLASS_BEGIN_WITH_SETUP(test_class_name, setup_method) \
  class test_class_name {                                                \
    TEST_CLASS(test_class_name);                                         \
    TEST_CLASS_SETUP(TestClassSetup) {                                   \
      getapi().setup_method();                                           \
      return true;                                                       \
    }

#define WINML_TEST_CLASS_END() \
  }                            \
  ;

#define WINML_TEST(group_name, test_name) \
  TEST_METHOD(test_name) {                \
    getapi().test_name();                 \
  }

#define WINML_SKIP_TEST(message)                                                                 \
  do {                                                                                           \
    Log::Result(TestResults::Skipped,                                                            \
                std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(message).c_str()); \
    return;                                                                                      \
  } while (0)

#define WINML_EXPECT_NO_THROW(statement) VERIFY_NO_THROW(statement)
#define WINML_EXPECT_TRUE(statement) VERIFY_IS_TRUE(statement)
#define WINML_EXPECT_FALSE(statement) VERIFY_IS_FALSE(statement)
#define WINML_EXPECT_EQUAL(val1, val2) VERIFY_ARE_EQUAL(val1, val2)
#define WINML_EXPECT_NOT_EQUAL(val1, val2) VERIFY_ARE_NOT_EQUAL(val1, val2)
#define WINML_LOG_ERROR(message) \
  VERIFY_FAIL(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(message).c_str())
#define WINML_LOG_COMMENT(message)\
  WEX::Logging::Log::Comment(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(message).c_str())
#define WINML_EXPECT_HRESULT_SUCCEEDED(hresult_expression) VERIFY_SUCCEEDED(hresult_expression)
#define WINML_EXPECT_THROW_SPECIFIC(statement, exception, condition) VERIFY_THROWS_SPECIFIC(statement, exception, condition)
#define WINML_EXPECT_HRESULT_FAILED(hresult_expression) VERIFY_FAILED(hresult_expression)

#ifndef USE_DML
#define GPUTEST \
  WINML_SUPRESS_UNREACHABLE_BELOW(WINML_SKIP_TEST("GPU tests disabled because this is a WinML only build (no DML)"))
#else
#define GPUTEST                                                                             \
  bool noGPUTests;                                                                          \
  if (SUCCEEDED(RuntimeParameters::TryGetValue(L"noGPUtests", noGPUTests)) && noGPUTests) { \
    WINML_SKIP_TEST("This test is disabled by the noGPUTests runtime parameter.");          \
    return;                                                                                 \
  }
#endif

#define SKIP_EDGECORE                                                                       \
  bool edgeCoreRun;                                                                         \
  if (SUCCEEDED(RuntimeParameters::TryGetValue(L"EdgeCore", edgeCoreRun)) && edgeCoreRun) { \
    WINML_SKIP_TEST("This test is disabled by the EdgeCore runtime parameter.");            \
    return;                                                                                 \
  }

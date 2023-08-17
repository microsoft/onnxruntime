// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "WexTestClass.h"

using namespace WEX::Logging;
using namespace WEX::Common;
using namespace WEX::TestExecution;

#define WINML_EXPECT_NO_THROW(statement) VERIFY_NO_THROW(statement)

#define WINML_TEST_CLASS_BEGIN(test_class_name) \
  class test_class_name {                       \
    TEST_CLASS(test_class_name);

#define WINML_TEST_CLASS_SETUP_CLASS(setup_class) \
  TEST_CLASS_SETUP(TestClassSetup) {              \
    getapi().setup_class();                       \
    return true;                                  \
  }

#define WINML_TEST_CLASS_TEARDOWN_CLASS(teardown_class) \
  TEST_CLASS_CLEANUP(TestClassCleanup) {                \
    getapi().teardown_class();                          \
    return true;                                        \
  }

#define WINML_TEST_CLASS_SETUP_METHOD(setup_method) \
  TEST_METHOD_SETUP(TestMethodSetup) {              \
    getapi().setup_method();                        \
    return true;                                    \
  }

#define WINML_TEST_CLASS_TEARDOWN_METHOD(teardown_method) \
  TEST_METHOD_CLEANUP(TestClassCleanup) {                 \
    getapi().teardown_method();                           \
    return true;                                          \
  }

#define WINML_TEST_CLASS_BEGIN_TESTS

#define WINML_TEST_CLASS_END() \
  }                            \
  ;

#define WINML_TEST(group_name, test_name) \
  TEST_METHOD(test_name) {                \
    getapi().test_name();                 \
  }

#define WINML_SKIP_TEST(message)                                                                                       \
  WINML_SUPRESS_UNREACHABLE_BELOW(                                                                                     \
    Log::Result(TestResults::Skipped, std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(message).c_str()); \
    return;                                                                                                            \
  )

#define WINML_EXPECT_NO_THROW(statement) VERIFY_NO_THROW(statement)
#define WINML_EXPECT_TRUE(statement) VERIFY_IS_TRUE(statement)
#define WINML_EXPECT_FALSE(statement) VERIFY_IS_FALSE(statement)
#define WINML_EXPECT_EQUAL(val1, val2) VERIFY_ARE_EQUAL(val1, val2)
#define WINML_EXPECT_NOT_EQUAL(val1, val2) VERIFY_ARE_NOT_EQUAL(val1, val2)
#define WINML_LOG_ERROR(message) \
  VERIFY_FAIL(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(message).c_str())
#define WINML_LOG_COMMENT(message) \
  WEX::Logging::Log::Comment(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(message).c_str())
#define WINML_EXPECT_HRESULT_SUCCEEDED(hresult_expression) VERIFY_SUCCEEDED(hresult_expression)
#define WINML_EXPECT_THROW_SPECIFIC(statement, exception, condition) \
  VERIFY_THROWS_SPECIFIC(statement, exception, condition)
#define WINML_EXPECT_HRESULT_FAILED(hresult_expression) VERIFY_FAILED(hresult_expression)

static bool RuntimeParameterExists(std::wstring param) {
  bool param_value;
  return SUCCEEDED(RuntimeParameters::TryGetValue(param.c_str(), param_value)) && param_value;
}

static bool SkipGpuTests() {
#ifndef USE_DML
  return true;
#else
  return RuntimeParameterExists(L"noGPUtests");
#endif
}

#define GPUTEST                            \
  if (SkipGpuTests()) {                    \
    WINML_SKIP_TEST("Gpu tests disabled"); \
  }

static bool SkipTestsImpactedByOpenMP() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

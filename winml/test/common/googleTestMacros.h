// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include "runtimeParameters.h"

#define TEST_GROUP_BEGIN(group_name)
#define TEST_GROUP_END()

#define WINML_TEST(group_name, test_name) \
  TEST_F(group_name, test_name) {         \
    getapi().test_name();                 \
  }

#define WINML_TEST_CLASS_BEGIN(test_class_name)      \
  namespace {                                        \
    class test_class_name : public ::testing::Test {

#define WINML_TEST_CLASS_SETUP_CLASS(setup_class) \
    protected:                                    \
      static void SetUpTestSuite() {              \
        getapi().setup_class();                   \
      }

#define WINML_TEST_CLASS_TEARDOWN_CLASS(teardown_class) \
    protected:                                          \
      static void TearDownTestSuite() {                 \
        getapi().teardown_class();                      \
      }

#define WINML_TEST_CLASS_SETUP_METHOD(setup_method) \
    protected:                                      \
      void SetUp() override {                       \
        getapi().setup_method();                    \
      }

#define WINML_TEST_CLASS_TEARDOWN_METHOD(teardown_method) \
    protected:                                            \
      void TearDown() override {                          \
        getapi().teardown_method();                       \
      }

#define WINML_TEST_CLASS_BEGIN_TESTS };
  
#define WINML_TEST_CLASS_END() }


// For old versions of gtest without GTEST_SKIP, stream the message and return success instead
#ifndef GTEST_SKIP
#define GTEST_SKIP_(message) \
  return GTEST_MESSAGE_(message, ::testing::TestPartResult::kSuccess)
#define GTEST_SKIP GTEST_SKIP_("")
#endif

#define EXPECT_THROW_SPECIFIC(statement, exception, condition)  \
    EXPECT_THROW(                                               \
        try {                                                   \
            statement;                                          \
        } catch (const exception& e) {                          \
            EXPECT_TRUE(condition(e));                          \
            throw;                                              \
        }                                                       \
    , exception);

#ifndef INSTANTIATE_TEST_SUITE_P
// Use the old name, removed in newer versions of googletest
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

#define WINML_SKIP_TEST(message) \
  WINML_SUPRESS_UNREACHABLE_BELOW(GTEST_SKIP() << message)

#define WINML_EXPECT_NO_THROW(statement) EXPECT_NO_THROW(statement)
#define WINML_EXPECT_TRUE(statement) EXPECT_TRUE(statement)
#define WINML_EXPECT_FALSE(statement) EXPECT_FALSE(statement)
#define WINML_EXPECT_EQUAL(val1, val2) EXPECT_EQ(val1, val2)
#define WINML_EXPECT_NOT_EQUAL(val1, val2) EXPECT_NE(val1, val2)

#define WINML_LOG_ERROR(message) \
  ADD_FAILURE() << message
#define WINML_LOG_COMMENT(message)\
  SCOPED_TRACE(message)
#define WINML_EXPECT_HRESULT_SUCCEEDED(hresult_expression) EXPECT_HRESULT_SUCCEEDED(hresult_expression)
#define WINML_EXPECT_HRESULT_FAILED(hresult_expression) EXPECT_HRESULT_FAILED(hresult_expression)
#define WINML_EXPECT_THROW_SPECIFIC(statement, exception, condition) EXPECT_THROW_SPECIFIC(statement, exception, condition)

#ifndef USE_DML
#define GPUTEST \
  WINML_SKIP_TEST("GPU tests disabled because this is a WinML only build (no DML)")
#define GPUTEST_ENABLED alwaysFalse()
#else
#define GPUTEST                                                                               \
  do {                                                                                        \
    if (auto no_gpu_tests = RuntimeParameters::Parameters.find("noGPUtests");                 \
        no_gpu_tests != RuntimeParameters::Parameters.end() && no_gpu_tests->second != "0") { \
      WINML_SKIP_TEST("GPU tests disabled");                                                  \
    }                                                                                         \
  } while (0)
#define GPUTEST_ENABLED auto _no_gpu_tests = RuntimeParameters::Parameters.find("noGPUtests");    \
      _no_gpu_tests == RuntimeParameters::Parameters.end() || _no_gpu_tests->second == "0"
#endif

#define SKIP_EDGECORE                                                                         \
  do {                                                                                        \
    if (auto is_edge_core = RuntimeParameters::Parameters.find("EdgeCore");                   \
        is_edge_core != RuntimeParameters::Parameters.end() && is_edge_core->second != "0") { \
      WINML_SKIP_TEST("Test can't be run in EdgeCore");                                       \
    }                                                                                         \
  } while (0)

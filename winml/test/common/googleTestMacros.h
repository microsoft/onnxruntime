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

#define WINML_TEST_CLASS_BEGIN_NO_SETUP(test_class_name) \
  namespace {                                            \
    class test_class_name : public ::testing::Test {     \
    };

#define WINML_TEST_CLASS_BEGIN_WITH_SETUP(test_class_name, setup_method) \
  namespace {                                                            \
    class test_class_name : public ::testing::Test {                     \
    protected:                                                           \
      void SetUp() override {                                            \
        getapi().setup_method();                                         \
      }                                                                  \
    };

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
  GTEST_SKIP() << message;

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
  WINML_SUPRESS_UNREACHABLE_BELOW(WINML_SKIP_TEST("GPU tests disabled because this is a WinML only build (no DML)"))
#else
#define GPUTEST                                                                         \
  if (auto noGpuTests = RuntimeParameters::Parameters.find("noGPUtests");               \
      noGpuTests != RuntimeParameters::Parameters.end() && noGpuTests->second != "0") { \
    WINML_SKIP_TEST("GPU tests disabled");                                              \
  }
#endif

#define SKIP_EDGECORE                                                                   \
  if (auto isEdgeCore = RuntimeParameters::Parameters.find("EdgeCore");                 \
      isEdgeCore != RuntimeParameters::Parameters.end() && isEdgeCore->second != "0") { \
    WINML_SKIP_TEST("Test can't be run in EdgeCore");                                   \
  }

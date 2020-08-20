// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_MSC_VER) && !defined(NDEBUG)

#include "core/common/common.h"

#include <iostream>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {

using namespace ::testing;
//TVM is not working with StackTrace now.
#if !(defined USE_TVM || (defined USE_NGRAPH && defined _WIN32))
TEST(StacktraceTests, BasicTests) {
  auto result = ::onnxruntime::GetStackTrace();

  // if we are running code coverage the Windows CaptureStackBackTrace function only returns a single
  // frame that is unknown. adjust for that.
  // works fine when running unit tests normally.
  const bool have_working_stacktrace = result.size() > 1;

  if (have_working_stacktrace)
    // this method name should be the first on the stack as we hide the calls to the infrastructure that
    // creates the stack trace
    EXPECT_THAT(result[0], HasSubstr("BasicTests"));

  try {
    ORT_THROW("Testing");
  } catch (const OnnxRuntimeException& ex) {
    auto msg = ex.what();
    std::cout << msg;

    if (have_working_stacktrace)
      // unit tests are run by main() in test_main.cc, so make sure that is present
      EXPECT_THAT(msg, HasSubstr("test_main.cc"));
    else
      // otherwise just make sure we captured where the throw was from
      EXPECT_THAT(msg, HasSubstr("BasicTests"));
  }
}
#endif
}  // namespace test
}  // namespace onnxruntime
#endif

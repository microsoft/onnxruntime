// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/run_options_c_api.h"
#include "test_fixture.h"
using namespace onnxruntime;

TEST_F(CApiTest, run_options) {
  std::unique_ptr<ONNXRuntimeRunOptions> options(ONNXRuntimeCreateRunOptions());
  ASSERT_NE(options, nullptr);
  ASSERT_EQ(ONNXRuntimeRunOptionsSetRunLogVerbosityLevel(options.get(), 1), nullptr);
  ASSERT_EQ(ONNXRuntimeRunOptionsSetRunTag(options.get(), "abc"), nullptr);
  ASSERT_STREQ(ONNXRuntimeRunOptionsGetRunTag(options.get()), "abc");
  ASSERT_EQ(ONNXRuntimeRunOptionsGetRunLogVerbosityLevel(options.get()), (unsigned)1);
}

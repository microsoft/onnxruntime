// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "test_fixture.h"
using namespace onnxruntime;

TEST_F(CApiTest, run_options) {
  std::unique_ptr<OrtRunOptions> options(OrtCreateRunOptions());
  ASSERT_NE(options, nullptr);
  ASSERT_EQ(OrtRunOptionsSetRunLogVerbosityLevel(options.get(), 1), nullptr);
  ASSERT_EQ(OrtRunOptionsSetRunTag(options.get(), "abc"), nullptr);
  ASSERT_STREQ(OrtRunOptionsGetRunTag(options.get()), "abc");
  ASSERT_EQ(OrtRunOptionsGetRunLogVerbosityLevel(options.get()), unsigned(1));
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include <gtest/gtest.h>

TEST(CApiTest, run_options) {
  Ort::RunOptions options;
  ASSERT_NE(options, nullptr);
  options.SetRunLogVerbosityLevel(1);
  options.SetRunTag("abc");
  ASSERT_STREQ(options.GetRunTag(), "abc");
  ASSERT_EQ(options.GetRunLogVerbosityLevel(), 1);
}

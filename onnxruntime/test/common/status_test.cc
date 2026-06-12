// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"

#include <cerrno>
#include <string>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace common {
namespace test {

// A SYSTEM-category status carries a raw OS errno, which must be mapped to the closest StatusCode.
// Unrecognized values fall back to FAIL.
TEST(StatusCodeFromSystemErrnoTest, KnownAndUnknownValues) {
  EXPECT_EQ(StatusCodeFromSystemErrno(ENOENT), NO_SUCHFILE);
  EXPECT_EQ(StatusCodeFromSystemErrno(EINVAL), INVALID_ARGUMENT);
  EXPECT_EQ(StatusCodeFromSystemErrno(EACCES), FAIL);
}

// ToString() on a SYSTEM-category status must use the status' own stored code, not the live global errno.
TEST(StatusTest, ToStringWithSystemCategoryUsesStoredCode) {
  errno = EINVAL;  // Poison the global errno; ToString() must not read it.
  const Status status(SYSTEM, ENOENT, "open file failed");
  const std::string text = status.ToString();
  EXPECT_NE(text.find(std::to_string(ENOENT)), std::string::npos);
  EXPECT_EQ(text.find(std::to_string(EINVAL)), std::string::npos);
}

}  // namespace test
}  // namespace common
}  // namespace onnxruntime

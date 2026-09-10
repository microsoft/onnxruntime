// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"

#include <cerrno>
#include <string>

#include "gsl/gsl"
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
  const int saved_errno = errno;
  auto restore_errno = gsl::finally([saved_errno]() { errno = saved_errno; });
  errno = EINVAL;  // Poison the global errno; ToString() must not read it.
  const Status status(SYSTEM, ENOENT, "open file failed");
  EXPECT_EQ(status.ToString(), "SystemError : " + std::to_string(ENOENT));
}

}  // namespace test
}  // namespace common
}  // namespace onnxruntime

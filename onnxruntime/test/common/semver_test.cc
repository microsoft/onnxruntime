// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/semver.h"

#include "gtest/gtest.h"

#include "test/util/include/asserts.h"

namespace onnxruntime::test {

TEST(SemVerParsingTest, Basic) {
  {
    auto semver = ParseSemVerVersion("1.2.3-abcde+fghij");
    EXPECT_EQ(semver.major, 1);
    EXPECT_EQ(semver.minor, 2);
    EXPECT_EQ(semver.patch, 3);
    EXPECT_EQ(semver.prerelease, "abcde");
    EXPECT_EQ(semver.build_metadata, "fghij");
  }

  {
    auto semver = ParseSemVerVersion("1.2.3");
    EXPECT_EQ(semver.major, 1);
    EXPECT_EQ(semver.minor, 2);
    EXPECT_EQ(semver.patch, 3);
    EXPECT_EQ(semver.prerelease, std::nullopt);
    EXPECT_EQ(semver.build_metadata, std::nullopt);
  }
}

TEST(SemVerParsingTest, Invalid) {
  SemVerVersion semver{};
  ASSERT_STATUS_NOT_OK(ParseSemVerVersion("version one point zero", &semver));
}

}  // namespace onnxruntime::test

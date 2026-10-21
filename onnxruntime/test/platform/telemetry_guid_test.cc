// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_guid.h"

#include "gtest/gtest.h"

namespace onnxruntime::test {

TEST(TelemetryGuidTest, ValidatesPersistedGuidFormat) {
  for (const char* value : {"11111111-2222-4333-8444-555555555555",
                            "01234567-89ab-cdef-ABCD-0123456789AB"}) {
    EXPECT_TRUE(IsValidGuid(value)) << value;
  }
  for (const char* value : {"", "corrupted", "11111111-2222-4333-8444-55555555555",
                            "11111111-2222-4333-8444-5555555555555",
                            "11111111_2222-4333-8444-555555555555",
                            "11111111-2222-4333-8444-55555555555g"}) {
    EXPECT_FALSE(IsValidGuid(value)) << value;
  }
  const std::string embedded_null("11111111-2222-4333-8444-555555555555\0junk", 41);
  EXPECT_FALSE(IsValidGuid(embedded_null));
}

TEST(TelemetryGuidTest, GeneratesVersionFourGuid) {
  const auto guid = GenerateGuidV4();
  ASSERT_TRUE(IsValidGuid(guid));
  EXPECT_EQ(guid[14], '4');
  EXPECT_NE(std::string_view("89ab").find(guid[19]), std::string_view::npos);
}

}  // namespace onnxruntime::test

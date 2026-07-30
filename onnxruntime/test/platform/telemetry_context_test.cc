// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/telemetry_context.h"
#include "core/platform/posix/telemetry_sha256.h"

#include <map>
#include <string>

#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {

class RecordingSemanticContext {
 public:
  void SetCommonField(const std::string& name, const std::string& value) {
    fields_[name] = value;
  }

  const std::map<std::string, std::string>& Fields() const {
    return fields_;
  }

 private:
  std::map<std::string, std::string> fields_;
};

TEST(TelemetryContextTest, SuppressesUnneededCommonContext) {
  RecordingSemanticContext context;
  telemetry_internal::SuppressUnneededCommonContext(context);

  ASSERT_EQ(context.Fields().size(), telemetry_internal::kSuppressedCommonContextFields.size());
  for (const char* field : telemetry_internal::kSuppressedCommonContextFields) {
    EXPECT_EQ(context.Fields().at(field), "");
  }
}

TEST(TelemetryContextTest, SuppressesNetworkContextAfterProcessInfo) {
  RecordingSemanticContext context;
  telemetry_internal::SuppressNetworkContext(context);

  ASSERT_EQ(context.Fields().size(), telemetry_internal::kProcessInfoOnlyNetworkContextFields.size());
  for (const char* field : telemetry_internal::kProcessInfoOnlyNetworkContextFields) {
    EXPECT_EQ(context.Fields().at(field), "");
  }
}

TEST(TelemetryContextTest, SharedDeviceIdUsesUnsaltedSha256) {
  EXPECT_EQ(telemetry_internal::Sha256::HashStringHex(""),
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855");
  EXPECT_EQ(telemetry_internal::Sha256::HashStringHex("abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD");
  EXPECT_EQ(telemetry_internal::Sha256::HashStringHex(
                "01234567-89ab-4def-8123-456789abcdef"),
            "6225BD190D6CCF87766A49C9986D174DEF3391FE175A61525E49A1D2334D6A43");
}

}  // namespace
}  // namespace onnxruntime::test

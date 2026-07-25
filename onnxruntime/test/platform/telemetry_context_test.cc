// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/telemetry_context.h"

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

}  // namespace
}  // namespace onnxruntime::test

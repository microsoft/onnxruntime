// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <stdexcept>
#include <string>

#include "core/platform/posix/telemetry_no_throw.h"

namespace onnxruntime::test {

#ifndef ORT_NO_EXCEPTIONS

TEST(TelemetryNoThrowTest, EventConstructionExceptionDoesNotEscape) {
  std::string warning;

  EXPECT_NO_THROW(telemetry_internal::RunTelemetryOperationNoThrow(
      []() {
        throw std::runtime_error("event construction failed");
      },
      [&](const char* message) {
        warning = message;
      }));

  EXPECT_EQ(warning, "event construction failed");
}

TEST(TelemetryNoThrowTest, SdkEmissionUnknownExceptionDoesNotEscape) {
  struct ThrowingSdkLogger {
    void LogEvent(int) const {
      throw 42;
    }
  } logger;
  bool warned = false;

  EXPECT_NO_THROW(telemetry_internal::RunTelemetryOperationNoThrow(
      [&]() {
        const int event = 1;
        logger.LogEvent(event);
      },
      [&](const char* message) {
        warned = true;
        EXPECT_EQ(message, nullptr);
      }));

  EXPECT_TRUE(warned);
}

TEST(TelemetryNoThrowTest, DiagnosticExceptionDoesNotEscape) {
  EXPECT_NO_THROW(telemetry_internal::RunTelemetryOperationNoThrow(
      []() {
        throw std::runtime_error("event failed");
      },
      [](const char*) {
        throw std::runtime_error("diagnostic failed");
      }));
}

#endif  // ORT_NO_EXCEPTIONS

}  // namespace onnxruntime::test

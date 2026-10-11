// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_no_throw.h"

#include <stdexcept>

#include "gtest/gtest.h"

namespace onnxruntime::test {

TEST(TelemetryNoThrowTest, SuccessfulOperationsDoNotWarn) {
  int calls = 0;
  EXPECT_TRUE(telemetry_internal::TryTelemetryOperationNoThrow([&]() { ++calls; }));
  telemetry_internal::RunTelemetryOperationNoThrow(
      [&]() { ++calls; }, [](const char*) { ADD_FAILURE() << "successful operation warned"; });
  EXPECT_EQ(calls, 2);
}

#ifndef ORT_NO_EXCEPTIONS

TEST(TelemetryNoThrowTest, ReportsStandardAndUnknownExceptionsWithoutEscaping) {
  for (bool standard_exception : {false, true}) {
    SCOPED_TRACE(standard_exception);
    const auto operation = [standard_exception]() {
      if (standard_exception) throw std::runtime_error("event failed");
      throw 42;
    };
    int warnings = 0;
    EXPECT_NO_THROW(telemetry_internal::RunTelemetryOperationNoThrow(
        operation, [&](const char* message) {
          ++warnings;
          if (standard_exception) {
            ASSERT_NE(message, nullptr);
            EXPECT_STREQ(message, "event failed");
          } else {
            EXPECT_EQ(message, nullptr);
          }
        }));
    EXPECT_EQ(warnings, 1);
    EXPECT_FALSE(telemetry_internal::TryTelemetryOperationNoThrow(operation));
  }
}

TEST(TelemetryNoThrowTest, DiagnosticExceptionDoesNotEscape) {
  EXPECT_NO_THROW(telemetry_internal::RunTelemetryOperationNoThrow(
      []() { throw std::runtime_error("event failed"); },
      [](const char*) { throw std::runtime_error("diagnostic failed"); }));
}

#endif  // ORT_NO_EXCEPTIONS

}  // namespace onnxruntime::test

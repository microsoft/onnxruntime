// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdint>
#include <string>

#include <EventProperties.hpp>
#include "gtest/gtest.h"

#include "core/platform/posix/telemetry.h"

namespace onnxruntime::test {
namespace {

using Microsoft::Applications::Events::EventProperty;

TEST(PosixTelemetryWindowsTest, BuildsExecutionProviderEvent) {
  LUID adapter_luid{};
  adapter_luid.LowPart = 0x89abcdef;
  adapter_luid.HighPart = static_cast<LONG>(0xfedcba98);

  const auto event = telemetry_internal::BuildExecutionProviderEvent(adapter_luid);
  EXPECT_EQ(event.GetName(), "ExecutionProviderEvent");

  const auto& properties = event.GetProperties();
  EXPECT_EQ(properties.at("adapterLuidLowPart").type, EventProperty::TYPE_INT64);
  EXPECT_EQ(properties.at("adapterLuidLowPart").as_int64, UINT64_C(0x89abcdef));
  EXPECT_EQ(properties.at("adapterLuidHighPart").type, EventProperty::TYPE_INT64);
  EXPECT_EQ(properties.at("adapterLuidHighPart").as_int64, UINT64_C(0xfedcba98));
}

TEST(PosixTelemetryWindowsTest, BuildsUtf8DriverInfoEvent) {
  const auto event = telemetry_internal::BuildDriverInfoEvent(
      "Display", L"Driver \u6d4b\u8bd5", L"Version \u7248\u672c");
  EXPECT_EQ(event.GetName(), "DriverInfo");

  const auto& properties = event.GetProperties();
  EXPECT_EQ(properties.at("schemaVersion").as_int64, 0);
  EXPECT_STREQ(properties.at("deviceClass").as_string, "Display");
  EXPECT_STREQ(properties.at("driverNames").as_string, "Driver \xe6\xb5\x8b\xe8\xaf\x95");
  EXPECT_STREQ(properties.at("driverVersions").as_string, "Version \xe7\x89\x88\xe6\x9c\xac");
}

}  // namespace
}  // namespace onnxruntime::test

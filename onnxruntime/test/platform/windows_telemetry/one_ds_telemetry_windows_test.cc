// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdint>
#include <string>

#include <EventProperties.hpp>
#include "gtest/gtest.h"

#include "core/platform/windows/telemetry_1ds.h"
#include "core/platform/telemetry_strings.h"
#include "core/common/logging/logging.h"
#include "test/common/logging/helpers.h"

namespace onnxruntime::test {
namespace {

using Microsoft::Applications::Events::EventProperty;

TEST(TelemetryEtwStringTest, BoundsCaptureViewWithoutChangingOtherLoggingSinks) {
  const std::string large(100000, 'a');
  auto sink = std::make_unique<MockSink>();
  EXPECT_CALL(*sink, SendImpl(testing::_, testing::_, testing::Property(&logging::Capture::Message, testing::Eq(large))))
      .Times(1);
  logging::LoggingManager manager{std::move(sink), logging::Severity::kWARNING, false,
                                  logging::LoggingManager::InstanceType::Temporal};
  auto logger = manager.CreateLogger("TelemetryEtwStringTest");
  {
    logging::Capture capture{*logger, logging::Severity::kWARNING, "telemetry",
                             logging::DataType::SYSTEM, ORT_WHERE};
    capture.Stream() << large;
    EXPECT_EQ(capture.MessageView(), large);
    EXPECT_EQ(telemetry_detail::BoundedTelemetryString(capture.MessageView()).size(), 1024);
  }
}

TEST(OneDsTelemetryWindowsTest, BuildsExecutionProviderEvent) {
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

TEST(OneDsTelemetryWindowsTest, BuildsUtf8DriverInfoEvent) {
  const auto event = telemetry_internal::BuildDriverInfoEvent(
      "Display", L"Driver \u6d4b\u8bd5", L"Version \u7248\u672c");
  EXPECT_EQ(event.GetName(), "DriverInfo");

  const auto& properties = event.GetProperties();
  EXPECT_EQ(properties.at("schemaVersion").as_int64, 0);
  EXPECT_STREQ(properties.at("deviceClass").as_string, "Display");
  EXPECT_STREQ(properties.at("driverNames").as_string, "Driver \xe6\xb5\x8b\xe8\xaf\x95");
  EXPECT_STREQ(properties.at("driverVersions").as_string, "Version \xe7\x89\x88\xe6\x9c\xac");
}

TEST(OneDsTelemetryWindowsTest, ProviderOptionsRedactPathsForNormalAndCaptureStateEvents) {
  for (bool capture_state : {false, true}) {
    const auto event = telemetry_internal::BuildProviderOptionsEvent(
        "CUDAExecutionProvider", "device_id:0,cache_dir:C:\\Users\\First Last\\cache", capture_state);
    EXPECT_EQ(event.GetName(), capture_state ? "ProviderOptions_CaptureState" : "ProviderOptions");
    const auto& properties = event.GetProperties();
    EXPECT_EQ(properties.at("schemaVersion").as_int64, 0);
    EXPECT_STREQ(properties.at("providerId").as_string, "CUDAExecutionProvider");
    EXPECT_STREQ(properties.at("providerOptions").as_string, "device_id:0,cache_dir:[path]");
  }
}

TEST(OneDsTelemetryWindowsTest, ProviderOptionsWithoutPathsArePreserved) {
  const auto event = telemetry_internal::BuildProviderOptionsEvent(
      "CUDAExecutionProvider", "device_id:0,arena_extend_strategy:kSameAsRequested", false);
  EXPECT_STREQ(event.GetProperties().at("providerOptions").as_string,
               "device_id:0,arena_extend_strategy:kSameAsRequested");
}

TEST(OneDsTelemetryWindowsTest, BoundsDriverAndProviderPropertiesBeforeConversion) {
  const std::string large(100000, 'a');
  const std::wstring wide(100000, L'\u20ac');
  const auto driver = telemetry_internal::BuildDriverInfoEvent(large, wide, wide);
  const auto& properties = driver.GetProperties();
  EXPECT_EQ(std::string_view(properties.at("deviceClass").as_string).size(), 1024);
  EXPECT_EQ(std::string_view(properties.at("driverNames").as_string).size(), 1023);
  EXPECT_EQ(std::string_view(properties.at("driverVersions").as_string).size(), 1023);
  const auto provider = telemetry_internal::BuildProviderOptionsEvent(large, large, false);
  EXPECT_EQ(std::string_view(provider.GetProperties().at("providerId").as_string).size(), 1024);
  EXPECT_EQ(std::string_view(provider.GetProperties().at("providerOptions").as_string).size(), 1024);
}

}  // namespace
}  // namespace onnxruntime::test

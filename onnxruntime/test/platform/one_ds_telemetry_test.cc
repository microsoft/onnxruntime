// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_1DS_TELEMETRY

#include <ILogConfiguration.hpp>
#include <NullObjects.hpp>
#include <map>
#include <string>

#include "core/platform/telemetry_1ds.h"
#include "core/platform/telemetry_strings.h"
#include "core/platform/telemetry_sampling.h"
#include "core/platform/telemetry_1ds_platform.h"
#include "gtest/gtest.h"

namespace onnxruntime::test {

class OneDsTelemetryTest : public testing::Test {
 protected:
  class RecordingLogger : public Microsoft::Applications::Events::NullLogger {
   public:
    void LogEvent(const Microsoft::Applications::Events::EventProperties& event) override {
      ++event_count;
      for (const auto& [name, property] : event.GetProperties()) {
        if (property.type == Microsoft::Applications::Events::EventProperty::TYPE_STRING) {
          strings[name] = property.as_string;
        }
      }
    }

    size_t event_count = 0;
    std::map<std::string, std::string> strings;
  };

  void SetUp() override {
    previous_logger_ = OneDsTelemetry::logger_.exchange(&logger_);
    previous_enabled_ = OneDsTelemetry::enabled_.exchange(true);
    previous_disabled_ = OneDsTelemetry::telemetry_disabled_.exchange(false);
    previous_process_info_logged_ = OneDsTelemetry::process_info_logged_.exchange(false);
  }

  void TearDown() override {
    OneDsTelemetry::logger_.store(previous_logger_);
    OneDsTelemetry::enabled_.store(previous_enabled_);
    OneDsTelemetry::telemetry_disabled_.store(previous_disabled_);
    OneDsTelemetry::process_info_logged_.store(previous_process_info_logged_);
  }

  bool ProcessInfoLogged() const {
    return OneDsTelemetry::process_info_logged_.load();
  }

  void SuppressProcess() {
    OneDsTelemetry::telemetry_disabled_.store(true);
    OneDsTelemetry::enabled_.store(false);
  }

  void RemoveLogger() {
    OneDsTelemetry::logger_.store(nullptr);
  }

  Microsoft::Applications::Events::ILogConfiguration* GetSdkConfiguration() const {
    return OneDsTelemetry::config_.get();
  }

  OneDsTelemetry telemetry_;
  RecordingLogger logger_;

 private:
  Microsoft::Applications::Events::ILogger* previous_logger_ = nullptr;
  bool previous_enabled_ = false;
  bool previous_disabled_ = false;
  bool previous_process_info_logged_ = false;
};

#if defined(_WIN32)
TEST_F(OneDsTelemetryTest, WindowsNetworkDetectorIsDisabled) {
  auto* config = GetSdkConfiguration();
  ASSERT_NE(config, nullptr);
  ASSERT_TRUE(config->HasConfig(Microsoft::Applications::Events::CFG_BOOL_ENABLE_NET_DETECT));
  EXPECT_FALSE(static_cast<bool>((*config)[Microsoft::Applications::Events::CFG_BOOL_ENABLE_NET_DETECT]));
}
#endif

TEST_F(OneDsTelemetryTest, DisabledProcessInfoIsNotEmittedOrConsumed) {
  ASSERT_TRUE(telemetry_.IsEnabled());
  telemetry_.DisableTelemetryEvents();
  telemetry_.LogProcessInfo();
  telemetry_.LogProcessInfo();

  EXPECT_EQ(logger_.event_count, size_t{0});
  EXPECT_FALSE(ProcessInfoLogged());

  telemetry_.EnableTelemetryEvents();
  EXPECT_TRUE(telemetry_.IsEnabled());
  EXPECT_FALSE(ProcessInfoLogged());
}

TEST_F(OneDsTelemetryTest, FullSuppressionCannotBeReenabledForProcessInfo) {
  SuppressProcess();
  telemetry_.EnableTelemetryEvents();
  telemetry_.LogProcessInfo();

  EXPECT_FALSE(telemetry_.IsEnabled());
  EXPECT_EQ(logger_.event_count, size_t{0});
  EXPECT_FALSE(ProcessInfoLogged());
}

TEST_F(OneDsTelemetryTest, UnavailableLoggerDoesNotConsumeProcessInfo) {
  RemoveLogger();
  telemetry_.LogProcessInfo();

  EXPECT_FALSE(telemetry_.IsEnabled());
  EXPECT_EQ(logger_.event_count, size_t{0});
  EXPECT_FALSE(ProcessInfoLogged());
}

TEST_F(OneDsTelemetryTest, BoundsRuntimeErrorPropertiesAtEmission) {
  const std::string large(1000000, 'a');
  const common::Status status(common::ONNXRUNTIME, common::FAIL, large);
  telemetry_.LogRuntimeError(0, status, large.c_str(), large.c_str(), 7);
  ASSERT_EQ(logger_.event_count, size_t{1});
  EXPECT_EQ(logger_.strings.at("errorMessage").size(), kMaxTelemetryStringLength);
  EXPECT_EQ(logger_.strings.at("file").size(), kMaxTelemetryStringLength);
  EXPECT_EQ(logger_.strings.at("function").size(), kMaxTelemetryStringLength);
  for (const auto& [name, value] : logger_.strings) {
    EXPECT_LE(value.size(), kMaxTelemetryStringLength) << name;
  }
}

TEST_F(OneDsTelemetryTest, BoundsSampledSessionPropertiesWithoutChangingModelMetadata) {
  uint32_t session_id = 0;
  while (session_id < 100000 &&
         !telemetry_internal::ShouldSampleSession(telemetry_internal::GetAppSessionGuid(), session_id)) {
    ++session_id;
  }
  ASSERT_LT(session_id, 100000u);
  const std::string large(100000, 'a');
  const std::unordered_map<std::string, std::string> metadata{{"key", large}};
  const std::unordered_map<std::string, int> domains{{large, 1}};
  const std::vector<std::string> providers{large, large};
  telemetry_.LogSessionCreation(
      session_id, 1, large, large, large, domains, large, large, large, large, large,
      metadata, large, providers, large, large, large, false, false);
  ASSERT_EQ(logger_.event_count, size_t{1});
  EXPECT_EQ(logger_.strings.at("modelMetaData").size(), kMaxTelemetryStringLength);
  EXPECT_EQ(logger_.strings.at("domainToVersionMap").size(), kMaxTelemetryStringLength);
  EXPECT_EQ(logger_.strings.at("executionProviderIds").size(), kMaxTelemetryStringLength);
  EXPECT_EQ(metadata.at("key"), large);
  for (const auto& [name, value] : logger_.strings) {
    EXPECT_LE(value.size(), kMaxTelemetryStringLength) << name;
  }
}

}  // namespace onnxruntime::test

#endif  // USE_1DS_TELEMETRY

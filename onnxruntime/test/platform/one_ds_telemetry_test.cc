// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_1DS_TELEMETRY

#include <ILogConfiguration.hpp>
#include <NullObjects.hpp>

#include "core/platform/telemetry_1ds.h"
#include "gtest/gtest.h"

namespace onnxruntime::test {

class OneDsTelemetryTest : public testing::Test {
 protected:
  class RecordingLogger : public Microsoft::Applications::Events::NullLogger {
   public:
    void LogEvent(const Microsoft::Applications::Events::EventProperties&) override {
      ++event_count;
    }

    size_t event_count = 0;
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

}  // namespace onnxruntime::test

#endif  // USE_1DS_TELEMETRY

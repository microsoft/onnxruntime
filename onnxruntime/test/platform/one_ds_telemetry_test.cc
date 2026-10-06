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

  class RecordingLocalTelemetry : public Telemetry {
   public:
    void SetLanguageProjection(uint32_t projection) const override {
      last_projection = projection;
    }

    void LogSessionCreation(
        uint32_t session_id, int64_t, const std::string&, const std::string&, const std::string&,
        const std::unordered_map<std::string, int>&, const std::string&, const std::string&,
        const std::string&, const std::string&, const std::string&,
        const std::unordered_map<std::string, std::string>&, const std::string&,
        const std::vector<std::string>&, const std::string&, const std::string&,
        const std::string&, bool, bool capture_state) const override {
      ++session_count;
      last_session_id = session_id;
      last_capture_state = capture_state;
    }

    void LogProviderOptions(const std::string&, const std::string& options, bool capture_state) const override {
      ++options_count;
      last_options = options;
      last_capture_state = capture_state;
    }

    mutable size_t session_count = 0;
    mutable size_t options_count = 0;
    mutable uint32_t last_session_id = 0;
    mutable uint32_t last_projection = 0;
    mutable bool last_capture_state = false;
    mutable std::string last_options;
  };

  void SetUp() override {
    previous_logger_ = OneDsTelemetry::logger_.exchange(&logger_);
    previous_enabled_ = OneDsTelemetry::enabled_.exchange(true);
    previous_disabled_ = OneDsTelemetry::telemetry_disabled_.exchange(false);
    previous_process_info_logged_ = OneDsTelemetry::process_info_logged_.exchange(false);
    previous_projection_ = OneDsTelemetry::projection_;
  }

  void TearDown() override {
    OneDsTelemetry::logger_.store(previous_logger_);
    OneDsTelemetry::enabled_.store(previous_enabled_);
    OneDsTelemetry::telemetry_disabled_.store(previous_disabled_);
    OneDsTelemetry::process_info_logged_.store(previous_process_info_logged_);
    OneDsTelemetry::projection_ = previous_projection_;
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

  static void ConfigureSdk(Microsoft::Applications::Events::ILogConfiguration& config) {
    OneDsTelemetry::ConfigureSdk(config);
  }

  static void LogCaptureState(const OneDsTelemetry& telemetry, uint32_t session_id) {
    telemetry.LogSessionCreation(
        session_id, 1, "", "", "", {}, "", "", "", "", "", {}, "", {}, "", "", "", false, true);
  }

  static uint32_t FindSessionId(bool sampled) {
    for (uint32_t session_id = 0; session_id < 100000; ++session_id) {
      if (telemetry_internal::ShouldSampleSession(telemetry_internal::GetAppSessionGuid(), session_id) == sampled) {
        return session_id;
      }
    }
    ORT_THROW("No telemetry session ID matched sampling decision ", sampled);
  }

  OneDsTelemetry telemetry_;
  RecordingLogger logger_;

 private:
  Microsoft::Applications::Events::ILogger* previous_logger_ = nullptr;
  bool previous_enabled_ = false;
  bool previous_disabled_ = false;
  bool previous_process_info_logged_ = false;
  uint32_t previous_projection_ = 0;
};

#if defined(_WIN32)
TEST_F(OneDsTelemetryTest, WindowsNetworkDetectorIsDisabled) {
  Microsoft::Applications::Events::ILogConfiguration config;
  ConfigureSdk(config);
  ASSERT_TRUE(config.HasConfig(Microsoft::Applications::Events::CFG_BOOL_ENABLE_NET_DETECT));
  EXPECT_FALSE(static_cast<bool>(config[Microsoft::Applications::Events::CFG_BOOL_ENABLE_NET_DETECT]));
}
#endif

TEST_F(OneDsTelemetryTest, CaptureStateUsesLocalProviderWithoutUploaderOrSampling) {
  RecordingLocalTelemetry local;
  OneDsTelemetry telemetry(local);
  RemoveLogger();
  const uint32_t session_id = FindSessionId(false);
  ASSERT_FALSE(telemetry.IsEnabled());
  LogCaptureState(telemetry, session_id);
  EXPECT_EQ(local.session_count, size_t{1});
  EXPECT_EQ(local.last_session_id, session_id);
  EXPECT_TRUE(local.last_capture_state);
  EXPECT_EQ(logger_.event_count, size_t{0});
}

TEST_F(OneDsTelemetryTest, CaptureStateHonorsRuntimeOptOutAndFullSuppression) {
  RecordingLocalTelemetry local;
  OneDsTelemetry telemetry(local);
  telemetry.DisableTelemetryEvents();
  LogCaptureState(telemetry, 0);
  EXPECT_EQ(local.session_count, size_t{0});
  telemetry.EnableTelemetryEvents();
  LogCaptureState(telemetry, 0);
  EXPECT_EQ(local.session_count, size_t{1});
  SuppressProcess();
  telemetry.EnableTelemetryEvents();
  LogCaptureState(telemetry, 0);
  EXPECT_EQ(local.session_count, size_t{1});
  EXPECT_EQ(logger_.event_count, size_t{0});
}

TEST_F(OneDsTelemetryTest, LocalProviderReceivesLanguageProjectionWithoutUploader) {
  RecordingLocalTelemetry local;
  OneDsTelemetry telemetry(local);
  RemoveLogger();
  telemetry.SetLanguageProjection(3);
  EXPECT_EQ(local.last_projection, 3u);
  EXPECT_EQ(logger_.event_count, size_t{0});
}

TEST_F(OneDsTelemetryTest, ProviderOptionsStayLocalAndHonorOptOut) {
  RecordingLocalTelemetry local;
  OneDsTelemetry telemetry(local);
  const std::string options = "custom_credential:private-value";
  for (bool capture_state : {false, true}) {
    telemetry.LogProviderOptions("CustomEP", options, capture_state);
    EXPECT_EQ(local.last_options, options);
    EXPECT_EQ(local.last_capture_state, capture_state);
  }
  EXPECT_EQ(local.options_count, size_t{2});
  telemetry.DisableTelemetryEvents();
  telemetry.LogProviderOptions("CustomEP", options, false);
  EXPECT_EQ(local.options_count, size_t{2});
  telemetry.EnableTelemetryEvents();
  SuppressProcess();
  telemetry.LogProviderOptions("CustomEP", options, true);
  EXPECT_EQ(local.options_count, size_t{2});
  EXPECT_EQ(logger_.event_count, size_t{0});
}

TEST_F(OneDsTelemetryTest, ProviderOptionsWithoutLocalProviderAreNotUploaded) {
  telemetry_.LogProviderOptions("CustomEP", "custom_credential:private-value", false);
  telemetry_.LogProviderOptions("CustomEP", "custom_credential:private-value", true);
  EXPECT_EQ(logger_.event_count, size_t{0});
}

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

TEST_F(OneDsTelemetryTest, BoundsAndSanitizesRuntimeErrorPropertiesAtEmission) {
  const struct {
    std::string input;
    std::string expected_message;
    std::string expected_function;
  } cases[] = {
      {std::string(telemetry_detail::kMaxTelemetryProbeBytes + 1, 'a'), "[path]",
       std::string(kMaxTelemetryStringLength, 'a')},
      {std::string(2000, '\x80'), std::string(kMaxTelemetryStringLength, '?'),
       std::string(kMaxTelemetryStringLength, '?')},
  };
  for (const auto& [input, expected_message, expected_function] : cases) {
    SCOPED_TRACE(input.size());
    logger_.event_count = 0;
    logger_.strings.clear();
    const common::Status status(common::ONNXRUNTIME, common::FAIL, input);
    telemetry_.LogRuntimeError(0, status, input.c_str(), input.c_str(), 7);
    ASSERT_EQ(logger_.event_count, size_t{1});
    EXPECT_EQ(logger_.strings.at("errorMessage"), expected_message);
    EXPECT_EQ(logger_.strings.at("function"), expected_function);
    EXPECT_EQ(logger_.strings.at("file"), expected_function);
    for (const auto& [name, value] : logger_.strings) {
      EXPECT_LE(value.size(), kMaxTelemetryStringLength) << name;
    }
  }
}

TEST_F(OneDsTelemetryTest, BoundsSampledSessionPropertiesWithoutChangingModelMetadata) {
  const uint32_t session_id = FindSessionId(true);
  const std::string large(kMaxTelemetryStringLength + 1, 'a');
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

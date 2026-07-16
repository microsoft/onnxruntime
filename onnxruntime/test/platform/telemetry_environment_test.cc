// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_environment.h"

#include <cstdlib>
#include <string>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

void UnsetEnv(const char* name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

// Saves an environment variable on construction and restores it on destruction so a test can mutate
// it without leaking state to sibling tests.
class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_(name) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // std::getenv is fine here; only reading a value to save/restore
#endif
    const char* value = std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    had_value_ = value != nullptr;
    if (had_value_) {
      saved_ = value;
    }
  }
  ~ScopedEnvVar() {
    if (had_value_) {
      SetEnv(name_, saved_.c_str());
    } else {
      UnsetEnv(name_);
    }
  }

 private:
  const char* name_;
  bool had_value_ = false;
  std::string saved_;
};

}  // namespace

TEST(TelemetryEnvironmentTest, IsTruthyCiValue) {
  using telemetry_detail::IsTruthyCiValue;
  // Any non-empty, non-falsey value counts as present.
  EXPECT_TRUE(IsTruthyCiValue("1"));
  EXPECT_TRUE(IsTruthyCiValue("true"));
  EXPECT_TRUE(IsTruthyCiValue("TRUE"));
  EXPECT_TRUE(IsTruthyCiValue("yes"));
  EXPECT_TRUE(IsTruthyCiValue(" 1 "));
  EXPECT_TRUE(IsTruthyCiValue("anything"));

  EXPECT_FALSE(IsTruthyCiValue(""));
  EXPECT_FALSE(IsTruthyCiValue("   "));
  EXPECT_FALSE(IsTruthyCiValue("0"));
  EXPECT_FALSE(IsTruthyCiValue("false"));
  EXPECT_FALSE(IsTruthyCiValue("FALSE"));
  EXPECT_FALSE(IsTruthyCiValue("no"));
  EXPECT_FALSE(IsTruthyCiValue("off"));
}

TEST(TelemetryEnvironmentTest, EnvVarOptOut) {
  ScopedEnvVar guard("ORT_TELEMETRY_DISABLED");

  SetEnv("ORT_TELEMETRY_DISABLED", "1");
  EXPECT_TRUE(IsTelemetryDisabledByEnvVar());
  EXPECT_TRUE(ShouldSuppressTelemetry());

  SetEnv("ORT_TELEMETRY_DISABLED", "TRUE");
  EXPECT_TRUE(IsTelemetryDisabledByEnvVar());

  SetEnv("ORT_TELEMETRY_DISABLED", "0");
  EXPECT_FALSE(IsTelemetryDisabledByEnvVar());

  SetEnv("ORT_TELEMETRY_DISABLED", "random");
  EXPECT_FALSE(IsTelemetryDisabledByEnvVar());

  UnsetEnv("ORT_TELEMETRY_DISABLED");
  EXPECT_FALSE(IsTelemetryDisabledByEnvVar());
}

TEST(TelemetryEnvironmentTest, CiDetectionSuppresses) {
  // Only the positive direction is asserted so the test is deterministic whether or not it itself
  // runs in a CI environment. APPVEYOR is not part of ORT's own CI, so save/restore stays clean.
  ScopedEnvVar guard("APPVEYOR");
  SetEnv("APPVEYOR", "true");
  EXPECT_TRUE(IsRunningInCI());
  EXPECT_TRUE(ShouldSuppressTelemetry());
}

TEST(TelemetryEnvironmentTest, RunningUnitTestsSuppresses) {
  // The unit-test entry point sets ORT_RUNNING_UNIT_TESTS process-wide; save/restore so this test can
  // exercise both directions without leaking state to siblings.
  ScopedEnvVar guard("ORT_RUNNING_UNIT_TESTS");

  SetEnv("ORT_RUNNING_UNIT_TESTS", "1");
  EXPECT_TRUE(IsRunningUnitTests());
  EXPECT_TRUE(ShouldSuppressTelemetry());

  // Only IsRunningUnitTests() is asserted in the negative direction; ShouldSuppressTelemetry() may
  // still hold from a CI variable when this test itself runs in CI.
  SetEnv("ORT_RUNNING_UNIT_TESTS", "0");
  EXPECT_FALSE(IsRunningUnitTests());

  UnsetEnv("ORT_RUNNING_UNIT_TESTS");
  EXPECT_FALSE(IsRunningUnitTests());
}

}  // namespace test
}  // namespace onnxruntime

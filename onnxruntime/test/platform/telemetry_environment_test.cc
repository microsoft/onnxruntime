// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_environment.h"
#include "test/util/include/scoped_env_vars.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

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
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TELEMETRY_DISABLED", "1"}}};
    EXPECT_TRUE(IsTelemetryDisabledByEnvVar());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TELEMETRY_DISABLED", "TRUE"}}};
    EXPECT_TRUE(IsTelemetryDisabledByEnvVar());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TELEMETRY_DISABLED", "0"}}};
    EXPECT_FALSE(IsTelemetryDisabledByEnvVar());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TELEMETRY_DISABLED", "random"}}};
    EXPECT_FALSE(IsTelemetryDisabledByEnvVar());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TELEMETRY_DISABLED", nullopt}}};
    EXPECT_FALSE(IsTelemetryDisabledByEnvVar());
  }
}

TEST(TelemetryEnvironmentTest, EnvironmentOptOutCannotBeReenabled) {
  EXPECT_TRUE(CanEnableTelemetryEvents(false));
  EXPECT_FALSE(CanEnableTelemetryEvents(true));
}

TEST(TelemetryEnvironmentTest, CiDetectionSuppresses) {
  // Only the positive direction is asserted so the test is deterministic whether or not it itself
  // runs in a CI environment. APPVEYOR is not part of ORT's own CI, so save/restore stays clean.
  ScopedEnvironmentVariables env_vars{EnvVarMap{{"APPVEYOR", "true"}}};
  EXPECT_TRUE(IsRunningInCI());
  EXPECT_TRUE(ShouldSuppressTelemetry());
}

TEST(TelemetryEnvironmentTest, RunningUnitTestsSuppresses) {
  // The unit-test entry point sets ORT_RUNNING_UNIT_TESTS process-wide; save/restore so this test can
  // exercise both directions without leaking state to siblings.
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_RUNNING_UNIT_TESTS", "1"}}};
    EXPECT_TRUE(IsRunningUnitTests());
    EXPECT_TRUE(ShouldSuppressTelemetry());
  }
  // Only IsRunningUnitTests() is asserted in the negative direction; ShouldSuppressTelemetry() may
  // still hold from a CI variable when this test itself runs in CI.
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_RUNNING_UNIT_TESTS", "0"}}};
    EXPECT_FALSE(IsRunningUnitTests());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_RUNNING_UNIT_TESTS", nullopt}}};
    EXPECT_FALSE(IsRunningUnitTests());
  }
}

}  // namespace test
}  // namespace onnxruntime

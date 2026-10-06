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
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_DISABLE_TELEMETRY", "1"}}};
    EXPECT_TRUE(IsTelemetryDisabledByEnvironment());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_DISABLE_TELEMETRY", "TRUE"}}};
    EXPECT_TRUE(IsTelemetryDisabledByEnvironment());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_DISABLE_TELEMETRY", "0"}}};
    EXPECT_FALSE(IsTelemetryDisabledByEnvironment());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_DISABLE_TELEMETRY", "random"}}};
    EXPECT_FALSE(IsTelemetryDisabledByEnvironment());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_DISABLE_TELEMETRY", nullopt}}};
    EXPECT_FALSE(IsTelemetryDisabledByEnvironment());
  }
}

TEST(TelemetryEnvironmentTest, CiDetectionSuppresses) {
  // Only the positive direction is asserted so the test is deterministic whether or not it itself
  // runs in a CI environment. APPVEYOR is not part of ORT's own CI, so save/restore stays clean.
  ScopedEnvironmentVariables env_vars{EnvVarMap{{"APPVEYOR", "true"}}};
  EXPECT_TRUE(IsRunningInCI());
}

TEST(TelemetryEnvironmentTest, RunningUnitTestsSuppresses) {
  // The unit-test entry point sets ORT_RUNNING_UNIT_TESTS process-wide; save/restore so this test can
  // exercise both directions without leaking state to siblings.
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_RUNNING_UNIT_TESTS", "1"}}};
    EXPECT_TRUE(IsRunningUnitTests());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_RUNNING_UNIT_TESTS", "0"}}};
    EXPECT_FALSE(IsRunningUnitTests());
  }
  {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_RUNNING_UNIT_TESTS", nullopt}}};
    EXPECT_FALSE(IsRunningUnitTests());
  }
}

TEST(TelemetryEnvironmentTest, ClassifiesContainersAndVirtualMachines) {
  using telemetry_detail::ClassifyHostEnvironment;
  using telemetry_detail::HostEnvironmentEvidence;

  {
    HostEnvironmentEvidence evidence;
    evidence.kubernetes = true;
    const auto info = ClassifyHostEnvironment(evidence);
    EXPECT_TRUE(info.is_container);
    EXPECT_STREQ(info.container_type, "kubernetes");
    EXPECT_STREQ(info.environment_class, "container");
    EXPECT_STREQ(info.detection_confidence, "high");
    EXPECT_STREQ(info.device_id_scope, "container");
  }
  {
    HostEnvironmentEvidence evidence;
    evidence.podman_marker = true;
    evidence.dmi = "Amazon EC2";
    const auto info = ClassifyHostEnvironment(evidence);
    EXPECT_TRUE(info.is_container);
    EXPECT_TRUE(info.is_virtual_machine);
    EXPECT_STREQ(info.container_type, "podman");
    EXPECT_STREQ(info.virtualization_type, "amazonEC2");
    EXPECT_STREQ(info.environment_class, "containerOnVirtualMachine");
  }
  {
    HostEnvironmentEvidence evidence;
    evidence.cgroup = "0::/system.slice/docker-012345.scope";
    const auto info = ClassifyHostEnvironment(evidence);
    EXPECT_TRUE(info.is_container);
    EXPECT_STREQ(info.container_type, "docker");
  }
  {
    HostEnvironmentEvidence evidence;
    evidence.aws_ecs = true;
    const auto info = ClassifyHostEnvironment(evidence);
    EXPECT_TRUE(info.is_container);
    EXPECT_STREQ(info.container_type, "amazonECS");
  }
  {
    HostEnvironmentEvidence evidence;
    evidence.dmi = "Microsoft Corporation Virtual Machine";
    const auto info = ClassifyHostEnvironment(evidence);
    EXPECT_TRUE(info.is_virtual_machine);
    EXPECT_STREQ(info.virtualization_type, "hyperV");
    EXPECT_STREQ(info.environment_class, "virtualMachine");
    EXPECT_STREQ(info.device_id_scope, "virtualMachine");
  }
  {
    HostEnvironmentEvidence evidence;
    evidence.kernel_release = "6.6.87.2-microsoft-standard-WSL2";
    const auto info = ClassifyHostEnvironment(evidence);
    EXPECT_TRUE(info.is_virtual_machine);
    EXPECT_STREQ(info.virtualization_type, "wsl");
  }
}

TEST(TelemetryEnvironmentTest, ClassifiesEmulatorAndUndetectedHostWithoutClaimingPhysicalDevice) {
  using telemetry_detail::ClassifyHostEnvironment;
  using telemetry_detail::HostEnvironmentEvidence;

  {
    HostEnvironmentEvidence evidence;
    evidence.android_emulator = true;
    const auto info = ClassifyHostEnvironment(evidence);
    EXPECT_TRUE(info.is_virtual_machine);
    EXPECT_TRUE(info.is_emulator);
    EXPECT_STREQ(info.virtualization_type, "androidEmulator");
    EXPECT_STREQ(info.environment_class, "emulator");
  }
  {
    const auto info = ClassifyHostEnvironment(HostEnvironmentEvidence{});
    EXPECT_FALSE(info.is_container);
    EXPECT_FALSE(info.is_virtual_machine);
    EXPECT_FALSE(info.is_emulator);
    EXPECT_STREQ(info.environment_class, "undetected");
    EXPECT_STREQ(info.detection_confidence, "none");
    EXPECT_STREQ(info.device_id_scope, "installation");
  }
}

}  // namespace test
}  // namespace onnxruntime

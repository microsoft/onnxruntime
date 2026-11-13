// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_environment.h"
#include "test/util/include/scoped_env_vars.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(TelemetryEnvironmentTest, BoundsValuesAndRejectsOversizedPaths) {
  const std::string large(20000, 'a');
  ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TEST_TELEMETRY_VALUE", large}}};
  EXPECT_FALSE(telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_VALUE"));
  EXPECT_FALSE(telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_VALUE", 4096));
  EXPECT_EQ(telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_VALUE", large.size()), large);
  EXPECT_FALSE(telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_VALUE", large.size() - 1));
}

TEST(TelemetryEnvironmentTest, RejectsEnvironmentValuesOverUtf8ByteBudget) {
  const std::string prefix = std::string(1021, 'a') + "\xe2\x82\xac";
  ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TEST_TELEMETRY_VALUE", prefix + "tail"}}};
#ifdef _WIN32
  const std::wstring wide = std::wstring(1021, L'a') + L"\u20actail";
  ASSERT_NE(::SetEnvironmentVariableW(L"ORT_TEST_TELEMETRY_VALUE", wide.c_str()), 0);
#endif
  EXPECT_FALSE(telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_VALUE", prefix.size()));
  EXPECT_EQ(telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_VALUE", prefix.size() + 4),
            prefix + "tail");
}

TEST(TelemetryEnvironmentTest, DistinguishesAbsentEmptyAndRejectedValues) {
  for (const auto& value : {std::optional<std::string>{}, std::optional<std::string>{""}}) {
    ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TEST_TELEMETRY_VALUE", value}}};
    EXPECT_EQ(telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_VALUE", 0), "");
  }
  ScopedEnvironmentVariables env_vars{EnvVarMap{{"ORT_TEST_TELEMETRY_VALUE", "x"}}};
  EXPECT_FALSE(telemetry_detail::ReadTelemetryEnvironment("ORT_TEST_TELEMETRY_VALUE", 0));
}

TEST(TelemetryEnvironmentTest, SuppressionFlagsAreNotTruncatedAndOversizedValuesFailClosed) {
  const std::string padded = std::string(kMaxTelemetryStringLength, ' ') + "true";
  const std::string oversized(telemetry_detail::kMaxTelemetryProbeBytes + 1, ' ');
  for (const auto& value : {padded, oversized}) {
    ScopedEnvironmentVariables env_vars{
        EnvVarMap{{"APPVEYOR", value}, {"ORT_RUNNING_UNIT_TESTS", value}, {"ORT_DISABLE_TELEMETRY", value}}};
    EXPECT_TRUE(IsRunningInCI());
    EXPECT_TRUE(IsRunningUnitTests());
    EXPECT_TRUE(IsTelemetryDisabledByEnvironment());
  }
  EXPECT_TRUE(telemetry_detail::IsTruthyCiValue(oversized));
}

TEST(TelemetryEnvironmentTest, SuppressionFlagsHonorValuesIndependentlyOfRunnerEnvironment) {
  EnvVarMap cleared_ci;
  for (const char* name : telemetry_detail::kCiEnvironmentVariableNames) {
    cleared_ci.emplace(name, nullopt);
  }
  ScopedEnvironmentVariables runner_environment{cleared_ci};
  const struct {
    optional<std::string> value;
    bool ci_or_unit_test;
    bool opt_out;
  } cases[] = {
      {nullopt, false, false},
      {"", false, false},
      {"   ", false, false},
      {"0", false, false},
      {"false", false, false},
      {"FALSE", false, false},
      {"no", false, false},
      {"off", false, false},
      {"1", true, true},
      {"true", true, true},
      {"TRUE", true, true},
      {"yes", true, true},
      {"on", true, true},
      {"y", true, true},
      {" 1 ", true, true},
      {"anything", true, false},
  };
  for (const auto& [value, ci_or_unit_test, opt_out] : cases) {
    SCOPED_TRACE(value.value_or("<unset>"));
    ScopedEnvironmentVariables env_vars{
        EnvVarMap{{"APPVEYOR", value}, {"ORT_RUNNING_UNIT_TESTS", value}, {"ORT_DISABLE_TELEMETRY", value}}};
    EXPECT_EQ(IsRunningInCI(), ci_or_unit_test);
    EXPECT_EQ(IsRunningUnitTests(), ci_or_unit_test);
    EXPECT_EQ(IsTelemetryDisabledByEnvironment(), opt_out);
    if (value) EXPECT_EQ(telemetry_detail::IsTruthyCiValue(*value), ci_or_unit_test);
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace onnxruntime {
namespace telemetry_detail {

// Well-known CI / build-pipeline environment variables. Mirrors the list used by Foundry Local /
// neutron-server so that ORT's CI telemetry suppression behaves consistently across stacks; keep the
// two lists in sync if either changes.
inline constexpr std::array<const char*, 13> kCiEnvironmentVariableNames = {
    "CI",                                  // Generic CI flag used by many providers
    "TF_BUILD",                            // Azure Pipelines
    "GITHUB_ACTIONS",                      // GitHub Actions
    "GITLAB_CI",                           // GitLab CI
    "CIRCLECI",                            // CircleCI
    "TRAVIS",                              // Travis CI
    "JENKINS_URL",                         // Jenkins
    "CODEBUILD_BUILD_ID",                  // AWS CodeBuild
    "BUILDKITE",                           // Buildkite
    "TEAMCITY_VERSION",                    // TeamCity
    "APPVEYOR",                            // AppVeyor
    "BITBUCKET_BUILD_NUMBER",              // Bitbucket Pipelines
    "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",  // Azure DevOps
};

// Read an environment variable, returning an empty string when unset.
inline std::string GetTelemetryEnv(const char* name) {
#ifdef _WIN32
  DWORD required_size = ::GetEnvironmentVariableA(name, nullptr, 0);
  while (required_size != 0) {
    std::string value(required_size, '\0');
    const DWORD written = ::GetEnvironmentVariableA(name, value.data(), required_size);
    if (written == 0) {
      return {};
    }
    if (written < required_size) {
      value.resize(written);
      return value;
    }

    // The value grew between calls. Windows returns its new required size, including the null.
    required_size = written;
  }
  return {};
#else
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string();
#endif
}

inline std::string_view TrimAscii(std::string_view s) {
  size_t begin = 0;
  size_t end = s.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  return s.substr(begin, end - begin);
}

inline std::string ToLowerAscii(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

// A CI variable counts as present unless its (trimmed) value is empty or an explicit falsey token, so
// that a runner exporting e.g. CI=false does not trip detection.
inline bool IsTruthyCiValue(std::string_view value) {
  const std::string v = ToLowerAscii(TrimAscii(value));
  return !v.empty() && v != "0" && v != "false" && v != "no" && v != "off";
}

}  // namespace telemetry_detail

// True if a well-known CI / build-pipeline environment variable is set to a truthy value. ORT's
// telemetry providers suppress all telemetry when this holds, matching Olive and Foundry Local.
inline bool IsRunningInCI() {
  for (const char* name : telemetry_detail::kCiEnvironmentVariableNames) {
    if (telemetry_detail::IsTruthyCiValue(telemetry_detail::GetTelemetryEnv(name))) {
      return true;
    }
  }
  return false;
}

// True if ORT_RUNNING_UNIT_TESTS is set to a truthy value. ORT's own unit-test entry points set this
// before creating any environment, so local (non-CI) test runs never initialize the telemetry uploader
// or emit events. This is an internal harness signal, not a user-facing opt-out.
inline bool IsRunningUnitTests() {
  return telemetry_detail::IsTruthyCiValue(telemetry_detail::GetTelemetryEnv("ORT_RUNNING_UNIT_TESTS"));
}

// True if ORT_TELEMETRY_DISABLED is set to a truthy value (1/true/yes/on/y, case-insensitive).
// The POSIX 1DS provider latches this opt-out during initialization. Windows ETW intentionally
// retains its separate API/trace-session control model and does not consult this environment variable.
inline bool IsTelemetryDisabledByEnvVar() {
  const std::string value = telemetry_detail::ToLowerAscii(
      telemetry_detail::TrimAscii(telemetry_detail::GetTelemetryEnv("ORT_TELEMETRY_DISABLED")));
  return value == "1" || value == "true" || value == "yes" || value == "on" || value == "y";
}

// Environment opt-out has higher priority than the runtime enable API for the lifetime of the process.
inline constexpr bool CanEnableTelemetryEvents(bool disabled_by_environment) noexcept {
  return !disabled_by_environment;
}

// True if telemetry should be fully suppressed for this process, including the initialization event.
// ORT_TELEMETRY_DISABLED is intentionally excluded because it disables only non-essential events.
inline bool ShouldSuppressTelemetry() {
  return IsRunningInCI() || IsRunningUnitTests();
}

}  // namespace onnxruntime

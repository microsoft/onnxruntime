// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/config_options.h"
#include "core/framework/moe_offload_plan.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

using moe_offload::MoeOffloadSessionPolicy;

Status ParsePolicy(const std::vector<std::pair<std::string, std::string>>& entries,
                   MoeOffloadSessionPolicy& policy) {
  ConfigOptions config_options;
  for (const auto& [key, value] : entries) {
    ORT_RETURN_IF_ERROR(config_options.AddConfigEntry(key.c_str(), value.c_str()));
  }
  return MoeOffloadSessionPolicy::ParseFromConfigOptions(config_options, std::filesystem::path{}, policy);
}

Status ParseDeviceMemLimit(const std::string& value, MoeOffloadSessionPolicy& policy) {
  return ParsePolicy({{kOrtSessionOptionsMoeOffloadGpuDeviceMemLimit, value}}, policy);
}

}  // namespace

// The key is part of the public session-option surface, so its spelling must not drift.
TEST(MoeOffloadSessionPolicyTest, GpuDeviceMemLimitKeyName) {
  EXPECT_STREQ(kOrtSessionOptionsMoeOffloadGpuDeviceMemLimit, "session.moe.offload.gpu_device_mem_limit");
}

TEST(MoeOffloadSessionPolicyTest, GpuDeviceMemLimitDefaultsToUnlimited) {
  MoeOffloadSessionPolicy policy;
  policy.gpu_device_mem_limit = 123;
  ASSERT_STATUS_OK(ParsePolicy({}, policy));
  EXPECT_EQ(policy.gpu_device_mem_limit, 0u);
}

TEST(MoeOffloadSessionPolicyTest, GpuDeviceMemLimitParsesByteCounts) {
  MoeOffloadSessionPolicy policy;

  ASSERT_STATUS_OK(ParseDeviceMemLimit("1073741824", policy));
  EXPECT_EQ(policy.gpu_device_mem_limit, 1073741824u);

  ASSERT_STATUS_OK(ParseDeviceMemLimit("0", policy));
  EXPECT_EQ(policy.gpu_device_mem_limit, 0u);

  ASSERT_STATUS_OK(ParseDeviceMemLimit("18446744073709551615", policy));
  EXPECT_EQ(policy.gpu_device_mem_limit, std::numeric_limits<uint64_t>::max());
}

// The limit is a budget knob only; it must not switch MoE offload on or disturb the other offload settings.
TEST(MoeOffloadSessionPolicyTest, GpuDeviceMemLimitLeavesOtherOffloadSettingsUntouched) {
  MoeOffloadSessionPolicy defaults;
  MoeOffloadSessionPolicy policy;
  ASSERT_STATUS_OK(ParseDeviceMemLimit("4096", policy));

  EXPECT_EQ(policy.mode, defaults.mode);
  EXPECT_EQ(policy.device_working_bytes, defaults.device_working_bytes);
  EXPECT_EQ(policy.pinned_bytes, defaults.pinned_bytes);
  EXPECT_EQ(policy.reserve_device_bytes, defaults.reserve_device_bytes);
  EXPECT_EQ(policy.budget_group, defaults.budget_group);
}

TEST(MoeOffloadSessionPolicyTest, GpuDeviceMemLimitRejectsMalformedValues) {
  const std::vector<std::string> bad_values = {
      "",
      "-1",
      "+1",
      "abc",
      "1.5",
      "1 GiB",
      " 1",
      "1 ",
      "0x10",
      "18446744073709551616",  // uint64 max + 1
  };

  for (const auto& bad_value : bad_values) {
    SCOPED_TRACE("value='" + bad_value + "'");
    MoeOffloadSessionPolicy policy;
    const Status status = ParseDeviceMemLimit(bad_value, policy);
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
    EXPECT_NE(status.ErrorMessage().find(kOrtSessionOptionsMoeOffloadGpuDeviceMemLimit), std::string::npos)
        << status.ErrorMessage();
  }
}

// The key lives under the session.moe.offload.* prefix, whose parser rejects names it does not know. A misspelled
// sibling must still be rejected rather than silently ignored.
TEST(MoeOffloadSessionPolicyTest, MisspelledOffloadKeyIsStillRejected) {
  MoeOffloadSessionPolicy policy;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      ParsePolicy({{"session.moe.offload.gpu_device_mem_limt", "4096"}}, policy),
      "Unknown session.moe.offload.* configuration key");
}

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD)

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <array>
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
#include "test/util/include/scoped_env_vars.h"

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

TEST(MoeOffloadSessionPolicyTest, TailExpertPruningOptionsDefaultAndParse) {
  EXPECT_STREQ(kOrtSessionOptionsMoeOffloadMinExpertWeight, "session.moe.offload.min_expert_weight");
  EXPECT_STREQ(kOrtSessionOptionsMoeOffloadSkipTailColdOnly, "session.moe.offload.skip_tail_cold_only");

  MoeOffloadSessionPolicy defaults;
  ASSERT_STATUS_OK(ParsePolicy({}, defaults));
  EXPECT_FLOAT_EQ(defaults.min_expert_weight, 0.0f);
  EXPECT_TRUE(defaults.skip_tail_cold_only);

  MoeOffloadSessionPolicy configured;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadMinExpertWeight, "0.065"},
                                {kOrtSessionOptionsMoeOffloadSkipTailColdOnly, "0"}},
                               configured));
  EXPECT_FLOAT_EQ(configured.min_expert_weight, 0.065f);
  EXPECT_FALSE(configured.skip_tail_cold_only);

  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadSkipTailColdOnly, "1"}}, configured));
  EXPECT_TRUE(configured.skip_tail_cold_only);

  for (const char* bad_weight : {"-0.1", "1.1", "abc", ""}) {
    MoeOffloadSessionPolicy bad;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
        ParsePolicy({{kOrtSessionOptionsMoeOffloadMinExpertWeight, bad_weight}}, bad),
        kOrtSessionOptionsMoeOffloadMinExpertWeight);
  }
  for (const char* bad_flag : {"2", "-1", "true", ""}) {
    MoeOffloadSessionPolicy bad;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
        ParsePolicy({{kOrtSessionOptionsMoeOffloadSkipTailColdOnly, bad_flag}}, bad),
        kOrtSessionOptionsMoeOffloadSkipTailColdOnly);
  }
}

TEST(MoeOffloadSessionPolicyTest, TailSubstitutionOptionsDefaultAndParse) {
  EXPECT_STREQ(kOrtSessionOptionsMoeOffloadTailSubstitutionThreshold,
               "session.moe.offload.tail_substitution_threshold");
  EXPECT_STREQ(kOrtSessionOptionsMoeOffloadTailSubstitutionMinRank,
               "session.moe.offload.tail_substitution_min_rank");

  ScopedEnvironmentVariables clean_env{{{"ORT_QMOE_TAIL_SUBSTITUTION_THRESHOLD", nullopt},
                                        {"ORT_QMOE_TAIL_SUBSTITUTION_MIN_RANK", nullopt}}};

  MoeOffloadSessionPolicy defaults;
  ASSERT_STATUS_OK(ParsePolicy({}, defaults));
  EXPECT_FLOAT_EQ(defaults.tail_substitution_threshold, 0.0f);
  EXPECT_EQ(defaults.tail_substitution_min_rank, 6);

  MoeOffloadSessionPolicy configured;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadTailSubstitutionThreshold, "0.04"},
                                {kOrtSessionOptionsMoeOffloadTailSubstitutionMinRank, "2"}},
                               configured));
  EXPECT_FLOAT_EQ(configured.tail_substitution_threshold, 0.04f);
  EXPECT_EQ(configured.tail_substitution_min_rank, 2);

  MoeOffloadSessionPolicy zero;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadTailSubstitutionThreshold, "0"},
                                {kOrtSessionOptionsMoeOffloadTailSubstitutionMinRank, "0"}},
                               zero));
  EXPECT_FLOAT_EQ(zero.tail_substitution_threshold, 0.0f);
  EXPECT_EQ(zero.tail_substitution_min_rank, 0);

  for (const char* bad_threshold : {"-0.1", "nan", "inf", "0.01junk", " 0.01", ""}) {
    MoeOffloadSessionPolicy bad;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
        ParsePolicy({{kOrtSessionOptionsMoeOffloadTailSubstitutionThreshold, bad_threshold}}, bad),
        kOrtSessionOptionsMoeOffloadTailSubstitutionThreshold);
  }
  for (const char* bad_rank : {"-1", "abc", "1.5", " 6", ""}) {
    MoeOffloadSessionPolicy bad;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
        ParsePolicy({{kOrtSessionOptionsMoeOffloadTailSubstitutionMinRank, bad_rank}}, bad),
        kOrtSessionOptionsMoeOffloadTailSubstitutionMinRank);
  }
}

TEST(MoeOffloadSessionPolicyTest, TailSubstitutionEnvironmentFallbackAndConfigPrecedence) {
  ScopedEnvironmentVariables env{{{"ORT_QMOE_TAIL_SUBSTITUTION_THRESHOLD", "0.02"},
                                  {"ORT_QMOE_TAIL_SUBSTITUTION_MIN_RANK", "3"}}};

  MoeOffloadSessionPolicy from_env;
  ASSERT_STATUS_OK(ParsePolicy({}, from_env));
  EXPECT_FLOAT_EQ(from_env.tail_substitution_threshold, 0.02f);
  EXPECT_EQ(from_env.tail_substitution_min_rank, 3);

  MoeOffloadSessionPolicy from_config;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadTailSubstitutionThreshold, "0.04"},
                                {kOrtSessionOptionsMoeOffloadTailSubstitutionMinRank, "2"}},
                               from_config));
  EXPECT_FLOAT_EQ(from_config.tail_substitution_threshold, 0.04f);
  EXPECT_EQ(from_config.tail_substitution_min_rank, 2);
}

TEST(MoeOffloadSessionPolicyTest, TailSubstitutionRejectsInvalidEnvironmentUnlessConfigured) {
  ScopedEnvironmentVariables clean_env{{{"ORT_QMOE_TAIL_SUBSTITUTION_THRESHOLD", nullopt},
                                        {"ORT_QMOE_TAIL_SUBSTITUTION_MIN_RANK", nullopt}}};
  for (const auto& [name, value, key] :
       {std::array{"ORT_QMOE_TAIL_SUBSTITUTION_THRESHOLD", "nan", kOrtSessionOptionsMoeOffloadTailSubstitutionThreshold},
        std::array{"ORT_QMOE_TAIL_SUBSTITUTION_MIN_RANK", "-1", kOrtSessionOptionsMoeOffloadTailSubstitutionMinRank}}) {
    SCOPED_TRACE(name);
    ScopedEnvironmentVariables env{{{name, value}}};
    MoeOffloadSessionPolicy policy;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(ParsePolicy({}, policy), key);
    ASSERT_STATUS_OK(ParsePolicy({{key, "0"}}, policy));
  }
}

TEST(MoeOffloadSessionPolicyTest, TailPruningEnvironmentFallbackAndConfigPrecedence) {
  ScopedEnvironmentVariables env{{{"ORT_QMOE_SKIP_ZERO_COLD_DOORBELL", "1"},
                                  {"ORT_QMOE_MIN_EXPERT_WEIGHT", "0.05"},
                                  {"ORT_QMOE_SKIP_TAIL_COLD_ONLY", "0"}}};

  MoeOffloadSessionPolicy from_env;
  ASSERT_STATUS_OK(ParsePolicy({}, from_env));
  EXPECT_TRUE(from_env.skip_zero_cold_doorbell);
  EXPECT_FLOAT_EQ(from_env.min_expert_weight, 0.05f);
  EXPECT_FALSE(from_env.skip_tail_cold_only);

  MoeOffloadSessionPolicy from_config;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadSkipZeroColdDoorbell, "0"},
                                {kOrtSessionOptionsMoeOffloadMinExpertWeight, "0"},
                                {kOrtSessionOptionsMoeOffloadSkipTailColdOnly, "1"}},
                               from_config));
  EXPECT_FALSE(from_config.skip_zero_cold_doorbell);
  EXPECT_FLOAT_EQ(from_config.min_expert_weight, 0.0f);
  EXPECT_TRUE(from_config.skip_tail_cold_only);
}

TEST(MoeOffloadSessionPolicyTest, EmptyEnvironmentFallbacksUseDefaults) {
  ScopedEnvironmentVariables env{{{"ORT_QMOE_TAIL_SUBSTITUTION_THRESHOLD", ""},
                                  {"ORT_QMOE_TAIL_SUBSTITUTION_MIN_RANK", ""},
                                  {"ORT_QMOE_SKIP_ZERO_COLD_DOORBELL", ""},
                                  {"ORT_QMOE_MIN_EXPERT_WEIGHT", ""},
                                  {"ORT_QMOE_SKIP_TAIL_COLD_ONLY", ""}}};
  MoeOffloadSessionPolicy policy;
  ASSERT_STATUS_OK(ParsePolicy({}, policy));
  EXPECT_FLOAT_EQ(policy.tail_substitution_threshold, 0.0f);
  EXPECT_EQ(policy.tail_substitution_min_rank, 6);
  EXPECT_FALSE(policy.skip_zero_cold_doorbell);
  EXPECT_FLOAT_EQ(policy.min_expert_weight, 0.0f);
  EXPECT_TRUE(policy.skip_tail_cold_only);
}

TEST(MoeOffloadSessionPolicyTest, DeviceExpertBytesAndGpuBudgetAliasPrecedenceAndConflict) {
  MoeOffloadSessionPolicy policy;
  ASSERT_STATUS_OK(ParsePolicy({{"session.moe.offload.gpu_budget_bytes", "8192"}}, policy));
  EXPECT_FALSE(policy.device_expert_bytes.is_auto);
  EXPECT_EQ(policy.device_expert_bytes.bytes, 8192u);

  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadDeviceExpertBytes, "8192"},
                                {"session.moe.offload.gpu_budget_bytes", "8192"}},
                               policy));
  EXPECT_FALSE(policy.device_expert_bytes.is_auto);
  EXPECT_EQ(policy.device_expert_bytes.bytes, 8192u);

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      ParsePolicy({{kOrtSessionOptionsMoeOffloadDeviceExpertBytes, "4096"},
                   {"session.moe.offload.gpu_budget_bytes", "8192"}},
                  policy),
      "Conflicting values");
}

TEST(MoeOffloadSessionPolicyTest, MoePackedGpuExpertsRespectsExplicitDisabledModeAndOrdering) {
  ScopedEnvironmentVariables env{{{"ORT_MOE_PACKED_GPU_EXPERTS", "16"}}};

  MoeOffloadSessionPolicy implicit_mode;
  ASSERT_STATUS_OK(ParsePolicy({}, implicit_mode));
  EXPECT_EQ(implicit_mode.mode, moe_offload::OffloadMode::kAuto);
  EXPECT_TRUE(implicit_mode.mode_inferred_from_moe_packed_only);
  ASSERT_EQ(implicit_mode.node_overrides.size(), 1u);
  EXPECT_EQ(implicit_mode.node_overrides[0].node_name, "*");
  EXPECT_EQ(implicit_mode.node_overrides[0].op_type, "MoEPacked");
  EXPECT_EQ(implicit_mode.node_overrides[0].max_resident_slots, 16);

  MoeOffloadSessionPolicy explicit_auto;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadMode, "auto"}}, explicit_auto));
  EXPECT_EQ(explicit_auto.mode, moe_offload::OffloadMode::kAuto);
  EXPECT_FALSE(explicit_auto.mode_inferred_from_moe_packed_only);

  MoeOffloadSessionPolicy explicit_disabled;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadMode, "disabled"}}, explicit_disabled));
  EXPECT_EQ(explicit_disabled.mode, moe_offload::OffloadMode::kDisabled);
  EXPECT_FALSE(explicit_disabled.mode_inferred_from_moe_packed_only);

  MoeOffloadSessionPolicy with_node_override;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadNodeOverrides, "layer.0:max_resident_slots=0"}},
                               with_node_override));
  ASSERT_EQ(with_node_override.node_overrides.size(), 2u);
  EXPECT_EQ(with_node_override.node_overrides[0].node_name, "*");
  EXPECT_EQ(with_node_override.node_overrides[0].op_type, "MoEPacked");
  EXPECT_EQ(with_node_override.node_overrides[0].max_resident_slots, 16);
  EXPECT_EQ(with_node_override.node_overrides[1].node_name, "layer.0");
  EXPECT_EQ(with_node_override.node_overrides[1].max_resident_slots, 0);

  for (const char* bad_val : {"-1", "abc", "25%"}) {
    MoeOffloadSessionPolicy bad;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
        ParsePolicy({{"ep.cuda.moe_packed_gpu_experts", bad_val}}, bad),
        "moe_packed_gpu_experts");
  }
}

TEST(MoeOffloadSessionPolicyTest, ReserveDeviceBytesDistinguishesAutoFromExplicitZero) {
  MoeOffloadSessionPolicy defaults;
  ASSERT_STATUS_OK(ParsePolicy({}, defaults));
  EXPECT_EQ(defaults.reserve_device_bytes, 0u);
  EXPECT_FALSE(defaults.reserve_device_bytes_explicit);

  MoeOffloadSessionPolicy explicit_zero;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadReserveDeviceBytes, "0"}}, explicit_zero));
  EXPECT_EQ(explicit_zero.reserve_device_bytes, 0u);
  EXPECT_TRUE(explicit_zero.reserve_device_bytes_explicit);

  MoeOffloadSessionPolicy explicit_auto;
  ASSERT_STATUS_OK(ParsePolicy({{kOrtSessionOptionsMoeOffloadReserveDeviceBytes, "auto"}}, explicit_auto));
  EXPECT_EQ(explicit_auto.reserve_device_bytes, 0u);
  EXPECT_FALSE(explicit_auto.reserve_device_bytes_explicit);
}

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD)

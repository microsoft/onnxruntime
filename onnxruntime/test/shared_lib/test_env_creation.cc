// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/session/onnxruntime_cxx_api.h"

#include "test/util/include/api_asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" void ortenv_setup();
extern "C" void ortenv_teardown();

TEST(EnvCreation, CreateEnvWithOptions) {
  const OrtApi& ort_api = Ort::GetApi();

  // Basic error checking when user passes an invalid version for OrtEnvCreationOptions
  {
    OrtEnv* test_env = nullptr;
    OrtEnvCreationOptions options{};
    options.version = 0;  // Invalid!
    options.logging_severity_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
    options.log_id = "test logger";

    Ort::Status status{ort_api.CreateEnvWithOptions(&options, &test_env)};

    ASSERT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("version set equal to ORT_API_VERSION"));
  }

  // Basic error checking when user passes an invalid log identifier to the API function
  {
    OrtEnv* test_env = nullptr;
    OrtEnvCreationOptions options{};
    options.version = ORT_API_VERSION;
    options.logging_severity_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
    options.log_id = nullptr;  // Invalid!

    Ort::Status status{ort_api.CreateEnvWithOptions(&options, &test_env)};

    ASSERT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("valid (non-null) log identifier string"));
  }

  // Basic error checking when user passes an invalid logging severity level
  {
    OrtEnv* test_env = nullptr;
    OrtEnvCreationOptions options{};
    options.version = ORT_API_VERSION;
    options.logging_severity_level = 100;  // Invalid!
    options.log_id = "EnvCreation.CreateEnvWithOptions";

    Ort::Status status{ort_api.CreateEnvWithOptions(&options, &test_env)};

    ASSERT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("valid logging severity level value from "
                                                             "the OrtLoggingLevel enumeration"));
  }

  // Create an OrtEnv with configuration entries. Use the CXX API.

  ortenv_teardown();  // Release current OrtEnv as we need to recreate it.

  auto run_test = [&]() -> void {
    // Create OrtEnv with some dummy config entry.
    Ort::KeyValuePairs env_configs;
    env_configs.Add("some_key", "some_val");

    OrtEnvCreationOptions options{};
    options.version = ORT_API_VERSION;
    options.logging_severity_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
    options.log_id = "EnvCreation.CreateEnvWithOptions_2";
    options.config_entries = env_configs.GetConst();

    Ort::Env tmp_env(&options);

    // Use EP API to retrieve environment configs and check contents
    Ort::KeyValuePairs env_configs_2 = Ort::GetEnvConfigEntries();

    auto configs_expected = env_configs.GetKeyValuePairs();
    auto configs_actual = env_configs_2.GetKeyValuePairs();
    ASSERT_EQ(configs_actual, configs_expected);
  };

  EXPECT_NO_FATAL_FAILURE(run_test());
  ortenv_setup();  // Restore OrtEnv
}

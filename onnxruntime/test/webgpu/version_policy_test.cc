// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "gtest/gtest.h"
#include "onnxruntime_c_api.h"

namespace onnxruntime {
namespace test {
namespace {

enum class MissingApi {
  None,
  Ort,
  Ep,
  ModelEditor,
};

struct VersionPolicyCase {
  const char* name;
  const char* runtime_version;
  MissingApi missing_api;
  uint32_t expected_api_version;
  OrtErrorCode expected_error;
  const char* expected_message;
};

#if GTEST_HAS_DEATH_TEST

struct MockStatus {
  OrtErrorCode code;
  std::string message;
};

// Only the APIs used before factory construction are supplied. No actual runtime or GPU is needed.
struct MockRuntime {
  inline static MockRuntime* current = nullptr;

  explicit MockRuntime(const VersionPolicyCase& policy_case) : test_case{policy_case} {
    current = this;
    api.CreateStatus = CreateStatus;
    api.GetEpApi = GetEpApi;
    api.GetModelEditorApi = GetModelEditorApi;
  }

  static const char* ORT_API_CALL GetVersionString() noexcept {
    return current->test_case.runtime_version;
  }

  static const OrtApi* ORT_API_CALL GetApi(uint32_t version) noexcept {
    if (version == 1) {
      ++current->error_api_requests;
      return &current->api;
    }
    ++current->runtime_api_requests;
    current->requested_api_version = version;
    return current->test_case.missing_api == MissingApi::Ort ? nullptr : &current->api;
  }

  static const OrtEpApi* ORT_API_CALL GetEpApi() noexcept {
    return current->test_case.missing_api == MissingApi::Ep ? nullptr : &current->ep_api;
  }

  static const OrtModelEditorApi* ORT_API_CALL GetModelEditorApi() noexcept {
    return current->test_case.missing_api == MissingApi::ModelEditor ? nullptr : &current->model_editor_api;
  }

  static OrtStatus* ORT_API_CALL CreateStatus(OrtErrorCode code, const char* message) noexcept {
    current->status = {code, message};
    return reinterpret_cast<OrtStatus*>(&current->status);
  }

  const VersionPolicyCase& test_case;
  OrtApi api{};
  OrtEpApi ep_api{};
  OrtModelEditorApi model_editor_api{};
  MockStatus status{};
  uint32_t requested_api_version = 0;
  size_t runtime_api_requests = 0;
  size_t error_api_requests = 0;
};

void CheckFactoryInitialization(const VersionPolicyCase& test_case) {
  const auto library_path = std::filesystem::absolute(ORT_WEBGPU_PLUGIN_LIBRARY_NAME);
#ifdef _WIN32
  const auto library = LoadLibraryW(library_path.c_str());
  ASSERT_NE(library, nullptr) << "LoadLibraryW failed: " << GetLastError();
  const auto create_factories = reinterpret_cast<CreateEpApiFactoriesFn>(GetProcAddress(library, "CreateEpFactories"));
#else
  void* library = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(library, nullptr) << dlerror();
  const auto create_factories = reinterpret_cast<CreateEpApiFactoriesFn>(dlsym(library, "CreateEpFactories"));
#endif
  ASSERT_NE(create_factories, nullptr);

  MockRuntime runtime{test_case};
  const OrtApiBase api_base{MockRuntime::GetApi, MockRuntime::GetVersionString};
  OrtEpFactory* factory = nullptr;
  size_t num_factories = 0;

  // Zero capacity stops immediately AFTER ApiInit succeeds, before logger/device/factory setup.
  // The returned error and API request counts distinguish this from a failed version/API check.
  const auto* status = create_factories("webgpu_version_policy_test", &api_base, nullptr,
                                        &factory, 0, &num_factories);
  ASSERT_EQ(status, reinterpret_cast<OrtStatus*>(&runtime.status));
  EXPECT_EQ(runtime.status.code, test_case.expected_error) << runtime.status.message;
  EXPECT_NE(runtime.status.message.find(test_case.expected_message), std::string::npos) << runtime.status.message;
  EXPECT_EQ(runtime.requested_api_version, test_case.expected_api_version);
  EXPECT_EQ(runtime.runtime_api_requests, test_case.expected_api_version == 0 ? 0u : 1u);
  EXPECT_EQ(runtime.error_api_requests, test_case.expected_error == ORT_FAIL ? 1u : 0u);
  EXPECT_EQ(factory, nullptr);
  EXPECT_EQ(num_factories, 0u);

#ifdef _WIN32
  EXPECT_NE(FreeLibrary(library), 0);
#else
  EXPECT_EQ(dlclose(library), 0);
#endif
}

#endif  // GTEST_HAS_DEATH_TEST

class WebGpuVersionPolicyDeathTest : public testing::TestWithParam<VersionPolicyCase> {};

TEST_P(WebGpuVersionPolicyDeathTest, FactoryInitializesWithRuntimeApiOrReturnsError) {
#if GTEST_HAS_DEATH_TEST
  // Re-exec instead of forking/inheriting a previously initialized plugin's ApiInit call_once state.
  const auto previous_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        CheckFactoryInitialization(GetParam());
        std::_Exit(testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
      },
      testing::ExitedWithCode(EXIT_SUCCESS), "");
  GTEST_FLAG_SET(death_test_style, previous_style);
#else
  GTEST_SKIP() << "Isolated factory initialization requires death-test support.";
#endif
}

constexpr const char* kInitialized = "Not enough space to return EP factory";
constexpr const char* kBelowMinimum = "below the minimum required version \"1.30.1\"";
constexpr const char* kMissingSubApi = "GetEpApi or GetModelEditorApi returned null";

INSTANTIATE_TEST_SUITE_P(
    RuntimeCompatibility, WebGpuVersionPolicyDeathTest,
    testing::Values(
        VersionPolicyCase{"ExactException", "1.28.3", MissingApi::None, 28, ORT_INVALID_ARGUMENT, kInitialized},
        VersionPolicyCase{"ContinuousMinimum", "1.30.1", MissingApi::None, 30, ORT_INVALID_ARGUMENT, kInitialized},
        VersionPolicyCase{"LaterPatch", "1.30.2", MissingApi::None, 30, ORT_INVALID_ARGUMENT, kInitialized},
        VersionPolicyCase{"LaterMinor", "1.31.0", MissingApi::None, 31, ORT_INVALID_ARGUMENT, kInitialized},
        VersionPolicyCase{"BeforeException", "1.28.2", MissingApi::None, 0, ORT_FAIL, kBelowMinimum},
        VersionPolicyCase{"AfterException", "1.28.4", MissingApi::None, 0, ORT_FAIL, kBelowMinimum},
        VersionPolicyCase{"InterveningMinor", "1.29.0", MissingApi::None, 0, ORT_FAIL, kBelowMinimum},
        VersionPolicyCase{"BeforeMinimum", "1.30.0", MissingApi::None, 0, ORT_FAIL, kBelowMinimum},
        VersionPolicyCase{"Prerelease", "1.28.3-dev", MissingApi::None, 0, ORT_FAIL, "could not parse ORT version"},
        VersionPolicyCase{"InvalidVersion", "invalid", MissingApi::None, 0, ORT_FAIL, "could not parse ORT version"},
        VersionPolicyCase{"UnavailableVersion", nullptr, MissingApi::None, 0, ORT_FAIL, "runtime version is unavailable"},
        VersionPolicyCase{"UnsupportedMajor", "2.0.0", MissingApi::None, 0, ORT_FAIL, "unsupported ORT major version"},
        VersionPolicyCase{"MissingExceptionApi", "1.28.3", MissingApi::Ort, 28, ORT_FAIL,
                          "does not support the parsed API version 28"},
        VersionPolicyCase{"MissingMinimumApi", "1.30.1", MissingApi::Ort, 30, ORT_FAIL,
                          "does not support the parsed API version 30"},
        VersionPolicyCase{"MissingEpApi", "1.28.3", MissingApi::Ep, 28, ORT_FAIL, kMissingSubApi},
        VersionPolicyCase{"MissingModelEditorApi", "1.28.3", MissingApi::ModelEditor, 28, ORT_FAIL, kMissingSubApi}),
    [](const testing::TestParamInfo<VersionPolicyCase>& info) { return info.param.name; });

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#include <filesystem>
#include <system_error>
#include <utility>

#if defined(ORT_NO_EXCEPTIONS) && (defined(__EXCEPTIONS) || defined(_CPPUNWIND))
#error "The optional-provider no-exceptions test must disable compiler exceptions."
#endif

#include "gtest/gtest.h"
#include "core/common/inlined_containers.h"
#include "core/platform/env.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/provider_bridge_library.h"
#include "core/session/provider_bridge_ort.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::test {

class OptionalProviderProbeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    runtime_directory_ = Env::Default().GetRuntimePath();
    shared_path_ = runtime_directory_ / GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_shared"));
    provider_path_ = runtime_directory_ / GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_openvino"));
    UnloadSharedProviders();
    ASSERT_FALSE(std::filesystem::exists(shared_path_));
    ASSERT_FALSE(std::filesystem::exists(provider_path_));
  }

  void TearDown() override {
    UnloadSharedProviders();
    for (const auto& path : installed_paths_) {
      std::error_code error;
      std::filesystem::remove(path, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  void Install(const char* source, const std::filesystem::path& destination) {
    std::error_code error;
    std::filesystem::copy_file(runtime_directory_ / "fixtures" / source, destination, error);
    ASSERT_FALSE(error) << error.message();
    installed_paths_.push_back(destination);
  }

  void CheckInitializedOnce() {
    void* handle = nullptr;
    ASSERT_STATUS_OK(Env::Default().LoadDynamicLibrary(provider_path_.native(), false, &handle));
    auto unload = gsl::finally([&] { EXPECT_STATUS_OK(Env::Default().UnloadDynamicLibrary(handle)); });
    void* symbol = nullptr;
    ASSERT_STATUS_OK(Env::Default().GetSymbolFromLibrary(handle, "GetInitializationCount", &symbol));
    EXPECT_EQ(reinterpret_cast<int (*)()>(symbol)(), 1);
  }

  std::filesystem::path runtime_directory_;
  std::filesystem::path shared_path_;
  std::filesystem::path provider_path_;
  InlinedVector<std::filesystem::path> installed_paths_;
};

TEST_F(OptionalProviderProbeTest, MissingSharedLibraryReturnsNullOnRetry) {
#if defined(ORT_NO_EXCEPTIONS) && GTEST_HAS_DEATH_TEST
  ASSERT_DEATH(
      {
        ProviderLibrary missing_library(ORT_TSTR("optional_provider_not_present"));
        (void)missing_library.Get();
      },
      "onnxruntime_providers_shared");
#endif
  EXPECT_EQ(TryGetProviderInfo_OpenVINO(), nullptr);
  EXPECT_EQ(TryGetProviderInfo_OpenVINO(), nullptr);
}

TEST_F(OptionalProviderProbeTest, MissingProviderReturnsNullOnRetry) {
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_shared_LIBRARY, shared_path_));
  EXPECT_EQ(TryGetProviderInfo_OpenVINO(), nullptr);
  EXPECT_EQ(TryGetProviderInfo_OpenVINO(), nullptr);
}

TEST_F(OptionalProviderProbeTest, MissingExportUnloadsAndAllowsReplacement) {
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_shared_LIBRARY, shared_path_));
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_missing_export_LIBRARY, provider_path_));
  EXPECT_EQ(TryGetProviderInfo_OpenVINO(), nullptr);
  EXPECT_EQ(TryGetProviderInfo_OpenVINO(), nullptr);
  std::error_code error;
  ASSERT_TRUE(std::filesystem::remove(provider_path_, error)) << error.message();
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_valid_LIBRARY, provider_path_));
  EXPECT_NE(TryGetProviderInfo_OpenVINO(), nullptr);
  ASSERT_NO_FATAL_FAILURE(CheckInitializedOnce());
}

TEST_F(OptionalProviderProbeTest, SuccessfulProbeInitializesOnceAndCanReload) {
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_shared_LIBRARY, shared_path_));
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_valid_LIBRARY, provider_path_));
  const ::ProviderInfo_OpenVINO* info = TryGetProviderInfo_OpenVINO();
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(TryGetProviderInfo_OpenVINO(), info);
  ASSERT_NO_FATAL_FAILURE(CheckInitializedOnce());
  void* retained_handle = nullptr;
  ASSERT_STATUS_OK(Env::Default().LoadDynamicLibrary(provider_path_.native(), false, &retained_handle));
  auto unload = gsl::finally([&] { EXPECT_STATUS_OK(Env::Default().UnloadDynamicLibrary(retained_handle)); });
  UnloadSharedProviders();
  EXPECT_NE(TryGetProviderInfo_OpenVINO(), nullptr);
  ASSERT_NO_FATAL_FAILURE(CheckInitializedOnce());
}

TEST_F(OptionalProviderProbeTest, PluginRegistrationInitializesOnceAndUnregisterUnloads) {
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_shared_LIBRARY, shared_path_));
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_plugin_LIBRARY, provider_path_));
  const ScopedEnvironmentVariables events{{"ORT_TEST_PLUGIN_LIFETIME_EVENTS", "_"}};
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "plugin_lifetime_test"};

  for (const char* expected : {"_L", "_LUL"}) {
    Ort::Status registered{Ort::GetApi().RegisterExecutionProviderLibrary(
        env, "plugin_lifetime_test", provider_path_.c_str())};
    ASSERT_TRUE(registered.IsOK()) << registered.GetErrorMessage();
    auto unregister = gsl::finally([&]() {
      Ort::Status status{Ort::GetApi().UnregisterExecutionProviderLibrary(env, "plugin_lifetime_test")};
      EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
    });
    EXPECT_EQ(Env::Default().GetEnvironmentVar("ORT_TEST_PLUGIN_LIFETIME_EVENTS"), expected);
  }

  EXPECT_EQ(Env::Default().GetEnvironmentVar("ORT_TEST_PLUGIN_LIFETIME_EVENTS"), "_LULU");
}

TEST_F(OptionalProviderProbeTest, FailedPluginLoadUnloadsRetainedProbe) {
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_shared_LIBRARY, shared_path_));
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "plugin_failure_test"};

  for (const auto& [library, registration_name] : {
           std::pair{ORT_OPTIONAL_PROBE_plugin_missing_export_LIBRARY, "missing_release_export"},
           std::pair{ORT_OPTIONAL_PROBE_plugin_LIBRARY, "fail_create_factories"}}) {
    SCOPED_TRACE(registration_name);
    ASSERT_NO_FATAL_FAILURE(Install(library, provider_path_));
    const ScopedEnvironmentVariables events{{"ORT_TEST_PLUGIN_LIFETIME_EVENTS", "_"}};
    Ort::Status status{Ort::GetApi().RegisterExecutionProviderLibrary(env, registration_name, provider_path_.c_str())};
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(Env::Default().GetEnvironmentVar("ORT_TEST_PLUGIN_LIFETIME_EVENTS"), "_LU");
    std::error_code error;
    ASSERT_TRUE(std::filesystem::remove(provider_path_, error)) << error.message();
  }
}

TEST_F(OptionalProviderProbeTest, RetainedPluginProbeCanRetryWithoutLeakingReferences) {
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_shared_LIBRARY, shared_path_));
  ASSERT_NO_FATAL_FAILURE(Install(ORT_OPTIONAL_PROBE_plugin_LIBRARY, provider_path_));
  const ScopedEnvironmentVariables events{{"ORT_TEST_PLUGIN_LIFETIME_EVENTS", "_"}};
  ProviderLibrary library{provider_path_.c_str(), true, ProviderLibraryPathType::Absolute};
  auto unload = gsl::finally([&]() { library.Unload(); });

  EXPECT_FALSE(library.Load(/*retain_library_on_missing_get_provider=*/true).IsOK());
  EXPECT_FALSE(library.Load(/*retain_library_on_missing_get_provider=*/true).IsOK());
  EXPECT_EQ(Env::Default().GetEnvironmentVar("ORT_TEST_PLUGIN_LIFETIME_EVENTS"), "_L");

  EXPECT_FALSE(library.Load().IsOK());
  EXPECT_EQ(Env::Default().GetEnvironmentVar("ORT_TEST_PLUGIN_LIFETIME_EVENTS"), "_LU");
}

}  // namespace onnxruntime::test

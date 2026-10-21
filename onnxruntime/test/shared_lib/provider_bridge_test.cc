#include <filesystem>
#include <system_error>

#include "gtest/gtest.h"
#include "core/platform/env.h"
#include "core/session/provider_bridge_ort.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"

namespace onnxruntime::test {

class ProviderBridgeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    runtime_directory_ = Env::Default().GetRuntimePath();
    shared_library_path_ = runtime_directory_ / GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_shared"));
    UnloadSharedProviders();
    ASSERT_FALSE(std::filesystem::exists(shared_library_path_));
  }

  void TearDown() override {
    UnloadSharedProviders();
    std::error_code error;
    std::filesystem::remove(shared_library_path_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void InstallLibrary(const char* file_name) {
    std::error_code error;
    std::filesystem::copy_file(runtime_directory_ / file_name, shared_library_path_, error);
    ASSERT_FALSE(error) << error.message();
  }

  std::filesystem::path runtime_directory_;
  std::filesystem::path shared_library_path_;
};

TEST_F(ProviderBridgeTest, MissingLibraryReturnsFalseOnRetry) {
  EXPECT_FALSE(InitProvidersSharedLibrary());
  EXPECT_FALSE(InitProvidersSharedLibrary());
}

TEST_F(ProviderBridgeTest, MissingExportUnloadsLibraryAndAllowsRetry) {
  ASSERT_NO_FATAL_FAILURE(InstallLibrary(ORT_PROVIDER_BRIDGE_MISSING_EXPORT_TEST_LIBRARY));
  EXPECT_FALSE(InitProvidersSharedLibrary());
  EXPECT_FALSE(InitProvidersSharedLibrary());

  std::error_code error;
  ASSERT_TRUE(std::filesystem::remove(shared_library_path_, error)) << error.message();
  EXPECT_FALSE(InitProvidersSharedLibrary());

  ASSERT_NO_FATAL_FAILURE(InstallLibrary(ORT_PROVIDER_BRIDGE_VALID_TEST_LIBRARY));
  EXPECT_TRUE(InitProvidersSharedLibrary());
}

TEST_F(ProviderBridgeTest, SuccessfulInitializationSetsHostOnceAndCanUnload) {
  ASSERT_NO_FATAL_FAILURE(InstallLibrary(ORT_PROVIDER_BRIDGE_VALID_TEST_LIBRARY));
  ASSERT_TRUE(InitProvidersSharedLibrary());

  {
    void* fixture_handle = nullptr;
    ASSERT_STATUS_OK(Env::Default().LoadDynamicLibrary(shared_library_path_.native(), false, &fixture_handle));
    auto unload_fixture = gsl::finally([&] { EXPECT_STATUS_OK(Env::Default().UnloadDynamicLibrary(fixture_handle)); });
    void* count_symbol = nullptr;
    ASSERT_STATUS_OK(Env::Default().GetSymbolFromLibrary(fixture_handle, "GetProviderSetHostCallCount", &count_symbol));
    const auto get_host_call_count = reinterpret_cast<int (*)()>(count_symbol);
    EXPECT_EQ(get_host_call_count(), 1);
    EXPECT_TRUE(InitProvidersSharedLibrary());
    EXPECT_EQ(get_host_call_count(), 1);
  }

  UnloadSharedProviders();
  std::error_code error;
  ASSERT_TRUE(std::filesystem::remove(shared_library_path_, error)) << error.message();
  EXPECT_FALSE(InitProvidersSharedLibrary());
}

}  // namespace onnxruntime::test
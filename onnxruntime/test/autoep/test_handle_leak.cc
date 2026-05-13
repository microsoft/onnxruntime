// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <gtest/gtest.h>

#include "core/common/path_string.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/util/include/temp_dir.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

namespace {

// Returns whether the library is currently mapped in the process, or std::nullopt if the platform
// does not support querying loaded-library state without side effects.
// On Windows, GetModuleHandleW queries by filename without incrementing the refcount.
// On Linux/macOS, dlopen with RTLD_NOLOAD probes without loading; if it succeeds it adds a
// refcount that we immediately release with dlclose.
std::optional<bool> IsLibraryLoaded(const std::filesystem::path& library_path) {
#if defined(_WIN32)
  return GetModuleHandleW(library_path.filename().wstring().c_str()) != nullptr;
#else
#ifdef RTLD_NOLOAD
  void* handle = dlopen(library_path.c_str(), RTLD_NOLOAD | RTLD_NOW);
  if (handle) {
    dlclose(handle);  // Undo the refcount added by the RTLD_NOLOAD probe.
    return true;
  }
  return false;
#else
  // RTLD_NOLOAD is not available on this platform; cannot probe without loading.
  (void)library_path;
  return std::nullopt;
#endif
#endif
}

}  // namespace

// Verify that registering and unregistering a plugin EP library does not leak the library handle.
//
// ProviderLibrary::Load() loads the library then probes for the "GetProvider" symbol. Most plugin EP
// libraries do not export "GetProvider", so the probe fails. Without the fix (PR #28396),
// Load() returned the error without calling Unload(), leaving a leaked refcount. After
// UnregisterExecutionProviderLibrary released only the EpLibraryPlugin's reference, the library
// remained mapped in the process.
//
// To ensure this test is independent of process state (other tests may load the same EP library),
// we copy the library to a temporary directory with a unique filename. This guarantees the copy
// has never been loaded, so we can reliably detect refcount leaks via IsLibraryLoaded.
TEST(OrtEpLibrary, RegisterUnregisterDoesNotLeakLibraryHandle) {
  const std::filesystem::path& original_library_path = Utils::example_ep_info.library_path;
  const std::string& registration_name = Utils::example_ep_info.registration_name;

  // Create a temporary directory (RAII: deleted on scope exit).
  TemporaryDirectory temp_dir(ORT_TSTR("test_handle_leak_temp"));

  // Copy the EP library to the temp directory with a unique filename so it is guaranteed to
  // not already be loaded in this process.
#if defined(_WIN32)
  const std::filesystem::path temp_library_path =
      std::filesystem::path(temp_dir.Path()) / "handle_leak_test_ep.dll";
#elif defined(__APPLE__)
  const std::filesystem::path temp_library_path =
      std::filesystem::path(temp_dir.Path()) / "libhandle_leak_test_ep.dylib";
#else
  const std::filesystem::path temp_library_path =
      std::filesystem::path(temp_dir.Path()) / "libhandle_leak_test_ep.so";
#endif

  std::error_code ec;
  std::filesystem::copy_file(original_library_path, temp_library_path,
                             std::filesystem::copy_options::overwrite_existing, ec);
  ASSERT_FALSE(ec) << "Failed to copy EP library to temp directory: " << ec.message();

  // Verify the platform supports loaded-library probes.
  std::optional<bool> loaded_before = IsLibraryLoaded(temp_library_path);
  if (!loaded_before.has_value()) {
    GTEST_SKIP() << "Platform does not support querying loaded-library state.";
  }

  // The copy should not be loaded yet since we just created it with a unique name.
  ASSERT_FALSE(*loaded_before)
      << "Freshly copied library should not already be loaded in the process.";

  // Register the plugin EP library. Internally this calls ProviderLibrary::Load() (which
  // loads the library and fails to find "GetProvider") and then EpLibraryPlugin::Load().
  ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(),
                                            temp_library_path.c_str());

  // The library should be loaded now.
  ASSERT_TRUE(IsLibraryLoaded(temp_library_path).value_or(false))
      << "Library should be loaded after registration.";

  // Unregister releases the EpLibraryPlugin's reference.
  ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());

  // If the fix is applied, the library should be fully unloaded (refcount == 0).
  // Without the fix, ProviderLibrary::Load() leaks a refcount so the library remains mapped.
  EXPECT_FALSE(IsLibraryLoaded(temp_library_path).value_or(true))
      << "Library handle leaked: EP library is still loaded after UnregisterExecutionProviderLibrary. "
         "This indicates ProviderLibrary::Load() did not call Unload() on GetProvider symbol miss.";
}

}  // namespace test
}  // namespace onnxruntime

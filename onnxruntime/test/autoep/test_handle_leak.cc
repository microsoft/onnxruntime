// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"

#include "test/autoep/test_autoep_utils.h"

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
// Limitation: This test can only detect the leak when the library starts fully unloaded (refcount 0).
// If another test in this binary already loaded the library, IsLibraryLoaded returns true both before
// and after, masking any refcount leak. In that case, we skip. GoogleTest does not guarantee test
// order, so this may or may not run before other tests that load the library.
//
// TODO: Build this test as a separate binary (e.g., onnxruntime_autoep_handle_leak_test) to guarantee
// a clean starting state. This was attempted but reverted because adding an extra link target caused
// the ARM64 Linux Debug CI to run out of memory during linking (the runner is at its limit).
TEST(OrtEpLibrary, RegisterUnregisterDoesNotLeakLibraryHandle) {
  const std::filesystem::path& library_path = Utils::example_ep_info.library_path;
  const std::string& registration_name = Utils::example_ep_info.registration_name;

  std::optional<bool> loaded_before = IsLibraryLoaded(library_path);
  if (!loaded_before.has_value()) {
    GTEST_SKIP() << "Platform does not support querying loaded-library state (RTLD_NOLOAD unavailable).";
  }

  if (*loaded_before) {
    GTEST_SKIP() << "Library is already loaded by another test in this binary. "
                    "Cannot detect refcount leaks with a boolean loaded/unloaded check. "
                    "See TODO above for the ideal separate-binary approach.";
  }

  // Register the plugin EP library. Internally this calls ProviderLibrary::Load() (which
  // loads the library and fails to find "GetProvider") and then EpLibraryPlugin::Load().
  ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

  // The library should be loaded now.
  ASSERT_TRUE(IsLibraryLoaded(library_path).value_or(false))
      << "Library should be loaded after registration.";

  // Unregister releases the EpLibraryPlugin's reference.
  ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());

  // If the fix is applied, the library should be fully unloaded (refcount == 0).
  // Without the fix, ProviderLibrary::Load() leaks a refcount so the library remains mapped.
  EXPECT_FALSE(IsLibraryLoaded(library_path).value_or(true))
      << "Library handle leaked: EP library is still loaded after UnregisterExecutionProviderLibrary. "
         "This indicates ProviderLibrary::Load() did not call Unload() on GetProvider symbol miss.";
}

}  // namespace test
}  // namespace onnxruntime

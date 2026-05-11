// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
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

// Returns true if the library is currently mapped in the process.
// On Windows, GetModuleHandleW queries by filename without incrementing the refcount.
// On Linux/macOS, dlopen with RTLD_NOLOAD probes without loading; if it succeeds it adds a
// refcount that we immediately release with dlclose.
bool IsLibraryLoaded(const std::filesystem::path& library_path) {
#if defined(_WIN32)
  return GetModuleHandleW(library_path.filename().wstring().c_str()) != nullptr;
#else
  void* handle = dlopen(library_path.c_str(), RTLD_NOLOAD | RTLD_NOW);
  if (handle) {
    dlclose(handle);  // Undo the refcount added by the RTLD_NOLOAD probe.
    return true;
  }
  return false;
#endif
}

}  // namespace

// Verify that registering and unregistering a plugin EP library does not leak the library handle.
//
// ProviderLibrary::Load() loads the library then probes for the "GetProvider" symbol. Plugin EP
// libraries do not export "GetProvider", so the probe fails. Without the fix (PR #28396),
// Load() returned the error without calling Unload(), leaving a leaked refcount. After
// UnregisterExecutionProviderLibrary released only the EpLibraryPlugin's reference, the library
// remained mapped in the process.
//
// This test lives in a separate binary to guarantee the plugin EP library starts fully unloaded
// (refcount 0). In the main autoep test binary, other tests may have already loaded the same
// library, making it impossible to detect refcount leaks with a boolean loaded/unloaded check.
TEST(OrtEpLibrary, RegisterUnregisterDoesNotLeakLibraryHandle) {
  const std::filesystem::path& library_path = Utils::example_ep_info.library_path;
  const std::string& registration_name = Utils::example_ep_info.registration_name;

  // This binary contains only this test, so the library must not be loaded yet.
  ASSERT_FALSE(IsLibraryLoaded(library_path))
      << "Library unexpectedly loaded before test. This test must run in an isolated binary.";

  // Register the plugin EP library. Internally this calls ProviderLibrary::Load() (which
  // loads the library and fails to find "GetProvider") and then EpLibraryPlugin::Load().
  ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

  // The library should be loaded now.
  ASSERT_TRUE(IsLibraryLoaded(library_path)) << "Library should be loaded after registration.";

  // Unregister releases the EpLibraryPlugin's reference.
  ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());

  // If the fix is applied, the library should be fully unloaded (refcount == 0).
  // Without the fix, ProviderLibrary::Load() leaks a refcount so the library remains mapped.
  EXPECT_FALSE(IsLibraryLoaded(library_path))
      << "Library handle leaked: EP library is still loaded after UnregisterExecutionProviderLibrary. "
         "This indicates ProviderLibrary::Load() did not call Unload() on GetProvider symbol miss.";
}

}  // namespace test
}  // namespace onnxruntime

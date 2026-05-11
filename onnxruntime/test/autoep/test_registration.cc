// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_env_config_keys.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" void ortenv_setup();
extern "C" void ortenv_teardown();

namespace onnxruntime {
namespace test {

TEST(OrtEpLibrary, LoadUnloadPluginLibrary) {
  const std::filesystem::path& library_path = Utils::example_ep_info.library_path;
  const std::string& registration_name = Utils::example_ep_info.registration_name;
  const std::string& ep_name = Utils::example_ep_info.ep_name;

  const OrtApi* c_api = &Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ASSERT_ORTSTATUS_OK(Ort::GetApi().RegisterExecutionProviderLibrary(*ort_env, registration_name.c_str(),
                                                                     library_path.c_str()));

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices = 0;

  ASSERT_ORTSTATUS_OK(Ort::GetApi().GetEpDevices(*ort_env, &ep_devices, &num_devices));
  // should be one device for the example EP
  auto num_test_ep_devices = std::count_if(ep_devices, ep_devices + num_devices,
                                           [&](const OrtEpDevice* device) {
                                             return c_api->EpDevice_EpName(device) == ep_name;
                                           });
  ASSERT_EQ(num_test_ep_devices, 1) << "Expected an OrtEpDevice to have been created by the test library.";

  // and this should unload it
  ASSERT_ORTSTATUS_OK(Ort::GetApi().UnregisterExecutionProviderLibrary(*ort_env,
                                                                       registration_name.c_str()));
}

TEST(OrtEpLibrary, LoadUnloadPluginLibraryCxxApi) {
  const std::filesystem::path& library_path = Utils::example_ep_info.library_path;
  const std::string& registration_name = Utils::example_ep_info.registration_name;
  const std::string& ep_name = Utils::example_ep_info.ep_name;

  // this should load the library and create OrtEpDevice
  ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

  std::vector<Ort::ConstEpDevice> ep_devices = ort_env->GetEpDevices();

  // should be one device for the example EP
  auto test_ep_device = std::find_if(ep_devices.begin(), ep_devices.end(),
                                     [&](Ort::ConstEpDevice& device) {
                                       return device.EpName() == ep_name;
                                     });
  ASSERT_NE(test_ep_device, ep_devices.end()) << "Expected an OrtEpDevice to have been created by the test library.";

  // test all the C++ getters.
  // expected values are from \onnxruntime\test\autoep\library\example_plugin_ep\*.cc
  ASSERT_STREQ(test_ep_device->EpVendor(), "Contoso");

  auto metadata = test_ep_device->EpMetadata();
  ASSERT_STREQ(metadata.GetValue(kOrtEpDevice_EpMetadataKey_Version), "0.1.0");
  ASSERT_STREQ(metadata.GetValue("supported_devices"), "CrackGriffin 7+");

  auto options = test_ep_device->EpOptions();
  ASSERT_STREQ(options.GetValue("run_really_fast"), "true");

  // Verify the library path is present in the EP metadata
  const char* metadata_library_path = metadata.GetValue(kOrtEpDevice_EpMetadataKey_LibraryPath);
  ASSERT_NE(metadata_library_path, nullptr) << "Expected library_path to be present in EP metadata.";

  // Verify the library path matches the registered path
  std::filesystem::path metadata_path{metadata_library_path};
  ASSERT_EQ(std::filesystem::canonical(metadata_path), std::filesystem::canonical(library_path))
      << "Expected library_path in EP metadata to match the registered library path.";

  // the CPU device info will vary by machine so check for the lowest common denominator values
  Ort::ConstHardwareDevice device = test_ep_device->Device();
  ASSERT_EQ(device.Type(), OrtHardwareDeviceType_CPU);
  ASSERT_GE(device.VendorId(), 0);
  ASSERT_GE(device.DeviceId(), 0);
  ASSERT_NE(device.Vendor(), nullptr);
  Ort::ConstKeyValuePairs device_metadata = device.Metadata();
  std::unordered_map<std::string, std::string> metadata_entries = device_metadata.GetKeyValuePairs();
#if defined(_WIN32)
  ASSERT_GT(metadata_entries.size(), 0);  // should have at least SPDRP_HARDWAREID on Windows
#endif

  // and this should unload it without throwing
  ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());
}

// Test loading example_plugin_ep_virt_gpu and its associated OrtEpDevice/OrtHardwareDevice.
// This EP creates a new OrtHardwareDevice instance that represents a virtual GPU and gives to ORT.
TEST(OrtEpLibrary, LoadUnloadPluginVirtGpuLibraryCxxApi) {
  const std::filesystem::path& library_path = Utils::example_ep_virt_gpu_info.library_path;
  const std::string& registration_name = "example_plugin_ep_virt_gpu";
  const std::string& ep_name = Utils::example_ep_virt_gpu_info.ep_name;

  auto get_plugin_ep_devices = [&](Ort::Env& env) -> std::vector<Ort::ConstEpDevice> {
    std::vector<Ort::ConstEpDevice> all_ep_devices = env.GetEpDevices();
    std::vector<Ort::ConstEpDevice> ep_devices;

    std::copy_if(all_ep_devices.begin(), all_ep_devices.end(), std::back_inserter(ep_devices),
                 [&](Ort::ConstEpDevice& device) {
                   return device.EpName() == ep_name;
                 });

    return ep_devices;
  };

  auto is_hw_device_virtual = [](Ort::ConstHardwareDevice hw_device) -> bool {
    std::unordered_map<std::string, std::string> metadata_entries = hw_device.Metadata().GetKeyValuePairs();
    auto iter = metadata_entries.find(kOrtHardwareDevice_MetadataKey_IsVirtual);

    if (iter == metadata_entries.end()) {
      return false;
    }

    return iter->second == "1";
  };

  // Test getting EP's supported OrtEpDevices. Do not allow virtual devices.
  // The EP should not return any OrtEpDevice instances.
  {
    ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

    // Find ep devices for this EP. Should not get any.
    std::vector<Ort::ConstEpDevice> ep_devices = get_plugin_ep_devices(*ort_env);
    ASSERT_EQ(ep_devices.size(), 0);

    ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());
  }

  // Test getting EP's supported OrtEpDevices, but ALLOW virtual devices.
  // The EP should return a OrtEpDevice for a virtual GPU.
  {
    // Use a registration name ending with ".virtual" to indicate to the EP library (factory) that creating virtual
    // devices is allowed.
    std::string registration_name_for_virtual_devices = registration_name + ".virtual";
    ort_env->RegisterExecutionProviderLibrary(registration_name_for_virtual_devices.c_str(), library_path.c_str());

    // Find ep devices for this EP. Should get a virtual gpu.
    std::vector<Ort::ConstEpDevice> ep_devices = get_plugin_ep_devices(*ort_env);
    ASSERT_EQ(ep_devices.size(), 1);

    auto virt_gpu_ep_device = std::find_if(ep_devices.begin(), ep_devices.end(),
                                           [](Ort::ConstEpDevice& ep_device) {
                                             return ep_device.Device().Type() == OrtHardwareDeviceType_GPU;
                                           });

    ASSERT_TRUE(is_hw_device_virtual(virt_gpu_ep_device->Device()));

    // test metadata and provider options attached to the virtual OrtEpDevice.
    // expected values are from \onnxruntime\test\autoep\library\example_plugin_ep_virt_gpu\*.cc
    ASSERT_STREQ(virt_gpu_ep_device->EpVendor(), "Contoso2");

    auto metadata = virt_gpu_ep_device->EpMetadata();
    ASSERT_STREQ(metadata.GetValue(kOrtEpDevice_EpMetadataKey_Version), "0.1.0");
    ASSERT_STREQ(metadata.GetValue("some_metadata"), "1");

    auto options = virt_gpu_ep_device->EpOptions();
    ASSERT_STREQ(options.GetValue("compile_optimization"), "O3");

    // Check the virtual GPU hw device info.
    ASSERT_EQ(virt_gpu_ep_device->Device().VendorId(), 0xB358);
    ASSERT_EQ(virt_gpu_ep_device->Device().DeviceId(), 0);
    ASSERT_STREQ(virt_gpu_ep_device->Device().Vendor(), virt_gpu_ep_device->EpVendor());

    ort_env->UnregisterExecutionProviderLibrary(registration_name_for_virtual_devices.c_str());
  }

  // Test using OrtApi::CreateEnvWithOptions to explicitly set a config that enables virtual devices.
  // The EP should return a OrtEpDevice for a virtual GPU.

  ortenv_teardown();  // Release current OrtEnv as we need to recreate it.

  auto run_test = [&]() -> void {
    // Create OrtEnv with config entry to enable virtual devices.
    Ort::KeyValuePairs env_configs;
    env_configs.Add(kOrtEnvAllowVirtualDevices, "1");

    OrtEnvCreationOptions env_options{};
    env_options.version = ORT_API_VERSION;
    env_options.logging_severity_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
    env_options.log_id = "LoadUnloadPluginVirtGpuLibraryCxxApi";
    env_options.config_entries = env_configs.GetConst();

    Ort::Env tmp_env(&env_options);

    // Register EP library. It should be able to extract the env config entry that enables virtual devices.
    tmp_env.RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

    // Find ep devices for this EP. Should get a virtual gpu.
    std::vector<Ort::ConstEpDevice> ep_devices = get_plugin_ep_devices(tmp_env);
    ASSERT_EQ(ep_devices.size(), 1);

    auto virt_gpu_ep_device = std::find_if(ep_devices.begin(), ep_devices.end(),
                                           [](Ort::ConstEpDevice& ep_device) {
                                             return ep_device.Device().Type() == OrtHardwareDeviceType_GPU;
                                           });

    ASSERT_TRUE(is_hw_device_virtual(virt_gpu_ep_device->Device()));
    tmp_env.UnregisterExecutionProviderLibrary(registration_name.c_str());
  };

  EXPECT_NO_FATAL_FAILURE(run_test());
  ortenv_setup();  // Restore OrtEnv
}

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
// ProviderLibrary::Load() loads the library then probes for the "GetProvider" symbol. Plugin EP
// libraries do not export "GetProvider", so the probe fails. Without the fix, Load() returned the
// error without calling Unload(), leaving a leaked refcount. After UnregisterExecutionProviderLibrary
// released only the EpLibraryPlugin's reference, the library remained mapped in the process.
TEST(OrtEpLibrary, RegisterUnregisterDoesNotLeakLibraryHandle) {
  const std::filesystem::path& library_path = Utils::example_ep_info.library_path;
  const std::string& registration_name = Utils::example_ep_info.registration_name;

  // Capture whether the library is already loaded (e.g., by a prior test in the same process).
  // If it is, we cannot verify the leak because IsLibraryLoaded returns a boolean, not a refcount —
  // it can't distinguish "loaded at refcount N" from "loaded at refcount N+1". A subprocess-based
  // approach could guarantee a clean starting state, but GTEST_SKIP is simpler and still catches
  // the regression when this test runs before other tests that load the same library.
  if (IsLibraryLoaded(library_path)) {
    GTEST_SKIP() << "Library already loaded by a prior test; cannot verify refcount leak.";
  }

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

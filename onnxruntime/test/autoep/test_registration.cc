// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

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

  auto get_plugin_ep_devices = [&]() -> std::vector<Ort::ConstEpDevice> {
    std::vector<Ort::ConstEpDevice> all_ep_devices = ort_env->GetEpDevices();
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
    std::vector<Ort::ConstEpDevice> ep_devices = get_plugin_ep_devices();
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
    std::vector<Ort::ConstEpDevice> ep_devices = get_plugin_ep_devices();
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
}
}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// registration/selection is only supported on windows as there's no device discovery on other platforms
#ifdef _WIN32

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
                                           [&ep_name, &c_api](const OrtEpDevice* device) {
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
                                     [&ep_name](Ort::ConstEpDevice& device) {
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
  ASSERT_GT(metadata_entries.size(), 0);  // should have at least SPDRP_HARDWAREID on Windows

  // and this should unload it without throwing
  ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());
}

// Test loading example_plugin_ep_virt_gpu and its associated OrtEpDevice/OrtHardwareDevice.
// This EP creates a new OrtHardwareDevice instance that represents a virtual GPU and gives to ORT.
TEST(OrtEpLibrary, LoadUnloadPluginVirtGpuLibraryCxxApi) {
  const std::filesystem::path& library_path = Utils::example_ep_virt_gpu_info.library_path;
  const std::string& registration_name = Utils::example_ep_virt_gpu_info.registration_name;
  const std::string& ep_name = Utils::example_ep_virt_gpu_info.ep_name;

  // this should load the library and create OrtEpDevice
  ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

  std::vector<Ort::ConstEpDevice> ep_devices = ort_env->GetEpDevices();

  // should be one device for the example EP
  auto test_ep_device = std::find_if(ep_devices.begin(), ep_devices.end(),
                                     [&ep_name](Ort::ConstEpDevice& device) {
                                       return device.EpName() == ep_name;
                                     });
  ASSERT_NE(test_ep_device, ep_devices.end()) << "Expected an OrtEpDevice to have been created by " << ep_name;

  // test all the C++ getters.
  // expected values are from \onnxruntime\test\autoep\library\example_plugin_ep_virt_gpu\*.cc
  ASSERT_STREQ(test_ep_device->EpVendor(), "Contoso2");

  auto metadata = test_ep_device->EpMetadata();
  ASSERT_STREQ(metadata.GetValue(kOrtEpDevice_EpMetadataKey_Version), "0.1.0");
  ASSERT_STREQ(metadata.GetValue("ex_key"), "ex_value");

  auto options = test_ep_device->EpOptions();
  ASSERT_STREQ(options.GetValue("compile_optimization"), "O3");

  // Check the virtual GPU device info.
  Ort::ConstHardwareDevice virt_gpu_device = test_ep_device->Device();
  ASSERT_EQ(virt_gpu_device.Type(), OrtHardwareDeviceType_GPU);
  ASSERT_EQ(virt_gpu_device.VendorId(), 0xB358);
  ASSERT_EQ(virt_gpu_device.DeviceId(), 0);
  ASSERT_STREQ(virt_gpu_device.Vendor(), test_ep_device->EpVendor());

  // OrtHardwareDevice should have 2 metadata entries ("DiscoveredBy" and "IsVirtual")
  Ort::ConstKeyValuePairs device_metadata = virt_gpu_device.Metadata();
  std::unordered_map<std::string, std::string> metadata_entries = device_metadata.GetKeyValuePairs();
  ASSERT_EQ(metadata_entries.size(), 2);
  ASSERT_EQ(metadata_entries[kOrtHardwareDevice_MetadataKey_DiscoveredBy], ep_name);
  ASSERT_EQ(metadata_entries[kOrtHardwareDevice_MetadataKey_IsVirtual], "1");

  // and this should unload it without throwing
  ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());
}
}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32

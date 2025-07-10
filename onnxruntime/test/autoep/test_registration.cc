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

  const OrtApi* c_api = &Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ASSERT_ORTSTATUS_OK(Ort::GetApi().RegisterExecutionProviderLibrary(*ort_env, registration_name.c_str(),
                                                                     library_path.c_str()));

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices = 0;

  ASSERT_ORTSTATUS_OK(Ort::GetApi().GetEpDevices(*ort_env, &ep_devices, &num_devices));
  // should be one device for the example EP
  auto num_test_ep_devices = std::count_if(ep_devices, ep_devices + num_devices,
                                           [&registration_name, &c_api](const OrtEpDevice* device) {
                                             // the example uses the registration name for the EP name
                                             // but that is not a requirement and the two can differ.
                                             return c_api->EpDevice_EpName(device) == registration_name;
                                           });
  ASSERT_EQ(num_test_ep_devices, 1) << "Expected an OrtEpDevice to have been created by the test library.";

  // and this should unload it
  ASSERT_ORTSTATUS_OK(Ort::GetApi().UnregisterExecutionProviderLibrary(*ort_env,
                                                                       registration_name.c_str()));
}

TEST(OrtEpLibrary, LoadUnloadPluginLibraryCxxApi) {
  const std::filesystem::path& library_path = Utils::example_ep_info.library_path;
  const std::string& registration_name = Utils::example_ep_info.registration_name;

  // this should load the library and create OrtEpDevice
  ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

  std::vector<Ort::ConstEpDevice> ep_devices = ort_env->GetEpDevices();

  // should be one device for the example EP
  auto test_ep_device = std::find_if(ep_devices.begin(), ep_devices.end(),
                                     [&registration_name](Ort::ConstEpDevice& device) {
                                       // the example uses the registration name for the EP name
                                       // but that is not a requirement and the two can differ.
                                       return device.EpName() == registration_name;
                                     });
  ASSERT_NE(test_ep_device, ep_devices.end()) << "Expected an OrtEpDevice to have been created by the test library.";

  // test all the C++ getters. expected values are from \onnxruntime\test\autoep\library\example_plugin_ep.cc
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

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32

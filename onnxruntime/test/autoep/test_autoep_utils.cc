// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/autoep/test_autoep_utils.h"

#include <algorithm>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/file_util.h"

namespace onnxruntime {
namespace test {

Utils::ExamplePluginInfo::ExamplePluginInfo(std::filesystem::path lib_path, const char* reg_name, const char* ep_name)
    : library_path(std::move(lib_path)), registration_name(reg_name), ep_name(ep_name) {}

const Utils::ExamplePluginInfo Utils::example_ep_info(
    GetSharedLibraryFileName(ORT_TSTR("example_plugin_ep")),
    // The example_plugin_ep always uses the registration name as the EP name.
    "example_ep",
    "example_ep");

const Utils::ExamplePluginInfo Utils::example_ep_virt_gpu_info(
    GetSharedLibraryFileName(ORT_TSTR("example_plugin_ep_virt_gpu")),
    "example_plugin_ep_virt_gpu.virtual",  // Ends in ".virtual" to allow creation of virtual devices.
    // This EP's name is hardcoded to the following
    "EpVirtualGpu");

const Utils::ExamplePluginInfo Utils::example_ep_kernel_registry_info(
    GetSharedLibraryFileName(ORT_TSTR("example_plugin_ep_kernel_registry")),
    "example_plugin_ep_kernel_registry",
    "ExampleKernelEp");

void Utils::GetEp(Ort::Env& env, const std::string& ep_name, const OrtEpDevice*& ep_device) {
  const OrtApi& c_api = Ort::GetApi();
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices;
  ASSERT_ORTSTATUS_OK(c_api.GetEpDevices(env, &ep_devices, &num_devices));

  auto it = std::find_if(ep_devices, ep_devices + num_devices,
                         [&c_api, &ep_name](const OrtEpDevice* ep_device) {
                           // example ep uses registration name as ep name
                           return c_api.EpDevice_EpName(ep_device) == ep_name;
                         });

  if (it == ep_devices + num_devices) {
    ep_device = nullptr;
  } else {
    ep_device = *it;
  }
}

void Utils::RegisterAndGetExampleEp(Ort::Env& env, const ExamplePluginInfo& ep_info,
                                    RegisteredEpDeviceUniquePtr& registered_ep) {
  const OrtApi& c_api = Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ASSERT_ORTSTATUS_OK(c_api.RegisterExecutionProviderLibrary(env,
                                                             ep_info.registration_name.c_str(),
                                                             ep_info.library_path.c_str()));
  const OrtEpDevice* example_ep = nullptr;
  GetEp(env, ep_info.ep_name, example_ep);
  ASSERT_NE(example_ep, nullptr);

  registered_ep = RegisteredEpDeviceUniquePtr(example_ep, [&env, &ep_info, c_api](const OrtEpDevice* /*ep*/) {
    Ort::Status ignored{c_api.UnregisterExecutionProviderLibrary(env, ep_info.registration_name.c_str())};
  });
}

}  // namespace test
}  // namespace onnxruntime

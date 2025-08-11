// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// registration/selection is only supported on windows as there's no device discovery on other platforms
#ifdef _WIN32

#include "test/autoep/test_autoep_utils.h"

#include <algorithm>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/api_asserts.h"

namespace onnxruntime {
namespace test {

Utils::ExamplePluginInfo Utils::example_ep_info;

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

void Utils::RegisterAndGetExampleEp(Ort::Env& env, RegisteredEpDeviceUniquePtr& registered_ep) {
  const OrtApi& c_api = Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ASSERT_ORTSTATUS_OK(c_api.RegisterExecutionProviderLibrary(env,
                                                             example_ep_info.registration_name.c_str(),
                                                             example_ep_info.library_path.c_str()));
  const OrtEpDevice* example_ep = nullptr;
  GetEp(env, example_ep_info.registration_name, example_ep);
  ASSERT_NE(example_ep, nullptr);

  registered_ep = RegisteredEpDeviceUniquePtr(example_ep, [&env, c_api](const OrtEpDevice* /*ep*/) {
    c_api.UnregisterExecutionProviderLibrary(env, example_ep_info.registration_name.c_str());
  });
}

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32

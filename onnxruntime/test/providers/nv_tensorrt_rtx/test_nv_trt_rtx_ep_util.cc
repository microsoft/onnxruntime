// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// registration/selection is only supported on windows as there's no device discovery on other platforms
#ifdef _WIN32

#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <algorithm>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/api_asserts.h"

namespace onnxruntime {
namespace test {

Utils::NvTensorRtRtxEpInfo Utils::nv_tensorrt_rtx_ep_info;

void Utils::GetEp(Ort::Env& env, const std::string& ep_name, const OrtEpDevice*& ep_device) {
  const OrtApi& c_api = Ort::GetApi();
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices;
  ASSERT_ORTSTATUS_OK(c_api.GetEpDevices(env, &ep_devices, &num_devices));

  auto it = std::find_if(ep_devices, ep_devices + num_devices,
                         [&c_api, &ep_name](const OrtEpDevice* ep_device) {
                           // NV TensorRT RTX EP uses registration name as ep name
                           return c_api.EpDevice_EpName(ep_device) == ep_name;
                         });

  if (it == ep_devices + num_devices) {
    ep_device = nullptr;
  } else {
    ep_device = *it;
  }
}

void Utils::RegisterAndGetNvTensorRtRtxEp(Ort::Env& env, RegisteredEpDeviceUniquePtr& registered_ep) {
  const OrtApi& c_api = Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ASSERT_ORTSTATUS_OK(c_api.RegisterExecutionProviderLibrary(env,
                                                             nv_tensorrt_rtx_ep_info.registration_name.c_str(),
                                                             nv_tensorrt_rtx_ep_info.library_path.c_str()));
  const OrtEpDevice* nv_tensorrt_rtx_ep = nullptr;
  GetEp(env, nv_tensorrt_rtx_ep_info.registration_name, nv_tensorrt_rtx_ep);
  ASSERT_NE(nv_tensorrt_rtx_ep, nullptr);

  registered_ep = RegisteredEpDeviceUniquePtr(nv_tensorrt_rtx_ep, [&env, c_api](const OrtEpDevice* /*ep*/) {
    c_api.UnregisterExecutionProviderLibrary(env, nv_tensorrt_rtx_ep_info.registration_name.c_str());
  });
}

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32

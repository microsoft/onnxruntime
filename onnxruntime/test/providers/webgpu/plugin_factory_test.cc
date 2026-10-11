// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <gsl/gsl>
#include "gtest/gtest.h"

#include "core/providers/webgpu/ep/factory.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"

namespace onnxruntime::test {

TEST(WebGpuPluginFactoryTest, VirtualDevice) {
  const auto& api = Ort::GetApi();
  const auto& ep_api = *api.GetEpApi();
  OrtHardwareDevice* real_gpu = nullptr;
  Ort::ThrowOnError(ep_api.CreateHardwareDevice(OrtHardwareDeviceType_GPU, 0x8086, 0x1234,
                                                "TestVendor", nullptr, &real_gpu));
  auto release_gpu = gsl::finally([&]() { ep_api.ReleaseHardwareDevice(real_gpu); });

  for (bool allow_virtual : {false, true}) {
    for (bool has_real_gpu : {false, true}) {
      SCOPED_TRACE(testing::Message() << "allow_virtual=" << allow_virtual << ", real_gpu=" << has_real_gpu);
      webgpu::ep::Factory factory({.allow_virtual_devices = allow_virtual});
      const OrtHardwareDevice* devices[] = {real_gpu};
      std::array<OrtEpDevice*, 4> ep_devices{};
      size_t count = 0;
      auto release_devices = gsl::finally([&]() {
        for (size_t i = 0; i < count; ++i) ep_api.ReleaseEpDevice(ep_devices[i]);
      });
      Ort::ThrowOnError(factory.GetSupportedDevices(&factory, devices, has_real_gpu ? 1 : 0,
                                                    ep_devices.data(), ep_devices.size(), &count));
      ASSERT_EQ(count, static_cast<size_t>(allow_virtual) + static_cast<size_t>(has_real_gpu));
      for (size_t i = 0; i < count; ++i) {
        EXPECT_STREQ(api.EpDevice_EpName(ep_devices[i]), "WebGpuExecutionProvider");
        const auto* device = api.EpDevice_Device(ep_devices[i]);
        EXPECT_EQ(api.HardwareDevice_Type(device), OrtHardwareDeviceType_GPU);
        const char* is_virtual = api.GetKeyValue(api.HardwareDevice_Metadata(device),
                                                 kOrtHardwareDevice_MetadataKey_IsVirtual);
        if (has_real_gpu && i == 0) {
          EXPECT_EQ(api.HardwareDevice_VendorId(device), 0x8086u);
          EXPECT_EQ(is_virtual, nullptr);
        } else {
          EXPECT_EQ(api.HardwareDevice_VendorId(device), 0u);
          ASSERT_NE(is_virtual, nullptr);
          EXPECT_STREQ(is_virtual, "1");
        }
      }
    }
  }
}

TEST(WebGpuPluginFactoryTest, RejectsVirtualDeviceWithoutCompileOnly) {
  const auto& api = Ort::GetApi();
  const auto& ep_api = *api.GetEpApi();
  webgpu::ep::Factory factory({.allow_virtual_devices = true});
  OrtEpDevice* device = nullptr;
  size_t count = 0;
  Ort::ThrowOnError(factory.GetSupportedDevices(&factory, nullptr, 0, &device, 1, &count));
  auto release_device = gsl::finally([&]() { ep_api.ReleaseEpDevice(device); });
  ASSERT_EQ(count, 1u);
  const OrtHardwareDevice* hardware[] = {api.EpDevice_Device(device)};
  Ort::SessionOptions options;
  OrtEp* ep = nullptr;
  Ort::Status status{factory.CreateEp(&factory, hardware, nullptr, 1, options, nullptr, &ep)};
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  EXPECT_EQ(ep, nullptr);
}

}  // namespace onnxruntime::test

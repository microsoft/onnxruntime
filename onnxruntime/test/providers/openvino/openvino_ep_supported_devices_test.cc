// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "core/providers/openvino/ov_factory.h"
#include "core/session/abi_devices.h"
#include "onnxruntime_cxx_api.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

struct HardwareDeviceCase {
  const char* name;
  OrtHardwareDeviceType type;
  uint32_t vendor_id;
  bool expected_eligible;
};

class OpenVINOHardwareEligibilityTest : public ::testing::TestWithParam<HardwareDeviceCase> {};

TEST_P(OpenVINOHardwareEligibilityTest, ReturnsExpectedEligibility) {
  const auto& test_case = GetParam();
  EXPECT_EQ(openvino_ep::OpenVINOEpPluginFactory::IsHardwareDeviceEligible(test_case.type, test_case.vendor_id),
            test_case.expected_eligible);
}

INSTANTIATE_TEST_SUITE_P(
    Hardware,
    OpenVINOHardwareEligibilityTest,
    ::testing::Values(
        HardwareDeviceCase{"IntelCpu", OrtHardwareDeviceType_CPU, 0x8086, true},
        HardwareDeviceCase{"AmdCpu", OrtHardwareDeviceType_CPU, 0x1022, true},
        HardwareDeviceCase{"IntelGpu", OrtHardwareDeviceType_GPU, 0x8086, true},
        HardwareDeviceCase{"IntelNpu", OrtHardwareDeviceType_NPU, 0x8086, true},
        HardwareDeviceCase{"AmdGpu", OrtHardwareDeviceType_GPU, 0x1022, false},
        HardwareDeviceCase{"AmdNpu", OrtHardwareDeviceType_NPU, 0x1022, false)),
    [](const ::testing::TestParamInfo<HardwareDeviceCase>& info) { return info.param.name; });

class OpenVINOGetSupportedDevicesTest : public ::testing::TestWithParam<HardwareDeviceCase> {};

TEST_P(OpenVINOGetSupportedDevicesTest, ReportsEligibleDevicesAndSkipsRejectedDevices) {
  const auto& test_case = GetParam();
  const OrtApi& ort_api = Ort::GetApi();
  openvino_ep::ApiPtrs api_ptrs{ort_api, Ort::GetEpApi(), Ort::GetModelEditorApi()};
  openvino_ep::OpenVINOEpPluginFactory factory{api_ptrs, "", std::make_shared<ov::Core>()};

  OrtHardwareDevice hardware_device{};
  hardware_device.type = test_case.type;
  hardware_device.vendor_id = test_case.vendor_id;
  const OrtHardwareDevice* devices[] = {&hardware_device};
  OrtEpDevice* ep_devices[1] = {nullptr};
  size_t num_ep_devices = 0;
  const size_t initial_num_ep_devices = num_ep_devices;

  OrtStatus* status = factory.GetSupportedDevices(devices, 1, ep_devices, 1, &num_ep_devices);
  ASSERT_EQ(status, nullptr) << test_case.name;

  if (test_case.expected_eligible) {
    ASSERT_EQ(num_ep_devices, initial_num_ep_devices + 1) << test_case.name;
    ASSERT_NE(ep_devices[0], nullptr) << test_case.name;
    EXPECT_EQ(ort_api.EpDevice_Device(ep_devices[0]), &hardware_device);

    if (test_case.type == OrtHardwareDeviceType_CPU) {
      const OrtKeyValuePairs* metadata = ort_api.EpDevice_EpMetadata(ep_devices[0]);
      ASSERT_NE(metadata, nullptr);
      const char* ov_device = ort_api.GetKeyValue(metadata, openvino_ep::OpenVINOEpPluginFactory::ov_device_key_);
      ASSERT_NE(ov_device, nullptr);
      EXPECT_STREQ(ov_device, "CPU");
    }

    ort_api.GetEpApi()->ReleaseEpDevice(ep_devices[0]);
    return;
  }

  EXPECT_EQ(num_ep_devices, initial_num_ep_devices) << test_case.name;
  EXPECT_EQ(ep_devices[0], nullptr) << test_case.name;
}

INSTANTIATE_TEST_SUITE_P(
    Hardware,
    OpenVINOGetSupportedDevicesTest,
    ::testing::Values(
        HardwareDeviceCase{"IntelCpu", OrtHardwareDeviceType_CPU, 0x8086, true},
        HardwareDeviceCase{"AmdCpu", OrtHardwareDeviceType_CPU, 0x1022, true},
        HardwareDeviceCase{"AmdGpu", OrtHardwareDeviceType_GPU, 0x1022, false},
        HardwareDeviceCase{"AmdNpu", OrtHardwareDeviceType_NPU, 0x1022, false},
        HardwareDeviceCase{"UnsupportedDevice", static_cast<OrtHardwareDeviceType>(42), 0x1022, false)),
    [](const ::testing::TestParamInfo<HardwareDeviceCase>& info) { return info.param.name; });

}  // namespace
}  // namespace test
}  // namespace onnxruntime

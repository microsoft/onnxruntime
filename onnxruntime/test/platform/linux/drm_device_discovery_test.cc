// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/drm_device_discovery.h"

#include <filesystem>
#include <fstream>
#include <string_view>

#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace fs = std::filesystem;

namespace onnxruntime::test {

namespace {

class DrmDeviceDiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    temp_dir_ = fs::temp_directory_path() / "ort_drm_discovery_test";
    fs::remove_all(temp_dir_);
    fs::create_directories(temp_dir_);
  }

  void TearDown() override {
    fs::remove_all(temp_dir_);
  }

  fs::path temp_dir_;
};

void CreatePlatformDrmCard(const fs::path& card_path, std::string_view driver) {
  fs::create_directories(card_path / "device");
  std::ofstream uevent(card_path / "device" / "uevent");
  uevent << "DRIVER=" << driver << "\n";
}

}  // namespace

TEST_F(DrmDeviceDiscoveryTest, DetectsJetsonGpuAndSkipsOtherPlatformCards) {
  CreatePlatformDrmCard(temp_dir_ / "card0", "nvgpu");
  CreatePlatformDrmCard(temp_dir_ / "card1", "amdgpu");
  CreatePlatformDrmCard(temp_dir_ / "card2", "nvgpu-extra");
  fs::create_directories(temp_dir_ / "card3" / "device");

  std::vector<drm_device_discovery::GpuSysfsPathInfo> gpu_paths;
  ASSERT_STATUS_OK(drm_device_discovery::DetectGpuSysfsPaths(temp_dir_, gpu_paths));

  ASSERT_EQ(gpu_paths.size(), 1u);
  EXPECT_EQ(gpu_paths[0].card_idx, 0u);
  EXPECT_TRUE(gpu_paths[0].is_nvidia_platform_gpu);
}

TEST_F(DrmDeviceDiscoveryTest, CreatesIntegratedNvidiaDeviceForJetsonGpu) {
  CreatePlatformDrmCard(temp_dir_ / "card1", "nvgpu");

  std::vector<drm_device_discovery::GpuSysfsPathInfo> gpu_paths;
  ASSERT_STATUS_OK(drm_device_discovery::DetectGpuSysfsPaths(temp_dir_, gpu_paths));
  ASSERT_EQ(gpu_paths.size(), 1u);

  OrtHardwareDevice gpu_device{};
  ASSERT_STATUS_OK(drm_device_discovery::GetGpuDeviceFromSysfs(gpu_paths[0], gpu_device));

  EXPECT_EQ(gpu_device.type, OrtHardwareDeviceType_GPU);
  EXPECT_EQ(gpu_device.vendor_id, 0x10deu);
  EXPECT_EQ(gpu_device.device_id, 0u);
  EXPECT_EQ(gpu_device.metadata.Entries().at("card_idx"), "1");
  EXPECT_EQ(gpu_device.metadata.Entries().at("Discrete"), "0");
  EXPECT_EQ(gpu_device.metadata.Entries().find("pci_bus_id"), gpu_device.metadata.Entries().end());
}

}  // namespace onnxruntime::test

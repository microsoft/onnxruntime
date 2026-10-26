// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/drm_device_discovery.h"

#include <algorithm>
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

void CreatePciDrmCard(const fs::path& card_path, std::string_view vendor_id, std::string_view driver) {
  fs::create_directories(card_path / "device");
  std::ofstream vendor(card_path / "device" / "vendor");
  vendor << vendor_id << "\n";
  std::ofstream uevent(card_path / "device" / "uevent");
  uevent << "DRIVER=" << driver << "\n";
}

void SetDeviceTreeCompatible(const fs::path& card_path, std::string_view compatible) {
  fs::create_directories(card_path / "device" / "of_node");
  std::ofstream file(card_path / "device" / "of_node" / "compatible", std::ios::binary);
  file << compatible << '\0';
}

// Directory iteration order is unspecified, so sort the detected paths by card index before checking them.
void SortByCardIdx(std::vector<drm_device_discovery::GpuSysfsPathInfo>& gpu_paths) {
  std::sort(gpu_paths.begin(), gpu_paths.end(),
            [](const auto& a, const auto& b) { return a.card_idx < b.card_idx; });
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

TEST_F(DrmDeviceDiscoveryTest, DetectsJetsonOrinPlatformGpu) {
  CreatePlatformDrmCard(temp_dir_ / "card1", "nv_platform");

  std::vector<drm_device_discovery::GpuSysfsPathInfo> gpu_paths;
  ASSERT_STATUS_OK(drm_device_discovery::DetectGpuSysfsPaths(temp_dir_, gpu_paths));

  ASSERT_EQ(gpu_paths.size(), 1u);
  EXPECT_EQ(gpu_paths[0].card_idx, 1u);
  EXPECT_TRUE(gpu_paths[0].is_nvidia_platform_gpu);
}

TEST_F(DrmDeviceDiscoveryTest, SkipsJetsonThorDisplayControllerWhenNvidiaPciGpuPresent) {
  // Jetson AGX Thor layout: card0 is host1x, card1 is the PCI GPU, card2 is the display controller.
  CreatePlatformDrmCard(temp_dir_ / "card0", "drm");
  CreatePciDrmCard(temp_dir_ / "card1", "0x10de", "nvidia");
  CreatePlatformDrmCard(temp_dir_ / "card2", "nv_platform");
  SetDeviceTreeCompatible(temp_dir_ / "card2", "nvidia,tegra264-display");

  std::vector<drm_device_discovery::GpuSysfsPathInfo> gpu_paths;
  ASSERT_STATUS_OK(drm_device_discovery::DetectGpuSysfsPaths(temp_dir_, gpu_paths));

  ASSERT_EQ(gpu_paths.size(), 1u);
  EXPECT_EQ(gpu_paths[0].card_idx, 1u);
  EXPECT_FALSE(gpu_paths[0].is_nvidia_platform_gpu);
}

TEST_F(DrmDeviceDiscoveryTest, SkipsJetsonThorDisplayControllerWithLowerCardIndexThanPciGpu) {
  CreatePlatformDrmCard(temp_dir_ / "card0", "nv_platform");
  SetDeviceTreeCompatible(temp_dir_ / "card0", "nvidia,tegra264-display");
  CreatePciDrmCard(temp_dir_ / "card1", "0x10de", "nvidia");

  std::vector<drm_device_discovery::GpuSysfsPathInfo> gpu_paths;
  ASSERT_STATUS_OK(drm_device_discovery::DetectGpuSysfsPaths(temp_dir_, gpu_paths));

  ASSERT_EQ(gpu_paths.size(), 1u);
  EXPECT_EQ(gpu_paths[0].card_idx, 1u);
  EXPECT_FALSE(gpu_paths[0].is_nvidia_platform_gpu);
}

TEST_F(DrmDeviceDiscoveryTest, KeepsJetsonOrinPlatformGpuWhenNvidiaPciGpuPresent) {
  // Jetson Orin with a discrete NVIDIA GPU, e.g. IGX Orin.
  CreatePciDrmCard(temp_dir_ / "card0", "0x10de", "nvidia");
  CreatePlatformDrmCard(temp_dir_ / "card1", "nv_platform");
  SetDeviceTreeCompatible(temp_dir_ / "card1", "nvidia,tegra234-display");

  std::vector<drm_device_discovery::GpuSysfsPathInfo> gpu_paths;
  ASSERT_STATUS_OK(drm_device_discovery::DetectGpuSysfsPaths(temp_dir_, gpu_paths));
  SortByCardIdx(gpu_paths);

  ASSERT_EQ(gpu_paths.size(), 2u);
  EXPECT_EQ(gpu_paths[0].card_idx, 0u);
  EXPECT_FALSE(gpu_paths[0].is_nvidia_platform_gpu);
  EXPECT_EQ(gpu_paths[1].card_idx, 1u);
  EXPECT_TRUE(gpu_paths[1].is_nvidia_platform_gpu);
}

TEST_F(DrmDeviceDiscoveryTest, KeepsNvidiaPlatformGpuWhenPciGpuIsNotNvidia) {
  CreatePciDrmCard(temp_dir_ / "card0", "0x1002", "amdgpu");
  CreatePlatformDrmCard(temp_dir_ / "card1", "nv_platform");
  SetDeviceTreeCompatible(temp_dir_ / "card1", "nvidia,tegra264-display");

  std::vector<drm_device_discovery::GpuSysfsPathInfo> gpu_paths;
  ASSERT_STATUS_OK(drm_device_discovery::DetectGpuSysfsPaths(temp_dir_, gpu_paths));
  SortByCardIdx(gpu_paths);

  ASSERT_EQ(gpu_paths.size(), 2u);
  EXPECT_EQ(gpu_paths[0].card_idx, 0u);
  EXPECT_FALSE(gpu_paths[0].is_nvidia_platform_gpu);
  EXPECT_EQ(gpu_paths[1].card_idx, 1u);
  EXPECT_TRUE(gpu_paths[1].is_nvidia_platform_gpu);
}

TEST_F(DrmDeviceDiscoveryTest, KeepsNvidiaPciGpuWithoutPlatformCards) {
  CreatePciDrmCard(temp_dir_ / "card0", "0x10de", "nvidia");

  std::vector<drm_device_discovery::GpuSysfsPathInfo> gpu_paths;
  ASSERT_STATUS_OK(drm_device_discovery::DetectGpuSysfsPaths(temp_dir_, gpu_paths));

  ASSERT_EQ(gpu_paths.size(), 1u);
  EXPECT_EQ(gpu_paths[0].card_idx, 0u);
  EXPECT_FALSE(gpu_paths[0].is_nvidia_platform_gpu);
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

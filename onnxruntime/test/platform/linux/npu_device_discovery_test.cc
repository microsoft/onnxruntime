// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/npu_device_discovery.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace fs = std::filesystem;

namespace onnxruntime::test {
namespace {

void WriteFile(const fs::path& path, const std::string& value) {
  std::ofstream f(path);
  f << value;
}

class NpuDeviceDiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    temp_dir_ = fs::temp_directory_path() / "ort_npu_discovery_test";
    fs::remove_all(temp_dir_);
    fs::create_directories(temp_dir_);
  }

  void TearDown() override {
    fs::remove_all(temp_dir_);
  }

  fs::path temp_dir_;
};

}  // namespace

TEST_F(NpuDeviceDiscoveryTest, ReturnsEmptyForNonexistentPath) {
  std::vector<npu_device_discovery::NpuSysfsPathInfo> npu_paths;
  ASSERT_STATUS_OK(npu_device_discovery::DetectNpuSysfsPaths(temp_dir_ / "nonexistent", npu_paths));
  EXPECT_TRUE(npu_paths.empty());
}

TEST_F(NpuDeviceDiscoveryTest, DetectsAccelDevices) {
  fs::create_directories(temp_dir_ / "accel0");
  fs::create_directories(temp_dir_ / "accel12");
  fs::create_directories(temp_dir_ / "renderD128");
  fs::create_directories(temp_dir_ / "accelabc");

  std::vector<npu_device_discovery::NpuSysfsPathInfo> npu_paths;
  ASSERT_STATUS_OK(npu_device_discovery::DetectNpuSysfsPaths(temp_dir_, npu_paths));

  ASSERT_EQ(npu_paths.size(), 2u);

  std::vector<size_t> accel_indices;
  accel_indices.reserve(npu_paths.size());
  for (const auto& npu_path : npu_paths) {
    accel_indices.push_back(npu_path.accel_idx);
  }

  std::sort(accel_indices.begin(), accel_indices.end());

  EXPECT_EQ(accel_indices[0], 0u);
  EXPECT_EQ(accel_indices[1], 12u);
}

TEST_F(NpuDeviceDiscoveryTest, GetNpuDeviceFromSysfsReadsVendorDeviceAndMetadata) {
  const auto pci_device_dir = temp_dir_ / "pci_devices" / "0000:65:00.0";
  const auto accel_dir = temp_dir_ / "class_accel" / "accel0";

  fs::create_directories(pci_device_dir);
  fs::create_directories(accel_dir);

  WriteFile(pci_device_dir / "vendor", "0x1022");
  WriteFile(pci_device_dir / "device", "0x1502");

  std::error_code error_code{};
  fs::create_directory_symlink(pci_device_dir, accel_dir / "device", error_code);
  ASSERT_FALSE(error_code) << error_code.message();

  npu_device_discovery::NpuSysfsPathInfo path_info{};
  path_info.accel_idx = 0;
  path_info.path = accel_dir;

  OrtHardwareDevice npu_device{};
  ASSERT_STATUS_OK(npu_device_discovery::GetNpuDeviceFromSysfs(path_info, npu_device));

  EXPECT_EQ(npu_device.type, OrtHardwareDeviceType_NPU);
  EXPECT_EQ(npu_device.vendor_id, 0x1022u);
  EXPECT_EQ(npu_device.device_id, 0x1502u);

  const auto& entries = npu_device.metadata.Entries();
  EXPECT_NE(entries.find("accel_idx"), entries.end());
  EXPECT_EQ(entries.at("accel_idx"), "0");
  EXPECT_NE(entries.find("pci_bus_id"), entries.end());
  EXPECT_EQ(entries.at("pci_bus_id"), "0000:65:00.0");
}

}  // namespace onnxruntime::test

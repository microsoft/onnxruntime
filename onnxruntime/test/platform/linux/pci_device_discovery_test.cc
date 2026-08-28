// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/pci_device_discovery.h"

#include <filesystem>
#include <fstream>

#include "test/util/include/asserts.h"
#include "gtest/gtest.h"

namespace fs = std::filesystem;

namespace onnxruntime::test {

namespace {

// Helper to create a fake PCI device directory with the given class, vendor, and device files.
void CreateFakePciDevice(const fs::path& device_dir, const std::string& pci_class,
                         const std::string& vendor, const std::string& device) {
  fs::create_directories(device_dir);
  {
    std::ofstream f(device_dir / "class");
    f << pci_class;
  }
  {
    std::ofstream f(device_dir / "vendor");
    f << vendor;
  }
  {
    std::ofstream f(device_dir / "device");
    f << device;
  }
}

class PciDeviceDiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    temp_dir_ = fs::temp_directory_path() / "ort_pci_discovery_test";
    fs::remove_all(temp_dir_);
    fs::create_directories(temp_dir_);
  }

  void TearDown() override {
    fs::remove_all(temp_dir_);
  }

  fs::path temp_dir_;
};

}  // namespace

TEST_F(PciDeviceDiscoveryTest, DetectsNvidiaVgaController) {
  // PCI class 0x030000 = VGA compatible controller
  CreateFakePciDevice(temp_dir_ / "0000:01:00.0", "0x030000", "0x10de", "0x2204");

  std::vector<pci_device_discovery::GpuPciPathInfo> gpu_paths;
  ASSERT_STATUS_OK(pci_device_discovery::DetectGpuPciPaths(temp_dir_, gpu_paths));
  ASSERT_EQ(gpu_paths.size(), 1u);
  EXPECT_EQ(gpu_paths[0].pci_bus_id, "0000:01:00.0");
}

TEST_F(PciDeviceDiscoveryTest, DetectsNvidia3DController) {
  // PCI class 0x030200 = 3D controller (common for NVIDIA datacenter GPUs like A100/H100)
  CreateFakePciDevice(temp_dir_ / "0000:65:00.0", "0x030200", "0x10de", "0x20b5");

  std::vector<pci_device_discovery::GpuPciPathInfo> gpu_paths;
  ASSERT_STATUS_OK(pci_device_discovery::DetectGpuPciPaths(temp_dir_, gpu_paths));
  ASSERT_EQ(gpu_paths.size(), 1u);
  EXPECT_EQ(gpu_paths[0].pci_bus_id, "0000:65:00.0");
}

TEST_F(PciDeviceDiscoveryTest, FiltersOutNonGpuDevices) {
  // PCI class 0x020000 = Network controller (should be skipped)
  CreateFakePciDevice(temp_dir_ / "0000:02:00.0", "0x020000", "0x8086", "0x1533");
  // PCI class 0x010600 = SATA controller (should be skipped)
  CreateFakePciDevice(temp_dir_ / "0000:00:1f.2", "0x010600", "0x8086", "0xa102");
  // PCI class 0x030000 = VGA controller (should be detected)
  CreateFakePciDevice(temp_dir_ / "0000:01:00.0", "0x030000", "0x10de", "0x2204");

  std::vector<pci_device_discovery::GpuPciPathInfo> gpu_paths;
  ASSERT_STATUS_OK(pci_device_discovery::DetectGpuPciPaths(temp_dir_, gpu_paths));
  ASSERT_EQ(gpu_paths.size(), 1u);
  EXPECT_EQ(gpu_paths[0].pci_bus_id, "0000:01:00.0");
}

TEST_F(PciDeviceDiscoveryTest, ReturnsEmptyForNonexistentPath) {
  std::vector<pci_device_discovery::GpuPciPathInfo> gpu_paths;
  ASSERT_STATUS_OK(pci_device_discovery::DetectGpuPciPaths(temp_dir_ / "nonexistent", gpu_paths));
  EXPECT_TRUE(gpu_paths.empty());
}

TEST_F(PciDeviceDiscoveryTest, ReturnsEmptyForEmptyDirectory) {
  std::vector<pci_device_discovery::GpuPciPathInfo> gpu_paths;
  ASSERT_STATUS_OK(pci_device_discovery::DetectGpuPciPaths(temp_dir_, gpu_paths));
  EXPECT_TRUE(gpu_paths.empty());
}

TEST_F(PciDeviceDiscoveryTest, DetectsMultipleGpus) {
  // Two NVIDIA GPUs
  CreateFakePciDevice(temp_dir_ / "0000:01:00.0", "0x030200", "0x10de", "0x20b5");
  CreateFakePciDevice(temp_dir_ / "0000:41:00.0", "0x030200", "0x10de", "0x20b5");

  std::vector<pci_device_discovery::GpuPciPathInfo> gpu_paths;
  ASSERT_STATUS_OK(pci_device_discovery::DetectGpuPciPaths(temp_dir_, gpu_paths));
  EXPECT_EQ(gpu_paths.size(), 2u);
}

TEST_F(PciDeviceDiscoveryTest, SkipsDevicesWithMissingClassFile) {
  // Device directory without a class file
  auto device_dir = temp_dir_ / "0000:03:00.0";
  fs::create_directories(device_dir);
  {
    std::ofstream f(device_dir / "vendor");
    f << "0x10de";
  }

  std::vector<pci_device_discovery::GpuPciPathInfo> gpu_paths;
  ASSERT_STATUS_OK(pci_device_discovery::DetectGpuPciPaths(temp_dir_, gpu_paths));
  EXPECT_TRUE(gpu_paths.empty());
}

TEST_F(PciDeviceDiscoveryTest, GetGpuDeviceFromPciReadsVendorAndDevice) {
  // Create a fake NVIDIA GPU PCI device
  CreateFakePciDevice(temp_dir_ / "0000:65:00.0", "0x030200", "0x10de", "0x20b5");

  pci_device_discovery::GpuPciPathInfo path_info;
  path_info.path = temp_dir_ / "0000:65:00.0";
  path_info.pci_bus_id = "0000:65:00.0";

  OrtHardwareDevice gpu_device{};
  ASSERT_STATUS_OK(pci_device_discovery::GetGpuDeviceFromPci(path_info, gpu_device));

  EXPECT_EQ(gpu_device.type, OrtHardwareDeviceType_GPU);
  EXPECT_EQ(gpu_device.vendor_id, 0x10deu);
  EXPECT_EQ(gpu_device.device_id, 0x20b5u);

  const auto& entries = gpu_device.metadata.Entries();
  EXPECT_NE(entries.find("pci_bus_id"), entries.end());
  EXPECT_EQ(entries.at("pci_bus_id"), "0000:65:00.0");
  EXPECT_NE(entries.find("Discrete"), entries.end());
  EXPECT_EQ(entries.at("Discrete"), "1");
}

TEST_F(PciDeviceDiscoveryTest, GetGpuDeviceFromPciNonNvidiaVendor) {
  // Create a fake AMD GPU PCI device
  CreateFakePciDevice(temp_dir_ / "0000:03:00.0", "0x030000", "0x1002", "0x731f");

  pci_device_discovery::GpuPciPathInfo path_info;
  path_info.path = temp_dir_ / "0000:03:00.0";
  path_info.pci_bus_id = "0000:03:00.0";

  OrtHardwareDevice gpu_device{};
  ASSERT_STATUS_OK(pci_device_discovery::GetGpuDeviceFromPci(path_info, gpu_device));

  EXPECT_EQ(gpu_device.type, OrtHardwareDeviceType_GPU);
  EXPECT_EQ(gpu_device.vendor_id, 0x1002u);
  EXPECT_EQ(gpu_device.device_id, 0x731fu);

  const auto& entries = gpu_device.metadata.Entries();
  // Non-NVIDIA vendor should not have the Discrete metadata entry
  EXPECT_EQ(entries.find("Discrete"), entries.end());
}

}  // namespace onnxruntime::test

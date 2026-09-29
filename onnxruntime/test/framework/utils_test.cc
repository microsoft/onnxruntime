// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/framework/utils.h"

namespace onnxruntime {
namespace test {

constexpr OrtDevice::VendorId kTestVendor1 = 0x1234;
constexpr OrtDevice::VendorId kTestVendor2 = 0x5678;

static OrtDevice Cpu() {
  return OrtDevice{OrtDevice::CPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0};
}

static OrtDevice HostAccessible(OrtDevice::VendorId vendor, OrtDevice::DeviceId id,
                                OrtDevice::Alignment align = 0) {
  return OrtDevice{OrtDevice::NPU, OrtDevice::MemType::HOST_ACCESSIBLE, vendor, id, align};
}

static OrtDevice Default(OrtDevice::VendorId vendor, OrtDevice::DeviceId id,
                         OrtDevice::Alignment align = 0) {
  return OrtDevice{OrtDevice::NPU, OrtDevice::MemType::DEFAULT, vendor, id, align};
}

TEST(CanSourceSatisfyTargetTest, CpuSourceHostAccessibleTarget) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(Cpu(), HostAccessible(kTestVendor1, 0)));
}

TEST(CanSourceSatisfyTargetTest, HostAccessibleSourceCpuTarget) {
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(HostAccessible(kTestVendor1, 0), Cpu()));
}

// src == tgt early return: identical devices are always compatible
TEST(CanSourceSatisfyTargetTest, BothHostAccessibleSameDevice) {
  auto dev = HostAccessible(kTestVendor1, 0, 16);
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(dev, dev));
}

TEST(CanSourceSatisfyTargetTest, BothHostAccessibleDifferentId) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0), HostAccessible(kTestVendor1, 1)));
}

TEST(CanSourceSatisfyTargetTest, BothHostAccessibleDifferentVendor) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0), HostAccessible(kTestVendor2, 0)));
}

TEST(CanSourceSatisfyTargetTest, HostAccessibleToDefaultSameDevice) {
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0), Default(kTestVendor1, 0)));
}

TEST(CanSourceSatisfyTargetTest, HostAccessibleToDefaultAlignmentSatisfied) {
  // src alignment >= tgt alignment: compatible
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0, 64), Default(kTestVendor1, 0, 32)));
}

TEST(CanSourceSatisfyTargetTest, HostAccessibleToDefaultAlignmentInsufficient) {
  // src alignment < tgt alignment: incompatible
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0, 16), Default(kTestVendor1, 0, 64)));
}

TEST(CanSourceSatisfyTargetTest, HostAccessibleToDefaultSrcAlignmentZero) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0, 0), Default(kTestVendor1, 0, 64)));
}

TEST(CanSourceSatisfyTargetTest, HostAccessibleToDefaultTgtAlignmentZero) {
  // 0 = unspecified, treated as wildcard
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0, 16), Default(kTestVendor1, 0, 0)));
}

TEST(CanSourceSatisfyTargetTest, HostAccessibleToDefaultDifferentDeviceId) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0), Default(kTestVendor1, 1)));
}

TEST(CanSourceSatisfyTargetTest, DefaultToHostAccessibleRejected) {
  // Reversed direction: CPU cannot read DEFAULT (device-only) memory
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      Default(kTestVendor1, 0), HostAccessible(kTestVendor1, 0)));
}

TEST(CanSourceSatisfyTargetTest, DefaultToDefaultRejected) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      Default(kTestVendor1, 0), Default(kTestVendor2, 0)));
}

// Early return: identical CPU devices are always compatible.
TEST(CanSourceSatisfyTargetTest, CpuToCpuIdentical) {
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(Cpu(), Cpu()));
}

// Early return: identical DEFAULT devices on the same physical device are compatible.
TEST(CanSourceSatisfyTargetTest, DefaultToDefaultSameDevice) {
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(Default(kTestVendor1, 0), Default(kTestVendor1, 0)));
}

// Both HOST_ACCESSIBLE, same physical device — alignment variations (not the early-return path
// because src and tgt differ in alignment, so src != tgt).
TEST(CanSourceSatisfyTargetTest, BothHostAccessibleSameDeviceAlignmentSatisfied) {
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0, 64), HostAccessible(kTestVendor1, 0, 32)));
}

TEST(CanSourceSatisfyTargetTest, BothHostAccessibleSameDeviceAlignmentInsufficient) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0, 16), HostAccessible(kTestVendor1, 0, 32)));
}

TEST(CanSourceSatisfyTargetTest, BothHostAccessibleSameDeviceSrcAlignmentZero) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0, 0), HostAccessible(kTestVendor1, 0, 32)));
}

// HOST_ACCESSIBLE → DEFAULT: same device id but different vendor fails is_same_physical_device.
TEST(CanSourceSatisfyTargetTest, HostAccessibleToDefaultDifferentVendor) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0), Default(kTestVendor2, 0)));
}

}  // namespace test
}  // namespace onnxruntime

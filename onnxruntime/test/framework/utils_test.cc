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

// Branch 3: both HOST_ACCESSIBLE, different physical device
TEST(CanSourceSatisfyTargetTest, BothHostAccessibleDifferentId) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0), HostAccessible(kTestVendor1, 1)));
}

TEST(CanSourceSatisfyTargetTest, BothHostAccessibleDifferentVendor) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0), HostAccessible(kTestVendor2, 0)));
}

TEST(CanSourceSatisfyTargetTest, BothHostAccessibleDifferentAlignment) {
  // Different alignment => OrtDevice::operator== returns false
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      HostAccessible(kTestVendor1, 0, 16), HostAccessible(kTestVendor1, 0, 32)));
}

// Branch 4: HOST_ACCESSIBLE (src) -> DEFAULT (tgt), same physical device
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
  // 0 = unspecified, treated as wildcard
  EXPECT_TRUE(utils::CanSourceSatisfyTarget(
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

// Branch 5: incompatible cases

TEST(CanSourceSatisfyTargetTest, DefaultToHostAccessibleRejected) {
  // Reversed direction: CPU cannot read DEFAULT (device-only) memory
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      Default(kTestVendor1, 0), HostAccessible(kTestVendor1, 0)));
}

TEST(CanSourceSatisfyTargetTest, DefaultToDefaultRejected) {
  EXPECT_FALSE(utils::CanSourceSatisfyTarget(
      Default(kTestVendor1, 0), Default(kTestVendor2, 0)));
}

}  // namespace test
}  // namespace onnxruntime

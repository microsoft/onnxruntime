// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/framework/ortdevice.h"
#include "core/providers/webgpu/data_transfer.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr OrtDevice Cpu() {
  return OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0);
}

// WebGPU buffers are allocated as generic GPU memory with no vendor id (see webgpu::WebGpuDevice).
constexpr OrtDevice WebGpu(OrtDevice::DeviceId device_id = 0) {
  return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, device_id);
}

constexpr OrtDevice VendorGpu(OrtDevice::VendorId vendor_id) {
  return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, vendor_id, 0);
}

bool CanCopy(const OrtDevice& src, const OrtDevice& dst, int context_id = 0) {
  return webgpu::DataTransfer::IsSupportedDevicePair(src, dst, context_id);
}

}  // namespace

TEST(WebGpuDataTransferTest, AcceptsHostAndWebGpuEndpoints) {
  EXPECT_TRUE(CanCopy(Cpu(), WebGpu()));
  EXPECT_TRUE(CanCopy(WebGpu(), Cpu()));
  EXPECT_TRUE(CanCopy(WebGpu(), WebGpu()));
  EXPECT_FALSE(CanCopy(WebGpu(1), WebGpu(0)));
}

TEST(WebGpuDataTransferTest, RestrictsCopiesToOwningContext) {
  for (const auto context_id : {OrtDevice::DeviceId{0}, OrtDevice::DeviceId{1}, OrtDevice::DeviceId{32767}}) {
    const auto own = WebGpu(context_id);
    const auto other = WebGpu(context_id == 0 ? 1 : 0);
    EXPECT_TRUE(CanCopy(Cpu(), own, context_id));
    EXPECT_TRUE(CanCopy(own, Cpu(), context_id));
    EXPECT_TRUE(CanCopy(own, own, context_id));
    EXPECT_FALSE(CanCopy(Cpu(), other, context_id));
    EXPECT_FALSE(CanCopy(other, Cpu(), context_id));
    EXPECT_FALSE(CanCopy(own, other, context_id));
    EXPECT_FALSE(CanCopy(other, own, context_id));
    EXPECT_FALSE(CanCopy(other, other, context_id));
  }
}

TEST(WebGpuDataTransferTest, RejectsHostToHost) {
  // CPU-to-CPU copies belong to the CPU data transfer, not this one.
  EXPECT_FALSE(CanCopy(Cpu(), Cpu()));
}

TEST(WebGpuDataTransferTest, RejectsForeignGpuVendors) {
  // A vendor-tagged GPU handle belongs to another EP, so this transfer must not claim it.
  for (const auto vendor_id : {OrtDevice::VendorIds::NVIDIA,
                               OrtDevice::VendorIds::AMD,
                               OrtDevice::VendorIds::INTEL,
                               OrtDevice::VendorIds::MICROSOFT}) {
    const OrtDevice foreign = VendorGpu(vendor_id);
    EXPECT_FALSE(CanCopy(Cpu(), foreign)) << "vendor " << vendor_id;
    EXPECT_FALSE(CanCopy(foreign, Cpu())) << "vendor " << vendor_id;
    EXPECT_FALSE(CanCopy(WebGpu(), foreign)) << "vendor " << vendor_id;
    EXPECT_FALSE(CanCopy(foreign, WebGpu())) << "vendor " << vendor_id;
    EXPECT_FALSE(CanCopy(foreign, foreign)) << "vendor " << vendor_id;
  }
}

}  // namespace test
}  // namespace onnxruntime

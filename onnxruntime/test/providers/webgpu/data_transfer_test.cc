// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/framework/ortdevice.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "test/util/include/asserts.h"

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

TEST(WebGpuDataTransferTest, BatchedUploadsPreserveDataAcrossChunksAndReuse) {
  ConfigOptions options;
  auto provider = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& ep = static_cast<WebGpuExecutionProvider&>(*provider);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = ep.Recording();
  CPUAllocator cpu_allocator;
  const OrtMemoryInfo gpu_info(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(0), OrtMemTypeDefault);
  webgpu::DataTransfer transfer(ep.BufferManager(), recording, 0);

  // Include empty/unaligned tensors, a batch boundary, and an upload larger than the batch limit.
  const std::vector<int64_t> sizes{0, 1, 3, 4, 31, 65, 4097, 9 * 1024 * 1024 + 1,
                                   9 * 1024 * 1024 + 3, 17 * 1024 * 1024 + 1};
  std::vector<std::vector<uint8_t>> data;
  std::vector<wgpu::Buffer> buffers;
  std::vector<Tensor> sources, targets;
  sources.reserve(sizes.size());
  targets.reserve(sizes.size());
  for (const auto size : sizes) {
    data.emplace_back(static_cast<size_t>(size));
    wgpu::BufferDescriptor desc{};
    // Match BufferManager's allocation alignment, which Download also relies on.
    desc.size = std::max<uint64_t>(16, (static_cast<uint64_t>(size) + 15) / 16 * 16);
    desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    buffers.push_back(context.Device().CreateBuffer(&desc));
    sources.emplace_back(DataTypeImpl::GetType<uint8_t>(), TensorShape({size}), data.back().data(), cpu_allocator.Info());
    targets.emplace_back(DataTypeImpl::GetType<uint8_t>(), TensorShape({size}), buffers.back().Get(), gpu_info);
  }
  std::vector<IDataTransfer::SrcDstPair> uploads;
  for (size_t i = 0; i < sizes.size(); ++i) uploads.push_back({sources[i], targets[i], nullptr});
  for (size_t pass = 0; pass < 2; ++pass) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      for (size_t j = 0; j < data[i].size(); ++j) data[i][j] = static_cast<uint8_t>(i * 17 + j * 7 + pass);
    }
    ASSERT_STATUS_OK(transfer.CopyTensors(uploads));
    for (size_t i = 0; i < sizes.size(); ++i) {
      std::vector<uint8_t> actual(static_cast<size_t>(sizes[i]));
      Tensor result(DataTypeImpl::GetType<uint8_t>(), TensorShape({sizes[i]}), actual.data(), cpu_allocator.Info());
      ASSERT_STATUS_OK(transfer.CopyTensor(targets[i], result));
      EXPECT_EQ(actual, data[i]) << "tensor " << i << " pass " << pass;
    }
  }
  ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), recording));
}

}  // namespace test
}  // namespace onnxruntime

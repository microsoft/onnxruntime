// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>

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

TEST(WebGpuDataTransferTest, BatchedUploadsRetainMappedDestinationFallback) {
  ConfigOptions options;
  auto provider = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& ep = static_cast<WebGpuExecutionProvider&>(*provider);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = ep.Recording();
  CPUAllocator cpu_allocator;
  const OrtMemoryInfo gpu_info(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(0), OrtMemTypeDefault);
  webgpu::DataTransfer transfer(ep.BufferManager(), recording, 0);

  // Exercise a mapped destination first, last, and at both positions, regardless of UMA support.
  for (const auto mapped_destinations : {1U, 2U, 3U}) {
    SCOPED_TRACE(mapped_destinations);
    std::array<std::array<uint8_t, 16>, 2> data{};
    std::array<wgpu::Buffer, 2> buffers;
    InlinedVector<Tensor> sources, targets;
    sources.reserve(data.size());
    targets.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      wgpu::BufferDescriptor desc{};
      desc.size = data[i].size();
      desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
      desc.mappedAtCreation = (mapped_destinations & (1U << i)) != 0;
      buffers[i] = context.Device().CreateBuffer(&desc);
      ASSERT_EQ(buffers[i].GetMapState(),
                desc.mappedAtCreation ? wgpu::BufferMapState::Mapped : wgpu::BufferMapState::Unmapped);
      sources.emplace_back(DataTypeImpl::GetType<uint8_t>(), TensorShape({16}), data[i].data(), cpu_allocator.Info());
      targets.emplace_back(DataTypeImpl::GetType<uint8_t>(), TensorShape({16}), buffers[i].Get(), gpu_info);
    }

    const std::vector<IDataTransfer::SrcDstPair> uploads{{sources[0], targets[0], nullptr},
                                                         {sources[1], targets[1], nullptr}};
    for (size_t pass = 0; pass < 2; ++pass) {
      SCOPED_TRACE(pass);
      for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) data[i][j] = static_cast<uint8_t>(i * 17 + j * 7 + pass);
      }

      context.PushErrorScope();
      const auto copy_status = transfer.CopyTensors(uploads);
      const auto validation_status = context.PopErrorScope();
      ASSERT_STATUS_OK(copy_status);
      ASSERT_STATUS_OK(validation_status);
      for (size_t i = 0; i < data.size(); ++i) {
        ASSERT_EQ(buffers[i].GetMapState(), wgpu::BufferMapState::Unmapped);
        std::array<uint8_t, 16> actual{};
        Tensor result(DataTypeImpl::GetType<uint8_t>(), TensorShape({16}), actual.data(), cpu_allocator.Info());
        ASSERT_STATUS_OK(transfer.CopyTensor(targets[i], result));
        EXPECT_EQ(actual, data[i]) << "tensor " << i;
      }
    }
  }
  ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), recording));
}

TEST(WebGpuDataTransferTest, BatchedAndSingleUploadsPreserveRecordedReaders) {
  ConfigOptions options;
  auto provider = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& ep = static_cast<WebGpuExecutionProvider&>(*provider);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = ep.Recording();
  auto& manager = ep.BufferManager();
  CPUAllocator cpu_allocator;
  const OrtMemoryInfo gpu_info(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(0), OrtMemTypeDefault);
  webgpu::DataTransfer transfer(manager, recording, 0);
  std::array<std::array<uint32_t, 16>, 2> data{};
  std::array<wgpu::Buffer, 2> buffers, snapshots;
  InlinedVector<Tensor> sources, targets;
  sources.reserve(data.size());
  targets.reserve(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    wgpu::BufferDescriptor desc{};
    desc.size = sizeof(data[i]);
    desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    buffers[i] = context.Device().CreateBuffer(&desc);
    snapshots[i] = context.Device().CreateBuffer(&desc);
    sources.emplace_back(DataTypeImpl::GetType<uint32_t>(), TensorShape({16}), data[i].data(), cpu_allocator.Info());
    targets.emplace_back(DataTypeImpl::GetType<uint32_t>(), TensorShape({16}), buffers[i].Get(), gpu_info);
    data[i].fill(static_cast<uint32_t>(i + 7));
  }
  const std::vector<IDataTransfer::SrcDstPair> uploads{{sources[0], targets[0], nullptr},
                                                       {sources[1], targets[1], nullptr}};
  ASSERT_STATUS_OK(transfer.CopyTensors(uploads));
  for (const bool batched : {false, true, false, true}) {
    SCOPED_TRACE(batched);
    const auto previous = data;
    for (size_t i = 0; i < data.size(); ++i) {
      manager.MemCpy(recording, buffers[i].Get(), snapshots[i].Get(), sizeof(data[i]));
      data[i].fill(previous[i][0] + 11);
    }
    const auto expected = data;
    if (batched) {
      ASSERT_STATUS_OK(transfer.CopyTensors(uploads));
    } else {
      for (size_t i = 0; i < data.size(); ++i) {
        ASSERT_STATUS_OK(transfer.CopyTensor(sources[i], targets[i]));
      }
    }
    for (auto& values : data) values.fill(99);
    for (size_t i = 0; i < data.size(); ++i) {
      std::array<uint32_t, 16> actual{};
      manager.Download(recording, snapshots[i].Get(), actual.data(), sizeof(actual));
      EXPECT_EQ(actual, previous[i]);
      manager.Download(recording, buffers[i].Get(), actual.data(), sizeof(actual));
      EXPECT_EQ(actual, expected[i]);
    }
    data = expected;
  }
  ASSERT_STATUS_OK(context.FlushAndWaitChecked(manager, recording));
}

}  // namespace test
}  // namespace onnxruntime

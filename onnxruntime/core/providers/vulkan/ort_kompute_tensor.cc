// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/ort_kompute_tensor.h"

namespace onnxruntime {
namespace vulkan {
KomputeTensor::KomputeTensor(VmaAllocator allocator, uint32_t size, bool allocate_device_memory)
    : Tensor(nullptr, nullptr, nullptr, size, 1, kp::Tensor::TensorDataTypes::eBool) {
  // empty to start with.
  // eBool so it's 1 byte per element. we sync the real info later.
  mSize = size;
  mDataTypeMemorySize = 1;

  // set mDevice so ~Tensor calls destroy. that should be all it's used for given we're manually managing
  // mPrimaryBuffer and mStagingBuffer.
  mDevice = std::make_shared<vk::Device>(allocator->m_hDevice);

  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;  // assuming aligned externally

  VmaAllocationCreateInfo allocInfo = {};

  if (allocate_device_memory) {
    // VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  } else {
    // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
    // 'Staging copy for upload' recommendation
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT;  // don't need to specifically map/unmap with this bit set
  }

  bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VkBuffer buffer;
  vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, &buffer, &allocation_, &allocation_info_);

  // need to use shared_ptr to match kp::Tensor implementation
  buffer_ = std::make_shared<vk::Buffer>(buffer);

  if (allocate_device_memory) {
    mPrimaryBuffer = buffer_;
  } else {
    mStagingBuffer = buffer_;
    mRawData = allocation_info_.pMappedData;
  }
}

void KomputeTensor::SyncWithOrtTensorShape(const onnxruntime::Tensor& ort_tensor) {
  const auto& shape = ort_tensor.Shape();

  switch (ort_tensor.GetElementType()) {
    case utils::ToTensorProtoElementType<float>():
      mDataType = kp::Tensor::TensorDataTypes::eFloat;
      break;
    default:
      // TODO: Figure out what we need and how this will work. e.g. there's no int64_t in Kompute but indexes
      // for operators such as Gather are int64_t. TBD if just the element size matters in which case we're fine.
      ORT_NOT_IMPLEMENTED("Unsupported data type");
  }

  mDataTypeMemorySize = narrow<uint32_t>(ort_tensor.DataType()->Size());
  mSize = narrow<uint32_t>(shape.Size());
}

void KomputeTensor::SyncWithOrtTensor(const onnxruntime::Tensor& ort_tensor) {
  ORT_ENFORCE(mStagingBuffer, "SyncWithOrtTensor called on instance that does not have a staging buffer.");

  SyncWithOrtTensor(ort_tensor);

  gsl::copy(gsl::make_span(static_cast<const std::byte*>(ort_tensor.DataRaw()), ort_tensor.SizeInBytes()),
            gsl::make_span(static_cast<std::byte*>(mRawData), mSize * mDataTypeMemorySize));
}

void KomputeTensor::CopyToOrtTensor(onnxruntime::Tensor& ort_tensor) const {
  ORT_ENFORCE(mStagingBuffer, "CopyToOrtTensor called on instance that does not have a staging buffer.");

  auto dst_size = ort_tensor.SizeInBytes();
  ORT_ENFORCE(dst_size >= mSize * mDataTypeMemorySize, "Size mismatch");
  gsl::copy(gsl::make_span(static_cast<const std::byte*>(mRawData), mSize * mDataTypeMemorySize),
            gsl::make_span(static_cast<std::byte*>(ort_tensor.MutableDataRaw()), dst_size));
}

void KomputeTensor::destroy() {
  ORT_ENFORCE(buffer_ && allocation_, "destroy was previously called.");
  vmaDestroyBuffer(allocator_, *buffer_, allocation_);
  buffer_ = nullptr;
  allocation_ = nullptr;

  // mRawData = nullptr;
  // mSize = 0;
  // mDataTypeMemorySize = 0;
  // mDevice = nullptr;
  Tensor::destroy();
}

}  // namespace vulkan
}  // namespace onnxruntime

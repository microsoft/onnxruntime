// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "opencl_allocator.h"
#include "opencl_utils.h"

#include <iostream>

namespace onnxruntime {
namespace opencl {

OpenCLBufferAllocator::OpenCLBufferAllocator(const cl::Context& ctx)
    : IAllocator(OrtMemoryInfo(BufferAllocatorName, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, CLMemType::OPENCL_BUFFER, /*device_id_=*/0))),
      ctx_(ctx) {
}

OpenCLBufferAllocator::~OpenCLBufferAllocator() {
  for (auto& [ptr, md] : meta_) {
    delete reinterpret_cast<cl::Buffer*>(ptr);
  }
}

void* OpenCLBufferAllocator::Alloc(size_t size) {
  auto it = cache_.find(size);

  if (it == cache_.end() || it->second.empty()) {
    cl_int err{};
    auto* ptr = new cl::Buffer(ctx_, CL_MEM_READ_WRITE, size, nullptr, &err);
    OPENCL_CHECK_ERROR(err);
    std::cerr << "OpenCL allocated " << ptr << " --> cl::Buffer(" << ptr->operator()() << ") in cl::Context(" << ctx_() << ")" << std::endl;
    meta_[ptr] = {size, MemoryKind::Buffer};
    return ptr;
  }

  auto* ptr = it->second.front();
  std::cerr << "OpenCL reused " << ptr << std::endl;
  it->second.pop_front();
  return ptr;
}

void OpenCLBufferAllocator::Free(void* p) {
  auto meta = meta_[p];
  auto it = cache_.find(meta.size);
  if (it == cache_.end()) {
    it = cache_.insert({meta.size, {}}).first;
  }
  it->second.push_front(p);
}

OpenCLImage2DAllocator::OpenCLImage2DAllocator(const cl::Context& ctx)
    : IAllocator(OrtMemoryInfo(Image2DAllocatorName, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, CLMemType::OPENCL_IMAGE_2D, /*device_id_=*/0),
                               /*id_*/ 0,
                               /*mem_type_=*/(OrtMemType)CLMemType::OPENCL_IMAGE_2D)),
      ctx_(ctx) {
}

OpenCLImage2DAllocator::~OpenCLImage2DAllocator() {
  for (auto& [ptr, md] : meta_) {
    delete reinterpret_cast<cl::Image2D*>(ptr);
  }
}

void* OpenCLImage2DAllocator::Alloc(size_t) {
  // not supported
  return nullptr;
}

void* OpenCLImage2DAllocator::Alloc(const TensorShape& shape) {
  auto it = cache_.find(shape);

  if (it == cache_.end() || it->second.empty()) {
    // TODO: support CL_HALF_FLOAT?
    cl_int err{};
    auto desc = Image2DDesc::PackFromTensor(shape);
    // FIXME: range limit is for NVIDIA GPU, adjust it for target gpu!
    ORT_ENFORCE(desc.Height() > 0 && desc.Height() <= 65535, "Image2D height invalid");
    ORT_ENFORCE(desc.Width() > 0 && desc.Width() <= 65535, "Image2D width invalid");
    auto ptr = new cl::Image2D(ctx_, CL_MEM_READ_WRITE, cl::ImageFormat{CL_RGBA, CL_FLOAT}, desc.Width(), desc.Height(), /*row_pitch=*/0, nullptr, &err);
    OPENCL_CHECK_ERROR(err);
    std::cerr << "OpenCL allocated " << ptr << " --> cl::Image2D(" << ptr->operator()() << ") in cl::Context(" << ctx_() << ")" << std::endl;
    meta_[ptr] = {shape, MemoryKind::Image2D};
    return ptr;
  }

  auto* ptr = it->second.front();
  std::cerr << "OpenCL reused " << ptr << std::endl;
  it->second.pop_front();
  return ptr;
}

void OpenCLImage2DAllocator::Free(void* p) {
  auto meta = meta_[p];
  auto it = cache_.find(meta.shape);
  if (it == cache_.end()) {
    it = cache_.insert({meta.shape, {}}).first;
  }
  it->second.push_front(p);
}

}  // namespace opencl
}  // namespace onnxruntime

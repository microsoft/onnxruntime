// Copyright (C) Intel Corporation
// Licensed under the MIT License
#ifdef USE_DEVICE_MEMORY
#include "core/providers/openvino/ov_allocator.h"
#include "core/providers/openvino/ov_interface.h"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace onnxruntime {

using namespace openvino_ep;

constexpr size_t default_alignment = 4096;

static inline size_t align_up(size_t size, size_t pow2_alignment) {
  return (size + pow2_alignment - 1) & ~(pow2_alignment - 1);
}

OVRTAllocator::OVRTAllocator(ov::Core& core, OrtDevice::DeviceType device_type, OrtDevice::DeviceId device_id, const char* name) : IAllocator(OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(device_type, OrtDevice::MemType::DEFAULT, device_id), device_id, OrtMemTypeCPUInput)), core_(core) {
  if (device_type == OrtDevice::NPU) {
    remote_ctx_ = core_.get_default_context("NPU").as<ov::intel_npu::level_zero::ZeroContext>();
  } else {
    ORT_THROW("Invalid device type");
  }
}

void* OVRTAllocator::Alloc(size_t size) {
  try {
    size_t alloc_size = align_up(size + sizeof(ov::Tensor*) + default_alignment, default_alignment);
    ov::Tensor* tensor = new ov::Tensor(remote_ctx_.create_host_tensor(ov::element::Type_t::u8,
                                                                       { alloc_size }));
    uintptr_t data_ptr = reinterpret_cast<uintptr_t>(tensor->data());

    ov::Tensor** ptr = reinterpret_cast<ov::Tensor**>(align_up(data_ptr + sizeof(ov::Tensor*), default_alignment));
    ptr[-1] = tensor;

    return reinterpret_cast<void*>(ptr);

  } catch (const ov::Exception& e) {
    ORT_THROW(std::string("Alloc failed: ") + e.what());
  }
  return nullptr;
}

void OVRTAllocator::Free(void* p) {
  try {
    ov::Tensor** ptr = reinterpret_cast<ov::Tensor**>(p);
    delete ptr[-1];
  } catch (const ov::Exception& e) {
    ORT_THROW(std::string("Free failed: ") + e.what());
  }
}

}  // namespace onnxruntime
#endif

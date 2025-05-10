// Copyright (C) Intel Corporation
// Licensed under the MIT License
#ifdef USE_OVEP_NPU_MEMORY
#include "core/providers/openvino/ov_allocator.h"
#include "core/providers/openvino/ov_interface.h"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace onnxruntime {

using namespace openvino_ep;

OVRTAllocator::OVRTAllocator(ov::Core& core, OrtDevice::DeviceId device_id, const char* name)
    : IAllocator(OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::CPU, OrtDevice::MemType::OPENVINO_NPU, device_id),
                               device_id)),
      core_(core) {
  remote_ctx_ = core_.get_default_context("NPU").as<ov::intel_npu::level_zero::ZeroContext>();
}

void* OVRTAllocator::Alloc(size_t size) {
  try {
    ov::Tensor* tensor = new ov::Tensor(remote_ctx_.create_host_tensor(ov::element::Type_t::u8,
                                                                       {size}));
    std::unique_lock lock(mutex_);
    allocated_.insert({tensor->data(), tensor});
    return reinterpret_cast<void*>(tensor->data());
  } catch (const ov::Exception& e) {
    ORT_THROW(std::string("Alloc failed: ") + e.what());
  }
}

void OVRTAllocator::Free(void* p) {
  try {
    std::unique_lock lock(mutex_);
    auto it = allocated_.find(p);
    if (it != allocated_.end()) {
      ov::Tensor* tensor = it->second;
      allocated_.erase(it);
      lock.unlock();
      delete tensor;
    }
  } catch (const ov::Exception& e) {
    ORT_THROW(std::string("Free failed: ") + e.what());
  }
}

}  // namespace onnxruntime
#endif

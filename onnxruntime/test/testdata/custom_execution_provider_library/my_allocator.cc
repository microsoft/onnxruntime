// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "my_allocator.h"
#include <stdint.h>

namespace onnxruntime {
MyEPAllocator::MyEPAllocator(OrtDevice::DeviceId device_id)
    : IAllocator(OrtMemoryInfo(MyEP, OrtAllocatorType::OrtArenaAllocator,
                               OrtDevice(MyEPDevice, OrtDevice::MemType::DEFAULT, device_id))) {
}

void* MyEPAllocator::Alloc(size_t size) {
  void* device_address = new (std::nothrow) uint8_t[size];
  return device_address;
}

void MyEPAllocator::Free(void* p) {
  delete[] reinterpret_cast<uint8_t*>(p);
}

}  // namespace onnxruntime

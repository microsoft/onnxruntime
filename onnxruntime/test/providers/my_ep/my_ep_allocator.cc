// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "my_ep_allocator.h"
#include "my_allocator.h"
#include <stdint.h>

namespace onnxruntime {
MyEPAllocator::MyEPAllocator(OrtDevice::DeviceId device_id)
    : IAllocator(OrtMemoryInfo(MyEP, OrtAllocatorType::OrtArenaAllocator,
                               OrtDevice(MyEPDevice, OrtDevice::MemType::DEFAULT, device_id))) {
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26400)
#pragma warning(disable : 26409)
#endif
void* MyEPAllocator::Alloc(size_t size) {
  return my_kernel_lib::my_alloc(size);
}

void MyEPAllocator::Free(void* p) {
  my_kernel_lib::my_free(p);
}
}  // namespace onnxruntime

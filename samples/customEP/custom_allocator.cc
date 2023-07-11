#include "custom_allocator.h"

namespace onnxruntime {

CustomAllocator::CustomAllocator(OrtDevice::DeviceId device_id) : IAllocator(OrtMemoryInfo("CustomAllocator", OrtAllocatorType::OrtArenaAllocator, OrtDevice(11, OrtDevice::MemType::DEFAULT, device_id))) {}

void* CustomAllocator::Alloc(size_t size) {
  void* device_address = new (std::nothrow) uint8_t[size];
  return device_address;
}

void CustomAllocator::Free(void* p) {
  delete[] reinterpret_cast<uint8_t*>(p);
}

}

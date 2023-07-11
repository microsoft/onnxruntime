#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {

class CustomAllocator : public IAllocator {
public:
  CustomAllocator(OrtDevice::DeviceId device_id);

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
};

}

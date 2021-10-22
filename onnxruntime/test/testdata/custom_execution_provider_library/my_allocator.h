// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
constexpr const char* MyEP = "MyEP";
static const OrtDevice::DeviceType MyEPDevice = 11;
}  // namespace onnxruntime

namespace onnxruntime {

class MyEPAllocator : public IAllocator {
 public:
  MyEPAllocator(OrtDevice::DeviceId device_id);

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;

};

}  // namespace onnxruntime

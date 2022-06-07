// Copyright (c) Microsoft Corporation. All rights reserved.
// Confidential and Proprietary.

#pragma once

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
constexpr const char* kMyProvider = "MyProvider";
}  // namespace onnxruntime

namespace onnxruntime {

struct MyProviderInfo {
  OrtDevice::DeviceId device_id{0};
  std::string some_config;
};

class MyExecutionProvider : public IExecutionProvider {
 public:
  explicit MyExecutionProvider(const MyProviderInfo& info);
  virtual ~MyExecutionProvider() {}

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  OrtDevice::DeviceId device_id_;
};

}  // namespace onnxruntime

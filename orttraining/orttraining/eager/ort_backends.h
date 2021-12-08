// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ort_eager_common.h"
#include <core/framework/ort_value.h>
#include "core/common/status.h"
#include <core/framework/provider_options.h>
#include <core/eager/ort_kernel_invoker.h>
#include <core/graph/schema_registry.h>
#include "onnx/defs/schema.h"
#include <core/graph/model.h>

namespace torch_ort {
namespace eager {

using ProviderInfoMap = std::unordered_map<std::string, onnxruntime::ProviderOptions >;

class ORTBackendsManager {
public:
  ORTBackendsManager(const onnxruntime::logging::Logger& logger);

  onnxruntime::Status set_device(size_t device_index, const std::string& provider_type,
                                 const onnxruntime::ProviderOptions& provider_options);

  onnxruntime::ORTInvoker& GetInvoker(const at::Device device);

  OrtDevice GetOrtDeviceInfo(size_t torch_device_index);

  size_t GetOrtDeviceIndex(const OrtMemoryInfo& ort_memory_info);

  const ProviderInfoMap& GetOrtDeviceProviderInfo(size_t torch_device_index) const;

private:
  std::map<at::DeviceIndex, std::unique_ptr<onnxruntime::ORTInvoker>> backends_;
  const onnxruntime::logging::Logger& logger_;
  //custom op schema registry
  //TODO: we might want to support load custom op schema on the fly
  onnxruntime::IOnnxRuntimeOpSchemaRegistryList custom_op_schema_ = {};

  // record the device associated provider information, so ortmodule can restore the ep
  std::unordered_map<at::DeviceIndex, ProviderInfoMap> device_ep_info_;
};

ORTBackendsManager& GetORTBackendsManager();

onnxruntime::ORTInvoker& GetORTInvoker(const at::Device device);

} // namespace eager
} // namespace torch_ort
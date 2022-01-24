// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>
#include <core/providers/cpu/cpu_provider_factory_creator.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/providers/get_execution_providers.h>
#include "ort_backends.h"
#include "ort_log.h"
#include "core/platform/env.h"
#include "orttraining/python/orttraining_python_module_eager.h"


//use the environment from python module
namespace onnxruntime{
namespace python{
  Environment& GetTrainingORTEnv();
  std::shared_ptr<IExecutionProvider> GetOrCreateExecutionProvider(const std::string& provider_type,
                                                                 const ProviderOptionsMap& provider_options_map,
                                                                 const SessionOptions& session_options);
}
}

namespace torch_ort {
namespace eager {

using namespace onnxruntime;
using namespace onnxruntime::python;

onnxruntime::ORTInvoker& GetORTInvoker(const at::Device device) {
  return GetORTBackendsManager().GetInvoker(device);
}

ORTBackendsManager::ORTBackendsManager(const onnxruntime::logging::Logger& logger): logger_(logger){
  // set device index 0 to cpu EP as default backend.
  auto status = set_device(0, kCpuExecutionProvider, {});
  if (!status.IsOK()){
    throw std::runtime_error("Init CPU device failed: " + status.ErrorMessage());
  }
}

onnxruntime::Status ORTBackendsManager::set_device(size_t device_index, const std::string& provider_type,
                                 const ProviderOptions& provider_options){
  auto ep = onnxruntime::python::GetOrCreateExecutionProvider(provider_type, 
                               ProviderOptionsMap{{provider_type, provider_options}},
                               SessionOptions{});

  auto invoker = 
  std::make_unique<onnxruntime::ORTInvoker>(
    std::move(ep),
    logger_,
    custom_op_schema_);

  backends_[device_index] = std::move(invoker);
  ProviderInfoMap provider_info;
  provider_info[provider_type] = provider_options;
  device_ep_info_[device_index] = provider_info;
  return onnxruntime::Status::OK();
}

OrtDevice ORTBackendsManager::GetOrtDeviceInfo(size_t torch_device_index){
  auto lookup = backends_.find(torch_device_index);
  ORT_ENFORCE(lookup != backends_.end());
  auto allocator = lookup->second->GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault);
  return allocator->Info().device;
}

size_t ORTBackendsManager::GetOrtDeviceIndex(const OrtMemoryInfo& ort_memory_info){
  for (auto it = backends_.begin(); it != backends_.end(); ++it){
    //eager mode currently only operate on EP's default memory type
    auto allocator = it->second->GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault);
    if (allocator->Info() == ort_memory_info)
      return it->first;
  }
  ORT_THROW("Can't find the eager mode ORT device index for target ort tensor");
}

const ProviderInfoMap& ORTBackendsManager::GetOrtDeviceProviderInfo(size_t torch_device_index) const {
  auto lookup = device_ep_info_.find(torch_device_index);
  ORT_ENFORCE(lookup != device_ep_info_.end());
  return lookup->second;
}

onnxruntime::ORTInvoker& ORTBackendsManager::GetInvoker(const at::Device device) {
  ORT_LOG_FN(device);

  auto device_index = 0;
  if (device.has_index()) {
    device_index = device.index();
  }

  TORCH_CHECK(device.type() == at::DeviceType::ORT, "must be an ORT device");
  TORCH_CHECK(device_index >= 0, "must have a valid index");

  auto lookup = backends_.find(device_index);
  if (lookup != backends_.end()) {
    return *lookup->second;
  }else{
    throw std::runtime_error("ORT device index: " + std::to_string(device_index) + " not initialized, \
                              please use 'torch_ort.set_device' to initialize it first.");
  }
}

} // namespace eager
} // namespace torch_ort
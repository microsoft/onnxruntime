// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>
#include <core/providers/cpu/cpu_provider_factory_creator.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/providers/get_execution_providers.h>
#include "ort_backends.h"
#include "ort_log.h"
#include "core/platform/env.h"
#include "core/providers/shared_library/provider_host_api.h"


//use the environment from python module
namespace onnxruntime{
namespace python{
  onnxruntime::Environment& GetEnv();
}
}

namespace torch_ort {
namespace eager {

using namespace onnxruntime;

ORTBackendsManager& GetORTBackendsManager() {
  auto& env = onnxruntime::python::GetEnv();
  static ORTBackendsManager instance {env.GetLoggingManager()->DefaultLogger()};
  return instance;
}

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

void ORTBackendsManager::RegisterProviderLib(const std::string& provider_type, 
                                             const std::string& lib_path,
                                             const std::string& entry_point){
  additional_provider_libs_.insert({provider_type, {lib_path, entry_point}});
}

onnxruntime::Status ORTBackendsManager::set_device(size_t device_index, const std::string& provider_type,
                                 const ProviderOptions& provider_options){
  // query avalible device
  auto& available_providers = GetAvailableExecutionProviderNames();
  std::unique_ptr<IExecutionProvider> provider_p;
  if (std::find(available_providers.begin(), available_providers.end(), provider_type) != available_providers.end()){
    if (provider_type == kCpuExecutionProvider){
      provider_p = onnxruntime::CreateExecutionProviderFactory_CPU(0)->CreateProvider();
    }
  }
  else{
    auto shared_lib_path_it = additional_provider_libs_.find(provider_type);
    if (shared_lib_path_it == additional_provider_libs_.end()){
      return onnxruntime::Status(common::StatusCategory::ONNXRUNTIME,
                          common::StatusCode::INVALID_ARGUMENT, 
                          "Execution provider: " + provider_type + " is not supported.");
    }

    void* handle;
    auto lib_path = shared_lib_path_it->second.first;
    auto entry_point = shared_lib_path_it->second.second;
    auto error = Env::Default().LoadDynamicLibrary(lib_path, false, &handle);
    if (!error.IsOK()) {
      return onnxruntime::Status(common::StatusCategory::ONNXRUNTIME,
                                 common::StatusCode::INVALID_ARGUMENT, 
                                 "Load shared execution provider: " + provider_type + " failed: "
                                 + error.ErrorMessage());
    }

    Provider* (*PGetProvider)();
    ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle, entry_point, (void**)&PGetProvider));

    Provider* provider = PGetProvider();
    std::shared_ptr<IExecutionProviderFactory> ep_factory = provider->CreateExecutionProviderFactory(&provider_options);
    provider_p = ep_factory->CreateProvider();
  }


  auto invoker = 
  std::make_unique<onnxruntime::ORTInvoker>(
    std::move(provider_p),
    logger_,
    custom_op_schema_);

  backends_[device_index] = std::move(invoker);
  return onnxruntime::Status::OK();
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
// Copyright (c) Xilinx Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vitisai/vitisai_provider_factory.h"
#include <atomic>
#include "vitisai_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct VitisAIProviderFactory : IExecutionProviderFactory {
  VitisAIProviderFactory(std::string&& backend_type, int device_id, std::string&& export_runtime_module,
                         std::string&& load_runtime_module)
    : backend_type_(std::move(backend_type)), device_id_(device_id),
      export_runtime_module_(std::move(export_runtime_module)),
      load_runtime_module_(std::move(load_runtime_module)) {}
  ~VitisAIProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  // The Vitis AI DPU target
  const std::string backend_type_;
  // Device ID (Unused for now)
  int device_id_;
  // If not empty, the path to the file where the PyXIR runtime module
  //	should be exported to (used for cross compilation)
  const std::string export_runtime_module_;
  // If not empty, the path to the file where the PyXIR runtime module
  //	should be loaded from
  const std::string load_runtime_module_;
};

std::unique_ptr<IExecutionProvider> VitisAIProviderFactory::CreateProvider() {
  VitisAIExecutionProviderInfo info;
  info.backend_type = backend_type_;
  info.device_id = device_id_;
  info.export_runtime_module = export_runtime_module_;
  info.load_runtime_module = load_runtime_module_;
  return std::make_unique<VitisAIExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_VITISAI(
    const char* backend_type, int device_id, const char* export_runtime_module,
    const char* load_runtime_module) {
  return std::make_shared<onnxruntime::VitisAIProviderFactory>(
    backend_type, device_id, export_runtime_module, load_runtime_module);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_VITISAI,
                    _In_ OrtSessionOptions* options, _In_ const char* backend_type, int device_id,
                    const char* export_runtime_module, const char* load_runtime_module) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_VITISAI(
    backend_type, device_id, export_runtime_module, load_runtime_module));
  return nullptr;
}


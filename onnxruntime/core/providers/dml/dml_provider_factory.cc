// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/dml/dml_provider_factory.h"
#include <atomic>
//#include "DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct DMLProviderFactory : IExecutionProviderFactory {
  DMLProviderFactory(int device_id) : device_id_(device_id) {}
  DMLProviderFactory(IDMLDevice* dml_device,
                     ID3D12CommandQueue* cmd_queue) : dml_device_(dml_device),
                                                      cmd_queue_(cmd_queue) {}
  ~DMLProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  int device_id_ = -1;
  IDMLDevice* dml_device_{};
  ID3D12CommandQueue* cmd_queue_{};
};

std::unique_ptr<IExecutionProvider> DMLProviderFactory::CreateProvider() {
  // TODO this is just a stub. Adrian, please fill this with the appropriate constructor.
  //std::shared_ptr<winrt::Windows::AI::MachineLearning::implementation::GraphNodeFactoryMap> graph_node_factory_map;
  std::unique_ptr<onnxruntime::IExecutionProvider> provider;
  //std::unique_ptr<onnxruntime::IDataTransfer> data_transfer;
  //Dml::CreateExecutionProviderObjects(nullptr, nullptr, graph_node_factory_map, provider, data_transfer);
  return provider;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_DML(int device_id) {
  return std::make_shared<onnxruntime::DMLProviderFactory>(device_id);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_DML(IDMLDevice* dml_device,
                                                                              ID3D12CommandQueue* cmd_queue) {
  return std::make_shared<onnxruntime::DMLProviderFactory>(dml_device, cmd_queue);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_DML(device_id));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProviderEx_DML, _In_ OrtSessionOptions* options,
                    IDMLDevice* dml_device, ID3D12CommandQueue* cmd_queue) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_DML(dml_device,
                                                                                        cmd_queue));
  return nullptr;
}

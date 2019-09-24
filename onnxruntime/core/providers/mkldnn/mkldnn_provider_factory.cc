// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#include <atomic>
#include "mkldnn_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {
struct MkldnnProviderFactory : IExecutionProviderFactory {
  MkldnnProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~MkldnnProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> MkldnnProviderFactory::CreateProvider() {
  MKLDNNExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<MKLDNNExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Mkldnn(int device_id) {
  return std::make_shared<onnxruntime::MkldnnProviderFactory>(device_id);
  //TODO: This is apparently a bug. The consructor parameter is create-arena-flag, not the device-id
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Mkldnn, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Mkldnn(use_arena));
  return nullptr;
}

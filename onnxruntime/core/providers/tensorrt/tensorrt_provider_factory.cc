// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include <atomic>
#include "tensorrt_execution_provider.h"
//#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct TensorrtProviderFactory : Provider_IExecutionProviderFactory {
  TensorrtProviderFactory(int device_id) : device_id_(device_id) {}
  ~TensorrtProviderFactory() override {}

  std::unique_ptr<Provider_IExecutionProvider> CreateProvider() override;

 private:
  int device_id_;
};

std::unique_ptr<Provider_IExecutionProvider> TensorrtProviderFactory::CreateProvider() {
  TensorrtExecutionProviderInfo info;
  info.device_id = device_id_;
  return onnxruntime::make_unique<TensorrtExecutionProvider>(info);
}

std::shared_ptr<Provider_IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id) {
  return std::make_shared<onnxruntime::TensorrtProviderFactory>(device_id);
}

struct Tensorrt_Provider : Provider {
  std::shared_ptr<Provider_IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    //TODO: This is apparently a bug. The consructor parameter is create-arena-flag, not the device-id
    // Will be fixed by PR #2850
    return std::make_shared<TensorrtProviderFactory>(device_id);
  }

  void SetProviderHost(ProviderHost& host) {
    onnxruntime::SetProviderHost(host);
  }
} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}

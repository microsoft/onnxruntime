// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include <atomic>
#include "dnnl_execution_provider.h"

using namespace onnxruntime;

extern onnxruntime::ProviderHost* g_host;

namespace onnxruntime {

void SetProviderHost(ProviderHost& host);

struct DnnlProviderFactory : Provider_IExecutionProviderFactory {
  DnnlProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~DnnlProviderFactory() override {}

  std::unique_ptr<Provider_IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<Provider_IExecutionProvider> DnnlProviderFactory::CreateProvider() {
  DNNLExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return onnxruntime::make_unique<DNNLExecutionProvider>(info);
}

struct Dnnl_Provider : Provider {
  std::shared_ptr<Provider_IExecutionProviderFactory> CreateExecutionProviderFactory(int use_arena) override {
    return std::make_shared<DnnlProviderFactory>(use_arena != 0);
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

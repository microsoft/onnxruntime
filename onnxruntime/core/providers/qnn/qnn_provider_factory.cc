// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/qnn_provider_factory_creator.h"

#include "core/session/abi_session_options_impl.h"
#include "core/providers/qnn/qnn_execution_provider.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
struct QNNProviderFactory : IExecutionProviderFactory {
  QNNProviderFactory(const ProviderOptions& provider_options_map) : provider_options_map_(provider_options_map) {
  }

  ~QNNProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<QNNExecutionProvider>(provider_options_map_);
  }

 private:
  ProviderOptions provider_options_map_;
};

std::shared_ptr<IExecutionProviderFactory> QNNProviderFactoryCreator::Create(const ProviderOptions& provider_options_map) {
  return std::make_shared<onnxruntime::QNNProviderFactory>(provider_options_map);
}

}  // namespace onnxruntime

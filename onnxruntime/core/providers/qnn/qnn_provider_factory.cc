// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/qnn_provider_factory_creator.h"

#include "core/session/abi_session_options_impl.h"
#include "core/providers/qnn/qnn_execution_provider.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
struct QNNProviderFactory : IExecutionProviderFactory {
  QNNProviderFactory(const ProviderOptions& provider_options_map, const SessionOptions* session_options)
      : provider_options_map_(provider_options_map), session_options_(session_options) {
  }

  ~QNNProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<QNNExecutionProvider>(provider_options_map_, session_options_);
  }

 private:
  ProviderOptions provider_options_map_;
  const SessionOptions* session_options_;
};

std::shared_ptr<IExecutionProviderFactory> QNNProviderFactoryCreator::Create(const ProviderOptions& provider_options_map,
                                                                             const SessionOptions* session_options) {
  return std::make_shared<onnxruntime::QNNProviderFactory>(provider_options_map, session_options);
}

}  // namespace onnxruntime

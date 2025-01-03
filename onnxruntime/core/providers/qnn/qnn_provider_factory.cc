// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/qnn_provider_factory_creator.h"
#include "core/providers/qnn/qnn_execution_provider.h"

namespace onnxruntime {
struct QNNProviderFactory : IExecutionProviderFactory {
  QNNProviderFactory(const ProviderOptions& provider_options_map, const ConfigOptions* config_options)
      : provider_options_map_(provider_options_map), config_options_(config_options) {
  }

  ~QNNProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<QNNExecutionProvider>(provider_options_map_, config_options_);
  }

 private:
  ProviderOptions provider_options_map_;
  const ConfigOptions* config_options_;
};

#if BUILD_QNN_EP_STATIC_LIB
std::shared_ptr<IExecutionProviderFactory> QNNProviderFactoryCreator::Create(const ProviderOptions& provider_options_map,
                                                                             const SessionOptions* session_options) {
  const ConfigOptions* config_options = nullptr;
  if (session_options != nullptr) {
    config_options = &session_options->config_options;
  }

  return std::make_shared<onnxruntime::QNNProviderFactory>(provider_options_map, config_options);
}
#else
struct QNN_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* param) override {
    if (param == nullptr) {
      LOGS_DEFAULT(ERROR) << "[QNN EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<const void*, 2> pointers_array = *reinterpret_cast<const std::array<const void*, 2>*>(param);
    const ProviderOptions* provider_options = reinterpret_cast<const ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<const ConfigOptions*>(pointers_array[1]);

    if (provider_options == nullptr) {
      LOGS_DEFAULT(ERROR) << "[QNN EP] Passed NULL ProviderOptions to CreateExecutionProviderFactory()";
      return nullptr;
    }

    return std::make_shared<onnxruntime::QNNProviderFactory>(*provider_options, config_options);
  }

  void Initialize() override {}
  void Shutdown() override {}
} g_provider;
#endif  // BUILD_QNN_EP_STATIC_LIB

}  // namespace onnxruntime

#if !BUILD_QNN_EP_STATIC_LIB
extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
#endif  // !BUILD_QNN_EP_STATIC_LIB

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <string>
#include <unordered_map>
#include "core/providers/qnn-abi/qnn_provider_factory_creator.h"
#include "core/providers/qnn-abi/qnn_execution_provider.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include <iostream>
#include "core/session/onnxruntime_c_api.h"
#include "core/session/abi_devices.h"

#if !BUILD_QNN_EP_STATIC_LIB
#include "core/providers/qnn-abi/qnn_ep_factory.h"
// #include "core/providers/qnn-abi/qnn_plugin_wrapper.h"
#include <gsl/gsl>
#endif


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

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override {
    const ConfigOptions& config_options = session_options.GetConfigOptions();
    const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    // We extract those EP options and pass them to QNN EP as separate "provider options".
    std::unordered_map<std::string, std::string> provider_options = provider_options_map_;
    std::string key_prefix = "ep.";
    key_prefix += qnn::utils::GetLowercaseString(kQnnExecutionProvider);
    key_prefix += ".";

    for (const auto& [key, value] : config_options_map) {
      if (key.rfind(key_prefix, 0) == 0) {
        provider_options[key.substr(key_prefix.size())] = value;
      }
    }

    auto qnn_ep = std::make_unique<QNNExecutionProvider>(provider_options, &config_options);
    qnn_ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
    return qnn_ep;
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

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                  const OrtKeyValuePairs* const* ep_metadata,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    if (num_devices != 1) {
      return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "QNN EP only supports one device.");
    }
    #if BUILD_QNN_EP_STATIC_LIB
      // Static library build - use built-in QNN EP creation
      const ConfigOptions* config_options = &session_options.GetConfigOptions();

      std::array<const void*, 2> configs_array = {&provider_options, config_options};
      const void* arg = reinterpret_cast<const void*>(&configs_array);
      auto ep_factory = CreateExecutionProviderFactory(arg);
      std::cout << "DEBUG: Creating IExecutionProvider for QNN EP static (1111)" << std::endl;
      ep = ep_factory->CreateProvider(session_options, logger);
    #else
      provider_options;
      ep_metadata;
      devices;
      const ConfigOptions* config_options = &session_options.GetConfigOptions();

      std::array<const void*, 2> configs_array = {&provider_options, config_options};
      const void* arg = reinterpret_cast<const void*>(&configs_array);
      auto ep_factory = CreateExecutionProviderFactory(arg);

      ep = ep_factory->CreateProvider(session_options, logger);
    #endif

    return Status::OK();
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "coreml_execution_provider.h"
#include "coreml_provider_factory_creator.h"

using namespace onnxruntime;

namespace onnxruntime {

namespace {
CoreMLOptions ParseProviderOption(const ProviderOptions& options) {
  CoreMLOptions coreml_options;
  const std::unordered_map<std::string, COREMLFlags> available_computeunits_options = {
      {"MLComputeUnitsCPUAndNeuralEngine", COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE},
      {"MLComputeUnitsCPUAndGPU", COREML_FLAG_USE_CPU_AND_GPU},
      {"MLComputeUnitsCPUOnly", COREML_FLAG_USE_CPU_ONLY},
      {"MLComputeUnitsAll", COREML_FLAG_USE_NONE},
  };
  const std::unordered_map<std::string, COREMLFlags> available_modelformat_options = {
      {"MLProgram", COREML_FLAG_CREATE_MLPROGRAM},
      {"NeuralNetwork", COREML_FLAG_USE_NONE},
  };
  std::unordered_set<std::string> valid_options = {
      kCoremlProviderOption_MLComputeUnits,
      kCoremlProviderOption_MLModelFormat,
      kCoremlProviderOption_MLAllowStaticInputShapes,
      kCoremlProviderOption_MLEnableOnSubgraphs,
      kCoremlProviderOption_MLModelCacheDir,
  };
  // Validate the options
  for (const auto& option : options) {
    if (valid_options.find(option.first) == valid_options.end()) {
      ORT_THROW("Unknown option: ", option.first);
    }
    if (kCoremlProviderOption_MLComputeUnits == option.first) {
      if (available_computeunits_options.find(option.second) == available_computeunits_options.end()) {
        ORT_THROW("Invalid value for option ", option.first, ": ", option.second);
      }else {
        coreml_options.coreml_flags |= available_computeunits_options.at(option.second);
      }
    } else if (kCoremlProviderOption_MLModelFormat == option.first) {
      if (available_modelformat_options.find(option.second) == available_modelformat_options.end()) {
        ORT_THROW("Invalid value for option ", option.first, ": ", option.second);
      } else {
        coreml_options.coreml_flags |= available_modelformat_options.at(option.second);
      }
    } else if (okCoremlProviderOption_MLAllowStaticInputShapes == option.first) {
      coreml_options.coreml_flags |= COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES;
    } else if (okCoremlProviderOption_MLEnableOnSubgraphs == option.first) {
      coreml_options.coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
    } else if (okCoremlProviderOption_MLModelCacheDir == option.first) {
      coreml_options.cache_path = option.second;
    }
  }

  return coreml_options;
}
}  // namespace
struct CoreMLProviderFactory : IExecutionProviderFactory {
  CoreMLProviderFactory(const CoreMLOptions& options)
      : options_(options) {}
  ~CoreMLProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  CoreMLOptions options_;
};

std::unique_ptr<IExecutionProvider> CoreMLProviderFactory::CreateProvider() {
  return std::make_unique<CoreMLExecutionProvider>(options_);
}

std::shared_ptr<IExecutionProviderFactory> CoreMLProviderFactoryCreator::Create(uint32_t coreml_flags) {
  CoreMLOptions coreml_options;
  coreml_options.coreml_flags = coreml_flags;
  return std::make_shared<onnxruntime::CoreMLProviderFactory>(coreml_options);
}

std::shared_ptr<IExecutionProviderFactory> CoreMLProviderFactoryCreator::Create(const ProviderOptions& options) {
  return std::make_shared<onnxruntime::CoreMLProviderFactory>(ParseProviderOption(options));
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CoreML,
                    _In_ OrtSessionOptions* options, uint32_t coreml_flags) {
  options->provider_factories.push_back(onnxruntime::CoreMLProviderFactoryCreator::Create(coreml_flags));
  return nullptr;
}

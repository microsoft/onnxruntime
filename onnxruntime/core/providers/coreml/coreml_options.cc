// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/coreml_execution_provider.h"
#include "core/providers/coreml/coreml_provider_factory.h"  // defines flags
#include "core/providers/coreml/model/host_utils.h"
#include "core/providers/coreml/builders/helper.h"

namespace onnxruntime {

CoreMLOptions::CoreMLOptions(uint32_t coreml_flags) {
  // validate the flags and populate the members. should be moving code from ctor to here
  require_static_shape_ = (coreml_flags & COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES) != 0;
  create_mlprogram_ = (coreml_flags & COREML_FLAG_CREATE_MLPROGRAM) != 0;
  enable_on_subgraph_ = (coreml_flags & COREML_FLAG_ENABLE_ON_SUBGRAPH) != 0;

#if defined(COREML_ENABLE_MLPROGRAM)
  if (coreml::util::CoreMLVersion() < MINIMUM_COREML_MLPROGRAM_VERSION && create_mlprogram_ != 0) {
    LOGS_DEFAULT(WARNING) << "ML Program is not supported on this OS version. Falling back to NeuralNetwork.";
    create_mlprogram_ = false;
  }
#else
  if (create_mlprogram_ != 0) {
    LOGS_DEFAULT(WARNING) << "ML Program is not supported in this build. Falling back to NeuralNetwork.";
    create_mlprogram_ = false;
  }
#endif

  compute_units_ = 0;  // 0 for all

  if (coreml_flags & COREML_FLAG_USE_CPU_ONLY) {
    compute_units_ |= COREML_FLAG_USE_CPU_ONLY;
  }
  if (coreml_flags & COREML_FLAG_USE_CPU_AND_GPU) {
    compute_units_ |= COREML_FLAG_USE_CPU_AND_GPU;
  }
  if (coreml_flags & COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE) {
    compute_units_ |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
  }

  // assure only one device option is selected
  if (compute_units_ & (compute_units_ - 1)) {
    // multiple device options selected
    ORT_THROW(
        "Multiple device options selected, you should use at most one of the following options:"
        "[COREML_FLAG_USE_CPU_ONLY, COREML_FLAG_USE_CPU_AND_GPU, COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE]");
  }

  const bool has_neural_engine = coreml::HasNeuralEngine();
  if (ComputeUnits(COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE) && !has_neural_engine) {
    ORT_THROW("The current system does not have Apple Neural Engine.");
  }
}

void CoreMLOptions::ValidateAndParseProviderOption(const ProviderOptions& options) {
  const std::unordered_map<std::string, COREMLFlags> available_computeunits_options = {
      {"CPUAndNeuralEngine", COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE},
      {"CPUAndGPU", COREML_FLAG_USE_CPU_AND_GPU},
      {"CPUOnly", COREML_FLAG_USE_CPU_ONLY},
      {"ALL", COREML_FLAG_USE_NONE},
  };
  const std::unordered_map<std::string, COREMLFlags> available_modelformat_options = {
      {"MLProgram", COREML_FLAG_CREATE_MLPROGRAM},
      {"NeuralNetwork", COREML_FLAG_USE_NONE},
  };
  const std::unordered_set<std::string_view> valid_options = {
      kCoremlProviderOption_MLComputeUnits,
      kCoremlProviderOption_ModelFormat,
      kCoremlProviderOption_RequireStaticInputShapes,
      kCoremlProviderOption_EnableOnSubgraphs,
      kCoremlProviderOption_SpecializationStrategy,
      kCoremlProviderOption_ProfileComputePlan,
      kCoremlProviderOption_AllowLowPrecisionAccumulationOnGPU,
  };
  // Validate the options
  for (const auto& option : options) {
    if (valid_options.find(option.first) == valid_options.end()) {
      ORT_THROW("Unknown option: ", option.first);
    }
    if (kCoremlProviderOption_MLComputeUnits == option.first) {
      if (available_computeunits_options.find(option.second) == available_computeunits_options.end()) {
        ORT_THROW("Invalid value for option `", option.first, "`: ", option.second);
      } else {
        compute_units_ = available_computeunits_options.at(option.second);
      }
    } else if (kCoremlProviderOption_ModelFormat == option.first) {
      if (available_modelformat_options.find(option.second) == available_modelformat_options.end()) {
        ORT_THROW("Invalid value for option ", option.first, ": ", option.second);
      } else {
        create_mlprogram_ = available_modelformat_options.at(option.second) & COREML_FLAG_CREATE_MLPROGRAM;
      }
    } else if (kCoremlProviderOption_RequireStaticInputShapes == option.first) {
      require_static_shape_ = option.second == "1";
    } else if (kCoremlProviderOption_EnableOnSubgraphs == option.first) {
      enable_on_subgraph_ = option.second == "1";
    } else if (kCoremlProviderOption_SpecializationStrategy == option.first) {
      if (option.second != "Default" && option.second != "FastPrediction") {
        ORT_THROW("Invalid value for option ", option.first, ": ", option.second,
                  ". Valid values are Default and FastPrediction.");
      }
      strategy_ = option.second;
    } else if (kCoremlProviderOption_ProfileComputePlan == option.first) {
      profile_compute_plan_ = option.second == "1";
    } else if (kCoremlProviderOption_AllowLowPrecisionAccumulationOnGPU == option.first) {
      allow_low_precision_accumulation_on_gpu_ = option.second == "1";
    }
  }
}
}  // namespace onnxruntime

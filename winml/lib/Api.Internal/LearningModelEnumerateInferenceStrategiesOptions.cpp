#include "lib/Api.Internal/pch/pch.h"
#include "LearningModelEnumerateInferenceStrategiesOptions.h"
#include "LearningModelBindingStrategyFilter.h"
#include "LearningModelReadModeFilter.h"
#include "LearningModelBatchingStrategyFilter.h"
#include "LearningModelPhaseFilter.h"
#include "LearningModelDeviceFilter.h"


namespace WINML_INTERNALP
{
    LearningModelEnumerateInferenceStrategiesOptions::LearningModelEnumerateInferenceStrategiesOptions()
    {
        input_strategy_filter_ = winrt::make<winml_internalp::LearningModelBindingStrategyFilter>();
        output_strategy_filter_ = winrt::make<winml_internalp::LearningModelBindingStrategyFilter>();
        output_read_mode_filter_ = winrt::make<winml_internalp::LearningModelReadModeFilter>();
        batching_strategy_filter_ = winrt::make<winml_internalp::LearningModelBatchingStrategyFilter>();
        phase_filter_ = winrt::make<winml_internalp::LearningModelPhaseFilter>();
        device_filter_ = winrt::make<winml_internalp::LearningModelDeviceFilter>();
    }


    uint32_t LearningModelEnumerateInferenceStrategiesOptions::NumberOfIterations()
    {
        return iterations_;
    }

    void LearningModelEnumerateInferenceStrategiesOptions::NumberOfIterations(uint32_t iterations)
    {
        iterations_ = iterations;
    }

    void LearningModelEnumerateInferenceStrategiesOptions::OverrideNamedDimension(const winrt::hstring& name, uint32_t dimension)
    {
        named_dimension_overrides[winrt::to_string(name)] = dimension;
    }

    std::unordered_map<std::string, uint32_t> LearningModelEnumerateInferenceStrategiesOptions::NamedDimensionOverrides()
    {
        return named_dimension_overrides;
    }

    winml_internal::LearningModelBindingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::InputStrategyFilter()
    {
        return input_strategy_filter_;
    }

    winml_internal::LearningModelBindingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::OutputStrategyFilter() {
      return output_strategy_filter_;
    }

    winml_internal::LearningModelReadModeFilter LearningModelEnumerateInferenceStrategiesOptions::OutputReadModeFilter() {
      return output_read_mode_filter_;
    }

    winml_internal::LearningModelBatchingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::BatchingStrategyFilter() {
      return batching_strategy_filter_;
    }

    winml_internal::LearningModelPhaseFilter LearningModelEnumerateInferenceStrategiesOptions::PhaseFilter() {
      return phase_filter_;
    }

    winml_internal::LearningModelDeviceFilter LearningModelEnumerateInferenceStrategiesOptions::DeviceFilter() {
      return device_filter_;
    }

    winml_internal::LearningModelOptimizationMode LearningModelEnumerateInferenceStrategiesOptions::OptimizationMode() {
      return optimization_mode_;
    }

    void LearningModelEnumerateInferenceStrategiesOptions::OptimizationMode(winml_internal::LearningModelOptimizationMode mode) {
      optimization_mode_ = mode;
    }
}

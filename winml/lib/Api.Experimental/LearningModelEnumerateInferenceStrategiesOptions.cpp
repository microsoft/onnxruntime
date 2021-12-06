#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelEnumerateInferenceStrategiesOptions.h"
#include "LearningModelBindingStrategyFilter.h"
#include "LearningModelReadModeFilter.h"
#include "LearningModelBatchingStrategyFilter.h"
#include "LearningModelPhaseFilter.h"
#include "LearningModelDeviceFilter.h"


namespace WINML_EXPERIMENTALP
{
    LearningModelEnumerateInferenceStrategiesOptions::LearningModelEnumerateInferenceStrategiesOptions()
    {
        input_strategy_filter_ = winrt::make<winml_experimentalp::LearningModelBindingStrategyFilter>();
        output_strategy_filter_ = winrt::make<winml_experimentalp::LearningModelBindingStrategyFilter>();
        output_read_mode_filter_ = winrt::make<winml_experimentalp::LearningModelReadModeFilter>();
        batching_strategy_filter_ = winrt::make<winml_experimentalp::LearningModelBatchingStrategyFilter>();
        phase_filter_ = winrt::make<winml_experimentalp::LearningModelPhaseFilter>();
        device_filter_ = winrt::make<winml_experimentalp::LearningModelDeviceFilter>();
    }

    void LearningModelEnumerateInferenceStrategiesOptions::OverrideNamedDimension(const winrt::hstring& name, uint32_t dimension)
    {
        named_dimension_overrides[winrt::to_string(name)] = dimension;
    }

    std::unordered_map<std::string, uint32_t> LearningModelEnumerateInferenceStrategiesOptions::NamedDimensionOverrides()
    {
        return named_dimension_overrides;
    }

    winml_experimental::LearningModelBindingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::InputStrategyFilter()
    {
        return input_strategy_filter_;
    }

    winml_experimental::LearningModelBindingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::OutputStrategyFilter() {
      return output_strategy_filter_;
    }

    winml_experimental::LearningModelReadModeFilter LearningModelEnumerateInferenceStrategiesOptions::OutputReadModeFilter() {
      return output_read_mode_filter_;
    }

    winml_experimental::LearningModelBatchingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::BatchingStrategyFilter() {
      return batching_strategy_filter_;
    }

    winml_experimental::LearningModelPhaseFilter LearningModelEnumerateInferenceStrategiesOptions::PhaseFilter() {
      return phase_filter_;
    }

    winml_experimental::LearningModelDeviceFilter LearningModelEnumerateInferenceStrategiesOptions::DeviceFilter() {
      return device_filter_;
    }

    winml_experimental::LearningModelOptimizationMode LearningModelEnumerateInferenceStrategiesOptions::OptimizationMode() {
      return optimization_mode_;
    }

    void LearningModelEnumerateInferenceStrategiesOptions::OptimizationMode(winml_experimental::LearningModelOptimizationMode mode) {
      optimization_mode_ = mode;
    }
}

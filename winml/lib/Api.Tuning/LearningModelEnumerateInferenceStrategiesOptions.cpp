#include "lib/Api.Tuning/pch/pch.h"
#include "LearningModelEnumerateInferenceStrategiesOptions.h"
#include "LearningModelBindingStrategyFilter.h"
#include "LearningModelReadModeFilter.h"
#include "LearningModelBatchingStrategyFilter.h"
#include "LearningModelPhaseFilter.h"
#include "LearningModelDeviceFilter.h"


namespace WINML_TUNINGP
{
    LearningModelEnumerateInferenceStrategiesOptions::LearningModelEnumerateInferenceStrategiesOptions()
    {
        input_strategy_filter_ = winrt::make<winml_tuningp::LearningModelBindingStrategyFilter>();
        output_strategy_filter_ = winrt::make<winml_tuningp::LearningModelBindingStrategyFilter>();
        output_read_mode_filter_ = winrt::make<winml_tuningp::LearningModelReadModeFilter>();
        batching_strategy_filter_ = winrt::make<winml_tuningp::LearningModelBatchingStrategyFilter>();
        phase_filter_ = winrt::make<winml_tuningp::LearningModelPhaseFilter>();
        device_filter_ = winrt::make<winml_tuningp::LearningModelDeviceFilter>();
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

    winml_tuning::LearningModelBindingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::InputStrategyFilter()
    {
        return input_strategy_filter_;
    }

    winml_tuning::LearningModelBindingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::OutputStrategyFilter() {
      return output_strategy_filter_;
    }

    winml_tuning::LearningModelReadModeFilter LearningModelEnumerateInferenceStrategiesOptions::OutputReadModeFilter() {
      return output_read_mode_filter_;
    }

    winml_tuning::LearningModelBatchingStrategyFilter LearningModelEnumerateInferenceStrategiesOptions::BatchingStrategyFilter() {
      return batching_strategy_filter_;
    }

    winml_tuning::LearningModelPhaseFilter LearningModelEnumerateInferenceStrategiesOptions::PhaseFilter() {
      return phase_filter_;
    }

    winml_tuning::LearningModelDeviceFilter LearningModelEnumerateInferenceStrategiesOptions::DeviceFilter() {
      return device_filter_;
    }

    winml_tuning::LearningModelOptimizationMode LearningModelEnumerateInferenceStrategiesOptions::OptimizationMode() {
      return optimization_mode_;
    }

    void LearningModelEnumerateInferenceStrategiesOptions::OptimizationMode(winml_tuning::LearningModelOptimizationMode mode) {
      optimization_mode_ = mode;
    }
}

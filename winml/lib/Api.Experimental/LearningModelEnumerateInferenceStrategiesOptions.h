#pragma once
#include "LearningModelEnumerateInferenceStrategiesOptions.g.h"

namespace WINML_EXPERIMENTALP
{
    struct LearningModelEnumerateInferenceStrategiesOptions : LearningModelEnumerateInferenceStrategiesOptionsT<LearningModelEnumerateInferenceStrategiesOptions>
    {
        LearningModelEnumerateInferenceStrategiesOptions();

        uint32_t NumberOfIterations();
        void NumberOfIterations(uint32_t iterations);

        void OverrideNamedDimension(const winrt::hstring& name, uint32_t dimension);
        std::unordered_map<std::string, uint32_t> NamedDimensionOverrides();

        winml_experimental::LearningModelBindingStrategyFilter InputStrategyFilter();
        winml_experimental::LearningModelBindingStrategyFilter OutputStrategyFilter();
        winml_experimental::LearningModelReadModeFilter OutputReadModeFilter();
        winml_experimental::LearningModelBatchingStrategyFilter BatchingStrategyFilter();
        winml_experimental::LearningModelPhaseFilter PhaseFilter();
        winml_experimental::LearningModelDeviceFilter DeviceFilter();
        winml_experimental::LearningModelOptimizationMode OptimizationMode();
        void OptimizationMode(winml_experimental::LearningModelOptimizationMode mode);

    private:
        uint32_t iterations_ = 100;
        winml_experimental::LearningModelBindingStrategyFilter input_strategy_filter_ = nullptr;
        winml_experimental::LearningModelBindingStrategyFilter output_strategy_filter_ = nullptr;
        winml_experimental::LearningModelReadModeFilter output_read_mode_filter_ = nullptr;
        winml_experimental::LearningModelBatchingStrategyFilter batching_strategy_filter_ = nullptr;
        winml_experimental::LearningModelPhaseFilter phase_filter_ = nullptr;
        winml_experimental::LearningModelDeviceFilter device_filter_ = nullptr;
        winml_experimental::LearningModelOptimizationMode optimization_mode_ = winml_experimental::LearningModelOptimizationMode::OptimizeForRuntimePerformance;
        std::unordered_map<std::string, uint32_t> named_dimension_overrides;
    };
}
namespace WINML_EXPERIMENTAL::factory_implementation
{
    struct LearningModelEnumerateInferenceStrategiesOptions : LearningModelEnumerateInferenceStrategiesOptionsT<LearningModelEnumerateInferenceStrategiesOptions, implementation::LearningModelEnumerateInferenceStrategiesOptions>
    {
    };
}

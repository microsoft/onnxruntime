#pragma once
#include "LearningModelEnumerateInferenceStrategiesOptions.g.h"

namespace WINML_INTERNALP
{
    struct LearningModelEnumerateInferenceStrategiesOptions : LearningModelEnumerateInferenceStrategiesOptionsT<LearningModelEnumerateInferenceStrategiesOptions>
    {
        LearningModelEnumerateInferenceStrategiesOptions();

        uint32_t NumberOfIterations();
        void NumberOfIterations(uint32_t iterations);

        void OverrideNamedDimension(const winrt::hstring& name, uint32_t dimension);
        std::unordered_map<std::string, uint32_t> NamedDimensionOverrides();

        winml_internal::LearningModelBindingStrategyFilter InputStrategyFilter();
        winml_internal::LearningModelBindingStrategyFilter OutputStrategyFilter();
        winml_internal::LearningModelReadModeFilter OutputReadModeFilter();
        winml_internal::LearningModelBatchingStrategyFilter BatchingStrategyFilter();
        winml_internal::LearningModelPhaseFilter PhaseFilter();
        winml_internal::LearningModelDeviceFilter DeviceFilter();
        winml_internal::LearningModelOptimizationMode OptimizationMode();
        void OptimizationMode(winml_internal::LearningModelOptimizationMode mode);

    private:
        // with a 95% confidence z-score (1.96), and .3ms error, and assuming a maximal std.dev of .5, only 11 samples are needed
        // sample size = 10.67 =  (1.96 * .5 / 3)^2
        // round up to 15 for buffer..
        uint32_t iterations_ = 15;
        winml_internal::LearningModelBindingStrategyFilter input_strategy_filter_ = nullptr;
        winml_internal::LearningModelBindingStrategyFilter output_strategy_filter_ = nullptr;
        winml_internal::LearningModelReadModeFilter output_read_mode_filter_ = nullptr;
        winml_internal::LearningModelBatchingStrategyFilter batching_strategy_filter_ = nullptr;
        winml_internal::LearningModelPhaseFilter phase_filter_ = nullptr;
        winml_internal::LearningModelDeviceFilter device_filter_ = nullptr;
        winml_internal::LearningModelOptimizationMode optimization_mode_ = winml_internal::LearningModelOptimizationMode::OptimizeForRuntimePerformance;
        std::unordered_map<std::string, uint32_t> named_dimension_overrides;
    };
}
namespace WINML_INTERNAL::factory_implementation
{
    struct LearningModelEnumerateInferenceStrategiesOptions : LearningModelEnumerateInferenceStrategiesOptionsT<LearningModelEnumerateInferenceStrategiesOptions, implementation::LearningModelEnumerateInferenceStrategiesOptions>
    {
    };
}

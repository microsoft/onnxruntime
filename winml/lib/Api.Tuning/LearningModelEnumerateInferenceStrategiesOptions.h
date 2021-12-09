#pragma once
#include "LearningModelEnumerateInferenceStrategiesOptions.g.h"

namespace WINML_TUNINGP
{
    struct LearningModelEnumerateInferenceStrategiesOptions : LearningModelEnumerateInferenceStrategiesOptionsT<LearningModelEnumerateInferenceStrategiesOptions>
    {
        LearningModelEnumerateInferenceStrategiesOptions();

        uint32_t NumberOfIterations();
        void NumberOfIterations(uint32_t iterations);

        void OverrideNamedDimension(const winrt::hstring& name, uint32_t dimension);
        std::unordered_map<std::string, uint32_t> NamedDimensionOverrides();

        winml_tuning::LearningModelBindingStrategyFilter InputStrategyFilter();
        winml_tuning::LearningModelBindingStrategyFilter OutputStrategyFilter();
        winml_tuning::LearningModelReadModeFilter OutputReadModeFilter();
        winml_tuning::LearningModelBatchingStrategyFilter BatchingStrategyFilter();
        winml_tuning::LearningModelPhaseFilter PhaseFilter();
        winml_tuning::LearningModelDeviceFilter DeviceFilter();
        winml_tuning::LearningModelOptimizationMode OptimizationMode();
        void OptimizationMode(winml_tuning::LearningModelOptimizationMode mode);

    private:
        // with a 95% confidence z-score (1.96), and .3ms error, and assuming a maximal std.dev of .5, only 11 samples are needed
        // sample size = 10.67 =  (1.96 * .5 / 3)^2
        // round up to 15 for buffer..
        uint32_t iterations_ = 15;
        winml_tuning::LearningModelBindingStrategyFilter input_strategy_filter_ = nullptr;
        winml_tuning::LearningModelBindingStrategyFilter output_strategy_filter_ = nullptr;
        winml_tuning::LearningModelReadModeFilter output_read_mode_filter_ = nullptr;
        winml_tuning::LearningModelBatchingStrategyFilter batching_strategy_filter_ = nullptr;
        winml_tuning::LearningModelPhaseFilter phase_filter_ = nullptr;
        winml_tuning::LearningModelDeviceFilter device_filter_ = nullptr;
        winml_tuning::LearningModelOptimizationMode optimization_mode_ = winml_tuning::LearningModelOptimizationMode::OptimizeForRuntimePerformance;
        std::unordered_map<std::string, uint32_t> named_dimension_overrides;
    };
}
namespace WINML_TUNING::factory_implementation
{
    struct LearningModelEnumerateInferenceStrategiesOptions : LearningModelEnumerateInferenceStrategiesOptionsT<LearningModelEnumerateInferenceStrategiesOptions, implementation::LearningModelEnumerateInferenceStrategiesOptions>
    {
    };
}

#pragma once
#include "LearningModelInferenceStrategy.g.h"

namespace WINML_TUNINGP
{
    struct LearningModelInferenceStrategy : LearningModelInferenceStrategyT<
                        LearningModelInferenceStrategy,
                        ILearningModelInferenceStrategyDetails>
    {
        LearningModelInferenceStrategy(
            winml::LearningModelDeviceKind kind,
            winml_tuning::LearningModelBindingStrategy input_strategy,
            winml_tuning::LearningModelBindingStrategy output_strategy,
            winml_tuning::LearningModelReadMode output_read_mode,
            uint32_t batch_size,
            float duration_in_milliseconds_mean,
            float duration_in_milliseconds_variance);

        winml::LearningModelDeviceKind DeviceKind();
        winml_tuning::LearningModelBindingStrategy InputStrategy();
        winml_tuning::LearningModelBindingStrategy OutputStrategy();
        winml_tuning::LearningModelReadMode OutputReadMode();
        float Metric();
        uint32_t BatchSize();

        // ILearningModelInferenceStrategyDetails
        float DurationInMillisecondsMean();
        float DurationInMillisecondsVariance();

    private:
        winml::LearningModelDeviceKind kind_;
        winml_tuning::LearningModelBindingStrategy input_strategy_;
        winml_tuning::LearningModelBindingStrategy output_strategy_;
        winml_tuning::LearningModelReadMode output_read_mode_;
        uint32_t batch_size_;
        float duration_in_milliseconds_mean_ = 0;
        float duration_in_milliseconds_variance_ = 0;

    };
}

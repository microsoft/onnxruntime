#pragma once
#include "LearningModelInferenceStrategy.g.h"

namespace WINML_EXPERIMENTALP
{
    struct LearningModelInferenceStrategy : LearningModelInferenceStrategyT<LearningModelInferenceStrategy>
    {
        LearningModelInferenceStrategy(
            winml::LearningModelDeviceKind kind,
            winml_experimental::LearningModelBindingStrategy input_strategy,
            winml_experimental::LearningModelBindingStrategy output_strategy,
            winml_experimental::LearningModelReadMode output_read_mode,
            winml_experimental::LearningModelBindMode bind_mode,
            uint32_t batch_size,
            float metric);

        winml::LearningModelDeviceKind DeviceKind();
        winml_experimental::LearningModelBindingStrategy InputStrategy();
        winml_experimental::LearningModelBindingStrategy OutputStrategy();
        winml_experimental::LearningModelReadMode OutputReadMode();
        winml_experimental::LearningModelBindMode OutputBindMode();
        float Metric();
        uint32_t BatchSize();

    private:
        winml::LearningModelDeviceKind kind_;
        winml_experimental::LearningModelBindingStrategy input_strategy_;
        winml_experimental::LearningModelBindingStrategy output_strategy_;
        winml_experimental::LearningModelReadMode output_read_mode_;
        winml_experimental::LearningModelBindMode output_bind_mode_;
        uint32_t batch_size_;
        float metric_;

    };
}

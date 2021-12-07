#pragma once
#include "LearningModelInferenceStrategy.g.h"

namespace WINML_INTERNALP
{
    struct LearningModelInferenceStrategy : LearningModelInferenceStrategyT<LearningModelInferenceStrategy>
    {
        LearningModelInferenceStrategy(
            winml::LearningModelDeviceKind kind,
            winml_internal::LearningModelBindingStrategy input_strategy,
            winml_internal::LearningModelBindingStrategy output_strategy,
            winml_internal::LearningModelReadMode output_read_mode,
            uint32_t batch_size,
            float metric);

        winml::LearningModelDeviceKind DeviceKind();
        winml_internal::LearningModelBindingStrategy InputStrategy();
        winml_internal::LearningModelBindingStrategy OutputStrategy();
        winml_internal::LearningModelReadMode OutputReadMode();
        float Metric();
        uint32_t BatchSize();

    private:
        winml::LearningModelDeviceKind kind_;
        winml_internal::LearningModelBindingStrategy input_strategy_;
        winml_internal::LearningModelBindingStrategy output_strategy_;
        winml_internal::LearningModelReadMode output_read_mode_;
        uint32_t batch_size_;
        float metric_;

    };
}

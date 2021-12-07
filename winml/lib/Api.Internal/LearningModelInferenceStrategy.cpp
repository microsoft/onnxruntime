#include "lib/Api.Internal/pch/pch.h"
#include "LearningModelInferenceStrategy.h"

namespace WINML_INTERNALP
{
    LearningModelInferenceStrategy::LearningModelInferenceStrategy(
            winml::LearningModelDeviceKind kind,
            winml_internal::LearningModelBindingStrategy input_strategy,
            winml_internal::LearningModelBindingStrategy output_strategy,
            winml_internal::LearningModelReadMode output_read_mode,
            uint32_t batch_size,
            float metric) :
                kind_(kind),
                input_strategy_(input_strategy),
                output_strategy_(output_strategy),
                output_read_mode_(output_read_mode),
                batch_size_(batch_size),
                metric_(metric)
    {
    }

    winml::LearningModelDeviceKind LearningModelInferenceStrategy::DeviceKind()
    {
        return kind_;
    }

    winml_internal::LearningModelBindingStrategy LearningModelInferenceStrategy::InputStrategy()
    {
        return input_strategy_;
    }

    winml_internal::LearningModelBindingStrategy LearningModelInferenceStrategy::OutputStrategy()
    {
        return output_strategy_;
    }

    float LearningModelInferenceStrategy::Metric() {
      return metric_;
    }

    uint32_t LearningModelInferenceStrategy::BatchSize() {
      return batch_size_;
    }

    winml_internal::LearningModelReadMode
        LearningModelInferenceStrategy::OutputReadMode() {
      return output_read_mode_;
    }
}

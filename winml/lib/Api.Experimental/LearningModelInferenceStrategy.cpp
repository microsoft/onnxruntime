#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelInferenceStrategy.h"

namespace WINML_EXPERIMENTALP
{
    LearningModelInferenceStrategy::LearningModelInferenceStrategy(
            winml::LearningModelDeviceKind kind,
            winml_experimental::LearningModelBindingStrategy input_strategy,
            winml_experimental::LearningModelBindingStrategy output_strategy,
            winml_experimental::LearningModelReadMode output_read_mode,
            winml_experimental::LearningModelBindMode bind_mode,
            uint32_t batch_size,
            float metric) :
                kind_(kind),
                input_strategy_(input_strategy),
                output_strategy_(output_strategy),
                output_read_mode_(output_read_mode),
                output_bind_mode_(bind_mode),
                batch_size_(batch_size),
                metric_(metric)
    {
    }

    winml::LearningModelDeviceKind LearningModelInferenceStrategy::DeviceKind()
    {
        return kind_;
    }

    winml_experimental::LearningModelBindingStrategy LearningModelInferenceStrategy::InputStrategy()
    {
        return input_strategy_;
    }

    winml_experimental::LearningModelBindingStrategy LearningModelInferenceStrategy::OutputStrategy()
    {
        return output_strategy_;
    }

    float LearningModelInferenceStrategy::Metric() {
      return metric_;
    }

    uint32_t LearningModelInferenceStrategy::BatchSize() {
      return batch_size_;
    }

    winml_experimental::LearningModelReadMode
        LearningModelInferenceStrategy::OutputReadMode() {
      return output_read_mode_;
    }

    winml_experimental::LearningModelBindMode LearningModelInferenceStrategy::OutputBindMode() {
      return output_bind_mode_;
    }
}

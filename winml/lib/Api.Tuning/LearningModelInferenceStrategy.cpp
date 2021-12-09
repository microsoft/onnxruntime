#include "lib/Api.Tuning/pch/pch.h"
#include "LearningModelInferenceStrategy.h"

namespace WINML_TUNINGP
{
    LearningModelInferenceStrategy::LearningModelInferenceStrategy(
            winml::LearningModelDeviceKind kind,
            winml_tuning::LearningModelBindingStrategy input_strategy,
            winml_tuning::LearningModelBindingStrategy output_strategy,
            winml_tuning::LearningModelReadMode output_read_mode,
            uint32_t batch_size,
            float duration_in_milliseconds_mean,
            float duration_in_milliseconds_variance) :
                kind_(kind),
                input_strategy_(input_strategy),
                output_strategy_(output_strategy),
                output_read_mode_(output_read_mode),
                batch_size_(batch_size),
                duration_in_milliseconds_mean_(duration_in_milliseconds_mean),
                duration_in_milliseconds_variance_(duration_in_milliseconds_variance)
    {
    }

    winml::LearningModelDeviceKind LearningModelInferenceStrategy::DeviceKind()
    {
        return kind_;
    }

    winml_tuning::LearningModelBindingStrategy LearningModelInferenceStrategy::InputStrategy()
    {
        return input_strategy_;
    }

    winml_tuning::LearningModelBindingStrategy LearningModelInferenceStrategy::OutputStrategy()
    {
        return output_strategy_;
    }

    float LearningModelInferenceStrategy::Metric() {
      // Currently there is only a single metric...
      return DurationInMillisecondsMean();
    }

    float LearningModelInferenceStrategy::DurationInMillisecondsMean() {
      return duration_in_milliseconds_mean_;
    }

    float LearningModelInferenceStrategy::DurationInMillisecondsVariance() {
      return duration_in_milliseconds_variance_;
    }


    uint32_t LearningModelInferenceStrategy::BatchSize() {
      return batch_size_;
    }

    winml_tuning::LearningModelReadMode
        LearningModelInferenceStrategy::OutputReadMode() {
      return output_read_mode_;
    }
}

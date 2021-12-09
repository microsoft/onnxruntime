#pragma once
#include "LearningModelInferenceStrategyEnumerator.g.h"

namespace WINML_TUNINGP
{
    struct LearningModelInferenceStrategyEnumerator
    {
        LearningModelInferenceStrategyEnumerator() = default;

        static wfc::IVectorView<winml_tuning::LearningModelInferenceStrategy> EnumerateInferenceStrategies(
            hstring const& path,
            winml_tuning::LearningModelEnumerateInferenceStrategiesOptions const& options);

        static wf::IAsyncOperationWithProgress<wfc::IVectorView<winml_tuning::LearningModelInferenceStrategy>, winml_tuning::EnumerateInferenceStrategiesProgress> EnumerateInferenceStrategiesAsync(
            hstring path,
            Microsoft::AI::MachineLearning::Tuning::LearningModelEnumerateInferenceStrategiesOptions options);

    };
}
namespace WINML_TUNING::factory_implementation
{
    struct LearningModelInferenceStrategyEnumerator : LearningModelInferenceStrategyEnumeratorT<LearningModelInferenceStrategyEnumerator, implementation::LearningModelInferenceStrategyEnumerator>
    {
    };
}

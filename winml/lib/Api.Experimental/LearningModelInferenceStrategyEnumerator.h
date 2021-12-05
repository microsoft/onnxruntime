#pragma once
#include "LearningModelInferenceStrategyEnumerator.g.h"

namespace WINML_EXPERIMENTALP
{
    struct LearningModelInferenceStrategyEnumerator
    {
        LearningModelInferenceStrategyEnumerator() = default;

        static wfc::IVectorView<winml_experimental::LearningModelInferenceStrategy> EnumerateInferenceStrategies(
            hstring const& path,
            winml_experimental::LearningModelEnumerateInferenceStrategiesOptions const& options);

        static wf::IAsyncOperationWithProgress<wfc::IVectorView<winml_experimental::LearningModelInferenceStrategy>, winml_experimental::EnumerateInferenceStrategiesProgress> EnumerateInferenceStrategiesAsync(
            hstring path,
            Microsoft::AI::MachineLearning::Experimental::LearningModelEnumerateInferenceStrategiesOptions options);

    };
}
namespace WINML_EXPERIMENTAL::factory_implementation
{
    struct LearningModelInferenceStrategyEnumerator : LearningModelInferenceStrategyEnumeratorT<LearningModelInferenceStrategyEnumerator, implementation::LearningModelInferenceStrategyEnumerator>
    {
    };
}

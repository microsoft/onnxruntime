#pragma once
#include "LearningModelInferenceStrategyEnumerator.g.h"

namespace WINML_INTERNALP
{
    struct LearningModelInferenceStrategyEnumerator
    {
        LearningModelInferenceStrategyEnumerator() = default;

        static wfc::IVectorView<winml_internal::LearningModelInferenceStrategy> EnumerateInferenceStrategies(
            hstring const& path,
            winml_internal::LearningModelEnumerateInferenceStrategiesOptions const& options);

        static wf::IAsyncOperationWithProgress<wfc::IVectorView<winml_internal::LearningModelInferenceStrategy>, winml_internal::EnumerateInferenceStrategiesProgress> EnumerateInferenceStrategiesAsync(
            hstring path,
            Microsoft::AI::MachineLearning::Internal::LearningModelEnumerateInferenceStrategiesOptions options);

    };
}
namespace WINML_INTERNAL::factory_implementation
{
    struct LearningModelInferenceStrategyEnumerator : LearningModelInferenceStrategyEnumeratorT<LearningModelInferenceStrategyEnumerator, implementation::LearningModelInferenceStrategyEnumerator>
    {
    };
}

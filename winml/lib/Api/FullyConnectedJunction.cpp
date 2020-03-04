#include "pch.h"
#include "FullyConnectedJunction.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    Windows::AI::MachineLearning::More::ILearningModelJunction FullyConnectedJunction::Then(Windows::AI::MachineLearning::More::ILearningModelJunction const& /*next_layer*/)
    {
        throw hresult_not_implemented();
    }

    Windows::AI::MachineLearning::More::ILearningModelJunction FullyConnectedJunction::Then(Windows::AI::MachineLearning::More::ILearningModelJunction const& /*next_layer*/, Windows::AI::MachineLearning::More::LearningModelJunctionResolutionPolicy const& /*policy*/)
    {
        throw hresult_not_implemented();
    }
}

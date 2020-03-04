#pragma once

#include "FullyConnectedJunction.g.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    struct FullyConnectedJunction : FullyConnectedJunctionT<FullyConnectedJunction>
    {
        FullyConnectedJunction() = delete;

        Windows::AI::MachineLearning::More::ILearningModelJunction Then(Windows::AI::MachineLearning::More::ILearningModelJunction const& next_junction);
        Windows::AI::MachineLearning::More::ILearningModelJunction Then(Windows::AI::MachineLearning::More::ILearningModelJunction const& next_junction, Windows::AI::MachineLearning::More::LearningModelJunctionResolutionPolicy const& policy);
    };
}

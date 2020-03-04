#pragma once

#include "LearningModelJunctionResolutionPolicy.g.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    struct LearningModelJunctionResolutionPolicy : LearningModelJunctionResolutionPolicyT<LearningModelJunctionResolutionPolicy>
    {
        LearningModelJunctionResolutionPolicy() = delete;

        void Foo();

        static Windows::AI::MachineLearning::More::LearningModelJunctionResolutionPolicy Create();
    };
}

namespace winrt::Windows::AI::MachineLearning::More::factory_implementation
{
    struct LearningModelJunctionResolutionPolicy : LearningModelJunctionResolutionPolicyT<LearningModelJunctionResolutionPolicy, implementation::LearningModelJunctionResolutionPolicy>
    {
    };
}

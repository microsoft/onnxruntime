#pragma once

#include "LearningModelBuilder.g.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder>
    {
        LearningModelBuilder() = delete;

        Windows::AI::MachineLearning::LearningModel CreateModel();
        void Close();

        static Windows::AI::MachineLearning::More::LearningModelBuilder Create();
    };
}

namespace winrt::Windows::AI::MachineLearning::More::factory_implementation
{
    struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder, implementation::LearningModelBuilder>
    {
    };
}

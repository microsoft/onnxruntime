#pragma once

#include "LearningModelBuilder.g.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder>
    {
        LearningModelBuilder() = delete;

        Windows::AI::MachineLearning::More::ILearningModelJunction InputJunction();
        Windows::AI::MachineLearning::More::ILearningModelJunction OutputJunction();
        Windows::AI::MachineLearning::More::ILearningModelJunction AddInput(Windows::AI::MachineLearning::ILearningModelFeatureDescriptor const& input_descriptor);
        Windows::AI::MachineLearning::More::ILearningModelJunction AddOutput(Windows::AI::MachineLearning::ILearningModelFeatureDescriptor const& output_descriptor);
        Windows::AI::MachineLearning::LearningModel CreateModel();
        void Close();

        static Windows::AI::MachineLearning::More::LearningModelBuilder Create();
        static Windows::AI::MachineLearning::More::ILearningModelJunction AfterAll(Windows::AI::MachineLearning::More::ILearningModelJunction const& target, Windows::Foundation::Collections::IVectorView<Windows::AI::MachineLearning::More::ILearningModelJunction> const& input_junctions);
        static Windows::AI::MachineLearning::More::ILearningModelJunction AfterAll(Windows::AI::MachineLearning::More::ILearningModelJunction const& target, Windows::Foundation::Collections::IVectorView<Windows::AI::MachineLearning::More::ILearningModelJunction> const& input_junctions, Windows::AI::MachineLearning::More::LearningModelJunctionResolutionPolicy const& policy);
    };
}

namespace winrt::Windows::AI::MachineLearning::More::factory_implementation
{
    struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder, implementation::LearningModelBuilder>
    {
    };
}

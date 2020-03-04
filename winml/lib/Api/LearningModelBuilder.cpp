#include "pch.h"
#include "LearningModelBuilder.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    Windows::AI::MachineLearning::More::ILearningModelJunction LearningModelBuilder::InputJunction()
    {
        throw hresult_not_implemented();
    }

    Windows::AI::MachineLearning::More::ILearningModelJunction LearningModelBuilder::OutputJunction()
    {
        throw hresult_not_implemented();
    }

    Windows::AI::MachineLearning::More::ILearningModelJunction LearningModelBuilder::AddInput(Windows::AI::MachineLearning::ILearningModelFeatureDescriptor const& /*input_descriptor*/)
    {
        throw hresult_not_implemented();
    }

    Windows::AI::MachineLearning::More::ILearningModelJunction LearningModelBuilder::AddOutput(Windows::AI::MachineLearning::ILearningModelFeatureDescriptor const& /*output_descriptor*/)
    {
        throw hresult_not_implemented();
    }

    Windows::AI::MachineLearning::LearningModel LearningModelBuilder::CreateModel()
    {
        throw hresult_not_implemented();
    }

    void LearningModelBuilder::Close()
    {
        throw hresult_not_implemented();
    }

    Windows::AI::MachineLearning::More::LearningModelBuilder LearningModelBuilder::Create(){
      return winrt::make<LearningModelBuilder>();
    }

    Windows::AI::MachineLearning::More::ILearningModelJunction LearningModelBuilder::AfterAll(Windows::AI::MachineLearning::More::ILearningModelJunction const& /*target*/, Windows::Foundation::Collections::IVectorView<Windows::AI::MachineLearning::More::ILearningModelJunction> const& /*input_junctions*/)
    {
        throw hresult_not_implemented();
    }

    Windows::AI::MachineLearning::More::ILearningModelJunction LearningModelBuilder::AfterAll(Windows::AI::MachineLearning::More::ILearningModelJunction const& /*target*/, Windows::Foundation::Collections::IVectorView<Windows::AI::MachineLearning::More::ILearningModelJunction> const& /*input_junctions*/, Windows::AI::MachineLearning::More::LearningModelJunctionResolutionPolicy const& /*policy*/)
    {
        throw hresult_not_implemented();
    }
}

#include "pch.h"
#include "LearningModelOperatorResolutionPolicy.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::ConnectInputs()
    {
        return *this;
    }

    Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::ConnectConstants()
    {
        return *this;
    }

    Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::AddInputMapping(hstring const& /*operator_input*/, hstring const& /*incoming_input*/)
    {
        return *this;
    }

    Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::GenerateMissingInputsAsModelConstants()
    {
        return *this;
    }

    Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::GenerateMissingInputsAsModelInputs()
    {
        return *this;
    }
}

#pragma once

#include "LearningModelOperatorResolutionPolicy.g.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation
{
    struct LearningModelOperatorResolutionPolicy : LearningModelOperatorResolutionPolicyT<LearningModelOperatorResolutionPolicy>
    {
        LearningModelOperatorResolutionPolicy() = default;

        Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy ConnectInputs();
        Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy ConnectConstants();
        Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy AddInputMapping(hstring const& operator_input, hstring const& incoming_input);
        Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy GenerateMissingInputsAsModelConstants();
        Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy GenerateMissingInputsAsModelInputs();

        static Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy Create();
    };
}
// namespace winrt::Windows::AI::MachineLearning::More::implementation

namespace winrt::Windows::AI::MachineLearning::More::factory_implementation {
struct LearningModelOperatorResolutionPolicy : LearningModelOperatorResolutionPolicyT<LearningModelOperatorResolutionPolicy, implementation::LearningModelOperatorResolutionPolicy> {
};
}  // namespace winrt::Windows::AI::MachineLearning::More::factory_implementation

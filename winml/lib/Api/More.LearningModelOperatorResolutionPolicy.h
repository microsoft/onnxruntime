#pragma once

#include "LearningModelOperatorResolutionPolicy.g.h"

namespace MOREP {

struct LearningModelOperatorResolutionPolicy : LearningModelOperatorResolutionPolicyT<LearningModelOperatorResolutionPolicy>
{
    LearningModelOperatorResolutionPolicy() = default;

    more::LearningModelOperatorResolutionPolicy ConnectInputs();
    more::LearningModelOperatorResolutionPolicy ConnectConstants();
    more::LearningModelOperatorResolutionPolicy AddInputMapping(hstring const& operator_input, hstring const& incoming_input);
    more::LearningModelOperatorResolutionPolicy GenerateMissingInputsAsModelConstants();
    more::LearningModelOperatorResolutionPolicy GenerateMissingInputsAsModelInputs();

    static more::LearningModelOperatorResolutionPolicy Create();
}

}
// namespace MOREP
namespace MORE::factory_implementation {

struct LearningModelOperatorResolutionPolicy : LearningModelOperatorResolutionPolicyT<LearningModelOperatorResolutionPolicy, implementation::LearningModelOperatorResolutionPolicy> {

};
}  // namespace winrt::more::factory_implementation

#include "pch.h"
#include "More.LearningModelOperatorResolutionPolicy.h"

namespace MOREP
{

more::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::ConnectInputs()
{
    return *this;
}

more::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::ConnectConstants()
{
    return *this;
}

more::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::AddInputMapping(hstring const& /*operator_input*/, hstring const& /*incoming_input*/)
{
    return *this;
}

more::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::GenerateMissingInputsAsModelConstants()
{
    return *this;
}

more::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::GenerateMissingInputsAsModelInputs()
{
    return *this;
}

}

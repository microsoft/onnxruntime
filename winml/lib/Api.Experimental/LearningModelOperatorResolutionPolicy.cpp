#include "pch.h"
#include "LearningModelOperatorResolutionPolicy.h"

namespace WINML_EXPERIMENTALP
{

winml_experimental::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::ConnectInputs()
{
    return *this;
}

winml_experimental::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::ConnectConstants()
{
    return *this;
}

winml_experimental::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::AddInputMapping(hstring const& /*operator_input*/, hstring const& /*incoming_input*/)
{
    return *this;
}

winml_experimental::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::GenerateMissingInputsAsModelConstants()
{
    return *this;
}

winml_experimental::LearningModelOperatorResolutionPolicy LearningModelOperatorResolutionPolicy::GenerateMissingInputsAsModelInputs()
{
    return *this;
}

}

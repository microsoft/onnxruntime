#pragma once

#include "LearningModelOperatorResolutionPolicy.g.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelOperatorResolutionPolicy : LearningModelOperatorResolutionPolicyT<LearningModelOperatorResolutionPolicy> {
  LearningModelOperatorResolutionPolicy() = default;

  winml_experimental::LearningModelOperatorResolutionPolicy ConnectInputs();
  winml_experimental::LearningModelOperatorResolutionPolicy ConnectConstants();
  winml_experimental::LearningModelOperatorResolutionPolicy AddInputMapping(hstring const& operator_input, hstring const& incoming_input);
  winml_experimental::LearningModelOperatorResolutionPolicy GenerateMissingInputsAsModelConstants();
  winml_experimental::LearningModelOperatorResolutionPolicy GenerateMissingInputsAsModelInputs();

  static winml_experimental::LearningModelOperatorResolutionPolicy Create();
};

}
// namespace WINML_EXPERIMENTALP
namespace WINML_EXPERIMENTAL::factory_implementation {

struct LearningModelOperatorResolutionPolicy : LearningModelOperatorResolutionPolicyT<LearningModelOperatorResolutionPolicy, implementation::LearningModelOperatorResolutionPolicy> {

};
}  // namespace winrt::winml_experimental::factory_implementation

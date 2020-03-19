#pragma once

#include <winrt/Windows.Foundation.Collections.h>
#include "LearningModelBuilder.g.h"
#include "iengine.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation {
struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder> {
  LearningModelBuilder();
  LearningModelBuilder(LearningModelBuilder& builder);

  more::LearningModelInputs Inputs();
  more::LearningModelOutputs Outputs();
  winml::LearningModel CreateModel();

  static Windows::AI::MachineLearning::More::LearningModelBuilder Create();
  static Windows::AI::MachineLearning::More::LearningModelOperator AfterAll(Windows::AI::MachineLearning::More::LearningModelOperator const& target, Windows::Foundation::Collections::IVectorView<Windows::AI::MachineLearning::More::LearningModelOperator> const& input_operators);
  static Windows::AI::MachineLearning::More::LearningModelOperator AfterAll(Windows::AI::MachineLearning::More::LearningModelOperator const& target, Windows::Foundation::Collections::IVectorView<Windows::AI::MachineLearning::More::LearningModelOperator> const& input_operators, Windows::AI::MachineLearning::More::LearningModelOperatorResolutionPolicy const& policy);

  WinML::IModel* UseModel();
  
 private:
  com_ptr<WinML::IEngineFactory> engine_factory_;
  com_ptr<WinML::IModel> model_;

  more::LearningModelInputs inputs_;
  more::LearningModelOutputs outputs_;
};
}  // namespace winrt::Windows::AI::MachineLearning::More::implementation

namespace winrt::Windows::AI::MachineLearning::More::factory_implementation {
struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder, implementation::LearningModelBuilder> {
};
}  // namespace winrt::Windows::AI::MachineLearning::More::factory_implementation

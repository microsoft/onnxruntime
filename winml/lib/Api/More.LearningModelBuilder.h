#pragma once

#include <winrt/Windows.Foundation.Collections.h>
#include "LearningModelBuilder.g.h"
#include "iengine.h"

namespace MOREP {

struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder> {
  LearningModelBuilder();
  LearningModelBuilder(LearningModelBuilder& builder);

  more::LearningModelInputs Inputs();
  more::LearningModelOutputs Outputs();
  winml::LearningModel CreateModel();

  static more::LearningModelBuilder Create();
  static more::LearningModelOperator AfterAll(more::LearningModelOperator const& target, wfc::IVectorView<more::LearningModelOperator> const& input_operators);
  static more::LearningModelOperator AfterAll(more::LearningModelOperator const& target, wfc::IVectorView<more::LearningModelOperator> const& input_operators, more::LearningModelOperatorResolutionPolicy const& policy);

  _winml::IModel* UseModel();
  
 private:
  com_ptr<_winml::IEngineFactory> engine_factory_;
  com_ptr<_winml::IModel> model_;

  more::LearningModelInputs inputs_;
  more::LearningModelOutputs outputs_;
};
}  // MOREP

namespace MORE::factory_implementation {
struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder, implementation::LearningModelBuilder> {
};
}  // namespace winrt::more::factory_implementation

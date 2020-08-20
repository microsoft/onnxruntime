#pragma once

#include <winrt/Windows.Foundation.Collections.h>
#include "LearningModelBuilder.g.h"
#include "iengine.h"

namespace WINML_EXPERIMENTALP {

struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder> {
  LearningModelBuilder();
  LearningModelBuilder(LearningModelBuilder& builder);

  winml_experimental::LearningModelInputs Inputs();
  winml_experimental::LearningModelOutputs Outputs();
  winml::LearningModel CreateModel();

  static winml_experimental::LearningModelBuilder Create();
  static winml_experimental::LearningModelOperator AfterAll(winml_experimental::LearningModelOperator const& target, wfc::IVectorView<winml_experimental::LearningModelOperator> const& input_operators);
  static winml_experimental::LearningModelOperator AfterAll(winml_experimental::LearningModelOperator const& target, wfc::IVectorView<winml_experimental::LearningModelOperator> const& input_operators, winml_experimental::LearningModelOperatorResolutionPolicy const& policy);

  _winml::IModel* UseModel();
  
 private:
  com_ptr<_winml::IEngineFactory> engine_factory_;
  com_ptr<_winml::IModel> model_;

  winml_experimental::LearningModelInputs inputs_;
  winml_experimental::LearningModelOutputs outputs_;
};
}  // WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {
struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder, implementation::LearningModelBuilder> {
};
}  // namespace winrt::winml_experimental::factory_implementation

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
  winml_experimental::LearningModelOperatorSet Operators();
  winml::LearningModel CreateModel();
  void Save(const winrt::hstring& file_name);

  static winml_experimental::LearningModelBuilder Create();

  _winml::IModel* UseModel();
  
 private:
  com_ptr<_winml::IEngineFactory> engine_factory_;
  com_ptr<_winml::IModel> model_;

  winml_experimental::LearningModelInputs inputs_;
  winml_experimental::LearningModelOutputs outputs_;
  winml_experimental::LearningModelOperatorSet operators_;
};
}  // WINML_EXPERIMENTALP

namespace WINML_EXPERIMENTAL::factory_implementation {
struct LearningModelBuilder : LearningModelBuilderT<LearningModelBuilder, implementation::LearningModelBuilder> {
};
}  // namespace winrt::winml_experimental::factory_implementation

#pragma once

#include "LearningModelInputs.g.h"
#include "LearningModelBuilder.h"

namespace winrt::Windows::AI::MachineLearning::More::implementation {
struct LearningModelInputs : LearningModelInputsT<LearningModelInputs> {
  LearningModelInputs(winml::More::LearningModelBuilder builder);

  more::LearningModelOperator Then(more::LearningModelOperator const& next_operator);
  more::LearningModelOperator Then(more::LearningModelOperator const& next_operator, more::LearningModelOperatorResolutionPolicy const& policy);
  more::LearningModelBuilder Add(winml::ILearningModelFeatureDescriptor const& input);
  more::LearningModelBuilder Add(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value);
  more::LearningModelBuilder AddConstant(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& value);

  more::LearningModelBuilder AddInput(winml::ILearningModelFeatureDescriptor const& input, Windows::Foundation::IInspectable const& default_value, bool is_constant);

  wfc::IVector<winml::ILearningModelFeatureDescriptor> Inputs();

 private:
  wfc::IVector<winml::ILearningModelFeatureDescriptor> input_descriptors_;
  wfc::IVector<wf::IInspectable> input_default_values_;
  wfc::IVector<winml::ILearningModelFeatureDescriptor> constant_descriptors_;
  wfc::IVector<wf::IInspectable> constant_values_;
  more::LearningModelBuilder builder_;
};
}  // namespace winrt::Windows::AI::MachineLearning::More::implementation
